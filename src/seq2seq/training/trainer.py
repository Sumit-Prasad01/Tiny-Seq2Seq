"""
Production Trainer for Tiny-Seq2Seq.
Implements:
- AMP Mixed Precision with GradScaler
- Gradient clipping at norm 5.0 applied strictly after unscaling
- Cross-entropy loss normalized by non-pad target tokens
- Resilient crash recovery and state-dict checkpointing
- Early stopping on dev perplexity
- Canary sample translation logging to MLflow
- Full performance visualization generation (loss, perplexity, VRAM, throughput, grad norm)
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from src.seq2seq.data.batcher import Batch, BucketBatcher
from src.seq2seq.data.c_batcher import CppBucketBatcher
from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.model import Seq2SeqModel
from src.seq2seq.training.scheduler import get_warmup_cosine_scheduler
from src.seq2seq.training.tracker import MLflowTracker
from src.seq2seq.training.visualizer import TrainingVisualizer
from utils.helpers import get_device, get_gpu_memory_mb
from utils.logger import logger
from utils.metrics import ThroughputTracker, compute_perplexity


class Seq2SeqTrainer:
    """Production Trainer encapsulating training, validation, checkpointing, and visualization."""

    def __init__(
        self,
        model: Seq2SeqModel,
        train_batcher: Any,
        dev_batcher: Optional[Any] = None,
        tokenizer: Optional[TokenizerManager] = None,
        config: Optional[Dict[str, Any]] = None,
        tracker: Optional[MLflowTracker] = None,
        checkpoint_dir: str = "checkpoints",
        plots_dir: str = "artifacts/plots",
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_batcher = train_batcher
        self.dev_batcher = dev_batcher
        self.tokenizer = tokenizer
        self.config = config or {}
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.tracker = tracker
        self.visualizer = TrainingVisualizer(output_dir=plots_dir)

        # Training configs
        train_cfg = self.config.get("training", {})
        self.lr = train_cfg.get("learning_rate", 1e-3)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 5.0)
        self.amp_enabled = train_cfg.get("amp_enabled", True) and (self.device.type == "cuda")
        self.epochs = train_cfg.get("epochs", 10)
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 2)
        self.log_steps = train_cfg.get("log_steps", 50)
        self.warmup_steps = train_cfg.get("warmup_steps", 1000)
        self.min_lr_ratio = train_cfg.get("min_lr_ratio", 0.05)

        # Optimization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

        # Scheduler is initialized in train() once total steps are known
        self.scheduler = None

        # State tracking
        self.global_step = 0
        self.start_epoch = 1
        self.best_dev_ppl = float("inf")
        self.patience_counter = 0

        # Diagnostics history for visualization
        self.history = {
            "steps": [],
            "step_losses": [],
            "step_lrs": [],
            "step_grad_norms": [],
            "tokens_per_sec": [],
            "vram_mb": [],
            "train_losses": [],
            "dev_losses": [],
            "train_ppls": [],
            "dev_ppls": [],
        }

    def train_epoch(self, epoch: int, throughput: ThroughputTracker) -> Tuple[float, float]:
        """Runs one full training epoch."""
        self.model.train()
        self.train_batcher.set_epoch(epoch)
        batches_plan = self.train_batcher.plan_batches()

        epoch_loss = 0.0
        epoch_tokens = 0
        step_start_time = time.perf_counter()

        for batch_indices in batches_plan:
            batch: Batch = self.train_batcher.collate_batch(batch_indices).to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled
            ):
                logits = self.model(batch.src_ids, batch.src_lens, batch.tgt_in_ids)
                vocab_size = logits.shape[-1]
                loss = self.criterion(logits.view(-1, vocab_size), batch.tgt_out_ids.view(-1))
                normalized_loss = loss / max(batch.num_tokens, 1)

            # Backward pass with scaler
            self.scaler.scale(normalized_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm).item()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            step_loss_val = normalized_loss.item()
            epoch_loss += loss.item()
            epoch_tokens += batch.num_tokens

            elapsed = time.perf_counter() - step_start_time
            step_start_time = time.perf_counter()
            t_stats = throughput.update(num_tokens=batch.num_tokens, elapsed_seconds=elapsed)

            current_lr = self.optimizer.param_groups[0]["lr"]
            vram_info = get_gpu_memory_mb()

            # Record history
            self.history["steps"].append(self.global_step)
            self.history["step_losses"].append(step_loss_val)
            self.history["step_lrs"].append(current_lr)
            self.history["step_grad_norms"].append(grad_norm)
            self.history["tokens_per_sec"].append(t_stats["tokens_per_second"])
            self.history["vram_mb"].append(vram_info["allocated_mb"])

            # Step logging
            if self.global_step % self.log_steps == 0 or self.global_step == 1:
                step_ppl = compute_perplexity(step_loss_val)
                if self.tracker is not None:
                    self.tracker.log_step_metrics(
                        step=self.global_step,
                        metrics={
                            "step/train_loss": step_loss_val,
                            "step/train_ppl": step_ppl,
                            "step/lr": current_lr,
                            "step/grad_norm": grad_norm,
                            "step/tokens_per_sec": t_stats["tokens_per_second"],
                            "step/vram_allocated_mb": vram_info["allocated_mb"],
                        },
                    )

        avg_loss = epoch_loss / max(epoch_tokens, 1)
        avg_ppl = compute_perplexity(avg_loss)
        return avg_loss, avg_ppl

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluates model performance on the held-out validation set."""
        if self.dev_batcher is None:
            return 0.0, 0.0

        self.model.eval()
        dev_batches_plan = self.dev_batcher.plan_batches()
        total_loss = 0.0
        total_tokens = 0

        for batch_indices in dev_batches_plan:
            batch: Batch = self.dev_batcher.collate_batch(batch_indices).to(self.device)
            with torch.amp.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled
            ):
                logits = self.model(batch.src_ids, batch.src_lens, batch.tgt_in_ids)
                vocab_size = logits.shape[-1]
                loss = self.criterion(logits.view(-1, vocab_size), batch.tgt_out_ids.view(-1))

            total_loss += loss.item()
            total_tokens += batch.num_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = compute_perplexity(avg_loss)
        return avg_loss, ppl

    @torch.no_grad()
    def generate_canary_translations(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Generates greedy Canary sample translations from the dev set for qualitative tracking."""
        if self.dev_batcher is None or self.tokenizer is None:
            return []

        self.model.eval()
        samples = []
        dataset = self.dev_batcher.dataset
        limit = min(num_samples, len(dataset))

        for idx in range(limit):
            s_raw, t_raw = dataset[idx]
            s_text = self.tokenizer.decode(s_raw.tolist())
            t_text = self.tokenizer.decode(t_raw.tolist())

            # Source Reversal: reversed tokens + EOS
            s_tokens = s_raw[::-1].tolist() + [TokenizerManager.EOS_ID]
            src_t = torch.tensor([s_tokens], dtype=torch.long, device=self.device)
            src_l = torch.tensor([len(s_tokens)], dtype=torch.long, device=self.device)

            enc_h, enc_c = self.model.encoder(src_t, src_l)
            dec_state = (enc_h, enc_c)
            curr_token = torch.tensor([[TokenizerManager.BOS_ID]], dtype=torch.long, device=self.device)
            generated_ids = []

            for _ in range(64):
                out, dec_state = self.model.decoder.step(curr_token, dec_state)
                logits = self.model.fc_out(out)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                if next_token == TokenizerManager.EOS_ID:
                    break
                generated_ids.append(next_token)
                curr_token = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

            pred_text = self.tokenizer.decode(generated_ids)
            samples.append({
                "source": s_text,
                "target": t_text,
                "prediction": pred_text,
                "beam_size": 1,
            })

        return samples

    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """Saves resilient checkpoint containing full model, optimizer, and training history."""
        checkpoint = {
            "epoch": self.start_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_dev_ppl": self.best_dev_ppl,
            "history": self.history,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        if is_best and self.tracker is not None:
            self.tracker.log_artifact(path, artifact_path="checkpoints")
        logger.info(f"Saved checkpoint to {path} (Best: {is_best})")

    def load_checkpoint(self, path: str) -> None:
        """Restores model, optimizer, scheduler, and step history from checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        logger.info(f"Loading checkpoint from {path}...")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scaler and ckpt.get("scaler_state_dict"):
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_dev_ppl = ckpt.get("best_dev_ppl", float("inf"))
        self.history = ckpt.get("history", self.history)
        logger.info(f"Resumed from checkpoint: next epoch {self.start_epoch}, step {self.global_step}")

    def train(self, resume_path: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop across all epochs with early stopping and visualization."""
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

        batches_per_epoch = len(self.train_batcher.plan_batches())
        total_training_steps = batches_per_epoch * self.epochs
        logger.info(
            f"Starting training: {self.epochs} epochs, {batches_per_epoch} batches/epoch, "
            f"~{total_training_steps} total steps."
        )

        if self.scheduler is None:
            self.scheduler = get_warmup_cosine_scheduler(
                self.optimizer,
                warmup_steps=self.warmup_steps,
                total_steps=total_training_steps,
                min_lr_ratio=self.min_lr_ratio,
            )

        throughput = ThroughputTracker()

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_ppl = self.train_epoch(epoch, throughput)
            dev_loss, dev_ppl = self.evaluate()
            epoch_duration = time.perf_counter() - epoch_start

            # Record history
            self.history["train_losses"].append(train_loss)
            self.history["dev_losses"].append(dev_loss)
            self.history["train_ppls"].append(train_ppl)
            self.history["dev_ppls"].append(dev_ppl)

            # Check best model
            is_best = dev_ppl < self.best_dev_ppl
            if is_best:
                self.best_dev_ppl = dev_ppl
                self.patience_counter = 0
                best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
                self.save_checkpoint(best_path, is_best=True)
            else:
                self.patience_counter += 1

            # Save latest checkpoint
            latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
            self.start_epoch = epoch
            self.save_checkpoint(latest_path, is_best=False)

            vram = get_gpu_memory_mb()
            logger.info(
                f"Epoch {epoch:02d}/{self.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
                f"Dev Loss: {dev_loss:.4f} (PPL: {dev_ppl:.2f}) | "
                f"Time: {epoch_duration:.1f}s | VRAM Peak: {vram['peak_mb']} MB"
            )

            # Generate Canary translations and log to MLflow
            canary_samples = self.generate_canary_translations(num_samples=5)
            if self.tracker is not None:
                self.tracker.log_epoch_metrics(
                    epoch=epoch,
                    metrics={
                        "epoch/train_loss": train_loss,
                        "epoch/train_ppl": train_ppl,
                        "epoch/dev_loss": dev_loss,
                        "epoch/dev_ppl": dev_ppl,
                        "epoch/duration_sec": epoch_duration,
                        "epoch/peak_vram_mb": vram["peak_mb"],
                    },
                )
                self.tracker.log_translation_samples(canary_samples, epoch=epoch)

            # Early stopping check
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (No improvement for {self.early_stopping_patience} epochs).")
                break

        # Generate final visualizations
        logger.info("Generating publication-quality training diagnostic plots...")
        plots = self.generate_visualizations()

        return {
            "best_dev_ppl": self.best_dev_ppl,
            "final_epoch": self.start_epoch,
            "plots": plots,
        }

    def generate_visualizations(self) -> List[str]:
        """Generates all training visual plots and uploads them to MLflow."""
        generated = []
        if self.history["train_losses"] and self.history["dev_losses"]:
            p1 = self.visualizer.plot_loss_and_perplexity(
                self.history["train_losses"],
                self.history["dev_losses"],
                self.history["train_ppls"],
                self.history["dev_ppls"],
            )
            generated.append(p1)

        if self.history["steps"] and self.history["tokens_per_sec"]:
            p2 = self.visualizer.plot_hardware_and_throughput(
                self.history["steps"],
                self.history["tokens_per_sec"],
                self.history["vram_mb"],
            )
            generated.append(p2)

        if self.history["steps"] and self.history["step_lrs"]:
            p3 = self.visualizer.plot_lr_schedule(
                self.history["steps"],
                self.history["step_lrs"],
            )
            generated.append(p3)

        if self.history["steps"] and self.history["step_grad_norms"]:
            p4 = self.visualizer.plot_gradient_norms(
                self.history["steps"],
                self.history["step_grad_norms"],
                clip_norm=self.grad_clip_norm,
            )
            generated.append(p4)

        if self.tracker is not None:
            for plot_path in generated:
                self.tracker.log_artifact(plot_path, artifact_path="plots")

        logger.info(f"Generated {len(generated)} visualization plots in {self.visualizer.output_dir}.")
        return generated
