"""
Production GRU Trainer for Tiny-Seq2Seq.
Optimized for 4-Layer GRU Seq2Seq model with single-tensor hidden states:
- Mixed precision AMP (FP16 autocast + GradScaler)
- Gradient clipping at norm 5.0 applied strictly after unscaling
- Cross-entropy loss normalized by non-pad target tokens
- Resilient crash recovery and state-dict checkpointing
- Early stopping on dev perplexity
- Canary sample translation logging to MLflow
- Full performance visualization generation
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from src.seq2seq.data.batcher import Batch
from src.seq2seq.gru_model import GRUSeq2SeqModel
from src.seq2seq.training.scheduler import get_warmup_cosine_scheduler
from src.seq2seq.training.tracker import MLflowTracker
from src.seq2seq.training.visualizer import TrainingVisualizer
from utils.helpers import get_device, get_gpu_memory_mb
from utils.logger import logger
from utils.metrics import ThroughputTracker, compute_perplexity


class GRUTrainer:
    """Production Trainer encapsulating training, validation, checkpointing for GRU models."""

    def __init__(
        self,
        model: GRUSeq2SeqModel,
        train_batcher: Any,
        dev_batcher: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        tracker: Optional[MLflowTracker] = None,
        checkpoint_dir: str = "checkpoints/gru_bbpe",
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

        # Training hyperparams
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

        self.scheduler = None

        # State tracking
        self.global_step = 0
        self.total_training_steps = 0
        self.start_epoch = 1
        self.best_dev_ppl = float("inf")
        self.patience_counter = 0

        # Diagnostics history for visualization
        self.history: Dict[str, List[Any]] = {
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

    def train(self, resume_path: Optional[str] = None) -> Dict[str, Any]:
        """Executes full multi-epoch training pipeline."""
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

        train_plan = self.train_batcher.plan_batches()
        steps_per_epoch = len(train_plan)
        self.total_training_steps = steps_per_epoch * self.epochs

        if self.scheduler is None:
            self.scheduler = get_warmup_cosine_scheduler(
                optimizer=self.optimizer,
                warmup_steps=self.warmup_steps,
                total_steps=self.total_training_steps,
                min_lr_ratio=self.min_lr_ratio,
            )

        logger.info(
            f"Beginning GRU training across {self.epochs} epochs: "
            f"{steps_per_epoch} batches/epoch, total steps = {self.total_training_steps}"
        )

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_loss, epoch_ppl = self._train_one_epoch(epoch)
            self.history["train_losses"].append(epoch_loss)
            self.history["train_ppls"].append(epoch_ppl)

            # Evaluate on held-out validation set
            dev_loss, dev_ppl = self.evaluate()
            self.history["dev_losses"].append(dev_loss)
            self.history["dev_ppls"].append(dev_ppl)

            logger.info(
                f"[Epoch {epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {epoch_loss:.4f} | Train PPL: {epoch_ppl:.2f} || "
                f"Dev Loss: {dev_loss:.4f} | Dev PPL: {dev_ppl:.2f}"
            )

            if self.tracker:
                self.tracker.log_metrics(
                    {
                        "train/loss": epoch_loss,
                        "train/perplexity": epoch_ppl,
                        "dev/loss": dev_loss,
                        "dev/perplexity": dev_ppl,
                        "epoch": epoch,
                    },
                    step=self.global_step,
                )

            # Qualitative canary translations
            canaries = self.generate_canary_translations(num_samples=3)
            for i, c in enumerate(canaries):
                logger.info(
                    f"Canary {i+1}: SRC: \"{c['src']}\" | REF: \"{c['ref']}\" | HYP: \"{c['hyp']}\""
                )

            # Checkpointing
            is_best = dev_ppl < self.best_dev_ppl
            if is_best:
                self.best_dev_ppl = dev_ppl
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, is_best=False)

            if self.patience_counter >= self.early_stopping_patience:
                logger.warning(
                    f"Early stopping triggered after {epoch} epochs (patience: {self.early_stopping_patience})"
                )
                break

        # Generate plots
        self._generate_visualizations()

        return {
            "best_dev_ppl": self.best_dev_ppl,
            "final_epoch": epoch,
            "global_step": self.global_step,
            "history": self.history,
        }

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Executes one epoch of training."""
        self.model.train()
        self.train_batcher.epoch = epoch
        batches_plan = self.train_batcher.plan_batches()

        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start_time = time.time()
        tracker = ThroughputTracker()

        for batch_indices in batches_plan:
            self.global_step += 1
            step_start = time.time()

            batch: Batch = self.train_batcher.collate_batch(batch_indices).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled
            ):
                logits = self.model(batch.src_ids, batch.src_lens, batch.tgt_in_ids)
                vocab_size = logits.shape[-1]
                loss = self.criterion(logits.view(-1, vocab_size), batch.tgt_out_ids.view(-1))
                norm_loss = loss / batch.num_tokens

            # Backward pass with GradScaler
            self.scaler.scale(norm_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_clip_norm).item()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            # Tracking
            batch_tokens = batch.num_tokens
            epoch_loss += loss.item()
            epoch_tokens += batch_tokens
            step_time = time.time() - step_start
            tracker.update(batch_tokens, step_time)

            if self.global_step % self.log_steps == 0 or self.global_step == 1:
                cur_lr = self.optimizer.param_groups[0]["lr"]
                step_loss = loss.item() / batch_tokens
                tps = tracker.get_rate()
                vram = get_gpu_memory_mb(self.device)

                self.history["steps"].append(self.global_step)
                self.history["step_losses"].append(step_loss)
                self.history["step_lrs"].append(cur_lr)
                self.history["step_grad_norms"].append(grad_norm)
                self.history["tokens_per_sec"].append(tps)
                self.history["vram_mb"].append(vram)

                if self.tracker:
                    self.tracker.log_metrics(
                        {
                            "train/step_loss": step_loss,
                            "train/step_ppl": compute_perplexity(step_loss),
                            "train/learning_rate": cur_lr,
                            "train/grad_norm": grad_norm,
                            "train/tokens_per_sec": tps,
                            "train/vram_allocated_mb": vram,
                        },
                        step=self.global_step,
                    )

        avg_loss = epoch_loss / max(epoch_tokens, 1)
        avg_ppl = compute_perplexity(avg_loss)
        return avg_loss, avg_ppl

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluates model performance on the validation set."""
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
        """Generates greedy Canary translations for qualitative tracking."""
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
            s_tokens = s_raw[::-1].tolist() + [3]  # EOS = 3
            src_t = torch.tensor([s_tokens], dtype=torch.long, device=self.device)
            src_l = torch.tensor([len(s_tokens)], dtype=torch.long, device=self.device)

            enc_h = self.model.encoder(src_t, src_l)
            dec_state = enc_h

            curr_t = torch.tensor([[2]], dtype=torch.long, device=self.device)  # BOS = 2
            pred_tokens = []

            for _ in range(64):
                out, dec_state = self.model.decoder.step(curr_t, dec_state)
                logits = self.model.fc_out(out.squeeze(1))
                tok = torch.argmax(logits, dim=-1).item()
                if tok in (0, 3):  # PAD or EOS
                    break
                pred_tokens.append(tok)
                curr_t = torch.tensor([[tok]], dtype=torch.long, device=self.device)

            hyp_text = self.tokenizer.decode(pred_tokens)
            samples.append({"src": s_text, "ref": t_text, "hyp": hyp_text})

        return samples

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """Saves resilient training checkpoint."""
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_dev_ppl": self.best_dev_ppl,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "history": self.history,
            "config": self.config,
        }
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        torch.save(state, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
            torch.save(state, best_path)
            logger.info(f"New best model saved to {best_path} (Dev PPL: {self.best_dev_ppl:.2f})")
            if self.tracker:
                self.tracker.log_artifact(best_path, artifact_path="checkpoints")

        return latest_path

    def load_checkpoint(self, path: str) -> None:
        """Loads model and optimizer states from checkpoint."""
        logger.info(f"Loading checkpoint from {path}...")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scaler_state_dict" in state and self.scaler:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.start_epoch = state.get("epoch", 0) + 1
        self.global_step = state.get("global_step", 0)
        self.best_dev_ppl = state.get("best_dev_ppl", float("inf"))
        self.history = state.get("history", self.history)
        logger.info(f"Resumed from epoch {self.start_epoch - 1}, global step {self.global_step}")

    def _generate_visualizations(self) -> None:
        """Renders diagnostic charts and uploads to MLflow."""
        if not self.history["train_losses"]:
            return

        p1 = self.visualizer.plot_loss_and_perplexity(
            self.history["train_losses"],
            self.history["dev_losses"],
            self.history["train_ppls"],
            self.history["dev_ppls"],
            save_name="gru_loss_perplexity.png",
        )
        if self.history["steps"]:
            p2 = self.visualizer.plot_hardware_and_throughput(
                self.history["steps"],
                self.history["tokens_per_sec"],
                self.history["vram_mb"],
                save_name="gru_hardware_throughput.png",
            )
            p3 = self.visualizer.plot_lr_schedule(
                self.history["steps"],
                self.history["step_lrs"],
                save_name="gru_lr_schedule.png",
            )
            p4 = self.visualizer.plot_gradient_norms(
                self.history["steps"],
                self.history["step_grad_norms"],
                clip_norm=self.grad_clip_norm,
                save_name="gru_gradient_norms.png",
            )
            if self.tracker:
                for p in [p1, p2, p3, p4]:
                    self.tracker.log_artifact(p, artifact_path="diagnostics")
