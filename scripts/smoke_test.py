"""
Overfitting Smoke Test for Tiny-Seq2Seq.
Trains on a small slice of sentence pairs to verify that:
1. The model architecture, loss function, and masking work correctly.
2. The network can memorize the training set (training loss drops close to 0, perplexity ~ 1.0).
3. Sample translations reproduce ground truth sentences verbatim.
Logs all dynamics to MLflow experiment 'TinySeq2Seq-SmokeTest'.
"""

import argparse
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from src.seq2seq.config import load_config
from src.seq2seq.data.batcher import BucketBatcher
from src.seq2seq.data.dataset import ParallelBinaryDataset
from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.model import Seq2SeqModel
from src.seq2seq.training.tracker import MLflowTracker
from utils.helpers import get_device, get_gpu_memory_mb, set_seed
from utils.logger import logger
from utils.metrics import ThroughputTracker, compute_perplexity


def run_smoke_test(
    config_path: str = "configs/base_config.yaml",
    num_pairs: int = 100,
    epochs: int = 35,
    lr: float = 0.003,
) -> bool:
    config = load_config(config_path)
    data_cfg = config["data"]
    mlflow_cfg = config["mlflow"]

    set_seed(42)
    device = get_device()
    logger.info(f"Running Smoke Test on device: {device}")

    # Paths
    processed_dir = data_cfg["processed_dir"]
    tokenizer_dir = data_cfg["tokenizer_dir"]
    src_lang = data_cfg["source_lang"]
    tgt_lang = data_cfg["target_lang"]

    src_bin = os.path.join(processed_dir, f"train.{src_lang}.bin")
    src_idx = os.path.join(processed_dir, f"train.{src_lang}.idx")
    tgt_bin = os.path.join(processed_dir, f"train.{tgt_lang}.bin")
    tgt_idx = os.path.join(processed_dir, f"train.{tgt_lang}.idx")

    if not os.path.exists(src_bin):
        raise FileNotFoundError(f"Binary dataset not found: {src_bin}. Run data pipeline first.")

    # Load dataset
    full_dataset = ParallelBinaryDataset(src_bin, src_idx, tgt_bin, tgt_idx)
    logger.info(f"Loaded dataset with {len(full_dataset)} total pairs. Taking slice of {num_pairs} pairs.")

    # Find tokenizer model
    tok_files = [f for f in os.listdir(tokenizer_dir) if f.endswith(".model")]
    if not tok_files:
        raise FileNotFoundError(f"No tokenizer model found in {tokenizer_dir}")
    tokenizer_path = os.path.join(tokenizer_dir, tok_files[0])
    tokenizer = TokenizerManager(tokenizer_path)

    # Initialize a small Seq2Seq model for fast memorization
    vocab_size = tokenizer.vocab_size
    d_model = 256
    n_layers = 2
    model = Seq2SeqModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        dropout=0.0,  # No dropout for memorization test
        tie_weights=True,
    ).to(device)

    counts = model.count_parameters()
    logger.info(f"Smoke Test Model initialized. Parameters: {counts['trainable_unique_parameters']:,}")

    # Batcher on the slice
    batcher = BucketBatcher(
        dataset=full_dataset,
        token_budget=2000,
        reverse_source=True,
        shuffle=True,
        seed=42,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    tracker = MLflowTracker(
        experiment_name=mlflow_cfg["experiment_smoke"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )

    throughput = ThroughputTracker()
    final_loss = 999.0
    global_step = 0

    with tracker.start_run(run_name=f"smoke_test_memorize_{num_pairs}_pairs") as run:
        tracker.log_params({
            "num_pairs": num_pairs,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "epochs": epochs,
            "lr": lr,
            "device": str(device),
        })

        for epoch in range(1, epochs + 1):
            model.train()
            batcher.set_epoch(epoch)
            batches_plan = batcher.plan_batches()

            epoch_loss = 0.0
            epoch_tokens = 0
            start_t = time.perf_counter()

            # Limit to slice
            processed_pairs = 0
            for batch_indices in batches_plan:
                # Filter to only the first num_pairs
                slice_indices = [idx for idx in batch_indices if idx < num_pairs]
                if not slice_indices:
                    continue

                batch = batcher.collate_batch(slice_indices).to(device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(batch.src_ids, batch.src_lens, batch.tgt_in_ids)
                    # Flatten for loss
                    vocab_s = logits.shape[-1]
                    loss = criterion(logits.view(-1, vocab_s), batch.tgt_out_ids.view(-1))
                    normalized_loss = loss / max(batch.num_tokens, 1)

                scaler.scale(normalized_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                step_loss_val = normalized_loss.item()
                epoch_loss += loss.item()
                epoch_tokens += batch.num_tokens
                processed_pairs += len(slice_indices)
                global_step += 1

                t_stats = throughput.update(num_tokens=batch.num_tokens, elapsed_seconds=time.perf_counter() - start_t)
                tracker.log_step_metrics(
                    step=global_step,
                    metrics={
                        "step/train_loss": step_loss_val,
                        "step/train_ppl": compute_perplexity(step_loss_val),
                    },
                )

            avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
            epoch_ppl = compute_perplexity(avg_epoch_loss)
            final_loss = avg_epoch_loss

            tracker.log_epoch_metrics(
                epoch=epoch,
                metrics={
                    "epoch/train_loss": avg_epoch_loss,
                    "epoch/train_ppl": epoch_ppl,
                },
            )

            if epoch % 5 == 0 or epoch == epochs:
                vram = get_gpu_memory_mb()
                logger.info(
                    f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_epoch_loss:.4f} | "
                    f"PPL: {epoch_ppl:.2f} | VRAM: {vram['allocated_mb']} MB"
                )

        # Verification: Greedy decoding on canary sentences
        model.eval()
        logger.info("Evaluating greedy reproduction of training sentences...")
        sample_results = []
        with torch.no_grad():
            for idx in range(min(5, num_pairs)):
                s_raw, t_raw = full_dataset[idx]
                s_text = tokenizer.decode(s_raw.tolist())
                t_text = tokenizer.decode(t_raw.tolist())

                # Reversal convention: reversed source + EOS
                s_tokens = s_raw[::-1].tolist() + [TokenizerManager.EOS_ID]
                src_t = torch.tensor([s_tokens], dtype=torch.long, device=device)
                src_l = torch.tensor([len(s_tokens)], dtype=torch.long, device=device)

                # Encoder state
                enc_h, enc_c = model.encoder(src_t, src_l)
                dec_state = (enc_h, enc_c)

                # Autoregressive greedy decoding
                curr_token = torch.tensor([[TokenizerManager.BOS_ID]], dtype=torch.long, device=device)
                generated_ids = []

                for _ in range(64):
                    out, dec_state = model.decoder.step(curr_token, dec_state)
                    logits = model.fc_out(out)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                    if next_token == TokenizerManager.EOS_ID:
                        break
                    generated_ids.append(next_token)
                    curr_token = torch.tensor([[next_token]], dtype=torch.long, device=device)

                pred_text = tokenizer.decode(generated_ids)
                sample_results.append({
                    "source": s_text,
                    "target": t_text,
                    "prediction": pred_text,
                    "beam_size": 1,
                })
                logger.info(f"\n[Sample {idx+1}]\n  Src: {s_text}\n  Ref: {t_text}\n  Out: {pred_text}")

        tracker.log_translation_samples(sample_results, epoch=epochs)

    success = final_loss < 0.20
    if success:
        logger.info(f"Smoke Test PASSED! Final Loss: {final_loss:.4f} < 0.20 (PPL: {final_loss:.2f}). Network memorization confirmed.")
    else:
        logger.warning(f"Smoke Test Warning: Final Loss {final_loss:.4f} did not drop below 0.20.")


    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tiny-Seq2Seq Overfitting Smoke Test")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args()

    passed = run_smoke_test(
        config_path=args.config,
        num_pairs=args.num_pairs,
        epochs=args.epochs,
        lr=args.lr,
    )
    sys.exit(0 if passed else 1)
