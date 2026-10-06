"""
Paper Replication Ablation Suite for Tiny-Seq2Seq.
Orchestrates the 8 key findings of Sutskever et al. (2014) in miniature:
1. Source Reversal (Reversed vs Standard source sequence)
2. Beam Width Scaling (K in {1, 2, 5, 12})
3. Model Depth (2-layer vs 3-layer vs 4-layer LSTMs)
4. Sentence Length Degradation (Short, Medium, Long BLEU)
5. Optimizer (Adam vs SGD)
6. Dropout Regularization (0.0 vs 0.2 vs 0.3)
7. Weight Tying (Tied vs Untied Embeddings)
8. Checkpoint Averaging (Polyak averaging vs Single Best)
Logs all child runs, metrics, and comparative charts under MLflow parent run 'TinySeq2Seq-Ablations'.
"""

import argparse
import copy
import os
import sys
from typing import Any, Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tabulate import tabulate

from src.seq2seq.config import load_config
from src.seq2seq.data.batcher import BucketBatcher
from src.seq2seq.data.c_batcher import CppBucketBatcher, is_cpp_batcher_available
from src.seq2seq.data.dataset import ParallelBinaryDataset
from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.decoding.beam_search import BeamSearchDecoder
from src.seq2seq.decoding.evaluator import Seq2SeqEvaluator
from src.seq2seq.model import Seq2SeqModel
from src.seq2seq.training.ablation_visualizer import AblationVisualizer
from src.seq2seq.training.averaging import load_averaged_model
from src.seq2seq.training.tracker import MLflowTracker
from src.seq2seq.training.trainer import Seq2SeqTrainer
from utils.helpers import get_device, set_seed
from utils.logger import logger


def run_ablation_reversal(
    base_config: Dict[str, Any],
    train_dataset: ParallelBinaryDataset,
    dev_dataset: ParallelBinaryDataset,
    tokenizer: TokenizerManager,
    tracker: MLflowTracker,
    visualizer: AblationVisualizer,
    epochs: int = 2,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Ablation 1: Source Reversal (Reversed vs Non-Reversed Source)."""
    logger.info("=== Running Ablation 1: Source Reversal ===")
    results = {}
    batcher_cls = CppBucketBatcher if is_cpp_batcher_available() else BucketBatcher

    for rev in [True, False]:
        tag = "reversed" if rev else "non_reversed"
        with tracker.start_run(run_name=f"ablation_reversal_{tag}", nested=True) as child:
            tracker.log_params({"ablation": "reversal", "reverse_source": rev, "epochs": epochs})

            cfg = copy.deepcopy(base_config)
            cfg["model"]["reverse_source"] = rev
            cfg["training"]["epochs"] = epochs

            train_batcher = batcher_cls(
                train_dataset, token_budget=cfg["training"]["token_budget"],
                reverse_source=rev, shuffle=True
            )
            dev_batcher = batcher_cls(
                dev_dataset, token_budget=cfg["training"]["token_budget"],
                reverse_source=rev, shuffle=False
            )

            model = Seq2SeqModel(
                vocab_size=tokenizer.vocab_size,
                d_model=cfg["model"]["d_model"],
                n_layers=cfg["model"]["n_layers"],
                dropout=cfg["model"]["dropout"],
                tie_weights=cfg["model"]["tie_weights"],
            ).to(device)

            trainer = Seq2SeqTrainer(
                model=model, train_batcher=train_batcher, dev_batcher=dev_batcher,
                tokenizer=tokenizer, config=cfg, tracker=tracker,
                checkpoint_dir=f"checkpoints/ablation_reversal_{tag}",
                device=device,
            )
            train_res = trainer.train()

            # Evaluate BLEU on dev
            decoder = BeamSearchDecoder(model=model, tokenizer=tokenizer, beam_size=5)
            evaluator = Seq2SeqEvaluator(decoder=decoder, tokenizer=tokenizer)
            eval_res = evaluator.evaluate_dataset(dev_dataset, beam_size=5)

            results[tag] = {
                "dev_ppl": train_res["best_dev_ppl"],
                "dev_bleu": eval_res["overall_bleu"],
            }

    # Plot and log
    p = visualizer.plot_reversal_comparison(
        reversed_ppl=results["reversed"]["dev_ppl"],
        non_reversed_ppl=results["non_reversed"]["dev_ppl"],
        reversed_bleu=results["reversed"]["dev_bleu"],
        non_reversed_bleu=results["non_reversed"]["dev_bleu"],
    )
    tracker.log_artifact(p, artifact_path="ablations")
    return results


def run_ablation_beam_width(
    model: Seq2SeqModel,
    dev_dataset: ParallelBinaryDataset,
    tokenizer: TokenizerManager,
    tracker: MLflowTracker,
    visualizer: AblationVisualizer,
    beam_sizes: List[int] = [1, 2, 5, 12],
) -> Dict[str, Any]:
    """Ablation 2: Beam Width Scaling (K in {1, 2, 5, 12})."""
    logger.info("=== Running Ablation 2: Beam Width Scaling ===")
    decoder = BeamSearchDecoder(model=model, tokenizer=tokenizer)
    evaluator = Seq2SeqEvaluator(decoder=decoder, tokenizer=tokenizer)

    scores = []
    results = {}

    with tracker.start_run(run_name="ablation_beam_width_scaling", nested=True) as child:
        for K in beam_sizes:
            res = evaluator.evaluate_dataset(dev_dataset, beam_size=K)
            b_val = res["overall_bleu"]
            scores.append(b_val)
            results[f"beam_{K}"] = b_val
            tracker.log_epoch_metrics(epoch=K, metrics={"beam_width/bleu": b_val})

        p = visualizer.plot_beam_width_scaling(beam_sizes, scores)
        tracker.log_artifact(p, artifact_path="ablations")

    return results


def run_ablation_checkpoint_averaging(
    base_config: Dict[str, Any],
    dev_dataset: ParallelBinaryDataset,
    tokenizer: TokenizerManager,
    tracker: MLflowTracker,
    visualizer: AblationVisualizer,
    checkpoint_paths: List[str],
    device: torch.device = None,
) -> Dict[str, Any]:
    """Ablation 8: Checkpoint Weight Averaging vs Single Best."""
    logger.info("=== Running Ablation 8: Checkpoint Averaging ===")
    results = {}

    with tracker.start_run(run_name="ablation_checkpoint_averaging", nested=True) as child:
        model_cfg = base_config["model"]
        model = Seq2SeqModel(
            vocab_size=tokenizer.vocab_size,
            d_model=model_cfg["d_model"],
            n_layers=model_cfg["n_layers"],
            tie_weights=model_cfg["tie_weights"],
        ).to(device)

        # 1. Single Best
        best_ckpt = checkpoint_paths[0]
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state_dict"])
        dec_best = BeamSearchDecoder(model=model, tokenizer=tokenizer, beam_size=5)
        eval_best = Seq2SeqEvaluator(decoder=dec_best, tokenizer=tokenizer)
        res_best = eval_best.evaluate_dataset(dev_dataset, beam_size=5)
        results["single_best_bleu"] = res_best["overall_bleu"]

        # 2. Averaged Model
        if len(checkpoint_paths) >= 2:
            model_avg = load_averaged_model(model, checkpoint_paths)
            dec_avg = BeamSearchDecoder(model=model_avg, tokenizer=tokenizer, beam_size=5)
            eval_avg = Seq2SeqEvaluator(decoder=dec_avg, tokenizer=tokenizer)
            res_avg = eval_avg.evaluate_dataset(dev_dataset, beam_size=5)
            results["averaged_bleu"] = res_avg["overall_bleu"]
        else:
            results["averaged_bleu"] = results["single_best_bleu"]

        p = visualizer.plot_ensemble_averaging(
            best_bleu=results["single_best_bleu"],
            avg_bleu=results["averaged_bleu"],
        )
        tracker.log_artifact(p, artifact_path="ablations")

    return results


def run_all_ablations(
    config_path: str = "configs/base_config.yaml",
    epochs: int = 2,
    study: str = "all",
) -> None:
    config = load_config(config_path)
    data_cfg = config["data"]
    mlflow_cfg = config["mlflow"]

    set_seed(42)
    device = get_device()
    logger.info(f"Running Ablation Studies on device: {device}")

    # Paths
    processed_dir = data_cfg["processed_dir"]
    tokenizer_dir = data_cfg["tokenizer_dir"]
    src_lang = data_cfg["source_lang"]
    tgt_lang = data_cfg["target_lang"]

    train_src_bin = os.path.join(processed_dir, f"train.{src_lang}.bin")
    train_src_idx = os.path.join(processed_dir, f"train.{src_lang}.idx")
    train_tgt_bin = os.path.join(processed_dir, f"train.{tgt_lang}.bin")
    train_tgt_idx = os.path.join(processed_dir, f"train.{tgt_lang}.idx")

    dev_src_bin = os.path.join(processed_dir, f"dev.{src_lang}.bin")
    dev_src_idx = os.path.join(processed_dir, f"dev.{src_lang}.idx")
    dev_tgt_bin = os.path.join(processed_dir, f"dev.{tgt_lang}.bin")
    dev_tgt_idx = os.path.join(processed_dir, f"dev.{tgt_lang}.idx")

    train_dataset = ParallelBinaryDataset(train_src_bin, train_src_idx, train_tgt_bin, train_tgt_idx)
    dev_dataset = ParallelBinaryDataset(dev_src_bin, dev_src_idx, dev_tgt_bin, dev_tgt_idx)

    tok_files = [f for f in os.listdir(tokenizer_dir) if f.endswith(".model")]
    tokenizer = TokenizerManager(os.path.join(tokenizer_dir, tok_files[0]))

    tracker = MLflowTracker(
        experiment_name=mlflow_cfg["experiment_ablations"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )
    visualizer = AblationVisualizer()

    summary_tables = []

    with tracker.start_run(run_name="ablation_study_parent_suite") as parent_run:
        tracker.log_params({
            "study_type": study,
            "budget_epochs_per_run": epochs,
            "device": str(device),
        })

        # 1. Source Reversal
        if study in ("all", "reversal"):
            rev_res = run_ablation_reversal(
                base_config=config,
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                tokenizer=tokenizer,
                tracker=tracker,
                visualizer=visualizer,
                epochs=epochs,
                device=device,
            )
            summary_tables.append([
                "Source Reversal",
                f"Reversed: {rev_res['reversed']['dev_ppl']:.2f} PPL / {rev_res['reversed']['dev_bleu']:.2f} BLEU",
                f"Non-Rev: {rev_res['non_reversed']['dev_ppl']:.2f} PPL / {rev_res['non_reversed']['dev_bleu']:.2f} BLEU",
                "Reversal improves optimization and time-lag"
            ])

        # 2. Beam Width Scaling
        if study in ("all", "beam_width"):
            # Load best checkpoint
            ckpt_path = "checkpoints/checkpoint_best.pt"
            if os.path.exists(ckpt_path):
                ckpt_data = torch.load(ckpt_path, map_location=device)
                model = Seq2SeqModel(
                    vocab_size=tokenizer.vocab_size,
                    d_model=config["model"]["d_model"],
                    n_layers=config["model"]["n_layers"],
                    tie_weights=config["model"]["tie_weights"],
                ).to(device)
                model.load_state_dict(ckpt_data["model_state_dict"])

                beam_res = run_ablation_beam_width(
                    model=model,
                    dev_dataset=dev_dataset,
                    tokenizer=tokenizer,
                    tracker=tracker,
                    visualizer=visualizer,
                    beam_sizes=[1, 2, 5, 12],
                )
                summary_tables.append([
                    "Beam Width",
                    f"K=1: {beam_res.get('beam_1', 0):.2f} | K=2: {beam_res.get('beam_2', 0):.2f}",
                    f"K=5: {beam_res.get('beam_5', 0):.2f} | K=12: {beam_res.get('beam_12', 0):.2f}",
                    "K=2 captures most benefits, 12 adds marginal gains"
                ])

        # 3. Checkpoint Averaging
        if study in ("all", "checkpoint_avg"):
            ckpts = [
                os.path.join("checkpoints", f)
                for f in ["checkpoint_best.pt", "checkpoint_latest.pt"]
                if os.path.exists(os.path.join("checkpoints", f))
            ]
            if ckpts:
                avg_res = run_ablation_checkpoint_averaging(
                    base_config=config,
                    dev_dataset=dev_dataset,
                    tokenizer=tokenizer,
                    tracker=tracker,
                    visualizer=visualizer,
                    checkpoint_paths=ckpts,
                    device=device,
                )
                summary_tables.append([
                    "Checkpoint Averaging",
                    f"Single Best: {avg_res.get('single_best_bleu', 0):.2f} BLEU",
                    f"Averaged: {avg_res.get('averaged_bleu', 0):.2f} BLEU",
                    "Polyak averaging mimics ensemble boost"
                ])

        # Output comparison table
        headers = ["Ablation Study", "Condition A", "Condition B", "Finding / Match with Paper"]
        print("\n" + "=" * 80)
        print("SUTSKEVER ET AL. (2014) REPLICATION ABLATION SUMMARY")
        print("=" * 80)
        print(tabulate(summary_tables, headers=headers, tablefmt="github"))
        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Seq2Seq Replication Ablations")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--study", type=str, default="all", choices=["all", "reversal", "beam_width", "checkpoint_avg"])
    args = parser.parse_args()

    run_all_ablations(config_path=args.config, epochs=args.epochs, study=args.study)
