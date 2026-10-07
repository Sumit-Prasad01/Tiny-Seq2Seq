"""
Data Preparation Script for Tiny-Seq2Seq.
Downloads/samples parallel text to ~250 MB, cleans and deduplicates,
applies length ratio filtering, and creates train/dev/test partitions.
Logs corpus statistics to MLflow.
"""

import argparse
import os
import random
import sys
from typing import List, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.seq2seq.config import load_config

from src.seq2seq.data.preprocessor import clean_sentence_pair
from src.seq2seq.training.tracker import MLflowTracker
from utils.logger import logger


def generate_synthetic_corpus(num_pairs: int = 10000) -> List[Tuple[str, str]]:
    """Generates synthetic En-Fr sentence pairs for offline testing and verification."""
    templates = [
        ("The quick brown fox jumps over the lazy dog.", "Le renard brun rapide saute par-dessus le chien paresseux."),
        ("Sequence to sequence learning with neural networks is powerful.", "L'apprentissage séquence à séquence avec des réseaux de neurones est puissant."),
        ("Machine translation systems have improved significantly.", "Les systèmes de traduction automatique se sont considérablement améliorés."),
        ("Deep learning models require careful hyperparameter tuning.", "Les modèles d'apprentissage profond nécessitent un réglage minutieux des hyperparamètres."),
        ("We evaluate the performance using SacreBLEU metrics.", "Nous évaluons les performances à l'aide des métriques SacreBLEU."),
        ("The encoder reads the source sequence and constructs representations.", "L'encodeur lit la séquence source et construit des représentations."),
        ("The decoder predicts the target sequence one token at a time.", "Le décodeur prédit la séquence cible un jeton à la fois."),
        ("Attention mechanisms allow models to focus on relevant context.", "Les mécanismes d'attention permettent aux modèles de se concentrer sur le contexte pertinent."),
        ("Beam search decoding maintains the top hypotheses at each step.", "Le décodage par recherche en faisceau maintient les meilleures hypothèses à chaque étape."),
        ("Natural language processing continues to advance rapidly.", "Le traitement automatique du langage naturel continue de progresser rapidement."),
    ]
    pairs = []
    for i in range(num_pairs):
        base_src, base_tgt = templates[i % len(templates)]
        if i >= len(templates):
            pairs.append((f"[{i}] {base_src}", f"[{i}] {base_tgt}"))
        else:
            pairs.append((base_src, base_tgt))
    return pairs


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def prepare_data(
    config_path: str = "configs/base_config.yaml",
    dataset_name: str = "opus-100",
    use_synthetic: bool = False,
    max_pairs: int = None,
) -> None:
    config = load_config(config_path)
    data_cfg = config["data"]
    mlflow_cfg = config["mlflow"]

    cleaned_dir = data_cfg["cleaned_dir"]
    os.makedirs(cleaned_dir, exist_ok=True)

    src_lang = data_cfg["source_lang"]
    tgt_lang = data_cfg["target_lang"]
    max_tokens = data_cfg["max_seq_len"]
    max_ratio = data_cfg["max_length_ratio"]
    dev_size = data_cfg["dev_size"]
    test_size = data_cfg["test_size"]
    target_bytes = data_cfg["target_bytes"]

    logger.info("Gathering parallel sentence pairs...")
    raw_pairs: List[Tuple[str, str]] = []

    if use_synthetic or dataset_name == "synthetic":
        logger.info("Generating synthetic En-Fr corpus for verification...")
        count = max_pairs or 15000
        raw_pairs = generate_synthetic_corpus(num_pairs=count)
    else:
        # Priority list of datasets to attempt
        candidates = []
        if dataset_name == "opus-100":
            candidates = [("Helsinki-NLP/opus-100", "en-fr"), ("Helsinki-NLP/opus_books", "en-fr")]
        elif dataset_name == "opus_books":
            candidates = [("Helsinki-NLP/opus_books", "en-fr"), ("Helsinki-NLP/opus-100", "en-fr")]
        else:
            candidates = [(dataset_name, "en-fr")]

        loaded = False
        for repo_id, sub_config in candidates:
            try:
                from datasets import load_dataset
                logger.info(f"Attempting download/load of parallel corpus from Hugging Face: '{repo_id}' ({sub_config})...")
                ds = load_dataset(repo_id, sub_config, split="train")
                logger.info(f"Successfully loaded '{repo_id}'. Extracting sentence pairs...")
                for item in ds:
                    trans = item["translation"]
                    if src_lang in trans and tgt_lang in trans:
                        raw_pairs.append((trans[src_lang], trans[tgt_lang]))
                    if max_pairs and len(raw_pairs) >= max_pairs:
                        break
                logger.info(f"Gathered {len(raw_pairs)} pairs from '{repo_id}'.")
                loaded = True
                break
            except Exception as e:
                logger.warning(f"Could not load '{repo_id}' ({e}). Trying next fallback...")

        if not loaded or len(raw_pairs) == 0:
            logger.warning("All online dataset downloads failed. Falling back to synthetic verification corpus.")
            raw_pairs = generate_synthetic_corpus(num_pairs=max_pairs or 15000)

    # Random shuffle before slicing to prevent domain concentration
    random.seed(config["project"]["seed"])
    random.shuffle(raw_pairs)

    # Clean and enforce length/ratio filtering
    logger.info("Cleaning and filtering sentence pairs...")
    cleaned_pairs: List[Tuple[str, str]] = []
    seen_pairs = set()
    total_bytes = 0

    for src, tgt in raw_pairs:
        pair = clean_sentence_pair(
            src, tgt, min_tokens=1, max_tokens=max_tokens, max_length_ratio=max_ratio
        )
        if pair is None:
            continue

        clean_src, clean_tgt = pair
        pair_key = (clean_src, clean_tgt)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        cleaned_pairs.append(pair)
        pair_bytes = len(clean_src.encode("utf-8")) + len(clean_tgt.encode("utf-8"))
        total_bytes += pair_bytes

        if total_bytes >= target_bytes:
            logger.info(f"Reached target size of {total_bytes / (1024**2):.2f} MB")
            break

        if max_pairs and len(cleaned_pairs) >= max_pairs:
            break

    logger.info(f"Total valid cleaned pairs: {len(cleaned_pairs)} ({total_bytes / (1024**2):.2f} MB)")

    # Partition dataset into train, dev, test
    actual_dev_size = min(dev_size, max(int(len(cleaned_pairs) * 0.1), 100))
    actual_test_size = min(test_size, max(int(len(cleaned_pairs) * 0.1), 100))
    actual_train_size = len(cleaned_pairs) - actual_dev_size - actual_test_size

    if actual_train_size <= 0:
        actual_dev_size = int(len(cleaned_pairs) * 0.1)
        actual_test_size = int(len(cleaned_pairs) * 0.1)
        actual_train_size = len(cleaned_pairs) - actual_dev_size - actual_test_size

    train_pairs = cleaned_pairs[:actual_train_size]
    dev_pairs = cleaned_pairs[actual_train_size : actual_train_size + actual_dev_size]
    test_pairs = cleaned_pairs[actual_train_size + actual_dev_size :]

    splits = {
        "train": train_pairs,
        "dev": dev_pairs,
        "test": test_pairs,
    }

    for split_name, pairs in splits.items():
        src_file = os.path.join(cleaned_dir, f"{split_name}.{src_lang}")
        tgt_file = os.path.join(cleaned_dir, f"{split_name}.{tgt_lang}")

        with open(src_file, "w", encoding="utf-8") as f_src, open(tgt_file, "w", encoding="utf-8") as f_tgt:
            for s, t in pairs:
                f_src.write(f"{s}\n")
                f_tgt.write(f"{t}\n")

        logger.info(f"Saved {split_name} split ({len(pairs)} pairs) to {src_file} and {tgt_file}")

    # Log to MLflow
    tracker = MLflowTracker(
        experiment_name=mlflow_cfg["experiment_data"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )
    with tracker.start_run(run_name="data_preparation") as run:
        tracker.log_params({
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "max_seq_len": max_tokens,
            "max_length_ratio": max_ratio,
            "total_cleaned_pairs": len(cleaned_pairs),
            "train_pairs": len(train_pairs),
            "dev_pairs": len(dev_pairs),
            "test_pairs": len(test_pairs),
            "total_megabytes": round(total_bytes / (1024 ** 2), 2),
        })
        tracker.log_epoch_metrics(epoch=1, metrics={
            "data/total_pairs": len(cleaned_pairs),
            "data/train_pairs": len(train_pairs),
            "data/dev_pairs": len(dev_pairs),
            "data/test_pairs": len(test_pairs),
        })

    logger.info("Data preparation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare parallel text data for Tiny-Seq2Seq")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config")
    parser.add_argument("--dataset", type=str, default="opus-100", choices=["opus-100", "opus_books", "synthetic"], help="Dataset source to download")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic corpus for fast verification")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs")
    args = parser.parse_args()

    prepare_data(
        config_path=args.config,
        dataset_name=args.dataset,
        use_synthetic=args.synthetic,
        max_pairs=args.max_pairs,
    )
