# Tiny-Seq2Seq: Complete Command Reference Guide

This document provides a comprehensive operational guide for the **Tiny-Seq2Seq** project — a reproduction of *Sequence to Sequence Learning with Neural Networks* (Sutskever et al., 2014) optimized for consumer GPUs (NVIDIA RTX 3050 4 GB VRAM) with end-to-end MLflow experiment tracking.

---

## 1. Environment & Server Setup

### Activate Virtual Environment
Before executing any script, activate the project environment:

**PowerShell (Windows):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt (cmd):**
```cmd
.\.venv\Scripts\activate.bat
```

### Launch MLflow Tracking Server UI
The project uses an embedded SQLite database (`sqlite:///mlruns.db`) to log parameters, metrics, system telemetry, and translation artifacts.

```powershell
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```
- **URL**: [http://localhost:5000](http://localhost:5000)
- **Experiments**:
  - `TinySeq2Seq-Pretraining`: Full model training runs and step/epoch metrics.
  - `TinySeq2Seq-Evaluation`: Multi-beam SacreBLEU evaluation and length analysis.
  - `TinySeq2Seq-Ablations`: Nested runs for paper replication studies.
  - `TinySeq2Seq-SmokeTest`: 2-batch memorization checks.

---

## 2. Phase-by-Phase Execution Commands

### Phase 2: Data Pipeline & Tokenization

#### 1. Clean Raw Parallel Corpus
Normalizes text with Unicode NFC, strips invalid characters, enforces sentence length limits (1 to 50 tokens), and outputs paired files.
```powershell
python scripts/prepare_data.py --config configs/base_config.yaml
```
- **Input**: Raw parallel files specified in `configs/base_config.yaml`
- **Output**: Cleaned text in `artifacts/cleaned/train.clean.en`, `artifacts/cleaned/train.clean.fr`, etc.

#### 2. Train Shared SentencePiece Tokenizer
Trains a joint byte-pair encoding (BPE) model with special tokens (`<pad>=0`, `<unk>=1`, `<s>=2`, `</s>=3`).
```powershell
python scripts/train_tokenizer.py --config configs/base_config.yaml --vocab_size 16000
```
- **Optional**: For rapid testing, train a smaller vocabulary:
  ```powershell
  python scripts/train_tokenizer.py --vocab_size 500
  ```
- **Output**: `artifacts/tokenizer/spm_<vocab_size>.model` and `.vocab`

#### 3. Serialize Data to Binary Format
Encodes text sentences into flat binary `uint16` memory-mapped arrays with index files (`.idx` and `.bin`) for instant zero-copy loading.
```powershell
python scripts/prepare_binary_data.py --config configs/base_config.yaml
```
- **Output**: 
  - `artifacts/processed/train.en.bin` & `artifacts/processed/train.en.idx`
  - `artifacts/processed/train.fr.bin` & `artifacts/processed/train.fr.idx`
  - Corresponding `dev` and `test` splits.

---

### Phase 4: Model Architecture Smoke Test

#### Run Memorization Smoke Test
Verifies forward-backward flow, FP16 AMP stability, source reversal, and parameter tying on a fixed 2-batch dataset. Checks that loss drops near 0 and canary translations reproduce verbatim.
```powershell
python scripts/smoke_test.py --config configs/base_config.yaml
```
- **Expected Outcome**: Loss drops < 0.1, Perplexity < 1.10. Canary translations logged to MLflow as a step table.

---

### Phase 5: High-Performance C++ Batch Builder

#### 1. Compile C++ Pybind11 Extension (If modifying C++ code)
Compiles `cpp/batch_builder.cpp` using MSVC and pybind11 with GIL-release optimization:
```powershell
python setup.py build_ext --inplace
```
- **Output**: Native binary `Release/seq2seq_c_batcher.cp312-win_amd64.pyd`

#### 2. Benchmark Batcher Throughput
Benchmarks pure Python batch collation vs. native C++ multithreaded collation:
```powershell
python scripts/benchmark_batcher.py --config configs/base_config.yaml --num_batches 50
```
- **Expected Outcome**: Reports collation time per batch and validates the ~180x C++ speedup.

---

### Phase 6: Production Training Loop

#### Train the Seq2Seq Model
Executes full model training with length-bucketing, FP16 AMP, cosine warmup schedule, gradient clipping at 5.0, periodic validation, and canary sentence logging.
```powershell
python scripts/train.py --config configs/base_config.yaml
```

**Common Flags & Customizations:**
- Override number of epochs:
  ```powershell
  python scripts/train.py --config configs/base_config.yaml --epochs 10
  ```
- Resume training from a checkpoint:
  ```powershell
  python scripts/train.py --resume
  ```

**Outputs Created:**
- Best checkpoint: `checkpoints/checkpoint_best.pt`
- Latest checkpoint: `checkpoints/checkpoint_latest.pt`
- Training Diagnostic Plots in `artifacts/plots/`:
  - `loss_perplexity_curve.png`
  - `learning_rate_schedule.png`
  - `gradient_norm_distribution.png`
  - `hardware_throughput_profile.png`

#### Train the Scaled 4-Layer 80.13M Model (New Experiment)
Executes training for the 4-layer LSTM architecture ($d_{model} = 896$, 4 layers, ~80.13M parameters) with checkpoints saved in `checkpoints/80m/`:
```powershell
python scripts/train.py --config configs/config_80m.yaml --epochs 10
```

- **Resume 80M training from checkpoint**:
  ```powershell
  python scripts/train.py --config configs/config_80m.yaml --resume
  ```
- **Outputs Created**:
  - Best checkpoint: `checkpoints/80m/checkpoint_best.pt`
  - Latest checkpoint: `checkpoints/80m/checkpoint_latest.pt`
  - MLflow Experiment: `TinySeq2Seq-Training-80M`

---

### Phase 7: GPU Beam Search & Multi-Metric Evaluation

#### Evaluate Checkpoint with SacreBLEU (Baseline 29M Model)
Runs GPU vectorized beam search across beam widths $K \in \{1, 2, 5, 12\}$ with length penalty $\alpha=0.6$, computes official SacreBLEU, and breaks down translation quality across sentence length buckets (short, medium, long).
```powershell
python scripts/evaluate.py --config configs/base_config.yaml --checkpoint checkpoints/checkpoint_best.pt
```

#### Evaluate Checkpoint with SacreBLEU (Scaled 80.13M Model)
Evaluates the 4-layer 80.13M model with beam search decoding:
```powershell
python scripts/evaluate.py --config configs/config_80m.yaml --checkpoint checkpoints/80m/checkpoint_best.pt
```

**Custom Evaluation Flags:**
- Evaluate specific beam sizes:
  ```powershell
  python scripts/evaluate.py --checkpoint checkpoints/checkpoint_best.pt --beam_sizes 1 2 5
  ```
- Evaluate on test split instead of dev:
  ```powershell
  python scripts/evaluate.py --split test --beam_sizes 5
  ```

**Outputs Created:**
- Metric logs in MLflow experiment `TinySeq2Seq-Evaluation`
- Sentence length degradation plot: `artifacts/plots/eval_beam_length_degradation.png`

---

### Phase 8: Paper Replication Ablation Suite

#### Run Ablation Studies
Executes miniature versions of the 8 core findings from Sutskever et al. (2014) under the `TinySeq2Seq-Ablations` MLflow experiment.

**1. Run All Ablations:**
```powershell
python scripts/run_ablations.py --config configs/base_config.yaml --study all
```

**2. Run Source Reversal Study (Reversed vs Standard Source):**
```powershell
python scripts/run_ablations.py --study reversal --epochs 2
```
- Output: `artifacts/plots/ablations/ablation_reversal.png`

**3. Run Beam Width Scaling ($K \in \{1, 2, 5, 12\}$):**
```powershell
python scripts/run_ablations.py --study beam_width
```
- Output: `artifacts/plots/ablations/ablation_beam_width.png`

**4. Run Polyak Checkpoint Averaging Study:**
```powershell
python scripts/run_ablations.py --study checkpoint_avg
```
- Output: `artifacts/plots/ablations/ablation_checkpoint_averaging.png`

---

## 3. Test Suite Commands

Run the full pytest suite (34 tests covering data, C++ batcher, model, training, beam search, and ablations):

```powershell
python -m pytest -v
```

### Run Specific Test Modules:
- **Phase 0 (Sanity & Helpers)**:
  ```powershell
  python -m pytest tests/test_phase0.py -v
  ```
- **Phase 1 (MLflow Tracker)**:
  ```powershell
  python -m pytest tests/test_tracker.py -v
  ```
- **Phase 2 (Data Pipeline & Serialization)**:
  ```powershell
  python -m pytest tests/test_data_pipeline.py -v
  ```
- **Phase 3 (Python Batcher & Prefetcher)**:
  ```powershell
  python -m pytest tests/test_batcher_parity.py -v
  ```
- **Phase 4 (Model & Embeddings)**:
  ```powershell
  python -m pytest tests/test_model.py -v
  ```
- **Phase 5 (C++ Native Batcher)**:
  ```powershell
  python -m pytest tests/test_c_batcher.py -v
  ```
- **Phase 6 (Trainer & Scheduler)**:
  ```powershell
  python -m pytest tests/test_trainer.py -v
  ```
- **Phase 7 (Beam Search & Evaluator)**:
  ```powershell
  python -m pytest tests/test_beam_search.py -v
  ```
- **Phase 8 (Polyak Averaging & Ablations)**:
  ```powershell
  python -m pytest tests/test_ablations.py -v
  ```

---

## 4. Hardware & Configuration Tips

- **Configuration File**: All hyperparameters (model dimensions, layer counts, token budgets, learning rates) are stored in [`configs/base_config.yaml`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/LLM_Engineering/Tiny-Seq2Seq/configs/base_config.yaml).
- **VRAM Budget on RTX 3050 (4 GB)**:
  - If CUDA Out of Memory occurs during long sequence training, reduce `token_budget` in `configs/base_config.yaml` from `4096` to `2048` or `1024`.
- **Plot Directory**: All diagnostic and comparative figures are saved in [`artifacts/plots/`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/LLM_Engineering/Tiny-Seq2Seq/artifacts/plots) and [`artifacts/plots/ablations/`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/LLM_Engineering/Tiny-Seq2Seq/artifacts/plots/ablations).
