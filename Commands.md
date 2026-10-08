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

---

## 5. 4-Layer GRU & Byte-Level BPE (BBPE) Experiment Commands

An experimental alternative architecture comparing **4-layer cuDNN GRU** against 4-layer LSTM, paired with a **Byte-Level BPE (BBPE)** tokenizer and accelerated by the compiled **C++ Batch Builder** (`seq2seq_c_batcher`).

### Architecture & Advantages
- **Parameters**: 67.27M ($d_{model}=896$, $n_{layers}=4$, $V=16,000$, tied embeddings).
- **VRAM Savings**: Single recurrent hidden state $h \in \mathbb{R}^{L \times B \times d}$ (saving 50% state memory vs. LSTM $(h, c)$). Peak VRAM is only ~700 MB.
- **BBPE Robustness**: Zero out-of-vocabulary (`<unk>`) rate via 256 base byte tokens.
- **C++ Batcher Acceleration**: 182x faster dataset bin-packing and collation with GIL release.
- **Dedicated Config**: [`configs/config_gru_bbpe.yaml`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/LLM_Engineering/Tiny-Seq2Seq/configs/config_gru_bbpe.yaml).

---

### Step 1: Train Byte-Level BPE Tokenizer
Trains a 16k Byte-Level BPE model on cleaned parallel text with special tokens (`<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`):

```powershell
python scripts/train_bbpe_tokenizer.py --config configs/config_gru_bbpe.yaml --vocab-size 16000
```
- **Inputs**: `artifacts/cleaned/train.en`, `artifacts/cleaned/train.fr`
- **Output**: `artifacts/tokenizer_bbpe/bbpe_16k.json`
- **MLflow Tracking**: Experiment `TinySeq2Seq-GRU-BBPE-Data`

---

### Step 2: High-Throughput Binary Serialization
Serializes cleaned corpora into memory-mappable `uint16` flat binary files and `int64` index offsets using native Rust/C-ABI multithreaded batch tokenization:

```powershell
python scripts/prepare_bbpe_binary_data.py --config configs/config_gru_bbpe.yaml
```
- **Inputs**: `artifacts/cleaned/*.en`, `artifacts/cleaned/*.fr`
- **Outputs**:
  - `artifacts/processed_bbpe/train.en.bin` & `train.en.idx`
  - `artifacts/processed_bbpe/train.fr.bin` & `train.fr.idx`
  - `artifacts/processed_bbpe/val.en.bin` & `val.en.idx`
  - `artifacts/processed_bbpe/val.fr.bin` & `val.fr.idx`
  - `artifacts/processed_bbpe/test.en.bin` & `test.en.idx`
  - `artifacts/processed_bbpe/test.fr.bin` & `test.fr.idx`

---

### Step 3: Train 4-Layer GRU Seq2Seq Model
Trains the 67.27M GRU model using compiled C++ batching (`CppBucketBatcher`), AMP mixed precision, warmup cosine scheduling, and gradient clipping at 5.0:

```powershell
python scripts/train_gru.py --config configs/config_gru_bbpe.yaml --epochs 10 --token-budget 4000
```

#### Quick 1-Epoch Validation Run:
```powershell
python scripts/train_gru.py --config configs/config_gru_bbpe.yaml --epochs 1 --token-budget 2000 --run-name quick_gru_check
```

#### Resume Training from Checkpoint:
```powershell
python scripts/train_gru.py --config configs/config_gru_bbpe.yaml --resume
```
- **Checkpoints**: `checkpoints/gru_bbpe/checkpoint_latest.pt` & `checkpoint_best.pt`
- **MLflow Tracking**: Experiment `TinySeq2Seq-GRU-BBPE-Training`
- **Diagnostic Plots**: `artifacts/plots/gru_loss_perplexity.png`, `gru_hardware_throughput.png`, etc.

---

### Step 4: Evaluate 4-Layer GRU with Vectorized Beam Search
Evaluates the trained GRU model using `GRUBeamSearchDecoder` across beam widths $K \in \{1, 2, 5, 12\}$ and computes SacreBLEU with length bucket degradation:

```powershell
python scripts/evaluate_gru.py --config configs/config_gru_bbpe.yaml --checkpoint checkpoints/gru_bbpe/checkpoint_best.pt --split test
```

#### Quick Evaluation on 200 Sentences:
```powershell
python scripts/evaluate_gru.py --config configs/config_gru_bbpe.yaml --checkpoint checkpoints/gru_bbpe/checkpoint_best.pt --split test --max-sentences 200
```
- **Metrics**: SacreBLEU overall, Short (<15 tokens), Medium (15-30 tokens), Long (>30 tokens).
- **MLflow Tracking**: Experiment `TinySeq2Seq-GRU-BBPE-Evaluation`

---

### Step 5: Run GRU & BBPE Unit Test Suite
Runs the dedicated unit test suite verifying BBPE tokenization, GRU forward/backward pass, and vectorized GRU beam search:

```powershell
python -m pytest tests/test_gru_bbpe.py -v
```

---

## 6. 4-Layer Bidirectional LSTM (BiLSTM) Experiment Commands

An experimental architecture comparing a **4-layer cuDNN Bidirectional LSTM (BiLSTM)** against unidirectional LSTM and GRU, featuring per-layer bridge projection and accelerated by the compiled **C++ Batch Builder** (`seq2seq_c_batcher`).

### Architecture & Specifications
- **Parameters**: 76.27M ($d_{model}=640$, $n_{layers}=4$, $V=16,000$, tied embeddings). Tensor Core aligned ($10 \times 64$).
- **Bidirectional Encoder**: 4-layer cuDNN BiLSTM reading source text forward and backward simultaneously.
- **Bridge Projection**: Per-layer non-linear bridge (`Linear(2 * d_model, d_model)` + `Tanh`) projecting concatenated states $[h_{fwd}; h_{bwd}]$ and $[c_{fwd}; c_{bwd}]$ into the unidirectional decoder initial state.
- **VRAM Budget**: Peak training VRAM is ~1.1 GB on NVIDIA RTX 3050 Laptop GPU (4 GB VRAM ceiling).
- **C++ Batcher Acceleration**: Uses `seq2seq_c_batcher` for 182x faster length-bucketed batch assembly with GIL release.
- **Dedicated Config**: [`configs/config_bilstm.yaml`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/LLM_Engineering/Tiny-Seq2Seq/configs/config_bilstm.yaml).

---

### Step 1: Train 4-Layer BiLSTM Seq2Seq Model
Trains the 76.27M BiLSTM model using compiled C++ batching (`CppBucketBatcher`), AMP mixed precision, warmup cosine scheduling, and gradient clipping at 5.0:

```powershell
python scripts/train_bilstm.py --config configs/config_bilstm.yaml --epochs 10 --token-budget 4000
```

#### Quick 1-Epoch Validation Run:
```powershell
python scripts/train_bilstm.py --config configs/config_bilstm.yaml --epochs 1 --token-budget 2000 --run-name quick_bilstm_check
```

#### Resume Training from Checkpoint:
```powershell
python scripts/train_bilstm.py --config configs/config_bilstm.yaml --resume
```
- **Checkpoints**: `checkpoints/bilstm/checkpoint_latest.pt` & `checkpoint_best.pt`
- **MLflow Tracking**: Experiment `TinySeq2Seq-BiLSTM-Training`
- **Diagnostic Plots**: `artifacts/plots/bilstm_loss_perplexity.png`, `bilstm_hardware_throughput.png`, etc.

---

### Step 2: Evaluate 4-Layer BiLSTM with Vectorized Beam Search
Evaluates the trained BiLSTM model using `BiLSTMBeamSearchDecoder` across beam widths $K \in \{1, 2, 5, 12\}$ and computes SacreBLEU with length bucket degradation:

```powershell
python scripts/evaluate_bilstm.py --config configs/config_bilstm.yaml --checkpoint checkpoints/bilstm/checkpoint_best.pt --split test
```

#### Quick Evaluation on 200 Sentences:
```powershell
python scripts/evaluate_bilstm.py --config configs/config_bilstm.yaml --checkpoint checkpoints/bilstm/checkpoint_best.pt --split test --max-sentences 200
```
- **Metrics**: SacreBLEU overall, Short (<15 tokens), Medium (15-30 tokens), Long (>30 tokens).
- **MLflow Tracking**: Experiment `TinySeq2Seq-BiLSTM-Evaluation`

---

### Step 3: Run BiLSTM Unit Test Suite
Runs the dedicated unit test suite verifying BiLSTM encoder, layer-wise bridge projection, decoder, gradient flow, and vectorized beam search:

```powershell
python -m pytest tests/test_bilstm.py -v
```


