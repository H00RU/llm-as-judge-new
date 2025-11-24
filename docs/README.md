# LLM-as-Judge: Mixed Training Baseline

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)

*Mixed-domain GRPO training baseline on 6 diverse datasets*

[⚡ Quick Start](#-quick-start) • [📚 Docs](#-documentation) • [🏗️ Architecture](#-architecture)

</div>

---

## 📖 Overview

Baseline training framework for evaluating LLMs on **6 diverse datasets** with mixed-domain training:

- **Datasets**: GSM8K, MATH (math), SQuAD2.0, HotpotQA (QA), HumanEval, MBPP (code)
- **Data Strategy**: Train:Test = 5:1 (83.3%:16.7%), domain-balanced mixing (4:3:3)
- **Models**: Qwen2.5-7B, Qwen-3-8B (LoRA rank-64)
- **Algorithm**: GRPO (Group Relative Policy Optimization) online learning
- **Evaluation**: Per-dataset metrics on all 6 test sets

---

## ⚡ Quick Start

### 1️⃣ Installation (5 min)

```bash
# Clone + setup environment
git clone <repo> && cd llm-as-judge
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Download & Process Data (10 min)

```bash
# Download 6 datasets from HuggingFace
python scripts/download_datasets.py

# Process with balanced mixing (Train:Test = 5:1)
python scripts/process_datasets.py
# Output: data/mixed/train_mixed.jsonl + test_mixed.jsonl
```

### 3️⃣ Train (configurable duration)

```bash
# Full pipeline: download → process → train → evaluate
./scripts/run_full_pipeline.sh --model qwen25-7b --device cuda:0

# Or individual steps
python train.py --model qwen25-7b --device cuda:0
```

### 4️⃣ Evaluate Results

```bash
# Results auto-saved to results/evaluation/qwen25-7b_results.json
cat results/evaluation/qwen25-7b_results.json | jq '.datasets[] | {name: .dataset, accuracy: .metrics.accuracy}'
```

---

## 🏗️ Architecture

```
Data Pipeline:
  download_datasets.py
     ↓
  process_datasets.py (Plan C mixing)
     ├─ 6 datasets → 5:1 split (train:test)
     ├─ Domain intra-balance (50:50)
     └─ Cross-domain 4:3:3 mix
     ↓
  data/mixed/{train,test}_mixed.jsonl
     ↓
Training Loop:
  train.py (GRPO)
     ├─ Base: Qwen2.5-7B or Qwen-3-8B
     ├─ LoRA: rank=64
     └─ Optimize on train_mixed.jsonl
     ↓
  checkpoints/qwen25-7b/grpo_mixed/step_*/
     ↓
Evaluation:
  eval_6datasets.py
     ├─ Eval on data/test/{gsm8k,math,squad2,hotpotqa,humaneval,mbpp}_test.jsonl
     └─ Save metrics to results/evaluation/
```

### Data Structure

```
data/
├── mixed/
│   ├── train_mixed.jsonl        ← For GRPO training (~160K samples)
│   ├── test_mixed.jsonl         ← For final eval (mixed 4:3:3)
│   └── info.json                ← Mixing metadata
└── test/
    ├── gsm8k_test.jsonl         ← Independent evals
    ├── math_test.jsonl
    ├── squad2_test.jsonl
    ├── hotpotqa_test.jsonl
    ├── humaneval_test.jsonl
    ├── mbpp_test.jsonl
    └── test_index.json
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[SETUP.md](SETUP.md)** | Installation, dependencies, model download |
| **[DATA.md](DATA.md)** | Data mixing strategy, formats, statistics |
| **[TRAINING.md](TRAINING.md)** | Training configs, monitoring, troubleshooting |

---

## 🔧 Key Features

### Data Processing (Plan C)
✅ **5:1 Split**: Train:Test = 83.3%:16.7% (no validation set)
✅ **Domain Balance**: Intra-domain 50:50, Inter-domain 4:3:3
✅ **Small-data Handling**: HumanEval/MBPP resampled to match larger peers
✅ **Clear Separation**: Train/Test fully isolated from raw data loading

### Training
✅ **Multi-model**: Both Qwen2.5-7B and Qwen-3-8B supported
✅ **LoRA Efficient**: Rank=64, trainable params only
✅ **Online Learning**: GRPO without replay buffer
✅ **Flexible Config**: All hyperparams in `config/training.yaml`

### Evaluation
✅ **Per-dataset Metrics**: Accuracy for all 6 datasets
✅ **Mixed Evaluation**: Overall performance on balanced mix
✅ **Reproducible**: Deterministic splits and fixed seeds

---

## 📊 Expected Data Volumes

| Dataset | Domain | Train | Test |
|---------|--------|-------|------|
| GSM8K | math | 6.2K | 1.2K |
| MATH | math | 6.3K | 1.3K |
| SQuAD2.0 | qa | 73K | 14.6K |
| HotpotQA | qa | 74K | 14.8K |
| HumanEval | code | 137 | 27 |
| MBPP | code | 356 | 71 |
| **Total** | - | **160K** | **32K** |

After mixing: `train_mixed.jsonl` = 160K samples (Math 40% + QA 30% + Code 30%)

---

## 💡 Usage Examples

### Change Training Parameters
```bash
# Edit config/training.yaml, then:
python train.py --config config/training.yaml --model qwen25-7b
```

### Use Different Model
```bash
./scripts/run_full_pipeline.sh --model qwen3-8b --device cuda:1
```

### Skip Data Processing
```bash
./scripts/run_full_pipeline.sh --skip-download --skip-process
```

### Evaluation Only
```bash
python scripts/eval_6datasets.py \
  --model qwen25-7b \
  --checkpoint checkpoints/qwen25-7b/grpo_mixed/step_100
```

---

## 🔍 Project Structure

```
llm-as-judge/
├── README.md                        # Overview (this file)
├── SETUP.md                         # Installation guide
├── DATA.md                          # Data strategy & format
├── TRAINING.md                      # Training configs
├── requirements.txt
│
├── scripts/
│   ├── download_datasets.py        # Download from HuggingFace
│   ├── process_datasets.py         # Unify & mix (5:1, 50:50, 4:3:3)
│   ├── eval_6datasets.py           # Evaluate all 6 datasets
│   └── run_full_pipeline.sh        # Automation script
│
├── src/
│   ├── grpo_trainer.py             # Main training loop
│   ├── data_manager.py             # Mixed data sampling
│   ├── reward_computer.py          # LLM judge + metrics
│   └── ...
│
├── config/
│   ├── training.yaml               # Training hyperparameters
│   ├── models.yaml                 # Model definitions
│   └── dataset.yaml                # Dataset metadata
│
├── data/
│   ├── raw/                        # Downloaded raw datasets
│   ├── processed/                  # Per-dataset splits
│   ├── mixed/                      # Mixed train/test
│   └── test/                       # Individual test sets
│
├── checkpoints/                    # Model checkpoints
│   ├── qwen25-7b/grpo_mixed/
│   └── qwen3-8b/grpo_mixed/
│
└── results/evaluation/             # Results JSON
```

---

## ❓ Troubleshooting

### CUDA Out of Memory
Reduce `rollout_batch_size` in `config/training.yaml`:
```yaml
rollout_batch_size: 2  # Default: 4
```

### Model Download Fails
Download manually:
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./models/Qwen2.5-7B-Instruct
```

### Data Processing Errors
Check log output and verify HuggingFace connectivity:
```bash
python -c "from datasets import load_dataset; print('✓ Datasets lib OK')"
```

For more help → see [SETUP.md](SETUP.md) or [TRAINING.md](TRAINING.md)

---

## 📖 Next Steps

1. Read [SETUP.md](SETUP.md) for detailed installation
2. Review [DATA.md](DATA.md) to understand data mixing
3. Check [TRAINING.md](TRAINING.md) for training specifics
4. Run `./scripts/run_full_pipeline.sh --model qwen25-7b` for end-to-end test

---

<div align="center">

**Built for reproducible baseline experiments with mixed-domain training**

Questions? Check the docs → [SETUP](SETUP.md) | [DATA](DATA.md) | [TRAINING](TRAINING.md)

</div>
