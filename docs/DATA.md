# Data Mixing Strategy & Format

## Overview

**Balanced mixed training on 6 diverse datasets** using a 3-stage mixing strategy:

1. **Original Split**: 5:1 ratio (83.3% train, 16.7% test) per dataset
2. **Domain Intra-Balance**: Each domain's 2 datasets balanced 50:50
3. **Cross-Domain Mix**: 4:3:3 ratio (math 40%, qa 30%, code 30%)

Result: Balanced `train_mixed.jsonl` and `test_mixed.jsonl` with no data leakage.

---

## Datasets

| Name | Domain | Type | Samples | Use |
|------|--------|------|---------|-----|
| **GSM8K** | math | Grade school math | 7.5K | Train/test |
| **MATH** | math | Competition math | 7.5K | Train/test |
| **SQuAD2.0** | qa | Reading comprehension | 87K | Train/test |
| **HotpotQA** | qa | Multi-hop QA | 89K | Train/test |
| **HumanEval** | code | Code generation | 164 | Train/test |
| **MBPP** | code | Python problems | 427 | Train/test |

---

## Mixing Strategy

### Stage 1: Original Split (5:1)

Each dataset split into train (83.3%) and test (16.7%):

```
GSM8K (7,473)           MATH (7,500)
├─ train: 6,227         ├─ train: 6,250
└─ test: 1,246          └─ test: 1,250

SQuAD2.0 (87,599)       HotpotQA (88,966)
├─ train: 72,999        ├─ train: 74,138
└─ test: 14,600         └─ test: 14,828

HumanEval (164)         MBPP (427)
├─ train: 137           ├─ train: 356
└─ test: 27             └─ test: 71
```

**Key**: Train and test are completely separated at this point.

### Stage 2: Domain Intra-Balance (50:50)

Within each domain, balance two datasets to 50:50:

**MATH Domain**:
- GSM8K_train: 6,227 → resample to 6,250 (match MATH)
- MATH_train: 6,250 → keep at 6,250
- Result: 12,500 samples (50:50)

**QA Domain**:
- SQuAD_train: 72,999 → resample to 74,138 (match HotpotQA)
- HotpotQA_train: 74,138 → keep at 74,138
- Result: 148,276 samples (50:50)

**Code Domain**:
- HumanEval_train: 137 → resample to 356 (match MBPP)
- MBPP_train: 356 → keep at 356
- Result: 712 samples (50:50)

**Note**: Small datasets (HumanEval, GSM8K) use `random.choices` for resampling (allows repetition).

### Stage 3: Cross-Domain Mix (4:3:3)

Mix balanced domain pools by ratio math:qa:code = 4:3:3:

```
Math pool (12,500) → sample 40% → math_samples
QA pool (148,276)  → sample 30% → qa_samples
Code pool (712)    → sample 30% → code_samples

Total available = min(
  int(12,500 / 0.4),
  int(148,276 / 0.3),
  int(712 / 0.3)
) = 2,373

Final amounts:
├─ math: 2,373 × 0.4 = ~950
├─ qa:   2,373 × 0.3 = ~712
└─ code: 2,373 × 0.3 = ~712
Total: ~2,374 (this scales with available data)
```

---

## Data Volumes (Actual After Processing)

### After 5:1 Split

| Dataset | Train | Test |
|---------|-------|------|
| GSM8K | 6,227 | 1,246 |
| MATH | 6,250 | 1,250 |
| SQuAD2.0 | 72,999 | 14,600 |
| HotpotQA | 74,138 | 14,828 |
| HumanEval | 137 | 27 |
| MBPP | 356 | 71 |
| **Total** | **160,107** | **32,022** |

### After Mixing (Balanced)

**train_mixed.jsonl**:
- Math: ~64,043 samples (40%)
- QA: ~48,032 samples (30%)
- Code: ~48,032 samples (30%)
- **Total: ~160,107 samples**

**test_mixed.jsonl**: Same ratio mix, ~32,022 samples

**data/test/\*_test.jsonl**: Individual test sets (not mixed, for separate eval)

---

## Data Format

All JSONL files share unified format:

```json
{
  "id": "gsm8k_0",
  "dataset": "gsm8k",
  "domain": "math",
  "question": "Natalia sold clips to 48 of her friends...",
  "reference_answer": "24",
  "answer_type": "numeric",
  "metadata": {
    "source": "gsm8k",
    "original_id": "0"
  }
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| **id** | Unique identifier across all datasets |
| **dataset** | Source dataset name (gsm8k, math, squad2, hotpotqa, humaneval, mbpp) |
| **domain** | Domain category (math, qa, code) |
| **question** | Problem statement / prompt |
| **reference_answer** | Ground truth answer |
| **answer_type** | numeric, text, or code |
| **metadata** | Additional info (source, original_id, etc.) |

---

## Directory Structure

```
data/
├── raw/
│   ├── math/
│   │   ├── gsm8k.jsonl
│   │   └── math.jsonl
│   ├── qa/
│   │   ├── squad2.jsonl
│   │   └── hotpotqa.jsonl
│   └── code/
│       ├── humaneval.jsonl
│       └── mbpp.jsonl
│
├── processed/
│   ├── gsm8k/
│   │   ├── train.jsonl (6,227 samples)
│   │   ├── test.jsonl (1,246 samples)
│   │   └── meta.json
│   ├── math/
│   ├── squad2/
│   ├── hotpotqa/
│   ├── humaneval/
│   └── mbpp/
│
├── mixed/
│   ├── train_mixed.jsonl ← Use this for GRPO training
│   ├── test_mixed.jsonl  ← Use this for validation/final eval
│   └── info.json
│
└── test/
    ├── gsm8k_test.jsonl ← Separate eval for math domain
    ├── math_test.jsonl
    ├── squad2_test.jsonl ← Separate eval for QA domain
    ├── hotpotqa_test.jsonl
    ├── humaneval_test.jsonl ← Separate eval for code domain
    ├── mbpp_test.jsonl
    └── test_index.json
```

---

## Processing Steps

### 1. Download Raw Data

```bash
python scripts/download_datasets.py
# Downloads from HuggingFace Hub to data/raw/{math,qa,code}/
```

### 2. Process & Mix

```bash
python scripts/process_datasets.py
# Outputs:
#  - data/processed/{dataset}/{train,test}.jsonl
#  - data/mixed/{train_mixed,test_mixed}.jsonl
#  - data/test/*_test.jsonl (6 files)
```

The script will print progress:
```
处理 GSM8K...
  ✅ GSM8K: 7473 样本 (train:6227 test:1246)

...

创建混合训练数据
📊 步骤1：TRAIN部分的域内均衡采样
  [TRAIN] MATH域均衡:
    gsm8k          6,227 →  6,250 (重采样)
    math           6,250 →  6,250 (欠采样)
  [TRAIN] QA域均衡:
    squad2        72,999 → 74,138 (重采样)
    hotpotqa      74,138 → 74,138 (欠采样)
  [TRAIN] CODE域均衡:
    humaneval        137 →    356 (重采样)
    mbpp             356 →    356 (欠采样)

🎯 步骤2：TRAIN部分的跨域4:3:3混合
  采样结果:
    math:    64,043 (40.0%)
    qa:      48,032 (30.0%)
    code:    48,032 (30.0%)
  总计: 160,107 样本
```

---

## Metadata Files

### data/mixed/info.json

```json
{
  "split_ratio": "5:1 (train:test = 83.3%:16.7%)",
  "domain_intra_balance": "50:50 per domain",
  "cross_domain_ratio": "4:3:3 (math:qa:code)",
  "total_train": 160107,
  "total_test": 32022,
  "domain_distribution_train": {
    "math": 64043,
    "qa": 48032,
    "code": 48032
  },
  "math_pct": 40.0,
  "qa_pct": 30.0,
  "code_pct": 30.0
}
```

### data/test/test_index.json

```json
{
  "gsm8k": "data/test/gsm8k_test.jsonl",
  "math": "data/test/math_test.jsonl",
  "squad2": "data/test/squad2_test.jsonl",
  "hotpotqa": "data/test/hotpotqa_test.jsonl",
  "humaneval": "data/test/humaneval_test.jsonl",
  "mbpp": "data/test/mbpp_test.jsonl"
}
```

---

## Key Design Decisions

✅ **5:1 Split (No Val Set)**
- Simpler data pipeline
- More training data
- You can create validation from train if needed

✅ **Domain Intra-Balance (50:50)**
- Prevents large datasets from dominating
- Small datasets (HumanEval: 164) resampled to match peers
- Fair representation of each dataset

✅ **Cross-Domain 4:3:3**
- User-specified ratio
- Math focus (40%) for challenging problems
- Balanced QA/Code (30% each)

✅ **Train/Test Isolation**
- Split at raw data level
- No overlap even after mixing
- Reproducible splits

---

## Usage Examples

### Use Mixed Data for Training

```python
# Data manager will load train_mixed.jsonl automatically
from src.data_manager import DataManager

manager = DataManager(
    data_dir="data/mixed",
    domain_ratios={"math": 0.4, "qa": 0.3, "code": 0.3}
)
batch = manager.get_batch(batch_size=4)
```

### Evaluate per Dataset

```bash
python scripts/eval_6datasets.py \
  --model qwen25-7b \
  --checkpoint checkpoints/qwen25-7b/grpo_mixed/step_100
# Evaluates on all 6 test sets separately
```

### Create Custom Mix

Edit `config/training.yaml`:
```yaml
domain_ratios:
  math: 0.5  # Increase math
  qa: 0.3
  code: 0.2  # Decrease code
```

Then reprocess:
```bash
python scripts/process_datasets.py
```

---

## References

- **GSM8K**: Cobbe et al., "Training Verifiers to Solve Math Word Problems"
- **MATH**: Hendrycks et al., "Measuring Mathematical Problem Solving"
- **SQuAD 2.0**: Rajpurkar et al., "Know What You Don't Know"
- **HotpotQA**: Yang et al., "HotpotQA: Diverse, Explainable Multi-hop QA"
- **HumanEval**: Chen et al., "Evaluating Large Language Models Trained on Code"
- **MBPP**: Austin et al., "Program Synthesis with Large Language Models"

---

**Next**: [TRAINING.md](TRAINING.md) to start training
