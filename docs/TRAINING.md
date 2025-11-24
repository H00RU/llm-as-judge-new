# Training Guide

## Quick Start

```bash
# Full automation (download → process → train → evaluate)
./scripts/run_full_pipeline.sh --model qwen25-7b --device cuda:0

# Or step by step
python scripts/download_datasets.py
python scripts/process_datasets.py
python train.py --model qwen25-7b --device cuda:0
python scripts/eval_6datasets.py --model qwen25-7b --checkpoint checkpoints/qwen25-7b/grpo_mixed/step_100
```

---

## Configuration

### config/training.yaml (Key Parameters)

```yaml
# Model
base_model: "Qwen/Qwen2.5-7B-Instruct"  # or "Qwen/Qwen-3-8B"

# LoRA fine-tuning
lora_rank: 64
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules: ["q_proj", "v_proj", "up_proj", "down_proj"]

# Training
max_steps: 100              # For baseline testing
rollout_batch_size: 4       # Per-step batch size
learning_rate: 1e-4
warmup_steps: 5

# Data
data_dir: "data/mixed"
domain_ratios:
  math: 0.4
  qa: 0.3
  code: 0.3
```

---

## Training Modes

### Full Pipeline (Recommended)

```bash
./scripts/run_full_pipeline.sh --model qwen25-7b --device cuda:0
```

### Training Only

```bash
python train.py --model qwen25-7b --device cuda:0
```

### Evaluation Only

```bash
python scripts/eval_6datasets.py \
  --model qwen25-7b \
  --checkpoint checkpoints/qwen25-7b/grpo_mixed/step_100
```

---

## Troubleshooting

### CUDA Out of Memory
```yaml
# In config/training.yaml
rollout_batch_size: 2  # Reduce from 4
```

### Model Download Fails
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./models/Qwen2.5-7B-Instruct
```

### Data Processing Errors
```bash
python -c "from datasets import load_dataset; print('✓ OK')"
```

---

**Next**: Run `./scripts/run_full_pipeline.sh --model qwen25-7b` to start training!
