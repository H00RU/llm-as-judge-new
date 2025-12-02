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
base_model: "/root/llm-as-judge-new/models"  # Local model path or HF model ID
device_mapping: [0]                          # GPU ID list
physical_gpus: [0]                           # Physical GPU IDs
bf16: false                                  # Use float16 precision

# LoRA fine-tuning
use_lora: true
lora_rank: 64
lora_alpha: 64                               # Maintained alpha/rank = 1.0
lora_dropout: 0.05
lora_target_modules: "q_proj,k_proj,v_proj,o_proj"

# Training
max_steps: 500                               # Total training steps
rollout_batch_size: 4                        # Samples per step
num_return_sequences_in_group: 6             # Workflows per problem (GRPO group)
learning_rate: 2.0e-5                        # Balanced learning rate
warmup_steps: 100                            # 20% of total steps
gradient_accumulation_steps: 1
clip_range: 0.2
max_grad_norm: 1.0
weight_decay: 0.01
use_kl_loss: true
kl_loss_coef: 0.1

# Temperature (Fixed)
temperature_schedule:
  enabled: false                             # Disable dynamic scheduling
  initial: 0.4                               # Fixed temperature
generation_config:
  temperature: 0.2                           # Generation temperature
  max_tokens: 4096                           # Prevent truncation
  top_p: 0.95
  top_k: 50

# Data
data_dir: "data"
train_dataset: "data/mixed/train_mixed.jsonl"
test_dataset: "data/mixed/test_mixed.jsonl"
domain_ratios:
  math: 0.4
  qa: 0.3
  code: 0.3

# AFlow Configuration (Workflow Execution)
aflow_config_path: "config/aflow_llm.yaml"
aflow_executor_model: "gpt-4o-mini"          # OpenAI gpt-4o-mini for execution
aflow_operator_descriptions_path: "config/aflow_operators.yaml"
execution_timeout: 180                        # Execution timeout (seconds)

# Reward Computation
reward_weights:
  correctness: 0.7
  efficiency: 0.2
  code_quality: 0.1

# Checkpointing
output_dir: "checkpoints/qwen25-7b/grpo_mixed"
checkpointing:
  save_dir: "checkpoints/qwen25-7b/grpo_mixed"
log_every: 5                                  # Log metrics every 5 steps
save_every: 20                                # Save checkpoint every 20 steps
eval_every: 0                                 # Disable online test eval (avoid leakage)

# W&B Monitoring
wandb:
  enabled: true
  project: "aflow-roll-integration"
  run_name: "grpo-500steps-4batch-6workflows-reference-restored"
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
