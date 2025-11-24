# Setup & Installation Guide

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (Qwen2.5-7B inference)
- **CPU**: 4+ cores
- **RAM**: 32GB+
- **Storage**: 100GB+ (for models + datasets)

### Software
- Python 3.10+
- CUDA 11.8+ (for GPU)
- Git

## Installation Steps

### 1. Clone & Create Environment

```bash
git clone <repo-url>
cd llm-as-judge

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')"
```

### 3. Install Models

**Option A: Automatic (recommended)**
Models auto-download on first training run.

**Option B: Manual**
```bash
# Qwen2.5-7B
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./models/Qwen2.5-7B-Instruct

# Qwen-3-8B (optional)
huggingface-cli download Qwen/Qwen-3-8B \
  --local-dir ./models/Qwen-3-8B
```

### 4. Configure Environment (Optional)

```bash
# For W&B monitoring (optional)
export WANDB_API_KEY=your_key_here

# For OpenAI API (optional, for LLM Judge)
export OPENAI_API_KEY=your_key_here
```

## Verification

```bash
# Test imports
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
print('✓ All imports successful')
"

# Test data pipeline
python scripts/download_datasets.py --help
python scripts/process_datasets.py --help

# Test training setup
python -c "
import yaml
with open('config/training.yaml') as f:
    cfg = yaml.safe_load(f)
    print(f'✓ Config loaded: {cfg[\"base_model\"]}')
"
```

## Troubleshooting

### CUDA/GPU Issues

**CUDA not found**:
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch for your CUDA version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Out of Memory**:
```bash
# Reduce batch size in config/training.yaml
rollout_batch_size: 2  # Default: 4
```

### Model Download Fails

**HuggingFace timeout**:
```bash
# Use different mirror
export HF_ENDPOINT=https://mirror.ghproxy.com/https://huggingface.co

# Or download manually
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

### Dataset/Dependencies

**Missing datasets library**:
```bash
pip install datasets --upgrade
```

**Permission denied when saving**:
```bash
chmod -R 755 data/
mkdir -p {data/raw/math,data/raw/qa,data/raw/code}
```

## Next Steps

1. → [DATA.md](DATA.md) to understand data mixing
2. → [TRAINING.md](TRAINING.md) to start training
3. → [README.md](README.md) for quick examples

---

**All set!** Now run:
```bash
python scripts/download_datasets.py
python scripts/process_datasets.py
python train.py --model qwen25-7b --device cuda:0
```
