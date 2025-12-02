# Installation Guide

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 32GB
- GPU: NVIDIA GPU with 16GB+ VRAM
- Storage: 50GB free space

**Recommended:**
- CPU: 8+ cores
- RAM: 64GB
- GPU: NVIDIA A100 (40GB/80GB) or equivalent
- Storage: 100GB+ free space

### Software Requirements

- Operating System: Linux (Ubuntu 20.04+ recommended) or macOS
- Python: 3.10 or higher
- CUDA: 11.8 or higher (for GPU support)
- Git: For cloning the repository

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/beita6969/llm-as-judge.git
cd llm-as-judge
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n llm-judge python=3.10
conda activate llm-judge
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True (if GPU is available)
```

### 4. Download Models

#### Qwen2.5-7B-Instruct

```bash
# Using Hugging Face CLI
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/qwen2.5-7b-instruct

# Or using Python
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='models/')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='models/')
"
```

#### gpt-4o-mini Configuration

This project uses OpenAI's gpt-4o-mini for:
- AFlow workflow execution
- LLM Judge for semantic equivalence checking

Configure in `config/aflow_llm.yaml` with your OpenAI API key.

### 5. Setup AFlow

```bash
# Clone AFlow (FoundationAgents)
git clone https://github.com/lupantech/FoundationAgents.git AFlow
cd AFlow
pip install -e .
cd ..

# Verify installation
python -c "import workspace.code.workflows.template.operator as operator"
```

### 6. Configure API Keys

Edit `config/aflow_llm.yaml` to add your OpenAI API key:

```yaml
models:
  "gpt-4o-mini":
    api_type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key: "your-openai-api-key-here"  # Replace with your key
    model_name: "gpt-4o-mini"
    temperature: 0
    top_p: 1
```

**Optional: W&B Monitoring**

```bash
# Set W&B API key (optional, for training visualization)
export WANDB_API_KEY=your-wandb-key
```

If not set, training will run in offline mode.

### 7. Prepare Configuration Files

```bash
# Copy AFlow config template and add your API key
cp config/aflow_llm.yaml.example config/aflow_llm.yaml
# Edit config/aflow_llm.yaml to add your OpenAI API key

# training.yaml is already configured with optimal defaults
# No need to modify unless you want custom settings
```

**Default Training Configuration** (`config/training.yaml`):
- Model: Qwen2.5-7B-Instruct (auto-downloads from HuggingFace)
- LoRA: rank=64, alpha=64
- Batch size: 4
- Max steps: 500
- Learning rate: 2.0e-5

All configurations are production-ready out of the box.

### 8. Download and Process Datasets

```bash
# Download 6 datasets from HuggingFace (GSM8K, MATH, SQuAD2.0, HotpotQA, HumanEval, MBPP)
python scripts/download_datasets.py

# Process and mix datasets (5:1 split, balanced 4:3:3 ratio)
python scripts/process_datasets.py

# This creates:
# - data/mixed/train_mixed.jsonl (2,071 samples)
# - data/mixed/test_mixed.jsonl (420 samples)
```

See [DATA.md](DATA.md) for detailed information on data mixing strategy.

### 9. Verify Installation

```bash
# Verify PyTorch and CUDA
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA: {torch.cuda.is_available()}')"

# Verify data is ready
ls -lh data/mixed/train_mixed.jsonl data/mixed/test_mixed.jsonl

# Verify config
python -c "import yaml; cfg = yaml.safe_load(open('config/training.yaml')); print('✓ Config loaded')"
```

If all checks pass, you're ready to start training!

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution:**
```bash
# Reduce batch size in config/training.yaml
rollout_batch_size: 2  # Default is 4

# Or use gradient accumulation
gradient_accumulation_steps: 4
```

#### 2. Module Not Found Errors

**Solution:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade

# Check PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/AFlow
```

#### 3. OpenAI API Key Issues

**Solution:**
```bash
# Verify API key is set in config/aflow_llm.yaml
grep "api_key" config/aflow_llm.yaml

# Test API connection
python -c "from openai import OpenAI; client = OpenAI(api_key='your-key'); print('✓ API connection OK')"
```

#### 4. Slow Training

**Solution:**
```bash
# Enable mixed precision
# In config/training.yaml:
mixed_precision: "bf16"

# Use flash attention
pip install flash-attn
```

### Getting Help

If you encounter issues:

1. Check the [FAQ](docs/FAQ.md)
2. Search [existing issues](https://github.com/beita6969/llm-as-judge/issues)
3. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details (`python --version`, `pip list`)

## Next Steps

After installation:

1. Review [SETUP.md](SETUP.md) for quick start guide
2. Read [DATA.md](DATA.md) to understand data mixing
3. Check [TRAINING.md](TRAINING.md) for training configuration
4. Start training: `python train.py --model qwen25-7b --device cuda:0`

For background training:
```bash
nohup python train.py --model qwen25-7b --device cuda:0 > training.log 2>&1 &
tail -f training.log  # Monitor progress
```

## Upgrading

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Check for breaking changes
git log --oneline --since="1 week ago"
```

## Uninstallation

```bash
# Remove virtual environment
rm -rf venv/

# Remove downloaded models (optional)
rm -rf models/

# Remove project directory
cd ..
rm -rf llm-as-judge/
```
