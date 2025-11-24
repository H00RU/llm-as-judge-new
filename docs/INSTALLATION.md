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

#### GPT OSS 120B (Optional)

If you have access to GPT OSS 120B or want to use a different LLM:

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/gpt-oss-120b \
    --port 8002 \
    --tensor-parallel-size 4
```

Or use OpenAI API directly by configuring in `.env`.

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

### 6. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

**Required Environment Variables:**

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-api-key-here  # For GPT OSS 120B or OpenAI
OPENAI_BASE_URL=http://localhost:8002/v1  # vLLM endpoint

# Wandb (Optional, for monitoring)
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=llm-as-judge

# Model Paths
QWEN_MODEL_PATH=models/qwen2.5-7b-instruct
GPT_OSS_MODEL_PATH=models/gpt-oss-120b

# AFlow Path
AFLOW_PATH=./AFlow

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```

### 7. Prepare Configuration Files

```bash
# Copy config templates
cp config/aflow_llm.yaml.example config/aflow_llm.yaml
cp config/training.yaml.example config/training.yaml

# Edit configs as needed
nano config/training.yaml
```

**Key Training Configuration:**

```yaml
# config/training.yaml
model_name: "models/qwen2.5-7b-instruct"
physical_gpus: [0]  # GPU indices to use
rollout_batch_size: 4
num_return_sequences_in_group: 6
ppo_epochs: 1
learning_rate: 5e-6
```

### 8. Download or Prepare Datasets

```bash
# Create data directory structure
mkdir -p data/mixed
mkdir -p data/experience_buffer

# Option 1: Download sample datasets
# wget https://example.com/sample_data.tar.gz
# tar -xzf sample_data.tar.gz -C data/

# Option 2: Prepare your own datasets
# See Data Preparation section below
```

**Dataset Format:**

Each JSONL file should contain:

```json
{"problem": "What is 2+2?", "answer": "4", "type": "math"}
{"problem": "def add(a, b):\n    return a + b", "answer": "def add(a, b):\n    return a + b", "type": "code"}
```

### 9. Verify Installation

```bash
# Run quick tests
python test_integration.py

# Test LLM Judge
python test_llm_judge.py

# Test workflow generation
python -c "
from src.rl_workflow_generator import RLWorkflowGenerator
gen = RLWorkflowGenerator(model_path='models/qwen2.5-7b-instruct')
print('✓ Workflow generator initialized')
"
```

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

#### 3. vLLM Server Not Responding

**Solution:**
```bash
# Check if server is running
curl http://localhost:8002/v1/models

# Restart server with correct parameters
pkill -f vllm
python -m vllm.entrypoints.openai.api_server --model /path/to/model --port 8002
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

1. Review the [Quick Start](README.md#quick-start) guide
2. Check out [example notebooks](examples/)
3. Read about [configuration options](docs/configuration.md)
4. Start training: `python train.py --config config/training.yaml`

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
