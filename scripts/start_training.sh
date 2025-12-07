#!/bin/bash
set -e

# Configuration
PROJECT_DIR="/root/llm-as-judge-new"
LOG_DIR="$PROJECT_DIR/logs"
TRAINING_LOG="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/training.pid"

echo "LLM-as-Judge Training Launcher"
echo ""

# Environment check
echo "Checking environment..."

if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi
echo "Python: $(python --version)"

if ! python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "Warning: CUDA not available, will use CPU (slow)"
else
    echo "CUDA available"
fi

if [ ! -d "$PROJECT_DIR/models/Qwen2.5-7B-Instruct" ]; then
    echo "Error: Model not found at $PROJECT_DIR/models/Qwen2.5-7B-Instruct"
    exit 1
fi
echo "Model ready: Qwen2.5-7B-Instruct"

if [ ! -f "$PROJECT_DIR/data/mixed/train_mixed.jsonl" ]; then
    echo "Error: Training data not found at $PROJECT_DIR/data/mixed/train_mixed.jsonl"
    exit 1
fi
echo "Training data ready"

if [ ! -f "$PROJECT_DIR/config/training.yaml" ]; then
    echo "Error: Training config not found at $PROJECT_DIR/config/training.yaml"
    exit 1
fi
echo "Training config ready"

echo "Environment OK"
echo ""

# Prepare
echo "Preparing environment..."
mkdir -p "$LOG_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export AFLOW_PATH="/root/AFlow"
echo "PYTHONPATH and AFLOW_PATH set"
echo ""

# Start training
echo "Starting training..."
START_TIME=$(date)
echo "Start time: $START_TIME" | tee -a "$TRAINING_LOG"
echo "" | tee -a "$TRAINING_LOG"

cd "$PROJECT_DIR"
nohup python train.py >> "$TRAINING_LOG" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

echo "Training started"
echo "  PID: $TRAIN_PID"
echo "  Log: $TRAINING_LOG"
echo "  PID file: $PID_FILE"
echo ""
echo "Monitor log: tail -f $TRAINING_LOG"
echo "Stop training: kill $TRAIN_PID"
echo "Check GPU: nvidia-smi"
