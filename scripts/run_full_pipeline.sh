#!/bin/bash

################################################################################
# Complete LLM Mixed Training Pipeline
#
# This script automates the entire workflow:
# 1. Download 6 datasets (GSM8K, MATH, SQuAD2.0, HotpotQA, HumanEval, MBPP)
# 2. Process datasets into unified JSONL format
# 3. Train model with GRPO on mixed data (40% math, 30% qa, 30% code)
# 4. Evaluate on 6 separate test datasets
# 5. Generate evaluation reports
#
# Usage:
#   ./scripts/run_full_pipeline.sh --model qwen25-7b [--skip-download] [--skip-process]
#
# Options:
#   --model         Model name: qwen25-7b (default) or qwen3-8b
#   --skip-download Skip dataset downloading (use existing data)
#   --skip-process  Skip dataset processing (use existing processed data)
#   --eval-only     Only run evaluation (skip download, process, train)
#   --device        GPU device: cuda:0 (default) or cuda:1, etc.
#
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL="qwen25-7b"
SKIP_DOWNLOAD=false
SKIP_PROCESS=false
EVAL_ONLY=false
DEVICE="cuda:0"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/pipeline_${TIMESTAMP}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-process)
            SKIP_PROCESS=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            SKIP_DOWNLOAD=true
            SKIP_PROCESS=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate model name
if [[ "$MODEL" != "qwen25-7b" && "$MODEL" != "qwen3-8b" ]]; then
    echo -e "${RED}❌ Invalid model: $MODEL (must be qwen25-7b or qwen3-8b)${NC}"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline.log"

# Helper function to log messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✅ SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[❌ ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}========== $1 ==========${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Main pipeline execution
main() {
    local start_time=$(date +%s)

    log_step "Pipeline Configuration"
    log_info "Model: $MODEL"
    log_info "Device: $DEVICE"
    log_info "Skip Download: $SKIP_DOWNLOAD"
    log_info "Skip Process: $SKIP_PROCESS"
    log_info "Eval Only: $EVAL_ONLY"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Log File: $LOG_FILE"

    # Step 1: Download datasets
    if [[ "$SKIP_DOWNLOAD" == false ]]; then
        log_step "Step 1: Downloading Datasets"
        log_info "Downloading 6 datasets (GSM8K, MATH, SQuAD2.0, HotpotQA, HumanEval, MBPP)..."

        if python scripts/download_datasets.py 2>&1 | tee -a "$LOG_FILE"; then
            log_success "Dataset download completed"
        else
            log_error "Dataset download failed"
            exit 1
        fi
    else
        log_step "Step 1: Skipping Dataset Download"
        log_info "Using existing datasets in data/raw/"
    fi

    # Step 2: Process datasets
    if [[ "$SKIP_PROCESS" == false ]]; then
        log_step "Step 2: Processing Datasets"
        log_info "Converting to unified format and creating train/val/test splits..."
        log_info "Mixing strategy: 40% math + 30% qa + 30% code"

        if python scripts/process_datasets.py 2>&1 | tee -a "$LOG_FILE"; then
            log_success "Dataset processing completed"
        else
            log_error "Dataset processing failed"
            exit 1
        fi
    else
        log_step "Step 2: Skipping Dataset Processing"
        log_info "Using existing processed datasets in data/processed/ and data/mixed/"
    fi

    # Step 3: Train model
    if [[ "$EVAL_ONLY" == false ]]; then
        log_step "Step 3: Training Model with GRPO"
        log_info "Starting GRPO training on mixed data..."
        log_info "Config: config/training.yaml"
        log_info "Model: $MODEL"
        log_info "Device: $DEVICE"

        # Create checkpoint directory
        CHECKPOINT_DIR="checkpoints/${MODEL}/grpo_mixed"
        mkdir -p "$CHECKPOINT_DIR"

        if python train.py \
            --config config/training.yaml \
            --model "$MODEL" \
            --device "$DEVICE" \
            --output_dir "$CHECKPOINT_DIR" 2>&1 | tee -a "$LOG_FILE"; then

            log_success "Model training completed"

            # Extract latest checkpoint
            LATEST_CHECKPOINT=$(ls -td "$CHECKPOINT_DIR/step_"* 2>/dev/null | head -1 || echo "")
            if [[ -n "$LATEST_CHECKPOINT" ]]; then
                log_info "Latest checkpoint: $LATEST_CHECKPOINT"
            fi
        else
            log_error "Model training failed"
            exit 1
        fi
    else
        log_step "Step 3: Skipping Model Training"
        log_info "Evaluation only mode - will evaluate base model or existing checkpoint"
        CHECKPOINT_DIR="checkpoints/${MODEL}/grpo_mixed"
        LATEST_CHECKPOINT=$(ls -td "$CHECKPOINT_DIR/step_"* 2>/dev/null | head -1 || echo "")
    fi

    # Step 4: Evaluate model
    log_step "Step 4: Evaluating Model on 6 Datasets"

    if [[ -n "$LATEST_CHECKPOINT" ]]; then
        log_info "Evaluating checkpoint: $LATEST_CHECKPOINT"
        eval_cmd="python scripts/eval_6datasets.py \
            --model $MODEL \
            --checkpoint $LATEST_CHECKPOINT \
            --device $DEVICE"
    else
        log_info "No checkpoint found, evaluating base model..."
        eval_cmd="python scripts/eval_6datasets.py \
            --model $MODEL \
            --device $DEVICE"
    fi

    # Create results directory
    mkdir -p "results/evaluation"

    if eval $eval_cmd 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Model evaluation completed"
    else
        log_error "Model evaluation failed"
        exit 1
    fi

    # Step 5: Generate summary report
    log_step "Step 5: Generating Summary Report"

    RESULTS_FILE="results/evaluation/${MODEL}_results.json"
    if [[ -f "$RESULTS_FILE" ]]; then
        log_success "Results saved to: $RESULTS_FILE"

        # Extract and display key metrics
        log_info "Evaluation Summary:"
        python << 'PYTHON_SCRIPT'
import json
import sys

try:
    with open("results/evaluation/${MODEL}_results.json") as f:
        results = json.load(f)

    print("\nDataset Performance:")
    for dataset_name, result in results.get("datasets", {}).items():
        if result:
            metrics = result.get("metrics", {})
            if metrics:
                acc = metrics.get("accuracy", 0)
                print(f"  {dataset_name}: {acc:.2%}")

    if "overall_accuracy" in results:
        print(f"\nOverall Accuracy: {results['overall_accuracy']:.4f}")
except Exception as e:
    print(f"Note: Could not extract metrics: {e}")
PYTHON_SCRIPT

    else
        log_error "Results file not found: $RESULTS_FILE"
    fi

    # Pipeline completion summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    log_step "Pipeline Complete"
    log_success "Total execution time: ${hours}h ${minutes}m ${seconds}s"
    log_info "Log file: $LOG_FILE"
    log_info "Results saved to: results/evaluation/"

    if [[ -d "checkpoints/${MODEL}/grpo_mixed" ]]; then
        local checkpoint_count=$(ls -d "checkpoints/${MODEL}/grpo_mixed/step_"* 2>/dev/null | wc -l)
        log_info "Checkpoints saved: $checkpoint_count"
    fi
}

# Run main pipeline
main

exit 0
