#!/bin/bash

################################################################################
# Minimal GRPO Training Pipeline (10 Steps)
#
# This script runs a quick 10-step GRPO training for testing/validation
# Perfect for verifying Plan B implementation, AFlow integration, etc.
#
# Usage:
#   ./scripts/run_minimal_training.sh [--device cuda:0]
#
# Options:
#   --device    GPU device: cuda:0 (default) or cuda:1, etc.
#   --skip-data Skip data preparation (use existing data)
#
# Runtime:
#   ~10-15 minutes depending on GPU speed
#   Produces 10 training steps, 240 total samples (10 steps × 4 batch × 6 workflows)
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
DEVICE="cuda:0"
SKIP_DATA=false
CONFIG="config/minimal_training.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/minimal_${TIMESTAMP}"
NOHUP_LOG="$LOG_DIR/nohup_minimal_training_${TIMESTAMP}.log"  # 🔧 修复：日志文件放在LOG_DIR里

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/minimal_pipeline.log"

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

    log_step "Minimal Training Pipeline Configuration"
    log_info "Config: $CONFIG"
    log_info "Device: $DEVICE"
    log_info "Skip Data: $SKIP_DATA"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Log File: $LOG_FILE"
    log_info "Nohup Log: $NOHUP_LOG"

    # Step 1: Verify/prepare data
    if [[ "$SKIP_DATA" == false ]]; then
        log_step "Step 1: Verifying Data Preparation"
        log_info "Checking if data exists..."

        # Check if mixed data exists
        if [[ -f "data/mixed/train_mixed.jsonl" ]]; then
            log_success "Training data exists (2,071 samples)"
        else
            log_error "Training data not found"
            log_info "Please run: python scripts/download_datasets.py && python scripts/process_datasets.py"
            exit 1
        fi

        # Check if model exists
        if [[ -d "models" && -f "models/config.json" ]]; then
            log_success "Model exists at /root/llm-as-judge-new/models"
        else
            log_error "Model not found"
            log_info "Please download: python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; ...\""
            exit 1
        fi
    else
        log_step "Step 1: Skipping Data Verification"
        log_info "Using existing data and model"
    fi

    # Step 2: Start training with nohup
    log_step "Step 2: Starting 10-Step GRPO Training"
    log_info "Starting training in background with nohup..."
    log_info "Output will be saved to: $NOHUP_LOG"
    log_info ""
    log_info "Monitor training with:"
    log_info "  tail -f $NOHUP_LOG"
    log_info "  tail -f $LOG_FILE"
    log_info ""

    # Start training
    if nohup python train.py --config "$CONFIG" --device "$DEVICE" > "$NOHUP_LOG" 2>&1 &
    then
        local PID=$!
        log_success "Training started (PID: $PID)"
        echo "$PID" > ".minimal_training_pid"

        # Wait a moment and check if process is still running
        sleep 5
        if ps -p $PID > /dev/null; then
            log_success "Training process is running successfully"
            log_info ""
            log_info "📊 Real-time monitoring:"
            log_info "  tail -f $NOHUP_LOG"
            log_info ""
            log_info "Expected completion time: 10-15 minutes"
            log_info "Total samples: 240 (10 steps × 4 batch × 6 workflows)"
        else
            log_error "Training process died unexpectedly"
            log_error "Check error log:"
            tail -50 "$NOHUP_LOG"
            exit 1
        fi
    else
        log_error "Failed to start training"
        exit 1
    fi

    # Step 3: Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_step "Pipeline Initialized"
    log_success "Training is running in background"
    log_info ""
    log_info "📋 Summary:"
    log_info "  Configuration: $CONFIG"
    log_info "  Training Steps: 10 (minimal test)"
    log_info "  Batch Size: 4"
    log_info "  Total Samples: 240"
    log_info "  Log File: $NOHUP_LOG"
    log_info "  PID File: .minimal_training_pid"
    log_info ""
    log_info "🎯 Next steps:"
    log_info "  1. Monitor progress: tail -f $NOHUP_LOG"
    log_info "  2. Wait for completion (~10-15 minutes)"
    log_info "  3. Check results in checkpoints/qwen25-7b/grpo_minimal/"
    log_info ""
    log_info "Setup completed in ${duration} seconds"
}

# Run main pipeline
main

exit 0
