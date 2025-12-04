#!/bin/bash
#
# LLM-as-Judge 训练启动脚本
# 使用 nohup 在后台运行完整 500 步训练
#
# 用法: bash start_training.sh
#

set -e

# ============================================
# 配置
# ============================================
PROJECT_DIR="/root/llm-as-judge-new"
LOG_DIR="$PROJECT_DIR/logs"
TRAINING_LOG="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/training.pid"

# ============================================
# 函数定义
# ============================================

print_header() {
    echo "========================================"
    echo "🚀 LLM-as-Judge 完整训练启动器"
    echo "========================================"
    echo ""
}

check_environment() {
    echo "📋 环境检查..."

    # 检查 Python
    if ! command -v python &> /dev/null; then
        echo "❌ Python 未安装"
        exit 1
    fi
    echo "✅ Python: $(python --version)"

    # 检查 CUDA
    if ! python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        echo "⚠️  警告: CUDA 不可用，将使用 CPU (非常慢)"
    else
        echo "✅ CUDA 可用"
    fi

    # 检查模型
    if [ ! -d "$PROJECT_DIR/models/Qwen2.5-7B-Instruct" ]; then
        echo "❌ 模型未下载: $PROJECT_DIR/models/Qwen2.5-7B-Instruct"
        exit 1
    fi
    echo "✅ 模型已就绪: Qwen2.5-7B-Instruct"

    # 检查数据
    if [ ! -f "$PROJECT_DIR/data/mixed/train_mixed.jsonl" ]; then
        echo "❌ 训练数据未准备: $PROJECT_DIR/data/mixed/train_mixed.jsonl"
        exit 1
    fi
    echo "✅ 训练数据已就绪"

    # 检查配置
    if [ ! -f "$PROJECT_DIR/config/training.yaml" ]; then
        echo "❌ 训练配置未找到: $PROJECT_DIR/config/training.yaml"
        exit 1
    fi
    echo "✅ 训练配置已就绪"

    echo ""
}

prepare_environment() {
    echo "🔧 准备环境..."

    # 创建日志目录
    mkdir -p "$LOG_DIR"
    echo "✅ 日志目录: $LOG_DIR"

    # 设置 Python 路径
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    echo "✅ PYTHONPATH 已设置"

    # 设置 AFlow 路径
    export AFLOW_PATH="/root/AFlow"
    echo "✅ AFLOW_PATH: $AFLOW_PATH"

    echo ""
}

show_training_info() {
    echo "📊 训练配置信息:"
    echo ""
    echo "  配置文件: $PROJECT_DIR/config/training.yaml"
    echo "  训练步数: 500"
    echo "  批次大小: 5"
    echo "  K值: 2 (工作流数/样本)"
    echo "  KL系数: 0.02"
    echo "  梯度累积: 4"
    echo "  温度调度: 启用 (0.5 → 0.15)"
    echo "  学习率: 2.0e-5"
    echo ""
    echo "📈 预期效果:"
    echo "  第50步: reward ~0.1-0.3"
    echo "  第100-200步: reward ~0.4-0.5+"
    echo "  第500步: reward ~0.6-0.8+ (收敛)"
    echo ""
}

start_training() {
    echo "🎯 启动训练..."
    echo ""

    # 记录启动时间
    START_TIME=$(date)
    echo "启动时间: $START_TIME" | tee -a "$TRAINING_LOG"
    echo "" | tee -a "$TRAINING_LOG"

    # 使用 nohup 在后台运行
    cd "$PROJECT_DIR"
    nohup python train.py >> "$TRAINING_LOG" 2>&1 &

    # 获取 PID
    TRAIN_PID=$!
    echo $TRAIN_PID > "$PID_FILE"

    echo "✅ 训练已启动"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📌 训练信息"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PID: $TRAIN_PID"
    echo "  日志: $TRAINING_LOG"
    echo "  状态文件: $PID_FILE"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📖 实时监控命令"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  查看日志 (实时):"
    echo "    tail -f $TRAINING_LOG"
    echo ""
    echo "  查看最后 100 行:"
    echo "    tail -100 $TRAINING_LOG"
    echo ""
    echo "  搜索特定信息:"
    echo "    grep 'reward' $TRAINING_LOG"
    echo "    grep 'Step' $TRAINING_LOG"
    echo ""
    echo "  查看进程状态:"
    echo "    ps aux | grep $TRAIN_PID"
    echo ""
    echo "  停止训练:"
    echo "    kill $TRAIN_PID"
    echo ""
    echo "  监控 GPU:"
    echo "    watch -n 1 nvidia-smi"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🌐 W&B 监控"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  实时指标: https://wandb.ai/yourproject/aflow-roll-integration"
    echo "  (需要设置 WANDB_API_KEY)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✨ 训练已在后台运行"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

show_quick_reference() {
    echo "⚡ 快速参考"
    echo ""
    echo "  实时查看日志:"
    echo "    tail -f $TRAINING_LOG"
    echo ""
    echo "  获取训练 PID:"
    echo "    cat $PID_FILE"
    echo ""
    echo "  停止训练:"
    echo "    kill \$(cat $PID_FILE)"
    echo ""
    echo "  强制停止:"
    echo "    kill -9 \$(cat $PID_FILE)"
    echo ""
}

# ============================================
# 主程序
# ============================================

main() {
    print_header
    check_environment
    prepare_environment
    show_training_info
    start_training
    show_quick_reference
}

# 运行主程序
main
