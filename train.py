#!/usr/bin/env python3
"""
训练入口 - 启动GRPO训练
支持多模型和设备配置
"""
import sys
import os
import asyncio
import argparse
import yaml

# 添加src到路径
sys.path.insert(0, 'src')

from grpo_trainer import GRPOTrainer
from scripts.setup_data_paths import DataPathSetup


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AFlow + ROLL GRPO训练")
    parser.add_argument(
        '--config',
        type=str,
        default='config/training.yaml',
        help='训练配置文件路径'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=['qwen25-7b', 'qwen3-8b'],
        help='模型名称 (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='使用的GPU设备，如 cuda:0 (overrides config)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='检查点输出目录 (overrides config)'
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     AFlow + ROLL 深度融合 - GRPO在线学习                    ║
║                                                              ║
║     多模型训练框架（支持Qwen2.5-7B和Qwen-3-8B）             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 验证必要的数据文件是否存在
    print("\n📋 验证数据环境...")
    from pathlib import Path

    code_data_files = {
        "humaneval": Path("data/raw/code/humaneval.jsonl"),
        "mbpp": Path("data/raw/code/mbpp.jsonl"),
    }

    code_data_ok = all(f.exists() for f in code_data_files.values())

    if code_data_ok:
        print("✅ 代码数据文件检查通过")
        # 如果源数据完整，自动设置数据路径映射
        print("📂 自动配置数据路径映射...")
        data_setup = DataPathSetup()
        data_setup.run_all(force=False)
    else:
        print("\n⚠️  警告：某些数据文件缺失")
        print("\n数据文件状态：")
        for name, path in code_data_files.items():
            status = "✅" if path.exists() else "❌"
            print(f"  {status} {path}")

        print("\n⚠️  虽然缺少部分数据，但将继续进行（可能会在训练时出错）")
        print("   如需完整训练，请先运行：")
        print("   python scripts/download_datasets.py")
        print("   python scripts/setup_data_paths.py")

    # 创建训练器
    trainer = GRPOTrainer(
        config_path=args.config,
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir
    )

    # 开始训练
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
