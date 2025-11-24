#!/usr/bin/env python3
"""
下载所有6个数据集
"""
import os
from pathlib import Path
from datasets import load_dataset
import json

class DatasetDownloader:
    """数据集下载器"""

    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.output_dir / "math").mkdir(exist_ok=True)
        (self.output_dir / "qa").mkdir(exist_ok=True)
        (self.output_dir / "code").mkdir(exist_ok=True)

    def download_gsm8k(self):
        """下载GSM8K"""
        print("\n📥 下载 GSM8K...")
        try:
            dataset = load_dataset("openai/gsm8k", "main")

            # 保存为jsonl
            output_path = self.output_dir / "math" / "gsm8k.jsonl"
            with open(output_path, "w") as f:
                for item in dataset["train"]:
                    f.write(json.dumps(item) + "\n")

            print(f"  ✅ GSM8K: {len(dataset['train'])} samples")
            return len(dataset['train'])
        except Exception as e:
            print(f"  ❌ GSM8K 下载失败: {e}")
            return 0

    def download_math(self):
        """下载MATH"""
        print("\n📥 下载 MATH...")
        try:
            # 使用qwedsacf镜像（原始hendrycks/competition_math不可用）
            dataset = load_dataset("qwedsacf/competition_math")

            if "train" in dataset:
                data = dataset["train"]
            else:
                available_splits = list(dataset.keys())
                data = dataset[available_splits[0]]

            output_path = self.output_dir / "math" / "math.jsonl"
            with open(output_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

            print(f"  ✅ MATH: {len(data)} samples")
            return len(data)
        except Exception as e:
            print(f"  ❌ MATH 下载失败: {e}")
            return 0

    def download_squad2(self):
        """下载SQuAD 2.0"""
        print("\n📥 下载 SQuAD 2.0...")
        try:
            dataset = load_dataset("rajpurkar/squad_v2")

            output_path = self.output_dir / "qa" / "squad2.jsonl"
            with open(output_path, "w") as f:
                for item in dataset["train"]:
                    f.write(json.dumps(item) + "\n")

            print(f"  ✅ SQuAD 2.0: {len(dataset['train'])} samples")
            return len(dataset['train'])
        except Exception as e:
            print(f"  ❌ SQuAD 2.0 下载失败: {e}")
            return 0

    def download_hotpotqa(self):
        """下载HotpotQA"""
        print("\n📥 下载 HotpotQA...")
        try:
            dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")

            output_path = self.output_dir / "qa" / "hotpotqa.jsonl"
            with open(output_path, "w") as f:
                for item in dataset["train"]:
                    f.write(json.dumps(item) + "\n")

            print(f"  ✅ HotpotQA: {len(dataset['train'])} samples")
            return len(dataset['train'])
        except Exception as e:
            print(f"  ❌ HotpotQA 下载失败: {e}")
            return 0

    def download_humaneval(self):
        """下载HumanEval"""
        print("\n📥 下载 HumanEval...")
        try:
            dataset = load_dataset("openai/openai_humaneval")

            output_path = self.output_dir / "code" / "humaneval.jsonl"
            with open(output_path, "w") as f:
                for item in dataset["test"]:
                    f.write(json.dumps(item) + "\n")

            print(f"  ✅ HumanEval: {len(dataset['test'])} samples")
            return len(dataset['test'])
        except Exception as e:
            print(f"  ❌ HumanEval 下载失败: {e}")
            return 0

    def download_mbpp(self):
        """下载MBPP"""
        print("\n📥 下载 MBPP...")
        try:
            dataset = load_dataset("google-research-datasets/mbpp", "full")

            output_path = self.output_dir / "code" / "mbpp.jsonl"
            with open(output_path, "w") as f:
                for item in dataset["train"]:
                    f.write(json.dumps(item) + "\n")

            print(f"  ✅ MBPP: {len(dataset['train'])} samples")
            return len(dataset['train'])
        except Exception as e:
            print(f"  ❌ MBPP 下载失败: {e}")
            return 0

    def run_all(self):
        """下载所有数据集"""
        print("=" * 60)
        print("开始下载6个数据集...")
        print("=" * 60)

        stats = {
            "gsm8k": self.download_gsm8k(),
            "math": self.download_math(),
            "squad2": self.download_squad2(),
            "hotpotqa": self.download_hotpotqa(),
            "humaneval": self.download_humaneval(),
            "mbpp": self.download_mbpp(),
        }

        total = sum(stats.values())
        print("\n" + "=" * 60)
        print(f"✅ 下载完成！总计: {total} 样本")
        print("=" * 60)
        print("\n数据统计:")
        for dataset, count in stats.items():
            print(f"  {dataset}: {count} samples")

        # 保存统计信息
        with open(self.output_dir / "download_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.run_all()
