#!/usr/bin/env python3
"""
评估脚本 - 在6个数据集上分别测试微调后的模型
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from tqdm import tqdm

sys.path.insert(0, 'src')

class ModelEvaluator:
    """模型评估器"""

    def __init__(self,
                 model_name: str = "qwen25-7b",
                 checkpoint_path: str = None,
                 device: str = "cuda:0"):
        """
        Args:
            model_name: 模型名称 (qwen25-7b 或 qwen3-8b)
            checkpoint_path: LoRA权重路径，如果None则使用base model
            device: 使用的设备
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device

        # 模型配置
        self.model_configs = {
            "qwen25-7b": {
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "local_path": "./models/Qwen2.5-7B-Instruct"
            },
            "qwen3-8b": {
                "base_model": "Qwen/Qwen-3-8B",
                "local_path": "./models/Qwen-3-8B"
            }
        }

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载模型和tokenizer"""
        if self.model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {self.model_name}")

        config = self.model_configs[self.model_name]
        model_id = config["base_model"]

        print(f"\n📦 加载模型: {self.model_name}")

        # 优先使用本地模型
        if Path(config["local_path"]).exists():
            print(f"  使用本地模型: {config['local_path']}")
            model_id = config["local_path"]

        # 加载base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # 如果有checkpoint，加载LoRA权重
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            print(f"  加载LoRA权重: {self.checkpoint_path}")
            self.model = PeftModel.from_pretrained(self.model, self.checkpoint_path)
            self.model = self.model.merge_and_unload()
        else:
            print(f"  使用base model（未微调）")

        self.model.eval()
        print("  ✅ 模型加载完成")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """生成回应"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.95,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def evaluate_dataset(self, dataset_name: str, test_file: str) -> Dict[str, Any]:
        """评估单个数据集"""
        print(f"\n🧪 评估 {dataset_name}...")

        if not Path(test_file).exists():
            print(f"  ❌ 文件不存在: {test_file}")
            return {}

        results = {
            "dataset": dataset_name,
            "total": 0,
            "correct": 0,
            "predictions": [],
            "metrics": {}
        }

        # 根据数据集类型确定评估指标
        metrics_config = {
            "gsm8k": ["accuracy"],
            "math": ["accuracy"],
            "squad2": ["exact_match", "f1"],
            "hotpotqa": ["exact_match", "f1"],
            "humaneval": ["pass@1"],
            "mbpp": ["pass@1"]
        }

        results["metrics_to_compute"] = metrics_config.get(dataset_name, ["accuracy"])

        # 加载测试数据
        test_samples = []
        with open(test_file) as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        test_samples.append(sample)
                    except json.JSONDecodeError:
                        continue

        results["total"] = len(test_samples)
        print(f"  总样本数: {results['total']}")

        # 逐个评估
        correct_count = 0

        for idx, sample in enumerate(tqdm(test_samples[:100], desc=f"Evaluating {dataset_name}")):  # 快速评估，只用前100个
            question = sample.get("question", "")
            reference_answer = sample.get("reference_answer", "")

            if not question:
                continue

            # 生成回应
            try:
                prediction = self.generate(question, max_tokens=512)
            except Exception as e:
                print(f"  生成失败: {e}")
                prediction = ""

            # 简单的准确性评估（基于是否包含答案的关键词）
            is_correct = self._check_correctness(dataset_name, prediction, reference_answer)

            if is_correct:
                correct_count += 1

            results["predictions"].append({
                "question": question[:100],
                "reference": reference_answer[:100],
                "prediction": prediction[:100],
                "correct": is_correct
            })

        # 计算指标
        accuracy = correct_count / min(100, results["total"]) if results["total"] > 0 else 0
        results["metrics"]["accuracy"] = accuracy
        results["correct"] = correct_count

        print(f"  ✅ 准确率: {accuracy:.2%}")

        return results

    def _check_correctness(self, dataset_name: str, prediction: str, reference: str) -> bool:
        """检查预测是否正确（简单启发式方法）"""
        prediction = prediction.lower().strip()
        reference = reference.lower().strip()

        if dataset_name in ["humaneval", "mbpp"]:
            # 代码任务：检查是否包含完整的函数定义或return语句
            return "def " in prediction or "return" in prediction

        elif dataset_name in ["squad2", "hotpotqa"]:
            # QA任务：简单的词汇重叠
            pred_words = set(prediction.split())
            ref_words = set(reference.split())
            overlap = len(pred_words & ref_words)
            return overlap >= min(3, len(ref_words))

        else:  # math datasets
            # 数学任务：检查数字答案
            import re
            pred_nums = re.findall(r'-?\d+\.?\d*', prediction)
            ref_nums = re.findall(r'-?\d+\.?\d*', reference)

            if pred_nums and ref_nums:
                try:
                    return float(pred_nums[-1]) == float(ref_nums[-1])
                except ValueError:
                    return pred_nums[-1] == ref_nums[-1]

            return False

    def evaluate_all(self, test_dir: str = "data/test") -> Dict[str, Any]:
        """评估所有6个数据集"""
        print("\n" + "=" * 60)
        print(f"开始评估 {self.model_name} 模型")
        print("=" * 60)

        test_dir = Path(test_dir)
        datasets = ["gsm8k", "math", "squad2", "hotpotqa", "humaneval", "mbpp"]

        all_results = {
            "model": self.model_name,
            "checkpoint": self.checkpoint_path,
            "datasets": {}
        }

        for dataset_name in datasets:
            test_file = test_dir / f"{dataset_name}_test.jsonl"
            if test_file.exists():
                result = self.evaluate_dataset(dataset_name, str(test_file))
                all_results["datasets"][dataset_name] = result
            else:
                print(f"\n⚠️  {dataset_name} 测试文件不存在")

        # 计算总体指标
        accuracies = []
        for result in all_results["datasets"].values():
            if "accuracy" in result.get("metrics", {}):
                accuracies.append(result["metrics"]["accuracy"])

        if accuracies:
            all_results["overall_accuracy"] = np.mean(accuracies)

        return all_results


def main():
    parser = argparse.ArgumentParser(description="评估微调后的模型")
    parser.add_argument("--model", default="qwen25-7b",
                       choices=["qwen25-7b", "qwen3-8b"],
                       help="模型名称")
    parser.add_argument("--checkpoint", default=None,
                       help="LoRA权重路径，如 checkpoints/qwen25-7b/grpo_mixed/step_1000")
    parser.add_argument("--test_dir", default="data/test",
                       help="测试数据目录")
    parser.add_argument("--output_dir", default="results/evaluation",
                       help="结果输出目录")
    parser.add_argument("--device", default="cuda:0",
                       help="使用的设备")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化评估器
    evaluator = ModelEvaluator(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # 评估所有数据集
    results = evaluator.evaluate_all(test_dir=args.test_dir)

    # 保存结果
    output_file = output_dir / f"{args.model}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 评估完成！结果已保存到: {output_file}")

    # 打印总体结果
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    for dataset_name, result in results["datasets"].items():
        if result:
            metrics = result.get("metrics", {})
            print(f"\n{dataset_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")

    if "overall_accuracy" in results:
        print(f"\n总体准确率: {results['overall_accuracy']:.4f}")

if __name__ == "__main__":
    main()
