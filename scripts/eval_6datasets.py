#!/usr/bin/env python3
"""
评估脚本 - 在6个数据集上分别测试微调后的模型

✅ 修复后的评估流程（与训练流程一致）：
问题 → Qwen生成workflow代码 → AFlow执行workflow → gpt-4o-mini运行算子 → 答案 → 准确性评估
      (RL策略模型)           (工作流引擎)       (执行引擎)      (精确匹配/LLM Judge)

❌ 旧版本错误流程（已废弃）：
问题 → Qwen直接生成答案 → 简单字符串匹配 → "准确率"
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from tqdm import tqdm
import yaml

sys.path.insert(0, 'src')

from rl_workflow_generator import RLWorkflowGenerator
from aflow_executor import AFlowExecutor
from reward_computer import RewardComputer

class ModelEvaluator:
    """模型评估器 - 使用与训练一致的workflow生成→执行流程"""

    def __init__(self,
                 config_path: str = "config/training.yaml",
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda:0"):
        """
        Args:
            config_path: 训练配置文件路径
            checkpoint_path: LoRA权重路径，如果None则使用base model
            device: 使用的设备
        """
        self.checkpoint_path = checkpoint_path
        self.device = device

        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 初始化workflow生成器
        print(f"\n📦 初始化评估器...")
        self.workflow_generator = RLWorkflowGenerator(
            model_name_or_path=checkpoint_path if checkpoint_path else self.config['base_model'],
            config=self.config,
            device=device
        )
        print("  ✅ Workflow生成器加载完成")

        # 初始化AFlow执行器
        self.executor = AFlowExecutor(
            aflow_config_path=self.config['aflow_config_path'],
            operator_descriptions_path=self.config['aflow_operator_descriptions_path']
        )
        print("  ✅ AFlow执行器初始化完成")

        # 初始化奖励计算器（用于评估答案正确性）
        self.reward_computer = RewardComputer(
            config=self.config,
            aflow_config_path=self.config['aflow_config_path']
        )
        print("  ✅ 奖励计算器初始化完成")

    async def generate_and_execute_workflow(self,
                                            problem: str,
                                            problem_type: str,
                                            entry_point: str = '',
                                            test: str = '') -> Dict[str, Any]:
        """
        生成workflow并执行（与训练流程一致）

        Returns:
            Dict包含:
            - answer: 最终答案
            - workflow_code: 生成的workflow代码
            - success: 是否执行成功
            - metadata: 执行元数据
        """
        # 1. 生成workflow代码
        workflow_code = self.workflow_generator.generate_workflow(
            problem=problem,
            problem_type=problem_type
        )

        # 2. 执行workflow
        try:
            answer, cost, metadata = await self.executor.execute_workflow(
                workflow_code=workflow_code,
                problem=problem,
                problem_type=problem_type,
                entry_point=entry_point,
                test=test
            )

            return {
                'answer': answer,
                'workflow_code': workflow_code,
                'success': metadata.get('success', False),
                'metadata': metadata
            }
        except Exception as e:
            return {
                'answer': None,
                'workflow_code': workflow_code,
                'success': False,
                'metadata': {'error': str(e), 'success': False}
            }

    async def evaluate_dataset(self, dataset_name: str, test_file: str) -> Dict[str, Any]:
        """评估单个数据集"""
        print(f"\n🧪 评估 {dataset_name}...")

        if not Path(test_file).exists():
            print(f"  ❌ 文件不存在: {test_file}")
            return {}

        # 映射数据集到问题类型
        dataset_to_type = {
            "gsm8k": "math",
            "math": "math",
            "squad2": "qa",
            "hotpotqa": "qa",
            "humaneval": "code",
            "mbpp": "code"
        }
        problem_type = dataset_to_type.get(dataset_name, "qa")

        results = {
            "dataset": dataset_name,
            "problem_type": problem_type,
            "total": 0,
            "correct": 0,
            "execution_success": 0,
            "predictions": [],
            "metrics": {}
        }

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

        # 逐个评估（限制前100个用于快速评估）
        correct_count = 0
        success_count = 0

        for idx, sample in enumerate(tqdm(test_samples[:100], desc=f"Evaluating {dataset_name}")):
            question = sample.get("question", "")
            reference_answer = sample.get("reference_answer", "")
            entry_point = sample.get('entry_point', '')
            test = sample.get('test', '')

            if not question:
                continue

            # 生成workflow并执行
            try:
                result = await self.generate_and_execute_workflow(
                    problem=question,
                    problem_type=problem_type,
                    entry_point=entry_point,
                    test=test
                )

                answer = result['answer']
                success = result['success']
                metadata = result['metadata']

                if success:
                    success_count += 1

                # 使用reward_computer评估正确性
                is_correct = self._evaluate_correctness(
                    prediction=answer,
                    reference=reference_answer,
                    problem_type=problem_type,
                    metadata=metadata,
                    problem=question
                )

                if is_correct:
                    correct_count += 1

                results["predictions"].append({
                    "question": question[:100],
                    "reference": str(reference_answer)[:100],
                    "prediction": str(answer)[:100],
                    "correct": is_correct,
                    "execution_success": success
                })

            except Exception as e:
                print(f"  评估失败: {e}")
                results["predictions"].append({
                    "question": question[:100],
                    "reference": str(reference_answer)[:100],
                    "prediction": "ERROR",
                    "correct": False,
                    "execution_success": False,
                    "error": str(e)
                })

        # 计算指标
        sample_count = min(100, results["total"])
        accuracy = correct_count / sample_count if sample_count > 0 else 0
        execution_rate = success_count / sample_count if sample_count > 0 else 0

        results["metrics"]["accuracy"] = accuracy
        results["metrics"]["execution_success_rate"] = execution_rate
        results["correct"] = correct_count
        results["execution_success"] = success_count

        print(f"  ✅ 准确率: {accuracy:.2%} | 执行成功率: {execution_rate:.2%}")

        return results

    def _evaluate_correctness(self,
                              prediction: Any,
                              reference: str,
                              problem_type: str,
                              metadata: Dict,
                              problem: str = '') -> bool:
        """
        评估答案正确性 - 使用与训练一致的LLM Judge评估

        Returns:
            bool: 是否正确
        """
        # 如果执行失败，直接返回False
        if not metadata.get('success', False):
            return False

        # 如果答案为None或空，返回False
        if prediction is None or str(prediction).strip() == '':
            return False

        # 使用reward_computer的LLM Judge评估（公共接口）
        try:
            is_correct = self.reward_computer.llm_judge_compare(
                problem=problem,
                prediction=str(prediction),
                ground_truth=str(reference),
                problem_type=problem_type
            )
            return is_correct
        except Exception as e:
            print(f"    ⚠️ LLM Judge评估失败: {e}")
            return False

    async def evaluate_all(self, test_dir: str = "data/test") -> Dict[str, Any]:
        """评估所有6个数据集"""
        print("\n" + "=" * 60)
        print(f"开始评估模型（使用workflow生成→执行流程）")
        print("=" * 60)

        test_dir = Path(test_dir)
        datasets = ["gsm8k", "math", "squad2", "hotpotqa", "humaneval", "mbpp"]

        all_results = {
            "checkpoint": self.checkpoint_path,
            "datasets": {}
        }

        for dataset_name in datasets:
            test_file = test_dir / f"{dataset_name}_test.jsonl"
            if test_file.exists():
                result = await self.evaluate_dataset(dataset_name, str(test_file))
                all_results["datasets"][dataset_name] = result
            else:
                print(f"\n⚠️  {dataset_name} 测试文件不存在")

        # 计算总体指标
        accuracies = []
        execution_rates = []
        for result in all_results["datasets"].values():
            if "accuracy" in result.get("metrics", {}):
                accuracies.append(result["metrics"]["accuracy"])
            if "execution_success_rate" in result.get("metrics", {}):
                execution_rates.append(result["metrics"]["execution_success_rate"])

        if accuracies:
            all_results["overall_accuracy"] = np.mean(accuracies)
        if execution_rates:
            all_results["overall_execution_success_rate"] = np.mean(execution_rates)

        return all_results


def main():
    parser = argparse.ArgumentParser(description="评估微调后的模型")
    parser.add_argument("--config", default="config/training.yaml",
                       help="训练配置文件路径")
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
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # 评估所有数据集（使用async）
    print("\n🚀 开始评估（workflow生成→AFlow执行流程）...")
    results = asyncio.run(evaluator.evaluate_all(test_dir=args.test_dir))

    # 保存结果
    checkpoint_name = Path(args.checkpoint).name if args.checkpoint else "base_model"
    output_file = output_dir / f"{checkpoint_name}_results.json"
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
    if "overall_execution_success_rate" in results:
        print(f"总体执行成功率: {results['overall_execution_success_rate']:.4f}")

if __name__ == "__main__":
    main()
