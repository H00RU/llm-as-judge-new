#!/usr/bin/env python3
"""
预处理数据集，统一为JSONL格式，支持混合训练

混合策略：
1. 原始分割：83.3% train : 16.7% test（5:1比例）
2. 域内均衡：每个域内两个数据集各占50%（小数据集上采样到大数据集大小）
3. 跨域混合：按4:3:3比例混合三个域（math:qa:code）
4. 输出：train_mixed、test_mixed两个文件（各自分别处理）
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

# 固定随机种子，确保数据分割的可重复性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

print(f"✅ 随机种子已固定为: {RANDOM_SEED} (确保数据分割可重复)")
print("=" * 80)

class DatasetProcessor:
    """数据集统一处理器"""

    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_data_separation(self, train_samples, test_samples, dataset_name):
        """
        确保训练和测试数据完全分离，避免数据泄露

        Args:
            train_samples: 训练样本列表
            test_samples: 测试样本列表
            dataset_name: 数据集名称（用于日志）

        Returns:
            (filtered_train_samples, filtered_test_samples, duplicate_count)
        """
        # 收集训练集中的所有original_id
        train_ids = set()
        for sample in train_samples:
            orig_id = sample.get("metadata", {}).get("original_id", "")
            if orig_id:
                train_ids.add(orig_id)

        # 过滤测试集中任何可能重复的ID
        filtered_test_samples = []
        duplicate_count = 0
        for sample in test_samples:
            orig_id = sample.get("metadata", {}).get("original_id", "")
            if orig_id and orig_id not in train_ids:
                filtered_test_samples.append(sample)
            elif not orig_id:  # 如果没有original_id，保留（为了向后兼容）
                filtered_test_samples.append(sample)
            else:
                duplicate_count += 1

        if duplicate_count > 0:
            print(f"    ⚠️  {dataset_name}: 发现并移除了 {duplicate_count} 个重复样本")

        return train_samples, filtered_test_samples, duplicate_count

    def process_gsm8k(self):
        """处理GSM8K"""
        print("\n处理 GSM8K...")
        input_file = self.raw_dir / "math" / "gsm8k.jsonl"
        output_dir = self.processed_dir / "gsm8k"
        output_dir.mkdir(exist_ok=True)

        if not input_file.exists():
            print(f"  ⚠️  文件不存在: {input_file}")
            return 0

        samples = []
        with open(input_file) as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    sample = {
                        "id": f"gsm8k_{idx}",
                        "dataset": "gsm8k",
                        "domain": "math",
                        "question": item.get("question", ""),
                        "reference_answer": item.get("answer", "").split("\n#### ")[-1].strip(),
                        "answer_type": "numeric",
                        "metadata": {
                            "source": "gsm8k",
                            "original_id": str(idx)
                        }
                    }
                    samples.append(sample)
                except Exception as e:
                    print(f"    处理第{idx}条失败: {e}")
                    continue

        random.shuffle(samples)
        n = len(samples)
        # 5:1分割 (83.3%:16.7%)
        train_idx = int(n * 5 / 6)

        train_samples = samples[:train_idx]
        test_samples = samples[train_idx:]

        # 确保数据完全分离，避免泄露
        final_train, final_test, duplicate_count = self._ensure_data_separation(
            train_samples, test_samples, "GSM8K"
        )

        self._save_jsonl(output_dir / "train.jsonl", final_train)
        self._save_jsonl(output_dir / "test.jsonl", final_test)

        meta = {
            "dataset": "gsm8k",
            "domain": "math",
            "total": len(samples),
            "train": len(final_train),
            "test": len(final_test),
            "filtered_duplicates": duplicate_count
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ✅ GSM8K: {len(samples)} 样本 (train:{len(samples[:train_idx])} test:{len(samples[train_idx:])})")
        return len(samples)

    def process_math(self):
        """处理MATH"""
        print("\n处理 MATH...")
        input_file = self.raw_dir / "math" / "math.jsonl"
        output_dir = self.processed_dir / "math"
        output_dir.mkdir(exist_ok=True)

        if not input_file.exists():
            print(f"  ⚠️  文件不存在: {input_file}")
            return 0

        samples = []
        with open(input_file) as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    sample = {
                        "id": f"math_{idx}",
                        "dataset": "math",
                        "domain": "math",
                        "question": item.get("problem", ""),
                        "reference_answer": item.get("solution", ""),
                        "answer_type": "text",
                        "metadata": {
                            "source": "math",
                            "original_id": str(idx)
                        }
                    }
                    samples.append(sample)
                except Exception as e:
                    print(f"    处理第{idx}条失败: {e}")
                    continue

        random.shuffle(samples)
        n = len(samples)
        train_idx = int(n * 5 / 6)

        self._save_jsonl(output_dir / "train.jsonl", samples[:train_idx])
        self._save_jsonl(output_dir / "test.jsonl", samples[train_idx:])

        meta = {
            "dataset": "math",
            "domain": "math",
            "total": len(samples),
            "train": len(samples[:train_idx]),
            "test": len(samples[train_idx:])
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ✅ MATH: {len(samples)} 样本 (train:{len(samples[:train_idx])} test:{len(samples[train_idx:])})")
        return len(samples)

    def process_squad2(self):
        """处理SQuAD2.0"""
        print("\n处理 SQuAD 2.0...")
        input_file = self.raw_dir / "qa" / "squad2.jsonl"
        output_dir = self.processed_dir / "squad2"
        output_dir.mkdir(exist_ok=True)

        if not input_file.exists():
            print(f"  ⚠️  文件不存在: {input_file}")
            return 0

        samples = []
        with open(input_file) as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    # SQuAD 2.0格式处理 - 包含无答案问题
                    answers = item.get("answers", {})
                    text_list = answers.get("text", [])

                    # 处理无答案问题（SQuAD 2.0特有）
                    if isinstance(text_list, list) and len(text_list) > 0:
                        answer_text = text_list[0]
                    elif isinstance(text_list, str):
                        answer_text = text_list
                    else:
                        # 无答案问题，使用特殊标记或跳过
                        answer_text = "无法回答"
                        # 或者跳过无答案问题：continue

                    sample = {
                        "id": f"squad2_{idx}",
                        "dataset": "squad2",
                        "domain": "qa",
                        "question": item.get("question", ""),
                        "reference_answer": answer_text,
                        "answer_type": "text",
                        "metadata": {
                            "source": "squad2",
                            "original_id": item.get("id", str(idx)),
                            "context": item.get("context", "")[:200],
                            "is_impossible": len(text_list) == 0  # 标记无答案问题
                        }
                    }
                    samples.append(sample)
                except Exception as e:
                    print(f"    处理第{idx}条失败: {e}")
                    continue

        random.shuffle(samples)
        n = len(samples)
        train_idx = int(n * 5 / 6)

        train_samples = samples[:train_idx]
        test_samples = samples[train_idx:]

        # 确保数据完全分离，避免泄露
        final_train, final_test, duplicate_count = self._ensure_data_separation(
            train_samples, test_samples, "SQuAD 2.0"
        )

        self._save_jsonl(output_dir / "train.jsonl", final_train)
        self._save_jsonl(output_dir / "test.jsonl", final_test)

        meta = {
            "dataset": "squad2",
            "domain": "qa",
            "total": len(samples),
            "train": len(final_train),
            "test": len(final_test),
            "filtered_duplicates": duplicate_count
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ✅ SQuAD 2.0: {len(samples)} 样本 (train:{len(samples[:train_idx])} test:{len(samples[train_idx:])})")
        return len(samples)

    def process_hotpotqa(self):
        """处理HotpotQA"""
        print("\n处理 HotpotQA...")
        input_file = self.raw_dir / "qa" / "hotpotqa.jsonl"
        output_dir = self.processed_dir / "hotpotqa"
        output_dir.mkdir(exist_ok=True)

        if not input_file.exists():
            print(f"  ⚠️  文件不存在: {input_file}")
            return 0

        samples = []
        with open(input_file) as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    sample = {
                        "id": f"hotpotqa_{idx}",
                        "dataset": "hotpotqa",
                        "domain": "qa",
                        "question": item.get("question", ""),
                        "reference_answer": item.get("answer", ""),
                        "answer_type": "text",
                        "metadata": {
                            "source": "hotpotqa",
                            "original_id": str(idx)
                        }
                    }
                    samples.append(sample)
                except Exception as e:
                    print(f"    处理第{idx}条失败: {e}")
                    continue

        random.shuffle(samples)
        n = len(samples)
        train_idx = int(n * 5 / 6)

        self._save_jsonl(output_dir / "train.jsonl", samples[:train_idx])
        self._save_jsonl(output_dir / "test.jsonl", samples[train_idx:])

        meta = {
            "dataset": "hotpotqa",
            "domain": "qa",
            "total": len(samples),
            "train": len(samples[:train_idx]),
            "test": len(samples[train_idx:])
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ✅ HotpotQA: {len(samples)} 样本 (train:{len(samples[:train_idx])} test:{len(samples[train_idx:])})")
        return len(samples)

    def process_humaneval(self):
        """处理HumanEval"""
        print("\n处理 HumanEval...")
        input_file = self.raw_dir / "code" / "humaneval.jsonl"
        output_dir = self.processed_dir / "humaneval"
        output_dir.mkdir(exist_ok=True)

        if not input_file.exists():
            print(f"  ⚠️  文件不存在: {input_file}")
            return 0

        samples = []
        with open(input_file) as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    sample = {
                        "id": f"humaneval_{idx}",
                        "dataset": "humaneval",
                        "domain": "code",
                        "question": item.get("prompt", ""),
                        "reference_answer": item.get("canonical_solution", ""),
                        "answer_type": "code",
                        "entry_point": item.get("entry_point", ""),  # ✅ 保留entry_point
                        "test": item.get("test", ""),  # ✅ 保留test字段（关键！）
                        "metadata": {
                            "source": "humaneval",
                            "original_id": str(item.get("task_id", idx))
                        }
                    }
                    samples.append(sample)
                except Exception as e:
                    print(f"    处理第{idx}条失败: {e}")
                    continue

        random.shuffle(samples)
        n = len(samples)
        train_idx = int(n * 5 / 6)

        self._save_jsonl(output_dir / "train.jsonl", samples[:train_idx])
        self._save_jsonl(output_dir / "test.jsonl", samples[train_idx:])

        meta = {
            "dataset": "humaneval",
            "domain": "code",
            "total": len(samples),
            "train": len(samples[:train_idx]),
            "test": len(samples[train_idx:])
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ✅ HumanEval: {len(samples)} 样本 (train:{len(samples[:train_idx])} test:{len(samples[train_idx:])})")
        return len(samples)

    def process_mbpp(self):
        """处理MBPP"""
        print("\n处理 MBPP...")
        input_file = self.raw_dir / "code" / "mbpp.jsonl"
        output_dir = self.processed_dir / "mbpp"
        output_dir.mkdir(exist_ok=True)

        if not input_file.exists():
            print(f"  ⚠️  文件不存在: {input_file}")
            return 0

        samples = []
        with open(input_file) as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line)

                    # 从code中提取函数名作为entry_point
                    code = item.get("code", "")
                    import re as regex_module
                    match = regex_module.search(r'def\s+(\w+)\s*\(', code)
                    entry_point = match.group(1) if match else f"func_{idx}"

                    # 处理测试用例（test_list转换为test字符串）
                    test_list = item.get("test_list", [])
                    if test_list:
                        # 合并多个测试用例为一个测试函数
                        test_code = f"def check(candidate):\n"
                        for test_case in test_list:
                            # 每个test_case是一个assert语句，需要将函数名替换为candidate
                            test_case = test_case.replace(entry_point, "candidate")
                            test_code += f"    {test_case}\n"
                        test = test_code
                    else:
                        test = ""

                    sample = {
                        "id": f"mbpp_{idx}",
                        "dataset": "mbpp",
                        "domain": "code",
                        "question": item.get("text", ""),
                        "reference_answer": item.get("code", ""),
                        "answer_type": "code",
                        "entry_point": entry_point,  # ✅ 从code提取函数名
                        "test": test,  # ✅ 保留test字段（转换后的测试函数）
                        "metadata": {
                            "source": "mbpp",
                            "original_id": str(item.get("task_id", idx))
                        }
                    }
                    samples.append(sample)
                except Exception as e:
                    print(f"    处理第{idx}条失败: {e}")
                    continue

        random.shuffle(samples)
        n = len(samples)
        train_idx = int(n * 5 / 6)

        self._save_jsonl(output_dir / "train.jsonl", samples[:train_idx])
        self._save_jsonl(output_dir / "test.jsonl", samples[train_idx:])

        meta = {
            "dataset": "mbpp",
            "domain": "code",
            "total": len(samples),
            "train": len(samples[:train_idx]),
            "test": len(samples[train_idx:])
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ✅ MBPP: {len(samples)} 样本 (train:{len(samples[:train_idx])} test:{len(samples[train_idx:])})")
        return len(samples)

    def _balance_domain_split(self, domain: str, split: str, datasets_structure: Dict) -> List[Dict]:
        """
        在单个域内进行均衡采样（50:50）

        Args:
            domain: 域名 (math/qa/code)
            split: 数据分割 (train/test)
            datasets_structure: 数据集结构

        Returns:
            均衡后的该域所有数据
        """
        dataset_names = datasets_structure[domain]

        # 加载该域的两个数据集的指定分割
        datasets_data = {}
        for ds_name in dataset_names:
            dataset_dir = self.processed_dir / ds_name
            split_file = dataset_dir / f"{split}.jsonl"

            if split_file.exists():
                with open(split_file) as f:
                    datasets_data[ds_name] = [json.loads(line) for line in f]
            else:
                datasets_data[ds_name] = []

        # 找到该域内最大的数据集大小
        max_size = max(len(datasets_data[ds_name]) for ds_name in dataset_names)

        if max_size == 0:
            return []

        # 对每个数据集进行采样至max_size（允许重复）
        print(f"    [{split.upper()}] {domain.upper()}域均衡:")
        balanced_data = []

        for ds_name in dataset_names:
            data = datasets_data[ds_name]
            if len(data) < max_size:
                # 小数据集：重复采样
                balanced = random.choices(data, k=max_size)
                print(f"      {ds_name:15} {len(data):6,} → {len(balanced):6,} (重采样)")
            else:
                # 大数据集或相等：直接采样
                balanced = random.sample(data, max_size)
                print(f"      {ds_name:15} {len(data):6,} → {len(balanced):6,} (欠采样)")

            balanced_data.extend(balanced)

        return balanced_data

    def create_mixed_dataset(self):
        """
        创建混合训练数据

        步骤：
        1. 原始数据集分割：83.3% train : 16.7% test
        2. 对train/test分别执行：
           a. 域内均衡：每个域内两个数据集各占50%
           b. 跨域4:3:3混合：math:qa:code = 4:3:3
        3. 输出：train_mixed、test_mixed两个文件
        """
        print("\n" + "=" * 80)
        print("创建混合训练数据")
        print("=" * 80)

        mixed_dir = self.processed_dir.parent / "mixed"
        mixed_dir.mkdir(exist_ok=True)

        datasets_structure = {
            "math": ["gsm8k", "math"],
            "qa": ["squad2", "hotpotqa"],
            "code": ["humaneval", "mbpp"]
        }

        # 对train/test两个分割分别处理
        mixed_data = {}

        for split in ["train", "test"]:
            print(f"\n📊 步骤1：{split.upper()}部分的域内均衡采样")

            # 步骤1：域内均衡
            domain_balanced_data = {}
            for domain in datasets_structure.keys():
                domain_balanced_data[domain] = self._balance_domain_split(domain, split, datasets_structure)

            # 步骤2：跨域4:3:3混合
            print(f"\n🎯 步骤2：{split.upper()}部分的跨域4:3:3混合")

            math_data = domain_balanced_data["math"]
            qa_data = domain_balanced_data["qa"]
            code_data = domain_balanced_data["code"]

            # 计算跨域采样大小（min原则确保比例一致）
            total_available = min(
                int(len(math_data) / 0.4),
                int(len(qa_data) / 0.3),
                int(len(code_data) / 0.3)
            )

            math_count = int(total_available * 0.4)
            qa_count = int(total_available * 0.3)
            code_count = int(total_available * 0.3)

            # 采样
            math_samples = random.choices(math_data, k=math_count) if len(math_data) > 0 else []
            qa_samples = random.choices(qa_data, k=qa_count) if len(qa_data) > 0 else []
            code_samples = random.choices(code_data, k=code_count) if len(code_data) > 0 else []

            print(f"  采样结果:")
            print(f"    math: {len(math_samples):8,} (40.0%)")
            print(f"    qa:   {len(qa_samples):8,} (30.0%)")
            print(f"    code: {len(code_samples):8,} (30.0%)")

            # 合并并shuffle
            all_mixed = math_samples + qa_samples + code_samples
            random.shuffle(all_mixed)

            mixed_data[split] = all_mixed
            print(f"  总计: {len(all_mixed):,} 样本")

        # 保存mixed数据
        print(f"\n💾 保存混合数据:")
        self._save_jsonl(mixed_dir / "train_mixed.jsonl", mixed_data["train"])
        self._save_jsonl(mixed_dir / "test_mixed.jsonl", mixed_data["test"])

        print(f"  ✅ train_mixed.jsonl: {len(mixed_data['train']):,} 样本")
        print(f"  ✅ test_mixed.jsonl:  {len(mixed_data['test']):,} 样本")

        # 保存信息
        info = {
            "split_ratio": "5:1 (train:test = 83.3%:16.7%)",
            "domain_intra_balance": "50:50 per domain (small dataset resampled to match large)",
            "cross_domain_ratio": "4:3:3 (math:qa:code)",
            "total_train": len(mixed_data["train"]),
            "total_test": len(mixed_data["test"]),
            "domain_distribution_train": {
                "math": sum(1 for x in mixed_data["train"] if x["domain"] == "math"),
                "qa": sum(1 for x in mixed_data["train"] if x["domain"] == "qa"),
                "code": sum(1 for x in mixed_data["train"] if x["domain"] == "code")
            }
        }

        # 计算百分比
        for domain_key, count in info["domain_distribution_train"].items():
            info[f"{domain_key}_pct"] = round(count / len(mixed_data["train"]) * 100, 2) if mixed_data["train"] else 0

        with open(mixed_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"\n📋 信息已保存到: {mixed_dir}/info.json")

        return info

    def create_test_sets(self):
        """创建单个数据集的test文件供后续评估"""
        print("\n" + "=" * 80)
        print("创建单个数据集的test集（用于分别评估各域）")
        print("=" * 80)

        test_dir = self.processed_dir.parent / "test"
        test_dir.mkdir(exist_ok=True)

        test_index = {}
        for dataset_name in ["gsm8k", "math", "squad2", "hotpotqa", "humaneval", "mbpp"]:
            dataset_dir = self.processed_dir / dataset_name
            test_file = dataset_dir / "test.jsonl"

            if test_file.exists():
                output_file = test_dir / f"{dataset_name}_test.jsonl"
                with open(test_file) as src, open(output_file, "w") as dst:
                    for line in src:
                        dst.write(line)

                with open(test_file) as f:
                    count = sum(1 for _ in f)

                # 处理路径：先尝试相对路径，失败则使用绝对路径
                try:
                    rel_path = output_file.relative_to(Path.cwd())
                    test_index[dataset_name] = str(rel_path)
                except ValueError:
                    # 如果不在当前工作目录下，直接使用输出文件的路径字符串
                    test_index[dataset_name] = str(output_file.relative_to(test_dir.parent))

                print(f"  ✅ {dataset_name}_test.jsonl: {count} 样本")

        with open(test_dir / "test_index.json", "w") as f:
            json.dump(test_index, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 所有test集已准备好，位置: {test_dir}/")

    def _save_jsonl(self, filepath: Path, samples: List[Dict]):
        """保存为JSONL"""
        with open(filepath, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def run_all(self):
        """运行所有处理"""
        print("\n" + "=" * 80)
        print("开始处理6个数据集（5:1分割）")
        print("=" * 80)

        total_counts = {}
        total_counts["gsm8k"] = self.process_gsm8k()
        total_counts["math"] = self.process_math()
        total_counts["squad2"] = self.process_squad2()
        total_counts["hotpotqa"] = self.process_hotpotqa()
        total_counts["humaneval"] = self.process_humaneval()
        total_counts["mbpp"] = self.process_mbpp()

        self.create_mixed_dataset()
        self.create_test_sets()

        # 保存总索引
        index = {
            "datasets": total_counts,
            "processed_dir": str(self.processed_dir),
            "mixed_dir": str(self.processed_dir.parent / "mixed"),
            "test_dir": str(self.processed_dir.parent / "test"),
            "total_samples": sum(total_counts.values()),
            "split_ratio": "5:1 (train:test)"
        }
        with open(self.processed_dir.parent / "index.json", "w") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 80)
        print("✅ 数据处理完成！")
        print("=" * 80)
        print(f"\n📂 输出目录结构:")
        print(f"  data/processed/          - 各数据集分别处理后（train/test）")
        print(f"  data/mixed/              - 混合后的train_mixed/test_mixed")
        print(f"  data/test/               - 各数据集的test文件（用于单独评估）")

if __name__ == "__main__":
    processor = DatasetProcessor()
    processor.run_all()
