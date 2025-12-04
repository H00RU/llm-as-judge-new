#!/usr/bin/env python3
"""
Phase 2: Data Schema Validation and Standardization
验证所有数据样本符合标准schema，检查字段完整性和格式一致性
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def validate_schema():
    """验证数据schema和字段完整性"""

    data_dir = Path("data/mixed")
    train_file = data_dir / "train_mixed.jsonl"
    test_file = data_dir / "test_mixed.jsonl"

    # 必需字段（基于process_datasets输出格式）
    REQUIRED_FIELDS = {
        "math": ["id", "dataset", "question", "reference_answer", "domain", "answer_type"],
        "code": ["id", "dataset", "question", "reference_answer", "domain", "answer_type", "entry_point", "test"],
        "qa": ["id", "dataset", "question", "reference_answer", "domain", "answer_type"],
    }

    def validate_file(filepath: Path, split_name: str) -> Dict:
        """验证单个文件"""
        result = {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "domain_distribution": defaultdict(int),
            "missing_fields": [],
            "type_mismatches": [],
            "format_issues": [],
            "samples_by_domain": defaultdict(list),
        }

        if not filepath.exists():
            print(f"❌ 文件不存在: {filepath}")
            return result

        print(f"\n📋 验证 {split_name}: {filepath}")

        with open(filepath, 'r') as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    sample = json.loads(line)
                    result["total_samples"] += 1

                    # 检查domain字段
                    domain = sample.get("domain", "unknown")
                    if domain not in REQUIRED_FIELDS:
                        result["format_issues"].append(
                            f"Line {line_no}: Unknown domain '{domain}' (must be math/code/qa)"
                        )
                        continue

                    result["domain_distribution"][domain] += 1

                    # 检查必需字段
                    required = REQUIRED_FIELDS[domain]
                    missing = [field for field in required if field not in sample or not sample.get(field)]

                    if missing:
                        result["missing_fields"].append({
                            "line": line_no,
                            "id": sample.get("id", "UNKNOWN"),
                            "domain": domain,
                            "missing": missing
                        })
                        continue

                    # 检查字���类型
                    type_checks = {
                        "id": str,
                        "dataset": str,
                        "question": str,
                        "reference_answer": str,
                        "domain": str,
                        "answer_type": str,
                    }

                    for field, expected_type in type_checks.items():
                        if field in sample and not isinstance(sample[field], expected_type):
                            result["type_mismatches"].append({
                                "line": line_no,
                                "field": field,
                                "expected": expected_type.__name__,
                                "actual": type(sample[field]).__name__
                            })

                    # 对于code类型，检查entry_point和test的格式
                    if domain == "code":
                        if "entry_point" in sample:
                            if not isinstance(sample["entry_point"], str) or not sample["entry_point"].strip():
                                result["format_issues"].append(
                                    f"Line {line_no}: Invalid entry_point format (should be non-empty string)"
                                )
                                continue
                        if "test" in sample:
                            if not isinstance(sample["test"], str) or not sample["test"].strip():
                                result["format_issues"].append(
                                    f"Line {line_no}: Invalid test format (should be non-empty string)"
                                )
                                continue

                    # 检查answer_type是否有效
                    valid_answer_types = {
                        "math": ["numeric", "text"],
                        "code": ["code"],
                        "qa": ["text"],
                    }
                    answer_type = sample.get("answer_type", "")
                    if answer_type not in valid_answer_types.get(domain, []):
                        result["format_issues"].append(
                            f"Line {line_no}: Invalid answer_type '{answer_type}' for domain {domain}"
                        )
                        continue

                    # 所有检查通过
                    result["valid_samples"] += 1
                    result["samples_by_domain"][domain].append(sample["id"])

                except json.JSONDecodeError as e:
                    result["format_issues"].append(f"Line {line_no}: JSON decode error - {str(e)}")
                except Exception as e:
                    result["format_issues"].append(f"Line {line_no}: Unexpected error - {str(e)}")

        result["invalid_samples"] = result["total_samples"] - result["valid_samples"]

        return result

    # 验证训练集
    train_result = validate_file(train_file, "TRAIN")

    # 验证测试集
    test_result = validate_file(test_file, "TEST")

    # 生成报告
    print("\n" + "="*70)
    print("📊 SCHEMA VALIDATION REPORT")
    print("="*70)

    print("\n【TRAIN SET】")
    print(f"  总样本数: {train_result['total_samples']}")
    print(f"  ✅ 有效样本: {train_result['valid_samples']}")
    print(f"  ❌ 无效样本: {train_result['invalid_samples']}")
    print(f"  有效率: {100*train_result['valid_samples']/max(train_result['total_samples'], 1):.1f}%")

    print("\n  域分布:")
    total_train = sum(train_result['domain_distribution'].values())
    for domain in ["math", "code", "qa"]:
        count = train_result['domain_distribution'].get(domain, 0)
        ratio = 100 * count / total_train if total_train > 0 else 0
        print(f"    - {domain:6s}: {count:5d} ({ratio:5.1f}%)")

    if train_result['missing_fields']:
        print(f"\n  ⚠️  缺失字段问题 ({len(train_result['missing_fields'])}):")
        for issue in train_result['missing_fields'][:5]:  # 只显示前5个
            print(f"    Line {issue['line']}: {issue['missing']} (ID: {issue['id']})")
        if len(train_result['missing_fields']) > 5:
            print(f"    ... 还有 {len(train_result['missing_fields']) - 5} 个问题")

    if train_result['type_mismatches']:
        print(f"\n  ⚠️  字段类型错误 ({len(train_result['type_mismatches'])}):")
        for issue in train_result['type_mismatches'][:3]:
            print(f"    Line {issue['line']}: {issue['field']} expected {issue['expected']}, got {issue['actual']}")
        if len(train_result['type_mismatches']) > 3:
            print(f"    ... 还有 {len(train_result['type_mismatches']) - 3} 个问题")

    if train_result['format_issues']:
        print(f"\n  ⚠️  格式问题 ({len(train_result['format_issues'])}):")
        for issue in train_result['format_issues'][:5]:
            print(f"    {issue}")
        if len(train_result['format_issues']) > 5:
            print(f"    ... 还有 {len(train_result['format_issues']) - 5} 个问题")

    print("\n【TEST SET】")
    print(f"  总样本数: {test_result['total_samples']}")
    print(f"  ✅ 有效样本: {test_result['valid_samples']}")
    print(f"  ❌ 无效样本: {test_result['invalid_samples']}")
    print(f"  有效率: {100*test_result['valid_samples']/max(test_result['total_samples'], 1):.1f}%")

    print("\n  域分布:")
    total_test = sum(test_result['domain_distribution'].values())
    for domain in ["math", "code", "qa"]:
        count = test_result['domain_distribution'].get(domain, 0)
        ratio = 100 * count / total_test if total_test > 0 else 0
        print(f"    - {domain:6s}: {count:5d} ({ratio:5.1f}%)")

    # 总结
    print("\n" + "="*70)
    total_valid = train_result['valid_samples'] + test_result['valid_samples']
    total_all = train_result['total_samples'] + test_result['total_samples']

    if train_result['valid_samples'] == train_result['total_samples'] and test_result['valid_samples'] == test_result['total_samples']:
        print("✅ 全部样本通过schema验证！")
    else:
        print(f"⚠️  验证完成: {total_valid}/{total_all} 样本有效 ({100*total_valid/max(total_all, 1):.1f}%)")

    # 检查训练比例
    train_total = sum(train_result['domain_distribution'].values())
    train_ratios = {
        domain: train_result['domain_distribution'].get(domain, 0) / train_total
        for domain in ["math", "code", "qa"]
    }

    print(f"\n【数据均衡检查】")
    print(f"  配置比例: math=0.4, code=0.3, qa=0.3")
    print(f"  实际比例: math={train_ratios['math']:.2%}, code={train_ratios['code']:.2%}, qa={train_ratios['qa']:.2%}")

    # 返回验证结果
    return {
        "train": train_result,
        "test": test_result,
        "overall_valid": total_valid == total_all,
        "total_valid_samples": total_valid,
        "total_samples": total_all,
    }

if __name__ == "__main__":
    result = validate_schema()
    print("\n✅ Schema validation complete!")
