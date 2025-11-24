#!/usr/bin/env python3
"""
设置数据路径映射 - 适配AFlow与项目的数据结构差异

作用：
1. 创建 data/datasets/ 目录结构
2. 从 data/raw/code/ 创建symlink到 data/datasets/
3. 确保AFlow的Test operator能找到测试数据
4. 保持数据单一真值源（data/raw/）

运行场景：
- 初次设置项目后
- 每次下载新数据后
- 训练前的环境检查
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

class DataPathSetup:
    """数据路径设置器"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).absolute()
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.datasets_dir = self.data_dir / "datasets"
        self.processed_dir = self.data_dir / "processed"
        self.code_raw_dir = self.raw_dir / "code"

    def check_source_data(self) -> Dict[str, bool]:
        """检查原始数据是否存在"""
        print("\n" + "=" * 80)
        print("📋 检查原始数据源")
        print("=" * 80)

        status = {
            "humaneval.jsonl": (self.code_raw_dir / "humaneval.jsonl").exists(),
            "mbpp.jsonl": (self.code_raw_dir / "mbpp.jsonl").exists(),
        }

        for dataset, exists in status.items():
            symbol = "✅" if exists else "❌"
            path = self.code_raw_dir / dataset
            if exists:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  {symbol} {dataset:25} ({size_mb:>6.1f} MB)")
            else:
                print(f"  {symbol} {dataset:25}")

        all_exist = all(status.values())
        if all_exist:
            print("\n✅ 所有原始数据源都存在")
        else:
            print("\n⚠️  某些原始数据源缺失！")
            print("请运行: python scripts/download_datasets.py")

        return status

    def create_datasets_dir(self) -> bool:
        """创建 data/datasets 目录"""
        try:
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n✅ 创建目录: {self.datasets_dir}")
            return True
        except Exception as e:
            print(f"\n❌ 创建目录失败: {e}")
            return False

    def setup_symlinks(self) -> Dict[str, Tuple[bool, str]]:
        """创建symlink（优先方案）"""
        print("\n" + "=" * 80)
        print("🔗 创建数据路径映射 (Symlink)")
        print("=" * 80)

        mapping = {
            "humaneval_public_test.jsonl": self.code_raw_dir / "humaneval.jsonl",
            "mbpp_public_test.jsonl": self.code_raw_dir / "mbpp.jsonl",
        }

        results = {}
        for target_name, source_path in mapping.items():
            target_path = self.datasets_dir / target_name
            results[target_name] = (False, "未设置")

            if not source_path.exists():
                print(f"  ⚠️  源文件不存在，跳过: {source_path}")
                results[target_name] = (False, "源文件不存在")
                continue

            # 如果target已存在，先删除（支持重新链接）
            if target_path.exists() or target_path.is_symlink():
                try:
                    if target_path.is_symlink():
                        target_path.unlink()
                        print(f"  ℹ️  移除旧symlink: {target_name}")
                    else:
                        # 如果是文件，备份它
                        import shutil
                        backup_path = target_path.with_suffix(".jsonl.backup")
                        shutil.move(str(target_path), str(backup_path))
                        print(f"  ℹ️  备份旧文件: {target_path.name} → {backup_path.name}")
                except Exception as e:
                    print(f"  ⚠️  清理target失败: {e}")
                    results[target_name] = (False, f"清理失败: {e}")
                    continue

            try:
                # 创建相对symlink（便于移动项目）
                relative_source = os.path.relpath(source_path, self.datasets_dir)
                os.symlink(relative_source, target_path)
                print(f"  ✅ {target_name:30} → {relative_source}")
                results[target_name] = (True, "symlink成功")
            except OSError as e:
                # Windows或不支持symlink的系统，改为复制
                print(f"  ⚠️  Symlink失败（{e.strerror}），使用复制方案...")
                try:
                    import shutil
                    shutil.copy2(source_path, target_path)
                    print(f"     ✅ 复制成功: {target_name}")
                    results[target_name] = (True, "文件复制")
                except Exception as e2:
                    print(f"     ❌ 复制也失败: {e2}")
                    results[target_name] = (False, f"复制失败: {e2}")
            except Exception as e:
                print(f"  ❌ 未知错误: {e}")
                results[target_name] = (False, f"未知错误: {e}")

        return results

    def verify_setup(self) -> bool:
        """验证设置是否成功"""
        print("\n" + "=" * 80)
        print("✔️  验证数据可访问性")
        print("=" * 80)

        required_files = [
            self.datasets_dir / "humaneval_public_test.jsonl",
            self.datasets_dir / "mbpp_public_test.jsonl",
        ]

        all_ok = True
        for file_path in required_files:
            if file_path.exists():
                try:
                    # 验证文件格式（至少能读第一行JSON）
                    with open(file_path, 'r') as f:
                        first_line = f.readline()
                        if first_line.strip():
                            json.loads(first_line)
                            line_count = sum(1 for _ in f) + 1
                        else:
                            line_count = 0

                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    is_link = "🔗" if file_path.is_symlink() else "📄"
                    print(f"  ✅ {is_link} {file_path.name:30} ({line_count:>6} lines, {size_mb:>6.1f} MB)")
                except Exception as e:
                    print(f"  ❌ {file_path.name:30} 格式错误: {e}")
                    all_ok = False
            else:
                print(f"  ❌ {file_path.name:30} 不存在")
                all_ok = False

        return all_ok

    def print_summary(self, symlink_results: Dict[str, Tuple[bool, str]], verify_ok: bool):
        """打印总结"""
        print("\n" + "=" * 80)
        print("📊 设置总结")
        print("=" * 80)

        successful = sum(1 for ok, _ in symlink_results.values() if ok)
        total = len(symlink_results)

        print(f"\n映射完成: {successful}/{total} 成功")
        for name, (ok, status) in symlink_results.items():
            symbol = "✅" if ok else "❌"
            print(f"  {symbol} {name:30} {status}")

        print(f"\n数据验证: {'✅ 通过' if verify_ok else '❌ 失败'}")

        if verify_ok and successful == total:
            print("\n✨ 所有路径映射已就绪！")
            print("   可以开始训练了。")
        else:
            print("\n⚠️  部分映射失败")
            if not all(Path(p).exists() for p in [
                self.datasets_dir / "humaneval_public_test.jsonl",
                self.datasets_dir / "mbpp_public_test.jsonl",
            ]):
                print("\n建议操作:")
                print("  1. 检查源文件: data/raw/code/{humaneval,mbpp}.jsonl")
                print("  2. 重新运行此脚本: python scripts/setup_data_paths.py")
                print("  3. 如果仍失败，手动复制:")
                print("     cp data/raw/code/*.jsonl data/datasets/")

    def run_all(self, force: bool = False) -> bool:
        """执行所有设置步骤"""
        print("\n" + "=" * 80)
        print("🚀 开始设置数据路径映射")
        print("=" * 80)

        # 1. 检查源数据
        source_data_status = self.check_source_data()
        if not all(source_data_status.values()) and not force:
            print("\n⚠️  源数据不完整，跳过路径映射")
            print("请先运行: python scripts/download_datasets.py")
            return False

        # 2. 创建 datasets 目录
        if not self.create_datasets_dir():
            return False

        # 3. 创建symlink/复制
        symlink_results = self.setup_symlinks()

        # 4. 验证
        verify_ok = self.verify_setup()

        # 5. 总结
        self.print_summary(symlink_results, verify_ok)

        return verify_ok


def main():
    """主程序入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="设置数据路径映射（为AFlow提供正确的数据位置）"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="项目根目录（默认为当前目录）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制执行，即使源数据不完整"
    )

    args = parser.parse_args()

    setup = DataPathSetup(args.project_root)
    success = setup.run_all(force=args.force)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
