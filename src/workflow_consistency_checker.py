#!/usr/bin/env python3
"""
工作流一致性检查器 - 全局验证import/initialization/call的一致性
"""
import re
import ast
from typing import Dict, Set, List, Tuple, Optional


class WorkflowConsistencyChecker:
    """
    检查工作流代码的全局一致性

    验证三个层级的一致性：
    1. Import层：哪些类被导入
    2. Initialization层：哪些operator被初始化
    3. Call层：哪些operator被调用

    规则：
    - imported_classes ⊇ called_operators (所有调用的类都被导入)
    - initialized_operators ⊇ called_operators (所有调用的operator都被初始化)
    """

    def __init__(self):
        """初始化一致性检查器"""
        self.valid_operators = {
            'Custom', 'AnswerGenerate', 'Programmer', 'Test',
            'Review', 'Revise', 'ScEnsemble'
        }

    def check_consistency(self, code: str) -> Dict:
        """
        检查工作流代码的全局一致性

        Args:
            code: Python工作流代码

        Returns:
            {
                'consistent': bool,  # 是否一致
                'imported_classes': Set[str],  # 导入的类名
                'initialized_operators': Set[str],  # 初始化的operator属性名
                'called_operators': Set[str],  # 被调用的operator属性名
                'missing_imports': Set[str],  # 未导入但被使用的类
                'missing_initializations': Set[str],  # 未初始化但被调用的operator
                'unused_initializations': Set[str],  # 已初始化但未被调用的operator
                'issues': List[str]  # 详细的问题描述
            }
        """
        # 1. 解析import语句
        imported_classes = self._parse_imports(code)

        # 2. 解析operator初始化
        initialized_operators = self._parse_initializations(code)

        # 3. 解析operator调用
        called_operators = self._parse_calls(code)

        # 4. 一致性检查
        issues = []
        missing_imports = set()
        missing_inits = set()
        unused_inits = set()

        # 检查缺失的导入（调用的类没有被导入）
        for op_attr, op_class in called_operators.items():
            if op_class and op_class not in imported_classes:
                missing_imports.add(op_class)
                issues.append(f"❌ 类 '{op_class}' 被调用但未导入 (via self.{op_attr})")

        # 检查缺失的初始化（被调用的operator没有被初始化）
        for op_attr in set(called_operators.keys()):
            if op_attr not in initialized_operators:
                issues.append(f"❌ Operator 'self.{op_attr}' 被调用但未初始化")
                missing_inits.add(op_attr)

        # 检查未使用的初始化
        for op_attr in initialized_operators.keys():
            if op_attr not in called_operators:
                issues.append(f"⚠️ Operator 'self.{op_attr}' 已初始化但未被调用")
                unused_inits.add(op_attr)

        is_consistent = len(missing_imports) == 0 and len(missing_inits) == 0

        return {
            'consistent': is_consistent,
            'imported_classes': imported_classes,
            'initialized_operators': initialized_operators,
            'called_operators': called_operators,
            'missing_imports': missing_imports,
            'missing_initializations': missing_inits,
            'unused_initializations': unused_inits,
            'issues': issues
        }

    def _parse_imports(self, code: str) -> Set[str]:
        """
        解析所有 from scripts.operators import XXX 中的类名

        Returns:
            Set of class names: {'Custom', 'AnswerGenerate', 'Programmer', ...}
        """
        imported = set()

        # 模式1: from scripts.operators import Custom, AnswerGenerate, ...
        pattern = r'from\s+scripts\.operators\s+import\s+([^#\n]+)'
        matches = re.findall(pattern, code)

        for match in matches:
            # 解析import列表: "Custom, AnswerGenerate, Programmer, Test, Review, Revise, ScEnsemble"
            classes = re.findall(r'\b([A-Z]\w+)\b', match)
            imported.update(classes)

        return imported

    def _parse_initializations(self, code: str) -> Dict[str, str]:
        """
        解析所有 self.xxx = ClassName(self.llm) 的operator初始化

        Returns:
            Dict mapping operator属性名 -> 类名
            例如: {'answer_generate': 'AnswerGenerate', 'test': 'Test'}
        """
        initialized = {}

        # 在__init__方法中查找initialization
        init_pattern = r'def __init__\([^)]*\):([\s\S]*?)(?=\n    (?:async\s+)?def|\n\nclass|\Z)'
        init_match = re.search(init_pattern, code)

        if not init_match:
            return initialized

        init_code = init_match.group(1)

        # 模式: self.attr_name = ClassName(self.llm)
        # 支持的形式：
        # self.answer_generate = AnswerGenerate(self.llm)
        # self.test = Test(self.llm)
        # self.programmer = Programmer(self.llm)
        init_patterns = re.findall(
            r'self\.(\w+)\s*=\s*([A-Z]\w+)\s*\(\s*self\.llm\s*\)',
            init_code
        )

        for attr_name, class_name in init_patterns:
            if class_name in self.valid_operators:
                initialized[attr_name] = class_name

        return initialized

    def _parse_calls(self, code: str) -> Dict[str, Optional[str]]:
        """
        解析所有 await self.xxx(...) 的operator调用

        Returns:
            Dict mapping operator属性名 -> 已知的类名（或None）
            例如: {'answer_generate': 'AnswerGenerate', 'test': 'Test', 'unknown_op': None}
        """
        called = {}

        # 在__call__方法中查找calls
        call_pattern = r'async\s+def\s+__call__\([^)]*\):([\s\S]+?)(?=\n    def|\n\nclass|\Z)'
        call_match = re.search(call_pattern, code)

        if not call_match:
            return called

        call_code = call_match.group(1)

        # 模式: await self.xxx(...)
        call_patterns = re.findall(r'await\s+self\.(\w+)\s*\(', call_code)

        for op_attr in set(call_patterns):
            # 尝试从initialized中推断类名
            # 如果不能推断，设为None
            called[op_attr] = None

        return called

    def get_summary(self, check_result: Dict) -> str:
        """
        生成一致性检查的人类可读总结

        Args:
            check_result: check_consistency()的返回值

        Returns:
            格式化的总结字符串
        """
        summary = []
        summary.append("=" * 70)
        summary.append("🔍 工作流一致性检查报告")
        summary.append("=" * 70)

        if check_result['consistent']:
            summary.append("✅ 状态: 一致性检查通过")
        else:
            summary.append("❌ 状态: 一致性检查失败")

        summary.append(f"\n📊 统计信息:")
        summary.append(f"  导入的类: {sorted(check_result['imported_classes'])}")
        summary.append(f"  初始化的operator: {sorted(check_result['initialized_operators'].keys())}")
        summary.append(f"  调用的operator: {sorted(check_result['called_operators'].keys())}")

        if check_result['missing_imports']:
            summary.append(f"\n❌ 缺失的导入: {sorted(check_result['missing_imports'])}")

        if check_result['missing_initializations']:
            summary.append(f"\n❌ 缺失的初始化: {sorted(check_result['missing_initializations'])}")

        if check_result['unused_initializations']:
            summary.append(f"\n⚠️ 未使用的初始化: {sorted(check_result['unused_initializations'])}")

        if check_result['issues']:
            summary.append(f"\n📝 详细问题:")
            for issue in check_result['issues']:
                summary.append(f"  {issue}")

        summary.append("=" * 70)

        return "\n".join(summary)

    def validate_and_report(self, code: str, verbose: bool = True) -> bool:
        """
        执行一致性检查并打印报告

        Args:
            code: Python工作流代码
            verbose: 是否打印详细报告

        Returns:
            True if consistent, False otherwise
        """
        result = self.check_consistency(code)

        if verbose:
            print(self.get_summary(result))

        return result['consistent']


# 测试函数
def test_checker():
    """测试工作流一致性检查器"""
    checker = WorkflowConsistencyChecker()

    # 测试1: 一致的代码
    valid_code = """
from scripts.operators import Custom, AnswerGenerate, Programmer
from scripts.async_llm import create_llm_instance

class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.answer_generate = AnswerGenerate(self.llm)
        self.programmer = Programmer(self.llm)

    async def __call__(self, problem: str):
        result = await self.answer_generate(input=problem)
        return result.get('answer', ''), self.llm.get_usage_summary()["total_cost"]
"""

    print("\n测试1: 一致的代码")
    checker.validate_and_report(valid_code, verbose=True)

    # 测试2: 未初始化的operator调用
    invalid_code_1 = """
from scripts.operators import Custom, AnswerGenerate
from scripts.async_llm import create_llm_instance

class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.answer_generate = AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        result = await self.answer_generate(input=problem)
        review = await self.review(problem=problem, solution=result['answer'])  # 未初始化!
        return result.get('answer', ''), self.llm.get_usage_summary()["total_cost"]
"""

    print("\n测试2: 未初始化的operator调用")
    checker.validate_and_report(invalid_code_1, verbose=True)

    # 测试3: 未导入的类
    invalid_code_2 = """
from scripts.operators import Custom, AnswerGenerate
from scripts.async_llm import create_llm_instance

class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.review = Review(self.llm)  # Review 未导入!

    async def __call__(self, problem: str):
        result = await self.review(problem=problem, solution="test")
        return result, self.llm.get_usage_summary()["total_cost"]
"""

    print("\n测试3: 未导入的类")
    checker.validate_and_report(invalid_code_2, verbose=True)


if __name__ == "__main__":
    test_checker()
