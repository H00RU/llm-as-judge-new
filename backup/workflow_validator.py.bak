#!/usr/bin/env python3
"""
工作流代码验证器 - 确保生成的工作流代码质量
"""
import ast
import re
from typing import Tuple, Dict, List

from src.workflow_consistency_checker import WorkflowConsistencyChecker


class WorkflowValidator:
    """
    验证RL模型生成的工作流代码

    功能：
    1. 语法检查
    2. 必需元素检查
    3. 算子名称规范检查
    4. 异步调用检查
    """

    def __init__(self):
        # 已知的算子列表
        self.valid_operators = [
            'Custom', 'AnswerGenerate', 'Programmer', 'ScEnsemble',
            'Test', 'Review', 'Revise', 'CustomCodeGenerate',
            'Format', 'MdEnsemble'
        ]

        # 算子参数要求
        self.operator_requirements = {
            'Custom': ['input', 'instruction'],
            'AnswerGenerate': ['input'],
            'Programmer': ['problem', 'analysis'],
            'ScEnsemble': ['solutions', 'problem'],
            'Test': ['problem', 'solution', 'entry_point'],
            'Review': ['problem', 'solution'],
            'Revise': ['problem', 'solution', 'feedback'],
            'CustomCodeGenerate': ['problem', 'entry_point', 'instruction'],
            'Format': ['problem', 'solution'],
            'MdEnsemble': ['solutions', 'problem']
        }

        # 初始化一致性检查器
        self.consistency_checker = WorkflowConsistencyChecker()

    def validate_workflow_code(self, code: str, problem_type: str = 'math') -> Tuple[bool, str, Dict]:
        """
        验证工作流代码

        Args:
            code: 生成的Python代码
            problem_type: 问题类型 (math/code/qa)

        Returns:
            (is_valid, error_message, validation_details)
        """
        validation_details = {
            'syntax_valid': False,
            'has_workflow_class': False,
            'has_call_method': False,
            'has_return': False,
            'operators_valid': False,
            'async_calls_valid': False,
            'warnings': []
        }

        # 1. 语法检查
        try:
            tree = ast.parse(code)
            validation_details['syntax_valid'] = True
        except SyntaxError as e:
            return False, f"语法错误: {e}", validation_details

        # 2. 检查Workflow类
        has_workflow_class = any(
            isinstance(node, ast.ClassDef) and node.name == 'Workflow'
            for node in ast.walk(tree)
        )
        validation_details['has_workflow_class'] = has_workflow_class
        if not has_workflow_class:
            return False, "缺少Workflow类定义", validation_details

        # 3. 检查__call__方法
        has_call_method = self._has_call_method(tree)
        validation_details['has_call_method'] = has_call_method
        if not has_call_method:
            return False, "缺少async def __call__方法", validation_details

        # 4. 检查return语句
        has_return = self._has_return_in_call(tree)
        validation_details['has_return'] = has_return
        if not has_return:
            return False, "__call__方法缺少return语句", validation_details

        # 5. 检查算子使用
        operator_issues = self._check_operators(code)
        if operator_issues:
            validation_details['operators_valid'] = False
            validation_details['warnings'].extend(operator_issues)
            # 算子问题作为警告，不直接失败
        else:
            validation_details['operators_valid'] = True

        # 6. 检查异步调用
        async_issues = self._check_async_calls(code)
        if async_issues:
            validation_details['async_calls_valid'] = False
            validation_details['warnings'].extend(async_issues)
        else:
            validation_details['async_calls_valid'] = True

        # 7. 特定类型检查
        # L2.2: QA 工作流检查（方案B：警告而非硬拒绝）
        # 改进：操作符冲突现在通过reward在aflow_executor中处理，不再硬拒绝
        if problem_type == 'qa':
            qa_issues = self._check_qa_workflow(code)
            if qa_issues:
                # 改为警告而非硬拒绝（方案B：软学习）
                # RL模型如果违反约束，会在metadata中标记，并在reward中受到-5.0惩罚
                validation_details['warnings'].extend(qa_issues)
                # 不再return False，允许workflow继续执行

        if problem_type == 'code':
            code_issues = self._check_code_workflow(tree, code)
            if code_issues:
                validation_details['warnings'].extend(code_issues)

        # 8. 全局一致性检查（新增 Phase 4 Step 0.6）
        consistency_result = self.consistency_checker.check_consistency(code)
        validation_details['consistency_check'] = consistency_result

        if not consistency_result['consistent']:
            # 一致性检查失败不阻止验证，但标记为警告
            validation_details['warnings'].extend(consistency_result['issues'])

        # 综合判断
        if validation_details['warnings']:
            warning_msg = '; '.join(validation_details['warnings'])
            return True, f"验证通过但有警告: {warning_msg}", validation_details

        return True, "验证通过", validation_details

    def _has_call_method(self, tree: ast.AST) -> bool:
        """检查是否有__call__方法"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == '__call__':
                        return True
        return False

    def _has_return_in_call(self, tree: ast.AST) -> bool:
        """检查__call__方法是否有return语句"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == '__call__':
                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Return):
                                return True
        return False

    def _check_operators(self, code: str) -> List[str]:
        """检查算子使用问题"""
        issues = []

        # 检查小写算子名（常见错误）
        lowercase_pattern = r'operator\.([a-z][a-zA-Z_]*?)\('
        lowercase_matches = re.findall(lowercase_pattern, code)
        for match in lowercase_matches:
            issues.append(f"算子名应使用PascalCase: operator.{match} -> operator.{match.capitalize()}")

        # 检查未知算子
        operator_pattern = r'operator\.([A-Z][a-zA-Z_]*?)\('
        operator_matches = re.findall(operator_pattern, code)
        for op in operator_matches:
            if op not in self.valid_operators:
                issues.append(f"未知算子: {op}")

        # 检查Test算子参数（Code工作流常见错误）
        if 'self.test' in code:
            test_pattern = r'self\.test\([^)]*\)'
            test_calls = re.findall(test_pattern, code)
            for call in test_calls:
                # 检查是否包含所有必需参数
                if not all(param in call for param in ['problem', 'solution', 'entry_point']):
                    issues.append("Test算子缺少必需参数: 需要problem, solution, entry_point")

        return issues

    def _check_async_calls(self, code: str) -> List[str]:
        """检查异步调用问题"""
        issues = []

        # 检查算子调用是否使用await
        operator_call_pattern = r'(self\.[a-z_]+)\([^)]*\)'
        calls = re.findall(operator_call_pattern, code)

        for call in calls:
            # 排除非算子调用
            if call in ['self.model', 'self.name', 'self.dataset']:
                continue

            # 检查是否有对应的await
            if f'await {call}' not in code:
                issues.append(f"异步调用缺少await: {call}")

        return issues

    def _check_qa_workflow(self, code: str) -> List[str]:
        """
        L2.2: 检查 QA 类型工作流的特殊要求（强制严格）

        QA 工作流不应该使用 Test 操作符，因为 QA 没有自动化测试用例。
        """
        issues = []

        # 规则1: QA 问题不应该使用 Test 操作符（强制严格）
        if "self.test(" in code or "await test(" in code or ".test(" in code:
            issues.append("QA 问题不应使用 Test 操作符（QA 没有自动化测试用例）")

        # 规则2: QA 问题不应该使用 Programmer 操作符（非代码相关）
        if "self.programmer(" in code or "await programmer(" in code or ".programmer(" in code:
            issues.append("QA 问题不应使用 Programmer 操作符（QA 是文本问题，不是代码问题）")

        # 规则3: QA 问题应该至少使用一个 QA-safe 操作符
        qa_safe_operators = ['Custom', 'AnswerGenerate', 'Review', 'Revise', 'ScEnsemble']
        has_qa_operator = any(f"self.{op_lower}(" in code for op_lower in
                             [op.lower() for op in qa_safe_operators])

        if not has_qa_operator:
            issues.append(f"QA 工作流应该至少使用一个 QA-safe 操作符: {', '.join(qa_safe_operators)}")

        return issues

    def _check_code_workflow(self, tree: ast.AST, code: str) -> List[str]:
        """检查Code类型工作流的特殊要求"""
        issues = []

        # 检查__call__方法签名
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == '__call__':
                        # 检查参数
                        args = [arg.arg for arg in item.args.args]
                        if 'entry_point' not in args:
                            issues.append("Code工作流的__call__方法应包含entry_point参数")

        return issues

    def fix_common_issues(self, code: str) -> str:
        """
        尝试自动修复常见问题

        Args:
            code: 有问题的代码

        Returns:
            修复后的代码
        """
        fixed_code = code

        # ✅ CRITICAL FIX: Normalize indentation FIRST (fixes 60-70% of errors)
        # Import WorkflowCodeBuilder for normalization method
        from src.workflow_code_builder import WorkflowCodeBuilder
        builder = WorkflowCodeBuilder()
        fixed_code = builder._normalize_indentation(fixed_code)

        # 1. 修复小写算子名
        lowercase_pattern = r'operator\.([a-z][a-zA-Z_]*?)\('
        def fix_case(match):
            name = match.group(1)
            # 智能大写转换
            if name == 'custom':
                return 'operator.Custom('
            elif name == 'answergenerae' or name == 'answer_generate':
                return 'operator.AnswerGenerate('
            elif name == 'programmer':
                return 'operator.Programmer('
            elif name == 'test':
                return 'operator.Test('
            elif name == 'review':
                return 'operator.Review('
            elif name == 'revise':
                return 'operator.Revise('
            elif name.startswith('sc'):
                return 'operator.ScEnsemble('
            else:
                # 默认：首字母大写
                return f'operator.{name.capitalize()}('

        fixed_code = re.sub(lowercase_pattern, fix_case, fixed_code)

        # 2. 修复缺少await的算子调用
        # 查找所有self.xxx()调用
        call_pattern = r'^(\s*)(self\.(?:custom|answer_generate|programmer|test|review|revise|sc_ensemble)\([^)]*\))'
        lines = fixed_code.split('\n')
        fixed_lines = []

        for line in lines:
            if re.match(call_pattern, line) and 'await' not in line:
                # 添加await
                line = re.sub(call_pattern, r'\1await \2', line)
            fixed_lines.append(line)

        fixed_code = '\n'.join(fixed_lines)

        # 3. 确保Test算子有完整参数（针对Code问题）
        if 'self.test' in fixed_code and 'entry_point' not in fixed_code:
            # 尝试添加entry_point参数
            test_pattern = r'self\.test\(([^)]+)\)'
            def add_entry_point(match):
                params = match.group(1)
                if 'entry_point' not in params:
                    # 添加entry_point参数
                    return f'self.test({params}, entry_point=entry_point)'
                return match.group(0)

            fixed_code = re.sub(test_pattern, add_entry_point, fixed_code)

        # 4. 修复 __call__ 方法的签名（关键！）
        # 将任何形式的 async def __call__ 改为标准签名
        call_sig_pattern = r'async def __call__\s*\([^)]*\):'
        if re.search(call_sig_pattern, fixed_code):
            fixed_code = re.sub(
                call_sig_pattern,
                'async def __call__(self, problem: str, entry_point: str = None):',
                fixed_code
            )

        return fixed_code

    def _detect_uninitialized_operators(self, code: str) -> tuple:
        """
        检测未初始化的operators

        对比 __init__ 中初始化的operators 和 __call__ 中使用的operators，找出差集

        Returns:
            (未初始化列表, 使用位置列表)
        """
        try:
            tree = ast.parse(code)
        except:
            return [], []

        # 找出 __init__ 中初始化的operators
        initialized_operators = set()
        call_method_node = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for item in node.body:
                    # 在 __init__ 中找初始化
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        for stmt in ast.walk(item):
                            # 查找 self.xxx = operator.YYY(...) 的赋值
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if isinstance(target, ast.Attribute):
                                        if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                            initialized_operators.add(target.attr)

                    # 保存 __call__ 方法节点
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == '__call__':
                        call_method_node = item

        # 找出 __call__ 中使用的operators (self.xxx)
        used_operators = set()
        if call_method_node:
            for node in ast.walk(call_method_node):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == 'self':
                        # 排除 self.model, self.name, self.dataset 等非operator属性
                        attr_name = node.attr
                        if attr_name not in ['model', 'name', 'dataset']:
                            used_operators.add(attr_name)

        # 找出差集：使用但未初始化的operators
        uninitialized = list(used_operators - initialized_operators)
        return uninitialized, list(used_operators)

    def fix_uninitialized_operators(self, code: str) -> tuple:
        """
        自动修复未初始化的operators

        在 __init__ 末尾添加缺失的初始化

        Returns:
            (修复后代码, 是否修复, 修复列表)
        """
        uninitialized, _ = self._detect_uninitialized_operators(code)

        if not uninitialized:
            return code, False, []

        try:
            tree = ast.parse(code)
        except:
            return code, False, []

        # 找到 __init__ 方法并在末尾添加初始化
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for i, item in enumerate(node.body):
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        # 在 __init__ 的最后一条语句后添加初始化
                        # 使用正则表达式方式修改（更安全）
                        init_pattern = r'(    def __init__\(self[^:]*\):\n(?:.*\n)*?)(    (?:async )?def |\Z)'

                        fixes = []
                        for op_name in uninitialized:
                            # 构造初始化语句
                            init_stmt = f"        self.{op_name} = operator.{op_name.title().replace('_', '')}(self.model)\n"
                            fixes.append(op_name)

                        # 在 __init__ 末尾添加初始化
                        if fixes:
                            # 找到 __init__ 的结束位置，在最后一个缩进语句后添加
                            init_section = re.search(r'(    def __init__\(self[^:]*\):\n((?:        .*\n)*?))(    (?:async )?def |\Z)', code, re.MULTILINE)
                            if init_section:
                                before = init_section.group(1)
                                after_start = init_section.start(3)
                                after = code[after_start:]

                                # 添加所有初始化语句
                                new_inits = ''.join([f"        self.{op} = operator.{op}(self.model)\n" for op in uninitialized])
                                code = code[:after_start] + new_inits + after

                        return code, len(fixes) > 0, fixes

        return code, False, []

    def fix_call_signature(self, code: str) -> tuple:
        """
        检查和修复 __call__ 方法的签名

        Returns:
            (修复后的代码, 是否进行了修复, 修复原因)
        """
        import re

        # 期望的正确签名
        expected_pattern = r'async def __call__\s*\(\s*self\s*,\s*problem\s*:\s*str\s*,\s*entry_point\s*:\s*str\s*=\s*None\s*\)'

        # 检查是否已经是正确的签名
        if re.search(expected_pattern, code):
            return code, False, None

        # 检查是否有 __call__ 方法（任何形式）
        call_pattern = r'async def __call__\s*\([^)]*\):'
        if re.search(call_pattern, code):
            # 有 __call__ 但签名错误，执行修复
            fixed_code = re.sub(
                call_pattern,
                'async def __call__(self, problem: str, entry_point: str = None):',
                code
            )
            return fixed_code, True, 'signature_mismatch'

        # 没有 __call__ 方法，返回原代码
        return code, False, None

    def validate_and_fix_workflow(self, code: str, problem_type: str = 'math') -> tuple:
        """
        验证工作流代码，同时进行必要的修复（综合方案）

        这个方法结合了：
        1. 签名修复（最关键）
        2. 未初始化operators修复
        3. 其他常见问题修复
        4. 完整的代码验证

        Returns:
            (修复后的代码, 是否有效, 错误信息, 修复操作列表, 签名错误标记, 未初始化operators标记)
        """
        fixes_applied = []
        had_signature_error = False
        had_uninitialized_operators = False

        # Step 1: 修复签名（最关键的）
        code, sig_fixed, sig_reason = self.fix_call_signature(code)
        if sig_fixed:
            fixes_applied.append('signature_fixed')
            had_signature_error = True
            print(f"  🔧 自动修复: __call__ 方法签名已正确")

        # Step 2: 修复未初始化的operators
        code, uninitialized_fixed, uninitialized_list = self.fix_uninitialized_operators(code)
        if uninitialized_fixed:
            fixes_applied.append('uninitialized_operators_fixed')
            had_uninitialized_operators = True
            print(f"  🔧 自动修复: 添加缺失的operator初始化 {uninitialized_list}")

        # Step 3: 修复其他常见问题
        fixed_code = self.fix_common_issues(code)
        if fixed_code != code:
            fixes_applied.append('common_issues_fixed')
            code = fixed_code

        # Step 4: 验证修复后的代码
        is_valid, msg, validation_details = self.validate_workflow_code(code, problem_type)

        return code, is_valid, msg, fixes_applied, had_signature_error, had_uninitialized_operators


def test_validator():
    """测试验证器"""
    validator = WorkflowValidator()

    # 测试用例1：正确的工作流
    good_code = '''
import operator
from scripts.async_llm import create_llm_instance

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.model = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.model)

    async def __call__(self, problem):
        result = await self.custom(input=problem, instruction="Solve")
        return result['response'], self.model.get_usage_summary()["total_cost"]
'''

    # 测试用例2：有问题的工作流
    bad_code = '''
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.custom = operator.custom(self.model)  # 小写错误

    async def __call__(self, problem):
        result = self.custom(input=problem)  # 缺少await
        # 缺少return
'''

    print("测试正确的工作流:")
    valid, msg, details = validator.validate_workflow_code(good_code)
    print(f"  结果: {valid}, 消息: {msg}")

    print("\n测试有问题的工作流:")
    valid, msg, details = validator.validate_workflow_code(bad_code)
    print(f"  结果: {valid}, 消息: {msg}")

    print("\n尝试自动修复:")
    fixed = validator.fix_common_issues(bad_code)
    print("修复后的代码:")
    print(fixed)


if __name__ == "__main__":
    test_validator()
