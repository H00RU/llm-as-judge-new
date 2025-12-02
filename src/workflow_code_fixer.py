#!/usr/bin/env python3
"""
Workflow代码自动修复器 - 修复LLM生成的常见错误
"""
import re
import ast
from typing import Tuple, List, Dict


class WorkflowCodeFixer:
    """自动修复workflow代码中的常见错误"""

    def __init__(self):
        # 变量名修复：确保使用'model'而非'llm'
        # 原因：tokenizer将'llm'分为['ll', 'm']两个token，导致生成错误
        self.typo_fixes = {
            r'\bself\.llm\b': 'self.model',           # self.llm → self.model
            r'\boperator\.(\w+)\(self\.llm\)': r'operator.\1(self.model)',  # operator.X(self.llm) → operator.X(self.model)
            r'\bself\.revise\(': 'self.review(',  # revise operator可能不存在
            r'await\s+self\.test\([^)]*test\s*=': 'await self.test(',  # 移除test参数
        }

        # 变量初始化模式
        self.uninitialized_vars = [
            'code', 'revised_code', 'solution', 'answer', 'response', 'result'
        ]

    def fix_code(self, code: str, problem_type: str = 'math') -> Tuple[str, List[str]]:
        """
        自动修复代码中的常见错误

        Args:
            code: 生成的workflow代码
            problem_type: 问题类型

        Returns:
            (fixed_code, fixes_applied)
        """
        fixes_applied = []
        original_code = code

        # 1. 修复拼写错误
        for pattern, replacement in self.typo_fixes.items():
            if re.search(pattern, code):
                code = re.sub(pattern, replacement, code)
                fixes_applied.append(f"修复拼写: {pattern} -> {replacement}")

        # 2. 检测并修复未初始化变量
        code, var_fixes = self._fix_uninitialized_variables(code)
        fixes_applied.extend(var_fixes)

        # 3. 修复dict访问错误 (str.get() -> 安全访问)
        code, dict_fixes = self._fix_dict_access_errors(code)
        fixes_applied.extend(dict_fixes)

        # 4. 确保正确的返回值类型
        code, return_fixes = self._fix_return_statement(code)
        fixes_applied.extend(return_fixes)

        # 5. 移除无效的operator使用
        code, op_fixes = self._fix_invalid_operators(code, problem_type)
        fixes_applied.extend(op_fixes)

        # 6. 验证语法
        try:
            ast.parse(code)
        except SyntaxError as e:
            fixes_applied.append(f"⚠️ 语法错误未完全修复: {e}")
            # 如果还有语法错误，尝试更激进的修复
            code = self._aggressive_fix(code, problem_type)
            fixes_applied.append("应用激进修复策略")

        return code, fixes_applied

    def _fix_uninitialized_variables(self, code: str) -> Tuple[str, List[str]]:
        """修复未初始化的变量"""
        fixes = []
        lines = code.split('\n')
        new_lines = []
        in_function = False
        func_indent = 0

        for i, line in enumerate(lines):
            new_lines.append(line)

            # 检测函数开始
            if 'async def __call__' in line:
                in_function = True
                func_indent = len(line) - len(line.lstrip())
                # 在函数开始后添加变量初始化
                init_line = ' ' * (func_indent + 4) + '# 初始化返回变量'
                new_lines.append(init_line)
                for var in self.uninitialized_vars:
                    init_line = ' ' * (func_indent + 4) + f'{var} = None'
                    new_lines.append(init_line)
                fixes.append("添加变量初始化")

        return '\n'.join(new_lines), fixes

    def _fix_dict_access_errors(self, code: str) -> Tuple[str, List[str]]:
        """修复字符串被当作字典访问的错误"""
        fixes = []

        # 查找 result.get() 但 result 可能是字符串的情况
        # 替换为: result.get() if isinstance(result, dict) else result
        pattern = r'(\w+)\.get\(([^)]+)\)'

        def safe_get_replacement(match):
            var_name = match.group(1)
            key = match.group(2)
            return f"{var_name}.get({key}) if isinstance({var_name}, dict) else {var_name}"

        new_code = re.sub(pattern, safe_get_replacement, code)

        if new_code != code:
            fixes.append("添加dict类型检查")
            code = new_code

        return code, fixes

    def _fix_return_statement(self, code: str) -> Tuple[str, List[str]]:
        """确保正确返回 (solution, cost) 元组"""
        fixes = []

        # 查找 return 语句
        lines = code.split('\n')
        new_lines = []

        for line in lines:
            # 如果return只有一个值，补充cost
            if 'return ' in line and line.strip().startswith('return'):
                # 提取return后的内容
                return_content = line.split('return', 1)[1].strip()

                # 检查是否已经是元组
                if ',' not in return_content:
                    # 单一返回值，需要添加cost
                    indent = len(line) - len(line.lstrip())
                    new_line = ' ' * indent + f'return {return_content}, self.model.get_usage_summary()["total_cost"]'
                    new_lines.append(new_line)
                    fixes.append("修复return语句 - 添加cost")
                    continue

            new_lines.append(line)

        return '\n'.join(new_lines), fixes

    def _fix_invalid_operators(self, code: str, problem_type: str) -> Tuple[str, List[str]]:
        """移除或替换无效的operator使用"""
        fixes = []

        # QA和Math问题不应该使用Test operator
        if problem_type in ['qa', 'math']:
            if 'self.test(' in code:
                # 注释掉test调用
                code = re.sub(
                    r'(\s+)(.*await self\.test\([^)]+\))',
                    r'\1# \2  # 自动注释: QA/Math不使用Test',
                    code
                )
                fixes.append(f"注释掉{problem_type}问题中的Test operator")

        return code, fixes

    def _aggressive_fix(self, code: str, problem_type: str) -> str:
        """激进的修复策略 - 当常规修���失败时使用"""
        # 返回一个安全的默认实现
        return f'''import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.model = create_llm_instance(llm_config)
        self.answer_generate = operator.AnswerGenerate(self.model)

    async def __call__(self, problem: str, entry_point: str = None):
        # 安全的默认实现
        result = await self.answer_generate(input=problem)
        answer = result.get('answer', '') if isinstance(result, dict) else str(result)
        cost = self.model.get_usage_summary()["total_cost"]
        return answer, cost
'''

    def validate_and_fix(self, code: str, problem_type: str = 'math', max_iterations: int = 3) -> Tuple[str, bool, List[str]]:
        """
        验证并修复代码，支持多轮修复

        Returns:
            (fixed_code, is_valid, all_fixes)
        """
        all_fixes = []

        for iteration in range(max_iterations):
            # 尝试修复
            code, fixes = self.fix_code(code, problem_type)
            all_fixes.extend(fixes)

            # 验证语法
            try:
                ast.parse(code)
                return code, True, all_fixes
            except SyntaxError as e:
                if iteration == max_iterations - 1:
                    # 最后一次尝试，使用默认实现
                    all_fixes.append(f"多轮修复失败，使用默认实现")
                    return self._aggressive_fix(code, problem_type), False, all_fixes
                continue

        return code, False, all_fixes


def test_fixer():
    """测试修复器"""
    fixer = WorkflowCodeFixer()

    # 测试用例1: 拼写错误
    bad_code1 = '''
class Workflow:
    def __init__(self, name, ll_config, dataset):
        self.ll_m = create_llm_instance(ll_config)

    async def __call__(self, problem, entry_point=None):
        result = await self.answer_generate(input=problem)
        return result['answer'], self.lll.get_usage_summary()["total_cost"]
'''

    print("\n测试1: 拼写错误修复")
    print("="*60)
    fixed, fixes = fixer.fix_code(bad_code1)
    print(f"应用的修复: {fixes}")
    print(f"修复后代码:\n{fixed}")

    # 测试用例2: 未初始化变量
    bad_code2 = '''
class Workflow:
    async def __call__(self, problem, entry_point=None):
        if some_condition:
            code = generate_code()
        return code, cost
'''

    print("\n测试2: 未初始化变量修复")
    print("="*60)
    fixed, fixes = fixer.fix_code(bad_code2)
    print(f"应用的修复: {fixes}")
    print(f"修复后代码:\n{fixed}")


if __name__ == '__main__':
    test_fixer()
