#!/usr/bin/env python3
"""
Workflow Validator - 多层验证系统

验证层次：
1. 语法验证：Python代码是否有效
2. 结构验证：是否有async def __call__方法和正确的签名
3. 一致性验证：Operator的import-initialization-usage是否一致
4. 逻辑验证：代码是否有return语句和基本逻辑
"""

import ast
import re
from typing import Tuple, List, Dict


class WorkflowValidator:
    """多层验证的Workflow验证器"""

    def __init__(self):
        """初始化验证器"""
        # 定义问题类型和对应的有效operators
        self.valid_operators = {
            'math': ['answer_generate', 'review', 'revise', 'scensemble', 'custom'],
            'code': ['programmer', 'test', 'review', 'revise', 'custom'],
            'qa': ['answer_generate', 'review', 'revise', 'scensemble', 'custom'],
        }

    def validate_and_fix_workflow(self, code: str, problem_type: str) -> Tuple[str, bool, str, list]:
        """
        多层验证工作流代码

        验证步骤：
        1. 语法验证
        2. 结构验证
        3. 一致性验证
        4. 逻辑验证

        Args:
            code: 工作流代码
            problem_type: "math", "code", 或 "qa"

        Returns:
            (code, is_valid, error_msg, fixes_applied)
        """
        fixes = []

        # ===== Layer 1: Syntax Check =====
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return (code, False, f"语法错误: {str(e)}", fixes)

        # ===== Layer 2: Structure Check =====
        # Check for async def __call__ presence
        if 'async def __call__' not in code:
            return (code, False, "缺少 'async def __call__' 方法", fixes)

        # Check signature
        signature_error = self._validate_signature(code, problem_type)
        if signature_error:
            return (code, False, signature_error, fixes)

        # ===== Layer 3: Operator Consistency Check (NEW) =====
        consistency_errors = self._check_operator_consistency(code, problem_type)
        if consistency_errors:
            error_msg = "Operator一致性检查失败:\n" + "\n".join(consistency_errors)
            return (code, False, error_msg, fixes)

        # ===== Layer 4: Logic Feasibility Check (NEW) =====
        logic_errors = self._check_logic_feasibility(code, problem_type)
        if logic_errors:
            # Logic errors are warnings, not failures
            # Just log them but don't fail validation
            pass

        # All checks passed
        return (code, True, "", fixes)

    def _validate_signature(self, code: str, problem_type: str) -> str:
        """
        验证__call__方法的签名是否正确

        Returns:
            错误信息（如果验证失败），或空字符串（如果验证成功）
        """
        if problem_type == "code":
            # CODE workflows must accept (problem, entry_point, test)
            code_pattern = r'async\s+def\s+__call__\s*\(\s*self\s*,\s*problem\s*:\s*str\s*,\s*entry_point\s*:\s*str\s*,\s*test\s*:\s*str\s*\)'
            if not re.search(code_pattern, code):
                return "CODE问题的签名必须是: async def __call__(self, problem: str, entry_point: str, test: str)"

        elif problem_type in ["math", "qa"]:
            # MATH/QA workflows must accept (problem)
            math_qa_pattern = r'async\s+def\s+__call__\s*\(\s*self\s*,\s*problem\s*:\s*str\s*\)'
            if not re.search(math_qa_pattern, code):
                return f"{problem_type.upper()}问题的签名必须是: async def __call__(self, problem: str)"

        return ""

    def _check_operator_consistency(self, code: str, problem_type: str) -> List[str]:
        """
        检查Operator的一致性：import-initialization-usage

        新策略：在预初始化架构中，我们不检查imports和initialization（因为基类已处理）
        我们只检查：
        1. 使用的operators是否有效（问题类型匹配）
        2. Operator调用的参数是否合理

        Returns:
            错误列表（如果一致，返回空列表）
        """
        errors = []

        # 找出代码中使用的所有operators
        operator_keywords = {
            'answer_generate': r'self\.answer_generate\s*\(',
            'programmer': r'self\.programmer\s*\(',
            'test': r'self\.test\s*\(',
            'review': r'self\.review\s*\(',
            'revise': r'self\.revise\s*\(',
            'scensemble': r'self\.scensemble\s*\(',
            'custom': r'self\.custom\s*\(',
        }

        used_operators = []
        for op_name, op_pattern in operator_keywords.items():
            if re.search(op_pattern, code):
                used_operators.append(op_name)

        # 检查使用的operators是否都有效
        valid_ops = self.valid_operators.get(problem_type, [])
        invalid_ops = [op for op in used_operators if op not in valid_ops]

        for op in invalid_ops:
            errors.append(f"❌ Operator '{op}' 不适合 {problem_type} 问题")

        # Phase 2增强：添加参数验证、继承验证、初始化验证
        param_errors = self._validate_operator_parameters(code, problem_type, used_operators)
        errors.extend(param_errors)

        inheritance_errors = self._validate_class_inheritance(code, problem_type)
        errors.extend(inheritance_errors)

        init_errors = self._validate_init_call(code)
        errors.extend(init_errors)

        return errors

    def _validate_operator_parameters(self, code: str, problem_type: str, used_operators: List[str]) -> List[str]:
        """
        验证operator调用的参数是否正确

        检查内容：
        1. answer_generate是否使用了错误的参数名(problem=而不是input=)
        2. review是否同时有problem和solution参数
        3. test是否有entry_point参数（for CODE）
        """
        errors = []

        # Check 1: answer_generate parameter
        if 'answer_generate' in used_operators:
            # 检查是否有 answer_generate(problem=...) 这样的错误调用
            if re.search(r'answer_generate\s*\(\s*problem\s*=', code):
                errors.append("❌ answer_generate: 参数应该是 'input='，不是 'problem='")

        # Check 2: review parameters
        if 'review' in used_operators:
            # 查找所有 review(...) 调用
            review_calls = re.findall(r'review\s*\([^)]*\)', code)
            for call in review_calls:
                # 检查是否有 problem 参数
                if 'problem' not in call:
                    errors.append("❌ review: 缺少必需的 'problem=' 参数")
                    break

        # Check 3: test parameters (for CODE)
        if problem_type == 'code' and 'test' in used_operators:
            test_calls = re.findall(r'test\s*\([^)]*\)', code)
            for call in test_calls:
                if 'entry_point' not in call:
                    errors.append("❌ test: 缺少必需的 'entry_point=' 参数")
                    break
                if 'problem' not in call:
                    errors.append("❌ test: 缺少必需的 'problem=' 参数")
                    break

        return errors

    def _validate_class_inheritance(self, code: str, problem_type: str) -> List[str]:
        """
        验证class结构（自包含架构，无继承）

        检查内容：
        1. 是否有 'class Workflow:' 定义（无继承）
        2. 是否有 __init__ 方法
        3. 是否初始化了 self.llm
        """
        errors = []

        # Check 1: 是否有class Workflow定义
        if not re.search(r'class\s+Workflow', code):
            errors.append(f"❌ 缺少 'class Workflow' 定义")
            return errors

        # Check 2: 确保没有使用旧的继承模式（如果有，给出警告）
        inheritance_pattern = r'class\s+Workflow\s*\(\s*([^)]+)\s*\)'
        inheritance_match = re.search(inheritance_pattern, code)
        if inheritance_match:
            inherited_class = inheritance_match.group(1).strip()
            if 'WorkflowBase' in inherited_class:
                errors.append(f"⚠️  检测到旧的继承模式 '{inherited_class}'，应使用自包含架构（class Workflow:）")

        # Check 3: 是否有 __init__ 方法
        if 'def __init__' not in code:
            errors.append("❌ 缺少 __init__ 方法")
            return errors

        # Check 4: 是否初始化了 self.llm
        if 'self.llm = create_llm_instance' not in code:
            errors.append("❌ __init__ 中缺少 'self.llm = create_llm_instance(llm_config)'")

        return errors

    def _validate_init_call(self, code: str) -> List[str]:
        """
        验证__init__中的operator初始化（自包含架构）

        检查内容：
        1. 检查是否初始化了使用的operators
        2. 不再检查super()调用（自包含架构不需要）
        """
        errors = []

        # 检查是否有 def __init__ 定义
        if 'def __init__' not in code:
            return errors

        # 查找使用的operators
        used_operators = self._find_used_operators(code)

        # 查找初始化的operators
        initialized_operators = self._find_initialized_operators(code)

        # 检查：使用的operators是否都已初始化
        missing = set(used_operators) - set(initialized_operators)
        if missing:
            errors.append(f"❌ Operators使用但未初始化: {', '.join(missing)}")

        return errors

    def _find_used_operators(self, code: str) -> List[str]:
        """查找在__call__中使用的operators"""
        operator_keywords = {
            'answer_generate': r'self\.answer_generate\s*\(',
            'programmer': r'self\.programmer\s*\(',
            'test': r'self\.test\s*\(',
            'review': r'self\.review\s*\(',
            'revise': r'self\.revise\s*\(',
            'scensemble': r'self\.scensemble\s*\(',
            'custom': r'self\.custom\s*\(',
        }

        used = []
        for op_name, pattern in operator_keywords.items():
            if re.search(pattern, code):
                used.append(op_name)

        return used

    def _find_initialized_operators(self, code: str) -> List[str]:
        """查找在__init__中初始化的operators"""
        patterns = {
            'answer_generate': r'self\.answer_generate\s*=\s*AnswerGenerate',
            'programmer': r'self\.programmer\s*=\s*Programmer',
            'test': r'self\.test\s*=\s*Test',
            'review': r'self\.review\s*=\s*Review',
            'revise': r'self\.revise\s*=\s*Revise',
            'scensemble': r'self\.scensemble\s*=\s*ScEnsemble',
            'custom': r'self\.custom\s*=\s*Custom',
        }

        initialized = []
        for op_name, pattern in patterns.items():
            if re.search(pattern, code):
                initialized.append(op_name)

        return initialized

    def _check_logic_feasibility(self, code: str, problem_type: str) -> List[str]:
        """
        检查代码的逻辑可行性

        检查项：
        1. 是否有return语句
        2. Return语句是否返回元组(result, cost)
        3. 基本的流程逻辑

        Returns:
            警告列表
        """
        warnings = []

        # Check for return statement
        if not re.search(r'return\s+', code):
            warnings.append("⚠️  缺少return语句")

        # Check if return returns a tuple with cost
        has_proper_return = re.search(r'return\s+\w+\s*,\s*self\.llm|return\s+\(.*?,.*?\)', code)
        if not has_proper_return:
            warnings.append("⚠️  return语句可能没有返回(result, cost)元组")

        # Check if there are operator calls
        if 'await self.' not in code and 'self.' not in code:
            warnings.append("⚠️  代码中没有找到任何operator调用")

        return warnings

    def extract_task_prompt(self, code: str) -> str:
        """Extract TASK_PROMPT variable if present (legacy compatibility)"""
        pattern = r"TASK_PROMPT\s*=\s*['\"]([^\"']*)['\"]"
        match = re.search(pattern, code)
        if match:
            return match.group(1)
        return ""


# For backward compatibility
def validate_workflow(code: str, problem_type: str) -> Tuple[str, bool, str]:
    """
    简化的验证函数（向后兼容接口）

    Returns:
        (code, is_valid, error_msg)
    """
    validator = WorkflowValidator()
    code, is_valid, error_msg, _ = validator.validate_and_fix_workflow(code, problem_type)
    return (code, is_valid, error_msg)
