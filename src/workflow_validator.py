#!/usr/bin/env python3
"""
Enhanced Workflow Validator - Consistency Enforcement

Prevents operator import-initialization-usage inconsistencies.
No auto-fixing, only preventive validation and fallback to consistent defaults.
"""

import ast
import re
from typing import List, Tuple, Set


class WorkflowValidator:
    """Enhanced validator with operator consistency checking"""

    def __init__(self):
        """Initialize validator"""
        # Available operators that can be imported
        self.available_operators = {
            'Custom', 'AnswerGenerate', 'Programmer', 'Test', 'Review',
            'Revise', 'Format', 'ScEnsemble', 'MdEnsemble', 'CustomCodeGenerate'
        }

    def extract_imports(self, code: str) -> Set[str]:
        """Extract operator names from import statements"""
        imports = set()
        # Match patterns like: from scripts.operators import Custom, AnswerGenerate
        pattern = r'from\s+scripts\.operators\s+import\s+([^#\n]+)'
        matches = re.findall(pattern, code)

        for match in matches:
            # Clean up and split by commas
            operators = [op.strip() for op in match.split(',')]
            for op in operators:
                # Remove any aliases (as X) and only keep operator names
                op_name = op.split(' as ')[0].strip()
                if op_name and op_name in self.available_operators:
                    imports.add(op_name)

        return imports

    def extract_initializations(self, code: str) -> Set[str]:
        """Extract operator names from self.X = Operator(...) patterns"""
        inits = set()
        # Match patterns like: self.custom = Custom(self.llm)
        pattern = r'self\.(\w+)\s*=\s*(\w+)\s*\('
        matches = re.findall(pattern, code)

        for attr_name, op_name in matches:
            # Only count if the operator name matches available operators
            if op_name in self.available_operators:
                inits.add(op_name)

        return inits

    def extract_usages(self, code: str) -> Set[str]:
        """Extract operator names from await self.X(...) patterns"""
        usages = set()

        # Simple approach: look for "await self." followed by word and "("
        # But be more robust by finding all occurrences
        import re
        pattern = r'await\s+self\.(\w+)\s*\('

        # Use a more comprehensive search that handles multi-line and complex cases
        for match in re.finditer(pattern, code):
            attr_name = match.group(1)

            # Convert attribute name back to operator name
            # Handle both patterns: custom->Custom, answer_generate->AnswerGenerate
            if attr_name in [op.lower() for op in self.available_operators]:
                # Direct match (e.g., custom -> Custom)
                for avail_op in self.available_operators:
                    if avail_op.lower() == attr_name:
                        usages.add(avail_op)
                        break
            elif '_' in attr_name:
                # Handle snake_case to CamelCase conversion
                # (answer_generate -> AnswerGenerate)
                op_name = ''.join(word.capitalize() for word in attr_name.split('_'))
                if op_name in self.available_operators:
                    usages.add(op_name)
            else:
                # Try case-insensitive match
                op_name = attr_name.capitalize()
                if op_name in self.available_operators:
                    usages.add(op_name)

        return usages

    def check_operator_consistency(self, code: str) -> Tuple[bool, List[str]]:
        """
        Check import-initialization-usage consistency.

        Returns:
            (is_consistent, error_messages)
        """
        imports = self.extract_imports(code)
        inits = self.extract_initializations(code)
        usages = self.extract_usages(code)

        errors = []

        # 1. All imports must be initialized
        for op in imports:
            if op not in inits:
                errors.append(f"❌ 导入{op}但未初始化：添加 self.{op.lower()} = {op}(self.llm)")

        # 2. All initializations must be imported
        for op in inits:
            if op not in imports:
                errors.append(f"❌ 初始化{op}但未导入：添加 from scripts.operators import {op}")

        # 3. All usages must be imported and initialized
        for op in usages:
            if op not in imports:
                errors.append(f"❌ 使用{op}但未导入：添加 from scripts.operators import {op}")
            if op not in inits:
                errors.append(f"❌ 使用{op}但未初始化：添加 self.{op.lower()} = {op}(self.llm)")

        # 4. Check for unused imports/initializations
        unused_imports = imports - usages
        unused_inits = inits - usages

        for op in unused_imports:
            errors.append(f"⚠️ 导入{op}但未使用（浪费资源）")

        for op in unused_inits:
            errors.append(f"⚠️ 初始化{op}但未使用（浪费资源）")

        is_consistent = len(errors) == 0
        return is_consistent, errors

    def get_consistent_default(self, problem_type: str) -> str:
        """Return a consistent default workflow for fallback"""
        if problem_type == "code":
            return '''
from scripts.operators import Custom, Test

class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)
        self.test = Test(self.llm)

    async def __call__(self, problem: str, entry_point: str = "solve"):
        analysis = await self.custom(problem, "Analyze requirements and design solution")
        solution_code = await self.custom(analysis['response'], "Generate complete Python solution")
        test_result = await self.test(problem, solution_code['response'], entry_point)

        return solution_code['response']

TASK_PROMPT = """Generate a complete Python solution that passes all test cases."""
'''
        elif problem_type == "math":
            return '''
from scripts.operators import Custom, AnswerGenerate

class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)
        self.answer_generate = AnswerGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str = "solve"):
        reasoning = await self.custom(problem, "Solve step-by-step with clear mathematical reasoning")
        final_answer = await self.answer_generate(reasoning['response'])

        return final_answer.get('answer', reasoning['response'])

TASK_PROMPT = """Solve the mathematical problem with step-by-step reasoning and provide the final answer."""
'''
        else:  # qa
            return '''
from scripts.operators import Custom, AnswerGenerate

class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)
        self.answer_generate = AnswerGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str = "solve"):
        research = await self.custom(problem, "Research and gather relevant information")
        final_answer = await self.answer_generate(research['response'])

        return final_answer.get('answer', research['response'])

TASK_PROMPT = """Provide a comprehensive and accurate answer to the question."""
'''

    def validate_and_fix_workflow(self, code: str, problem_type: str):
        """
        Enhanced validation with consistency checking (no fixing).

        Returns:
            (code, is_valid, error_msg, fixes_applied)
        """
        # Phase 1: Basic syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return (code, False, f"Syntax error: {str(e)}", [])

        # Phase 2: Basic structure check
        if 'class Workflow' not in code:
            return (code, False, "Missing 'class Workflow' definition", [])

        if 'def __call__' not in code and 'async def __call__' not in code:
            return (code, False, "Missing '__call__' method", [])

        # Phase 3: Operator consistency check (CRITICAL)
        is_consistent, consistency_errors = self.check_operator_consistency(code)

        if not is_consistent:
            # ✅ 只检查，不替换：返回原始代码+错误信息
            error_msg = "Operator consistency violations: " + "; ".join(consistency_errors)
            return (code, False, error_msg, [])  # 返回原始代码，不是默认工作流

        # All checks passed
        return (code, True, "", [])

    def extract_task_prompt(self, code: str) -> str:
        """Extract TASK_PROMPT variable if present"""
        # Simple pattern to extract TASK_PROMPT
        pattern = r"TASK_PROMPT\s*=\s*['\"]([^\"']*)['\"]"
        match = re.search(pattern, code)
        if match:
            return match.group(1)
        return ""
