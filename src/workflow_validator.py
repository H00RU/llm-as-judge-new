#!/usr/bin/env python3
"""
Simplified Workflow Validator - No Auto-Fix

Only performs basic syntax validation.
No patching, no fixing, no reactive mechanisms.
"""


class WorkflowValidator:
    """Minimal validator - basic syntax checks only"""

    def __init__(self):
        """Initialize validator"""
        pass

    def validate_and_fix_workflow(self, code: str, problem_type: str):
        """
        Validate workflow code (no fixing).

        Returns:
            (code, is_valid, error_msg, fixes_applied)
        """
        # Basic syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return (code, False, f"Syntax error: {str(e)}", [])

        # Check required structures
        if 'class Workflow' not in code:
            return (code, False, "Missing 'class Workflow' definition", [])

        if 'def __call__' not in code and 'async def __call__' not in code:
            return (code, False, "Missing '__call__' method", [])

        # Code is valid
        return (code, True, "", [])

    def extract_task_prompt(self, code: str) -> str:
        """Extract TASK_PROMPT variable if present"""
        import re

        # Simple pattern to extract TASK_PROMPT
        pattern = r"TASK_PROMPT\s*=\s*['\"]([^\"']*)['\"]"
        match = re.search(pattern, code)
        if match:
            return match.group(1)
        return ""
