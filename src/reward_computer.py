#!/usr/bin/env python3
"""
5-Tier Reward System - Clear Learning Signal

Tier Structure:
- Tier 1 (0.0): Execution failure or completely wrong
- Tier 2 (0.25): 20-40% correct
- Tier 3 (0.5): 40-70% correct
- Tier 4 (0.75): 70-95% correct
- Tier 5 (1.0): 95%+ or perfect

Clean learning gradient without complex adjustments or caps.
"""
import sys
import os
import re
from typing import Any, Dict, Optional, Tuple
from fractions import Fraction

# Add AFlow to path
sys.path.insert(0, os.getenv("AFLOW_PATH", "./AFlow"))

# Import enhanced answer extractor
try:
    from .answer_extractor import AnswerExtractor
except ImportError:
    from answer_extractor import AnswerExtractor


class RewardComputer:
    """
    5-Tier reward system with clear learning gradient.

    Tier Structure:
    ├── Tier 1 (0.0): Execution failure or completely wrong (0% correct)
    ├── Tier 2 (0.25): Minimal progress (20-40% correct)
    ├── Tier 3 (0.5): Medium progress (40-70% correct)
    ├── Tier 4 (0.75): Strong progress (70-95% correct)
    └── Tier 5 (1.0): Perfect or near-perfect (95%+ correct)
    """

    def __init__(
        self,
        use_answer_extractor: bool = True,
        use_llm_judge: bool = False,
        llm_config: Optional[Dict] = None
    ):
        """
        Initialize 3-tier reward computer.

        Args:
            use_answer_extractor: Enable enhanced answer extraction
            use_llm_judge: Enable gpt-4o-mini for semantic comparison (optional)
            llm_config: LLM configuration (not used in simplified version)
        """
        self.use_answer_extractor = use_answer_extractor

        # Initialize enhanced answer extractor
        if use_answer_extractor:
            self.extractor = AnswerExtractor(use_llm_fallback=False)
        else:
            self.extractor = None

        print(f"✅ 5-Tier Reward System with Consistency Enforcement Initialized")
        print(f"   Tiers: [0.0, 0.25, 0.5, 0.75, 1.0]")
        print(f"   Answer Extractor: {'Enabled' if use_answer_extractor else 'Disabled'}")
        print(f"   Consistency Check: Enabled (violations = immediate 0.0 reward)")

    def compute_reward(
        self,
        problem: str,
        prediction: Any,
        ground_truth: Any,
        problem_type: str = "math",
        metadata: Optional[Dict] = None,
        execution_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        5-tier reward computation with consistency checking.

        Args:
            problem: Problem text
            prediction: Model's prediction (raw output)
            ground_truth: Expected answer
            problem_type: "math" | "code" | "qa"
            metadata: Additional context
            execution_metadata: Execution success/failure info

        Returns:
            {
                'reward': float,  # [0.0, 0.25, 0.5, 0.75, 1.0]
                'tier': int,      # 1-5
                'breakdown': {...}
            }
        """
        metadata = metadata or {}
        execution_metadata = execution_metadata or {}

        # Step 0: Check for real execution errors - IMMEDIATE 0.0 reward with learning feedback
        execution_error = execution_metadata.get('execution_error')
        if execution_error:
            # Phase 3增强：根据错误类型提供细粒度的学习信号
            error_type = execution_error.get('type')
            error_msg = execution_error.get('message', '')

            if error_type == 'AttributeError':
                # 细分AttributeError：可能是缺少继承、super()未调用、或operator不存在
                return self._handle_attribute_error(error_msg, problem_type)
            elif error_type in ['TypeError', 'NameError']:
                # 类型错误或名称错误：通常是参数错误或拼写错误
                return self._handle_type_or_name_error(error_msg, problem_type)
            else:
                # 其他错误：使用通用处理
                return {
                    'reward': 0.0,
                    'tier': 1,
                    'breakdown': {
                        'reason': 'execution_failure',
                        'error_type': error_type,
                        'error_message': error_msg,
                        'learning_point': execution_error.get('learning_point'),
                        'is_consistency_error': execution_error.get('is_consistency_error', False),
                        'problem_type': problem_type
                    }
                }

        # Step 1: Check operator consistency violations (pre-execution check)
        validation_metadata = execution_metadata.get('validation_metadata', {})
        is_consistent = validation_metadata.get('is_consistent', True)
        consistency_errors = validation_metadata.get('consistency_errors', [])

        if not is_consistent:
            # 预检查发现不一致，但代码仍被执行了
            return {
                'reward': 0.0,
                'tier': 1,
                'breakdown': {
                    'reason': 'operator_consistency_violation',
                    'errors': consistency_errors,
                    'message': 'Import-Initialization-Usage inconsistency detected during validation',
                    'learning_message': '导入-初始化-使用必���一致',
                    'problem_type': problem_type
                }
            }

        # Step 2: Check for execution failure - direct 0.0 reward
        if not execution_metadata.get('success', True):
            return {
                'reward': 0.0,
                'tier': 1,
                'breakdown': {
                    'reason': 'execution_failed',
                    'error': execution_metadata.get('error', 'Unknown error'),
                    'problem_type': problem_type
                }
            }

        # Step 3: Extract answers using enhanced extractor
        if self.use_answer_extractor and self.extractor:
            try:
                pred_extracted = self.extractor.extract_answer(
                    str(prediction), problem_type, is_ground_truth=False
                )
                gt_extracted = self.extractor.extract_answer(
                    str(ground_truth), problem_type, is_ground_truth=True
                )
            except Exception as e:
                # Fallback to raw strings if extraction fails
                print(f"⚠️  Answer extraction failed: {e}")
                pred_extracted = str(prediction)
                gt_extracted = str(ground_truth)
        else:
            pred_extracted = str(prediction)
            gt_extracted = str(ground_truth)

        # Step 4: Compute problem-type-specific reward (5-tier)
        if problem_type == "math":
            reward, reason = self._compute_math_reward(pred_extracted, gt_extracted)
        elif problem_type == "code":
            reward, reason = self._compute_code_reward(pred_extracted, execution_metadata)
        elif problem_type == "qa":
            reward, reason = self._compute_qa_reward(pred_extracted, gt_extracted)
        else:
            reward, reason = self._compute_qa_reward(pred_extracted, gt_extracted)

        # Snap to nearest tier
        tier_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        reward = min(tier_levels, key=lambda x: abs(x - reward))
        tier_index = tier_levels.index(reward) + 1

        return {
            'reward': reward,
            'tier': tier_index,
            'breakdown': {
                'reason': reason,
                'problem_type': problem_type,
                'prediction': str(pred_extracted)[:100],
                'ground_truth': str(gt_extracted)[:100]
            }
        }

    def _is_math_correct(self, prediction: str, ground_truth: str) -> bool:
        """
        Comprehensive math answer comparison - handles all edge cases.

        Comparison strategy:
        1. Exact string match (case-insensitive, whitespace-normalized)
        2. Extract and compare \boxed{} content
        3. Numerical comparison with tolerance (handles fractions, decimals)
        4. Extract last number from text and compare

        This method solves the \boxed{13} vs 13 issue by extracting content first.

        Args:
            prediction: Model's predicted answer (may contain \boxed{})
            ground_truth: Expected answer (may contain \boxed{})

        Returns:
            True if answers are mathematically equivalent, False otherwise
        """
        pred_str = str(prediction).strip()
        gt_str = str(ground_truth).strip()

        # 1. Exact string match (normalize whitespace and case)
        pred_norm = ' '.join(pred_str.lower().split())
        gt_norm = ' '.join(gt_str.lower().split())
        if pred_norm == gt_norm:
            return True

        # 2. Extract \boxed{} content and compare
        pred_boxed = self._extract_boxed_content(pred_str)
        gt_boxed = self._extract_boxed_content(gt_str)

        # If one has boxed and the other doesn't, compare boxed content with raw
        if pred_boxed and not gt_boxed:
            # Compare pred_boxed with gt_str
            if self._answers_equal_normalized(pred_boxed, gt_str):
                return True
        elif gt_boxed and not pred_boxed:
            # Compare pred_str with gt_boxed
            if self._answers_equal_normalized(pred_str, gt_boxed):
                return True
        elif pred_boxed and gt_boxed:
            # Both have boxed, compare the contents
            if self._answers_equal_normalized(pred_boxed, gt_boxed):
                return True

        # 3. Numerical comparison with tolerance
        try:
            pred_num = self._parse_number_strict(pred_str)
            gt_num = self._parse_number_strict(gt_str)
            if pred_num is not None and gt_num is not None:
                if self._numbers_equal(pred_num, gt_num):
                    return True
        except:
            pass

        # Also try comparing boxed contents numerically
        if pred_boxed and gt_boxed:
            try:
                pred_num = self._parse_number_strict(pred_boxed)
                gt_num = self._parse_number_strict(gt_boxed)
                if pred_num is not None and gt_num is not None:
                    if self._numbers_equal(pred_num, gt_num):
                        return True
            except:
                pass

        # 4. Extract last number from text and compare
        pred_numbers = self._extract_numbers_from_text(pred_str)
        gt_numbers = self._extract_numbers_from_text(gt_str)
        if pred_numbers and gt_numbers:
            try:
                pred_last = self._parse_number_strict(pred_numbers[-1])
                gt_last = self._parse_number_strict(gt_numbers[-1])
                if pred_last is not None and gt_last is not None:
                    if self._numbers_equal(pred_last, gt_last):
                        return True
            except:
                pass

        return False

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """Extract content from \boxed{...}"""
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None

    def _answers_equal_normalized(self, ans1: str, ans2: str) -> bool:
        """Compare two answers after normalization"""
        # Normalize strings
        norm1 = ' '.join(ans1.lower().strip().split())
        norm2 = ' '.join(ans2.lower().strip().split())
        if norm1 == norm2:
            return True

        # Try numerical comparison
        try:
            num1 = self._parse_number_strict(ans1)
            num2 = self._parse_number_strict(ans2)
            if num1 is not None and num2 is not None:
                return self._numbers_equal(num1, num2)
        except:
            pass

        return False

    def _parse_number_strict(self, text: str) -> Optional[float]:
        """
        Parse a number from text with better fraction support.

        Handles:
        - Regular floats: "3.14", "-5.2"
        - Fractions: "1/2", "3/4"
        - Mixed: "2 1/2" → 2.5
        """
        if not text:
            return None

        text = str(text).strip()

        # Remove common LaTeX commands
        text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)

        # Try fraction first (e.g., "1/2", "3/4")
        if '/' in text:
            try:
                parts = text.split('/')
                if len(parts) == 2:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    if denominator != 0:
                        return numerator / denominator
            except:
                pass

        # Try direct float conversion
        try:
            return float(text)
        except:
            pass

        # Extract first number from text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass

        return None

    def _extract_numbers_from_text(self, text: str) -> list:
        """Extract all number strings from text (including fractions)"""
        numbers = []

        # Extract fractions first
        fractions = re.findall(r'-?\d+/\d+', text)
        numbers.extend(fractions)

        # Extract regular numbers
        regular = re.findall(r'-?\d+\.?\d*', text)
        for num in regular:
            # Skip if it's part of a fraction we already found
            if not any(num in frac for frac in fractions):
                numbers.append(num)

        return numbers

    def _numbers_equal(self, a: float, b: float, rel_tol: float = 1e-6) -> bool:
        """
        Compare two numbers with relative tolerance.

        Uses relative error to handle both small and large numbers correctly.
        For numbers near zero, uses absolute tolerance.
        """
        # Handle exact equality
        if a == b:
            return True

        # Handle near-zero cases with absolute tolerance
        if abs(b) < 1e-9:
            return abs(a - b) < 1e-9

        # Use relative error for general case
        rel_error = abs(a - b) / abs(b)
        return rel_error < rel_tol

    def _compute_math_reward(self, prediction: str, ground_truth: str) -> Tuple[float, str]:
        """
        Compute reward for math problems - uses exact matching first, then error-based tiers.

        Strategy:
        1. Check if answer is correct using _is_math_correct() → 1.0 reward
        2. If not exact match, compute error-based tiered reward

        Returns:
            (reward, reason) tuple
        """
        # First check if the answer is correct (handles all edge cases)
        if self._is_math_correct(prediction, ground_truth):
            return (1.0, "Correct answer")

        # If not correct, compute error-based partial credit
        try:
            # Try to parse both as numbers
            pred_num = self._parse_number(prediction)
            gt_num = self._parse_number(ground_truth)

            if pred_num is None or gt_num is None:
                return (0.0, "Cannot parse as number")

            # Calculate relative error
            if abs(gt_num) < 1e-9:  # Ground truth near zero
                error = abs(pred_num)
            else:
                error = abs((pred_num - gt_num) / gt_num)

            # 5-tier classification based on error thresholds
            if error < 0.0001:  # < 0.01% (near perfect - catches any rounding)
                return (1.0, f"Near-perfect (error={error:.6f})")
            elif error < 0.1:  # < 10%
                return (0.75, f"Strong (error={error:.2%})")
            elif error < 0.3:  # < 30%
                return (0.5, f"Medium progress (error={error:.2%})")
            elif error < 0.5:  # < 50%
                return (0.25, f"Minimal progress (error={error:.2%})")
            else:  # >= 50%
                return (0.0, f"Wrong answer (error={error:.2%})")

        except Exception as e:
            # Fallback to string comparison
            pred_clean = prediction.strip().lower()
            gt_clean = ground_truth.strip().lower()

            if pred_clean == gt_clean:
                return (1.0, "Exact string match")
            elif pred_clean in gt_clean or gt_clean in pred_clean:
                return (0.5, "Partial string match")
            else:
                return (0.0, f"No match (error: {str(e)})")

    def _compute_code_reward(self, prediction: str, metadata: Dict) -> Tuple[float, str]:
        """
        Compute 5-tier reward for code problems based on test pass rate.

        Tiers based on pass_rate:
        - Tier 1 (0.0): 0% or syntax error
        - Tier 2 (0.25): 1-25% tests passed
        - Tier 3 (0.5): 25-75% tests passed
        - Tier 4 (0.75): 75-99% tests passed
        - Tier 5 (1.0): 99%+ tests passed (all or nearly all)

        Returns:
            (reward, reason) tuple
        """
        # Check if test results are available
        test_results = metadata.get('test_results', {})

        if not test_results:
            # No test results - use basic syntax check
            if self._is_valid_python_syntax(prediction):
                return (0.25, "Valid Python syntax but no test results")
            else:
                return (0.0, "Invalid Python syntax")

        # Use test pass rate
        passed = test_results.get('passed', 0)
        total = test_results.get('total', 0)

        if total == 0:
            return (0.0, "No tests available")

        pass_rate = passed / total

        # 5-tier classification based on pass rate
        if pass_rate >= 0.99:  # 99%+ (all or nearly all)
            return (1.0, f"Perfect ({passed}/{total})")
        elif pass_rate >= 0.75:  # 75-99%
            return (0.75, f"Strong ({passed}/{total} tests passed)")
        elif pass_rate >= 0.25:  # 25-75%
            return (0.5, f"Medium progress ({passed}/{total} tests passed)")
        elif pass_rate > 0:  # 1-25%
            return (0.25, f"Minimal progress ({passed}/{total} tests passed)")
        else:  # 0%
            return (0.0, f"Failed ({passed}/{total} tests passed)")

    def _compute_qa_reward(self, prediction: str, ground_truth: str) -> Tuple[float, str]:
        """
        Compute 5-tier reward for QA problems based on word overlap.

        Tiers based on overlap:
        - Tier 1 (0.0): 0-20% overlap or empty
        - Tier 2 (0.25): 20-40% overlap
        - Tier 3 (0.5): 40-70% overlap
        - Tier 4 (0.75): 70-90% overlap
        - Tier 5 (1.0): 90%+ overlap (near/exact match)

        Returns:
            (reward, reason) tuple
        """
        # Normalize both texts
        pred_norm = self._normalize_text(prediction)
        gt_norm = self._normalize_text(ground_truth)

        # Exact match
        if pred_norm == gt_norm:
            return (1.0, "Exact match")

        # Calculate word overlap
        pred_words = set(pred_norm.split())
        gt_words = set(gt_norm.split())

        if not gt_words:
            return (0.0, "Empty ground truth")

        overlap = len(pred_words & gt_words) / len(gt_words)

        # 5-tier classification based on overlap
        if overlap >= 0.9:  # 90%+
            return (1.0, f"Near-perfect ({overlap:.0%})")
        elif overlap >= 0.7:  # 70-90%
            return (0.75, f"Strong ({overlap:.0%})")
        elif overlap >= 0.4:  # 40-70%
            return (0.5, f"Medium progress ({overlap:.0%})")
        elif overlap >= 0.2:  # 20-40%
            return (0.25, f"Minimal progress ({overlap:.0%})")
        else:  # < 20%
            return (0.0, f"Low overlap ({overlap:.0%})")

    # ==================== HELPER METHODS ====================

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse a number from text, handling various formats."""
        if not text:
            return None

        text = str(text).strip()

        # Try direct float conversion
        try:
            return float(text)
        except:
            pass

        # Extract last number from text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                pass

        # Try fraction
        try:
            return float(Fraction(text))
        except:
            pass

        return None

    def _is_valid_python_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except:
            return False

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = str(text).lower().strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def evaluate_code_quality(self, code_metadata: Dict) -> Tuple[float, str]:
        """
        根据代码质量元数据评估代码质量得分

        Args:
            code_metadata: 代码质量检查的结果字典，包含：
                - has_syntax_error: bool
                - has_call_method: bool
                - signature_correct: bool
                - operators_valid: bool
                - operator_calls_valid: bool
                - has_return_statement: bool
                - issues: List[str]

        Returns:
            (quality_score, reason) - quality_score 在 0.0 到 1.0 之间
        """
        if not code_metadata:
            return (0.5, "无代码质量元数据")

        # 统计通过的检查项
        score = 1.0
        issues = code_metadata.get('issues', [])

        # 检查项权重（从严重到轻微）
        if code_metadata.get('has_syntax_error', False):
            score = 0.0  # 语法错误无法运行
            return (score, "代码有语法错误，无法执行")

        if not code_metadata.get('has_call_method', False):
            score *= 0.0  # 没有__call__方法是致命的
            return (score, "代码缺少async def __call__方法")

        if not code_metadata.get('signature_correct', False):
            score *= 0.1  # 签名错误会导致运行时错误

        if not code_metadata.get('operators_valid', False):
            score *= 0.3  # operator类型不匹配会导致运行失败
            if issues:
                reason = "Operator不适合该问题类型: " + "; ".join([i for i in issues if "Operator" in i])
                return (score, reason)

        if not code_metadata.get('operator_calls_valid', True):
            score *= 0.5  # operator调用参数错误会导致运行时错误
            if issues:
                reason = "Operator调用参数错误: " + "; ".join([i for i in issues if "参数" in i or "错误" in i])
                return (score, reason)

        if not code_metadata.get('has_return_statement', False):
            score *= 0.2  # 没有return语句是严重问题

        # 根据最终分数返回
        if score >= 0.9:
            return (1.0, "代码质量优秀")
        elif score >= 0.7:
            return (0.75, "代码质量良好，但有轻微问题")
        elif score >= 0.5:
            return (0.5, f"代码质量一般，存在多个问题")
        else:
            reason = "代码质量差: " + "; ".join(issues) if issues else "代码质量差"
            return (score, reason)

    def _handle_attribute_error(self, error_msg: str, problem_type: str) -> Dict:
        """
        细粒度的AttributeError处理

        不同的AttributeError原因给予不同的学习指导：
        - 缺少继承声明：class Workflow应该继承MathWorkflowBase等
        - super()未调用：__init__应该调用super().__init__()
        - operator不存在：使用的operator没有被初始化
        """
        error_msg_lower = error_msg.lower()

        # Case 1: 缺少继承声明（特征：提到"workflow"或直接缺少attribute）
        if "'nonetype' object" in error_msg_lower or 'self.' in error_msg_lower:
            # 这通常表示self本身是None或缺少属性
            if any(op in error_msg_lower for op in ['answer_generate', 'programmer', 'review', 'revise', 'test']):
                base_class = {
                    'math': 'MathWorkflowBase',
                    'code': 'CodeWorkflowBase',
                    'qa': 'QAWorkflowBase'
                }.get(problem_type, 'MathWorkflowBase')

                return {
                    'reward': 0.0,
                    'tier': 1,
                    'breakdown': {
                        'reason': 'missing_inheritance',
                        'error': error_msg,
                        'learning_message': '⚠️  代码缺少class继承声明或super()未调用',
                        'instruction': f'MUST add: class Workflow({base_class}): and super().__init__(name, llm_config, dataset)',
                        'error_category': 'code_structure',
                        'error_priority': 'critical',
                        'problem_type': problem_type
                    }
                }

        # Case 2: super().__init__() 未调用
        if 'super' in error_msg_lower or 'init' in error_msg_lower:
            return {
                'reward': 0.0,
                'tier': 1,
                'breakdown': {
                    'reason': 'super_init_not_called',
                    'error': error_msg,
                    'learning_message': '⚠️  __init__方法必须调用 super().__init__()',
                    'instruction': 'Add super().__init__(name, llm_config, dataset) at the beginning of __init__',
                    'error_category': 'code_structure',
                    'error_priority': 'critical',
                    'problem_type': problem_type
                }
            }

        # Case 3: Generic operator不存在或未初始化
        return {
            'reward': 0.0,
            'tier': 1,
            'breakdown': {
                'reason': 'operator_not_initialized',
                'error': error_msg,
                'learning_message': '⚠️  Operator初始化失败或使用不一致',
                'instruction': '检查: (1) class是否继承基类 (2) __init__是否调用super() (3) operator名称是否正确',
                'error_category': 'operator_initialization',
                'error_priority': 'critical',
                'problem_type': problem_type
            }
        }

    def _handle_type_or_name_error(self, error_msg: str, problem_type: str) -> Dict:
        """
        处理TypeError和NameError

        通常是参数错误或拼写错误
        """
        error_msg_lower = error_msg.lower()

        # 判断是哪种错误
        if 'got an unexpected keyword argument' in error_msg_lower:
            return {
                'reward': 0.0,
                'tier': 1,
                'breakdown': {
                    'reason': 'operator_parameter_error',
                    'error': error_msg,
                    'learning_message': '⚠️  Operator调用参数名错误',
                    'instruction': '检查operator的参数名是否正确（例如：answer_generate应该用input=，不是problem=）',
                    'error_category': 'operator_parameters',
                    'error_priority': 'major',
                    'problem_type': problem_type
                }
            }
        elif 'not defined' in error_msg_lower or 'undefined' in error_msg_lower:
            return {
                'reward': 0.0,
                'tier': 1,
                'breakdown': {
                    'reason': 'name_error',
                    'error': error_msg,
                    'learning_message': '⚠️  变量或函数未定义或拼写错误',
                    'instruction': '检查变量名拼写是否正确，以及是否所有变量都已定义',
                    'error_category': 'syntax_or_naming',
                    'error_priority': 'major',
                    'problem_type': problem_type
                }
            }
        else:
            return {
                'reward': 0.0,
                'tier': 1,
                'breakdown': {
                    'reason': 'parameter_or_syntax_error',
                    'error': error_msg,
                    'learning_message': '⚠️  Operator调用参数错误或拼写错误',
                    'instruction': '检查operator的参数名和调用方式是否正确',
                    'error_category': 'operator_or_syntax',
                    'error_priority': 'major',
                    'problem_type': problem_type
                }
            }
