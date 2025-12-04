#!/usr/bin/env python3
"""
5-Tier Reward System - Complete Rewrite
Implements granular [0.0, 0.2, 0.4, 0.7, 1.0] tier-based rewards.

Design:
- Tier 1 (0.0): Completely wrong or no output
- Tier 2 (0.2): Minimal/some correct output
- Tier 3 (0.4): Partial/majority correct
- Tier 4 (0.7): Very good/close/mostly correct
- Tier 5 (1.0): Perfect/all correct

For each problem type (math/code/qa), tiers are defined with clear metrics.
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
    from .answer_extractor_v2 import AnswerExtractor
except ImportError:
    from answer_extractor_v2 import AnswerExtractor


class RewardComputer:
    """
    5-Tier granular reward system with problem-type-specific logic.

    Tier Structure:
    ├── Math: 1.0 (error<1e-4) | 0.7 (error<5%) | 0.4 (error<50%) | 0.2 (has output) | 0.0 (wrong)
    ├── Code: 1.0 (100% pass) | 0.7 (>80% pass) | 0.4 (>50% pass) | 0.2 (>20% pass) | 0.0 (fail)
    └── QA:   1.0 (exact/equiv) | 0.7 (answer+context) | 0.4 (high overlap) | 0.2 (partial) | 0.0 (wrong)
    """

    def __init__(
        self,
        use_answer_extractor: bool = True,
        use_llm_judge: bool = False,
        llm_config: Optional[Dict] = None
    ):
        """
        Initialize 5-tier reward computer.

        Args:
            use_answer_extractor: Enable enhanced answer extraction
            use_llm_judge: Enable gpt-4o-mini for semantic comparison
            llm_config: LLM configuration (base_url, api_key, model_name)
        """
        self.use_answer_extractor = use_answer_extractor

        # Initialize enhanced answer extractor
        if use_answer_extractor:
            self.extractor = AnswerExtractor(use_llm_fallback=False)
        else:
            self.extractor = None

        # Initialize LLM Judge
        self.use_llm_judge = use_llm_judge
        self.llm_judge_client = None
        if use_llm_judge:
            self._init_llm_judge_client(llm_config)

        print(f"✅ 5-Tier Reward System Initialized")
        print(f"   Tiers: [0.0, 0.2, 0.4, 0.7, 1.0]")
        print(f"   Answer Extractor: {'Enabled' if use_answer_extractor else 'Disabled'}")
        print(f"   LLM Judge: {'Enabled (gpt-4o-mini)' if use_llm_judge else 'Disabled'}")

    def _init_llm_judge_client(self, llm_config: Optional[Dict]):
        """Initialize LLM Judge client (OpenAI gpt-4o-mini)"""
        try:
            from openai import OpenAI
            import yaml

            aflow_config_path = "config/aflow_llm.yaml"
            default_config = {
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv("OPENAI_API_KEY", "sk-xxx"),
                "model_name": "gpt-4o-mini"
            }

            # Try to read aflow config
            try:
                with open(aflow_config_path, 'r', encoding='utf-8') as f:
                    aflow_config = yaml.safe_load(f)
                    models_config = aflow_config.get('models', {})
                    if 'gpt-4o-mini' in models_config:
                        gpt_cfg = models_config['gpt-4o-mini']
                        default_config = {
                            "base_url": gpt_cfg.get('base_url', default_config["base_url"]),
                            "api_key": gpt_cfg.get('api_key', default_config["api_key"]),
                            "model_name": gpt_cfg.get('model_name', default_config["model_name"])
                        }
                        print(f"   ✅ Loaded gpt-4o-mini config from {aflow_config_path}")
            except Exception as e:
                print(f"   ⚠️  Could not read {aflow_config_path}: {e}")

            config = llm_config or default_config

            self.llm_judge_client = OpenAI(
                base_url=config.get("base_url", default_config["base_url"]),
                api_key=config.get("api_key", default_config["api_key"])
            )
            self.llm_judge_model = config.get("model_name", default_config["model_name"])
            print(f"   ✅ LLM Judge client initialized ({self.llm_judge_model})")

        except Exception as e:
            print(f"   ⚠️  LLM Judge initialization failed: {e}")
            self.use_llm_judge = False
            self.llm_judge_client = None

    # ==================== MAIN ENTRY POINT ====================

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
        Unified 5-tier reward computation.

        Args:
            problem: Problem text
            prediction: Model's prediction (raw output)
            ground_truth: Expected answer
            problem_type: "math" | "code" | "qa"
            metadata: Additional context
            execution_metadata: Execution success/failure info

        Returns:
            {
                'reward': float,  # [0.0, 0.2, 0.4, 0.7, 1.0]
                'tier': int,      # 1-5
                'breakdown': {...}
            }
        """
        metadata = metadata or {}
        execution_metadata = execution_metadata or {}

        # Step 1: Extract answers using enhanced extractor
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

        # Step 2: Compute problem-type-specific base tier
        if problem_type == "math":
            base_tier = self._compute_math_reward(pred_extracted, gt_extracted)
        elif problem_type == "code":
            base_tier = self._compute_code_reward(pred_extracted, execution_metadata)
        elif problem_type == "qa":
            base_tier = self._compute_qa_reward(pred_extracted, gt_extracted)
        else:
            base_tier = self._compute_qa_reward(pred_extracted, gt_extracted)

        # Step 3: Apply execution penalties
        execution_penalty = 0.0
        penalty_reason = ""

        if not execution_metadata.get('success', True):
            error_type = execution_metadata.get('error_type', 'unknown')

            if error_type == 'operator_mismatch':
                execution_penalty = -0.4
                penalty_reason = "Operator-problem type mismatch"
            elif error_type == 'validation_failed':
                execution_penalty = -0.2
                penalty_reason = "Validation/syntax error"
            else:
                execution_penalty = -0.3
                penalty_reason = f"Execution error: {error_type}"

        # Step 4: Compute final reward (ensure stays in tier boundaries)
        final_reward = max(0.0, base_tier + execution_penalty)

        # Snap to nearest tier
        tier_levels = [0.0, 0.2, 0.4, 0.7, 1.0]
        final_reward = min(tier_levels, key=lambda x: abs(x - final_reward))

        tier_index = tier_levels.index(final_reward)

        return {
            'reward': final_reward,
            'tier': tier_index + 1,  # 1-5
            'breakdown': {
                'base_tier': base_tier,
                'execution_penalty': execution_penalty,
                'final_reward': final_reward,
                'penalty_reason': penalty_reason,
                'problem_type': problem_type
            }
        }

    # ==================== MATH REWARD COMPUTATION ====================

    def _compute_math_reward(self, prediction: str, ground_truth: str) -> float:
        """
        Math reward: 5-tier based on numerical accuracy.

        Tier 5 (1.0): Perfect match (error < 1e-4)
        Tier 4 (0.7): Close match (error < 5%)
        Tier 3 (0.4): Partial (error < 50% OR format OK but value wrong)
        Tier 2 (0.2): Has output (boxed/answer tag but wrong)
        Tier 1 (0.0): Completely wrong or no output
        """
        if not prediction or not ground_truth:
            return 0.0

        # Step 1: Try numerical comparison
        pred_value = self._extract_final_number(prediction)
        gt_value = self._extract_final_number(ground_truth)

        if pred_value is not None and gt_value is not None:
            try:
                rel_error = abs(pred_value - gt_value) / (abs(gt_value) + 1e-9)

                if rel_error < 1e-4:
                    return 1.0  # Perfect
                elif rel_error < 0.05:  # 5% threshold
                    return 0.7  # Very good
                elif rel_error < 0.5:   # 50% threshold
                    return 0.4  # Partial
                else:
                    # Check if has valid format
                    if self._has_valid_math_format(prediction):
                        return 0.2  # Has output but wrong
                    return 0.0
            except (ValueError, ZeroDivisionError, TypeError):
                pass

        # Step 2: Try algebraic equivalence
        if self._is_algebraic_equivalent(prediction, ground_truth):
            return 1.0

        # Step 3: LLM Judge fallback
        if self.use_llm_judge and self.llm_judge_client:
            is_equiv = self._llm_judge_compare(
                problem="",
                prediction=prediction,
                ground_truth=ground_truth,
                problem_type="math"
            )
            if is_equiv:
                return 0.7  # LLM says equivalent

        # Step 4: Format-only credit
        if self._has_valid_math_format(prediction):
            return 0.2  # Has formatting but no valid content

        return 0.0

    def _extract_final_number(self, text: str) -> Optional[float]:
        """
        Extract final numerical value from text.
        Handles: floats, fractions, scientific notation, units.
        """
        if not text:
            return None

        text = str(text).strip()

        # Try \boxed{} first
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            content = boxed_match.group(1).strip()
            num = self._parse_number(content)
            if num is not None:
                return num

        # Try fraction
        frac_match = re.search(r'(\d+)/(\d+)', text)
        if frac_match:
            try:
                return float(frac_match.group(1)) / float(frac_match.group(2))
            except (ValueError, ZeroDivisionError):
                pass

        # Extract all numbers and return last
        numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse a single number from text"""
        try:
            # Try float first
            return float(text)
        except ValueError:
            pass

        # Try fraction
        if '/' in text:
            parts = text.split('/')
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    pass

        return None

    def _has_valid_math_format(self, text: str) -> bool:
        """Check if text has valid mathematical formatting"""
        return bool(
            re.search(r'\\boxed\{', text) or
            re.search(r'<answer>', text) or
            re.search(r'#### ', text) or  # GSM8K format
            re.search(r'-?\d+', text)      # Any number
        )

    def _is_algebraic_equivalent(self, expr1: str, expr2: str) -> bool:
        """Check algebraic equivalence using SymPy"""
        try:
            from sympy import sympify, simplify

            # Skip if contains variables
            for var in ['x', 'y', 'z', 'n', 'a', 'b', 'c']:
                if var in expr1.lower() or var in expr2.lower():
                    return False

            e1 = sympify(expr1)
            e2 = sympify(expr2)

            return simplify(e1 - e2) == 0
        except Exception:
            return False

    # ==================== CODE REWARD COMPUTATION ====================

    def _compute_code_reward(self, prediction: str, metadata: Dict) -> float:
        """
        Code reward: 5-tier based on test pass rate.

        Tier 5 (1.0): All tests pass (100%)
        Tier 4 (0.7): Mostly correct (>80%)
        Tier 3 (0.4): Majority correct (>50%)
        Tier 2 (0.2): Some correct (>20% OR structurally valid)
        Tier 1 (0.0): No tests pass or runtime error
        """
        if not prediction or not isinstance(prediction, str):
            return 0.0

        pred_clean = prediction.strip()

        if not pred_clean or len(pred_clean) < 10:
            return 0.0

        # Extract test pass rate from various metadata formats
        test_pass_rate = metadata.get('test_pass_rate', None)

        if test_pass_rate is None:
            # Try to compute from test_results (list of booleans)
            if 'test_results' in metadata:
                test_results = metadata['test_results']
                if isinstance(test_results, list) and len(test_results) > 0:
                    test_pass_rate = sum(test_results) / len(test_results)
                else:
                    test_pass_rate = 0.0
            # Try to compute from test_passed / test_total
            elif 'test_passed' in metadata and 'test_total' in metadata:
                test_passed = metadata['test_passed']
                test_total = metadata['test_total']
                if test_total > 0:
                    test_pass_rate = test_passed / test_total
                else:
                    test_pass_rate = 0.0
            else:
                test_pass_rate = 0.0

        has_syntax_error = metadata.get('syntax_error', False)
        has_runtime_error = metadata.get('runtime_error', False)

        # Execution errors
        if has_syntax_error or has_runtime_error:
            if 'def ' in pred_clean and ('return' in pred_clean or 'print' in pred_clean):
                return 0.0  # Broken structure
            return 0.0

        # Tier based on test pass rate
        # Tiers: 1.0 (100%), 0.7 (75%+), 0.4 (50%+), 0.2 (25%+), 0.0 (<25%)
        if test_pass_rate >= 0.99:  # 100% pass
            return 1.0
        elif test_pass_rate >= 0.75:  # 75%+ pass
            return 0.7
        elif test_pass_rate >= 0.5:  # 50%+ pass
            return 0.4
        elif test_pass_rate >= 0.2:  # 20%+ pass
            return 0.2
        else:
            # <20% pass - return 0.0 (structurally valid but mostly broken)
            return 0.0

    def _is_valid_python_syntax(self, code: str) -> bool:
        """Validate Python syntax"""
        try:
            import ast
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False

    # ==================== QA REWARD COMPUTATION ====================

    def _compute_qa_reward(self, prediction: str, ground_truth: str) -> float:
        """
        QA reward: 5-tier based on semantic similarity & token overlap.

        Tier 5 (1.0): Exact match or semantic equivalence
        Tier 4 (0.7): Contains answer + context OR F1 > 0.75
        Tier 3 (0.4): High token overlap (F1 > 0.5)
        Tier 2 (0.2): Moderate overlap (F1 > 0.2)
        Tier 1 (0.0): Low overlap or empty
        """
        if not prediction or not ground_truth:
            return 0.0

        pred_norm = self._normalize_text(prediction)
        gt_norm = self._normalize_text(ground_truth)

        # Tier 5: Exact match
        if pred_norm == gt_norm:
            return 1.0

        # Compute token F1 score (for tier decisions)
        pred_tokens = set(pred_norm.split())
        gt_tokens = set(gt_norm.split())

        if not gt_tokens:
            return 0.0

        if pred_tokens:
            intersection = len(pred_tokens & gt_tokens)
            precision = intersection / len(pred_tokens) if pred_tokens else 0
            recall = intersection / len(gt_tokens) if gt_tokens else 0
            f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        else:
            f1 = 0.0

        # Tier 4: Contains answer + extra context OR very high F1
        if (gt_norm in pred_norm and len(pred_norm) > len(gt_norm) + 5) or f1 > 0.75:
            return 0.7

        # Tier 3: High F1
        if f1 > 0.5:
            return 0.4

        # Tier 2: Moderate F1
        elif f1 > 0.2:
            return 0.2

        # Tier 1: Very low overlap
        else:
            return 0.0

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = ' '.join(text.split())  # Normalize whitespace
        return text.strip()

    # ==================== LLM JUDGE COMPARISON ====================

    def _llm_judge_compare(
        self,
        problem: str,
        prediction: str,
        ground_truth: str,
        problem_type: str
    ) -> bool:
        """
        Use LLM Judge (gpt-4o-mini) for semantic equivalence.
        Returns True if equivalent, False otherwise.
        """
        if not self.llm_judge_client:
            return False

        prompt = f"""You are a precise answer equivalence evaluator.

**Task:** Determine if the Model Response contains an answer equivalent to the Ground Truth.

**Ground Truth:** {ground_truth}
**Model Response:** {prediction}

**Instructions:**
1. Extract the final answer from both
2. Normalize (numbers, units, formatting)
3. Compare for semantic equivalence
4. Allow reasonable rounding

**Output Format:**
<analysis>Brief reasoning</analysis>
<true_false>True or False</true_false>
"""

        try:
            for attempt in range(2):
                response = self.llm_judge_client.chat.completions.create(
                    model=self.llm_judge_model,
                    messages=[
                        {"role": "system", "content": "You are a precise answer evaluator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=200
                )

                content = response.choices[0].message.content
                if content is None:
                    if attempt == 0:
                        continue
                    else:
                        return False

                result_text = content.strip()
                break

            # Parse response - 5 fallback formats
            patterns = [
                r'<true_false>\s*(True|False)\s*</true_false>',
                r'<true_false>\s*:\s*(True|False)',
                r'\*\*true_false\*\*\s*:?\s*(True|False)',
                r'true_false\s*:?\s*(True|False)',
                r'\b(True|False)\b',
            ]

            for pattern in patterns:
                match = re.search(pattern, result_text, re.IGNORECASE)
                if match:
                    return match.group(1).lower() == "true"

            return False

        except Exception as e:
            print(f"⚠️  LLM Judge failed: {e}")
            return False
