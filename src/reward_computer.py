#!/usr/bin/env python3
"""
3-Tier Reward System - Simplified

Design:
- Tier 1 (0.0): Completely wrong or execution failure
- Tier 2 (0.5): Partially correct
- Tier 3 (1.0): Perfect/correct

Simple, clean learning signal without complex adjustments or caps.
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
    3-Tier simplified reward system.

    Tier Structure:
    ├── Tier 1 (0.0): Completely wrong or execution failure
    ├── Tier 2 (0.5): Partially correct
    └── Tier 3 (1.0): Perfect/correct
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

        print(f"✅ 3-Tier Reward System Initialized")
        print(f"   Tiers: [0.0, 0.5, 1.0]")
        print(f"   Answer Extractor: {'Enabled' if use_answer_extractor else 'Disabled'}")

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
        Simplified 3-tier reward computation.

        Args:
            problem: Problem text
            prediction: Model's prediction (raw output)
            ground_truth: Expected answer
            problem_type: "math" | "code" | "qa"
            metadata: Additional context (unused in simplified version)
            execution_metadata: Execution success/failure info

        Returns:
            {
                'reward': float,  # [0.0, 0.5, 1.0]
                'tier': int,      # 1-3
                'breakdown': {...}
            }
        """
        metadata = metadata or {}
        execution_metadata = execution_metadata or {}

        # Step 1: Check for execution failure - direct 0.0 reward
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

        # Step 2: Extract answers using enhanced extractor
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

        # Step 3: Compute problem-type-specific reward (3-tier)
        if problem_type == "math":
            reward, reason = self._compute_math_reward(pred_extracted, gt_extracted)
        elif problem_type == "code":
            reward, reason = self._compute_code_reward(pred_extracted, execution_metadata)
        elif problem_type == "qa":
            reward, reason = self._compute_qa_reward(pred_extracted, gt_extracted)
        else:
            reward, reason = self._compute_qa_reward(pred_extracted, gt_extracted)

        # Snap to nearest tier
        tier_levels = [0.0, 0.5, 1.0]
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

    def _compute_math_reward(self, prediction: str, ground_truth: str) -> Tuple[float, str]:
        """
        Compute 3-tier reward for math problems.

        Returns:
            (reward, reason) tuple
        """
        try:
            # Try to parse both as numbers
            pred_num = self._parse_number(prediction)
            gt_num = self._parse_number(ground_truth)

            if pred_num is None:
                return (0.0, "Cannot parse prediction as number")

            if gt_num is None:
                return (0.0, "Cannot parse ground truth as number")

            # Calculate error
            if gt_num == 0:
                error = abs(pred_num)
            else:
                error = abs((pred_num - gt_num) / gt_num)

            # 3-tier classification
            if error < 1e-4:
                return (1.0, f"Exact match (error={error:.6f})")
            elif error < 0.1:  # Within 10%
                return (0.5, f"Partially correct (error={error:.2%})")
            else:
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
        Compute 3-tier reward for code problems.

        Returns:
            (reward, reason) tuple
        """
        # Check if test results are available
        test_results = metadata.get('test_results', {})

        if not test_results:
            # No test results - use basic syntax check
            if self._is_valid_python_syntax(prediction):
                return (0.5, "Valid Python syntax but no test results")
            else:
                return (0.0, "Invalid Python syntax")

        # Use test pass rate
        passed = test_results.get('passed', 0)
        total = test_results.get('total', 0)

        if total == 0:
            return (0.0, "No tests available")

        pass_rate = passed / total

        # 3-tier classification
        if pass_rate >= 0.99:  # Allow for rounding
            return (1.0, f"All tests passed ({passed}/{total})")
        elif pass_rate >= 0.5:
            return (0.5, f"Partially correct ({passed}/{total} tests passed)")
        else:
            return (0.0, f"Most tests failed ({passed}/{total} tests passed)")

    def _compute_qa_reward(self, prediction: str, ground_truth: str) -> Tuple[float, str]:
        """
        Compute 3-tier reward for QA problems.

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

        # 3-tier classification
        if overlap >= 0.8:
            return (1.0, f"High overlap ({overlap:.0%})")
        elif overlap >= 0.4:
            return (0.5, f"Partial overlap ({overlap:.0%})")
        else:
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
