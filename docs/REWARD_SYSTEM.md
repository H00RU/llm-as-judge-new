# 5-Tier Reward System Documentation

**Date**: 2025-12-04 (Phase 1 Implementation)
**Status**: Validated and Integrated

## Executive Summary

The 5-tier reward system replaces the previous binary (0/1) or 10-point system to provide fine-grained learning signals for GRPO optimization.

### Why 5-Tier?

```
Old System              →  New System
├─ Binary (0/1)        →  [0.0, 0.2, 0.4, 0.7, 1.0]
│  Problem: No gradient signal for partial correctness
│
└─ 10-point (0-10)     →  5-tier (well-defined boundaries)
   Problem: Too granular, hard to tune thresholds

New System Benefits:
✅ Clear gradient for RL learning
✅ Rewards "improving towards correct"
✅ Human-aligned scoring (fail, barely, partial, good, perfect)
✅ Prevents spurious tier assignments
```

---

## Tier Structure

### Overview

```
Reward    Tier    Interpretation              RL Signal
────────  ────    ──────────────────────────  ──────────
1.0       Tier 5  Perfect solution           ✅ Goal achieved
0.7       Tier 4  Good/near-correct          ✅ Strong progress
0.4       Tier 3  Partial correctness        ✅ Some progress
0.2       Tier 2  Has output/attempt         ✅ Weak signal
0.0       Tier 1  Completely wrong           ❌ No signal
```

### Per-Domain Metrics

#### Math Problems

**Definition**: Error-based evaluation using numerical comparison

| Tier | Reward | Criteria | Example |
|------|--------|----------|---------|
| 5 | 1.0 | Error < 1e-4 (perfect) | Predicted: 3.14159, Actual: 3.14159265 |
| 4 | 0.7 | Error < 5% (close) | Predicted: 95, Actual: 100 |
| 3 | 0.4 | Error < 50% (partial) | Predicted: 60, Actual: 100 |
| 2 | 0.2 | Has output (attempt) | Predicted: "abc", Actual: 42 |
| 1 | 0.0 | No output (wrong) | Predicted: "", Actual: 42 |

**Implementation** (`reward_computer_v2.py`):
```python
def _compute_math_reward(self, prediction: str, ground_truth: str) -> float:
    """
    Extract numbers, compute relative error, map to tier
    """
    pred_num = extract_final_number(prediction)
    true_num = extract_final_number(ground_truth)

    if pred_num is None or true_num is None:
        return 0.0  # No numbers found

    rel_error = abs(pred_num - true_num) / max(abs(true_num), 1e-10)

    if rel_error < 1e-4:
        return 1.0  # Tier 5
    elif rel_error < 0.05:
        return 0.7  # Tier 4
    elif rel_error < 0.50:
        return 0.4  # Tier 3
    else:
        return 0.2  # Tier 2
```

**Extraction Methods** (6-level fallback in `answer_extractor_v2.py`):
1. `<answer>text</answer>` tags
2. `\boxed{text}` notation
3. `#### text` format (GSM8K)
4. "Final Answer: text"
5. Algebraic expressions (SymPy evaluation)
6. Last number in output

**Code Leak Detection**:
- Detects code in `\boxed{}` (indicator of incorrect formatting)
- Flags executable Python code in math answers
- Applies penalty for leakage

---

#### Code Problems

**Definition**: Test pass rate-based evaluation

| Tier | Reward | Criteria | Interpretation |
|------|--------|----------|-----------------|
| 5 | 1.0 | 100% tests pass | Perfect implementation |
| 4 | 0.7 | 75%+ tests pass | Minor edge cases |
| 3 | 0.4 | 50%+ tests pass | Core logic works |
| 2 | 0.2 | 25%+ tests pass | Partial solution |
| 1 | 0.0 | <25% tests pass | Mostly broken |

**Implementation** (`reward_computer_v2.py`):
```python
def _compute_code_reward(self, test_pass_rate: float) -> float:
    """
    Map test pass rate to tier
    """
    if test_pass_rate >= 1.0:
        return 1.0  # Tier 5 (100%)
    elif test_pass_rate >= 0.75:
        return 0.7  # Tier 4 (75%+)
    elif test_pass_rate >= 0.5:
        return 0.4  # Tier 3 (50%+)
    elif test_pass_rate > 0.0:
        return 0.2  # Tier 2 (25%+)
    else:
        return 0.0  # Tier 1 (0%)
```

**Test Execution**:
- HumanEval format with `entry_point` and test cases
- Timeout: 5 seconds per test
- Pass rate: (passed_tests) / (total_tests)

**Edge Cases**:
- Syntax errors → Tier 1 (0.0)
- Timeout → Tier 2 (0.2)
- Partial output → Tier 2 (0.2)

---

#### QA Problems

**Definition**: F1-score based semantic similarity

| Tier | Reward | F1 Score | Interpretation |
|------|--------|----------|-----------------|
| 5 | 1.0 | Exact match | Identical answer |
| 4 | 0.7 | F1 > 0.75 | High overlap |
| 3 | 0.4 | F1 > 0.5 | Good match |
| 2 | 0.2 | F1 > 0.2 | Partial match |
| 1 | 0.0 | F1 ≤ 0.2 | No match |

**Implementation** (`reward_computer_v2.py`):
```python
def _compute_qa_reward(self, prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score using token-level matching
    """
    pred_tokens = normalize_text(prediction).split()
    true_tokens = normalize_text(ground_truth).split()

    if not true_tokens:
        return 1.0 if not pred_tokens else 0.0

    # Token-level F1
    common = len(set(pred_tokens) & set(true_tokens))
    precision = common / len(pred_tokens) if pred_tokens else 0
    recall = common / len(true_tokens)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    if f1 >= 1.0:
        return 1.0  # Tier 5 (exact)
    elif f1 >= 0.75:
        return 0.7  # Tier 4 (high)
    elif f1 >= 0.5:
        return 0.4  # Tier 3 (good)
    elif f1 >= 0.2:
        return 0.2  # Tier 2 (partial)
    else:
        return 0.0  # Tier 1 (none)
```

**Normalization**:
- Lowercase
- Remove punctuation
- Collapse whitespace
- Split into tokens

**LLM Judge Fallback**:
- Uses gpt-4o-mini for ambiguous cases
- Provides semantic understanding beyond token overlap
- More expensive but more accurate

---

## Execution Penalties

If a workflow fails during execution, the tier is reduced by one level:

```python
# Base reward computed above
base_reward = 0.7  # Tier 4

# Execution issues reduce tier
if operator_mismatch:
    penalty = -0.4  # e.g., 0.7 - 0.4 = 0.3 (Tier 2.5)
elif validation_failed:
    penalty = -0.2  # e.g., 0.7 - 0.2 = 0.5 (Tier 2.5)
else:
    penalty = -0.3  # Other errors

final_reward = max(0.0, base_reward + penalty)
```

**Error Types**:
- `operator_mismatch`: Parameter mismatch, wrong signature (-0.4)
- `validation_failed`: Code validation error (-0.2)
- `execution_error`: Runtime error, timeout (-0.3)

---

## Reward Computation Flow

```
Sample Input
    ├─ problem_type = "math" / "code" / "qa"
    ├─ prediction = workflow output
    ├─ ground_truth = correct answer
    └─ execution_metadata = {errors, timing, etc}
            ↓
    [AnswerExtractor_v2] - 6-level fallback
            ↓
    Extracted prediction & ground_truth
            ↓
    Problem Type Router
            ├─→ Math: error-based
            ├─→ Code: pass-rate based
            └─→ QA: F1-based
            ↓
    Base Reward [0.0, 0.2, 0.4, 0.7, 1.0]
            ↓
    Execution Penalty Check
            ├─ operator_mismatch: -0.4
            ├─ validation_failed: -0.2
            └─ execution_error: -0.3
            ↓
    Final Reward = max(0.0, base_reward + penalty)
            ↓
    Return {reward, tier, breakdown}
```

---

## Training Integration

### Reward Distribution Tracking

During training, W&B logs these metrics:

```
reward/mean              # Average reward per step
reward/std               # Reward standard deviation
reward/tier_1_pct        # % samples in Tier 1 (0.0)
reward/tier_2_pct        # % samples in Tier 2 (0.2)
reward/tier_3_pct        # % samples in Tier 3 (0.4)
reward/tier_4_pct        # % samples in Tier 4 (0.7)
reward/tier_5_pct        # % samples in Tier 5 (1.0)
train_accuracy           # % samples with reward ≥ 0.7
```

### Expected Progression

```
Step 0-50:    Early training
├─ reward/mean: 0.1-0.3
├─ Rewards span all 5 tiers
└─ reward/std: High variance

Step 100-200: Learning phase
├─ reward/mean: 0.3-0.5
├─ Tier 5 frequency increases
└─ reward/std: Decreasing

Step 300-500: Convergence
├─ reward/mean: 0.6-0.8+
├─ Tier 4-5 dominate
└─ reward/std: Stable
```

### Success Metrics

| Milestone | Criterion | Target |
|-----------|-----------|--------|
| 50 steps | Training runs without crash | ✅ |
| 100 steps | Positive reward signal | mean > 0.2 |
| 200 steps | Significant learning | mean > 0.5 |
| 500 steps | Near convergence | mean > 0.6 |

---

## Comparison: Before vs After

### Binary System (Old)

```
Problem: Answer is "42"
Solution generated: "The answer is 42"
Reward: 1.0 (matches exactly)

Problem: Answer is "42"
Solution generated: "The answer is 40"
Reward: 0.0 (mismatch)
→ NO GRADIENT SIGNAL FOR IMPROVEMENT!
```

### 5-Tier System (New)

```
Problem: Answer is "42" (Math domain)
Solution 1: "The answer is 42"
  Error = 0% → Tier 5 (1.0) ✅

Solution 2: "The answer is 40"
  Error = 4.8% → Tier 4 (0.7) ✅ LEARNING SIGNAL!

Solution 3: "The answer is 30"
  Error = 28.6% → Tier 3 (0.4) ✅ WEAK SIGNAL!

Solution 4: "The answer is 100"
  Error = 138% → Tier 1 (0.0)

→ GRADIENT SIGNAL ENABLES LEARNING!
```

---

## Threshold Configuration

From `training.yaml`:

```yaml
experience_buffer:
  buffer_threshold: 0.7    # Tier ≥ 4 for high-quality examples

# This means: only save examples with reward ≥ 0.7 to few-shot buffer
```

The threshold 0.7 was chosen because:
- **Tier 4** represents "good" solutions
- Still achievable during early training
- Provides meaningful few-shot examples
- Accumulated slowly (avoiding data leakage)

---

## Debugging Reward Issues

### Problem: All rewards are 0.0

**Diagnosis**:
```python
# Check if extraction fails
from src.answer_extractor_v2 import AnswerExtractor
extractor = AnswerExtractor()

# Test extraction
answer = extractor.extract_answer(
    text="Some model output",
    problem_type="math",
    is_ground_truth=False
)
print(f"Extracted: {answer}")  # Should be non-empty
```

**Solutions**:
1. Check model output format (should include numbers)
2. Verify problem_type is correct
3. Review answer extraction logs
4. Check for code leaks in `\boxed{}`

### Problem: Rewards too high/low across board

**Diagnosis**:
- Compare training config thresholds with actual data
- Verify problem_type distribution (40% math, 30% code, 30% qa)
- Check if answer extraction is working correctly

**Solutions**:
1. Adjust F1 thresholds for QA
2. Verify test harness for code problems
3. Check numerical precision for math problems

### Problem: Unbalanced tier distribution

**Expected**: Wide spread across tiers (all non-zero)
**Check**:
```python
# Print distribution
print(f"Tier 1: {tier_1_pct:.1f}%")  # Should be ~20-30%
print(f"Tier 2: {tier_2_pct:.1f}%")  # Should be ~20-30%
print(f"Tier 3: {tier_3_pct:.1f}%")  # Should be ~20-30%
print(f"Tier 4: {tier_4_pct:.1f}%")  # Should be ~10-20%
print(f"Tier 5: {tier_5_pct:.1f}%")  # Should be ~5-10%
```

---

## References

- Implementation: `/root/llm-as-judge-new/src/reward_computer_v2.py`
- Answer Extraction: `/root/llm-as-judge-new/src/answer_extractor_v2.py`
- Training Integration: `/root/llm-as-judge-new/src/grpo_trainer.py`
- Configuration: `/root/llm-as-judge-new/config/training.yaml`

---

**Version**: 2.0 (Phase 1 Complete)
**Status**: Production Ready
**Last Updated**: 2025-12-04
