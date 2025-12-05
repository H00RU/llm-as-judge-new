# System Architecture: LLM-as-Judge Training Pipeline

**Date**: 2025-12-04
**Status**: Phase 6 - Complete Integration & Documentation

## Overview

This document describes the complete architecture of the LLM-as-Judge training system after comprehensive Phase 1-6 fixes.

## System Components

### 1. Data Layer (`data/mixed/`)

**Input**: Raw datasets (GSM8K, HumanEval, QA)

**Processing**:
- `scripts/process_datasets.py`: Standardizes all datasets to unified schema
- Output: `data/mixed/{train,test}_mixed.jsonl`

**Schema** (mandatory fields):
```json
{
  "id": "unique_identifier",
  "dataset": "source_dataset_name",
  "domain": "math|code|qa",
  "question": "problem_description",
  "reference_answer": "ground_truth",
  "answer_type": "numeric|code|text",
  "entry_point": "function_name (code only)",
  "test": "test_code (code only)",
  "metadata": {}
}
```

**Validation**: `scripts/validate_data_schema.py`
- Checks all 2,071 training + 420 test samples
- Verifies domain distribution (40% math, 30% code, 30% qa)
- All samples must pass validation before training

### 2. Reward System (`src/reward_computer_v2.py`)

**Purpose**: Provide fine-grained gradient signals to GRPO optimizer

**Architecture**: 5-Tier System
```
Tier 5 (1.0)  ← Perfect solution
Tier 4 (0.7)  ← Good, near-correct
Tier 3 (0.4)  ← Partial correctness
Tier 2 (0.2)  ← Has output/attempt
Tier 1 (0.0)  ← Completely wrong
```

**Problem-Type Metrics**:

| Type | Tier 5 | Tier 4 | Tier 3 | Tier 2 | Tier 1 |
|------|--------|--------|--------|--------|--------|
| **Math** | Error < 1e-4 | Error < 5% | Error < 50% | Has output | Wrong |
| **Code** | 100% tests pass | 75%+ pass | 50%+ pass | 25%+ pass | <25% pass |
| **QA** | Exact match | F1 > 0.75 | F1 > 0.5 | F1 > 0.2 | No match |

**Execution Penalties**:
- Operator mismatch: -0.4 reward
- Validation failed: -0.2 reward
- Other errors: -0.3 reward

**Dependencies**:
- `answer_extractor_v2.py`: Enhanced answer extraction with 6-level fallback
- `reward_computer_v2.py`: 5-tier reward computation

### 3. Workflow Generation (`src/prompt_optimizer.py`, `src/workflow_code_builder.py`)

**Generator**: Qwen2.5-7B-Instruct (trained with GRPO)

**Prompt Engineering**:
- Complete operator specifications
- Problem-type-specific guidance
- Few-shot examples from `ExperienceBuffer`
- Variable initialization rules
- Parameter requirement emphasis

**Code Quality Assurance**:

1. **Generation** → PromptOptimizer produces draft code
2. **Building & Fixing** → WorkflowCodeBuilder constructs and auto-fixes code
3. **Validation** → WorkflowValidator checks:
   - Syntax correctness
   - Class/method presence
   - Operator names (PascalCase)
   - Return statements

**Validation Flow**:
```
Generated Code
    ↓
AST Parse Check (syntax_valid)
    ↓
Workflow Class Check (has_workflow_class)
    ↓
__call__ Method Check (has_call_method)
    ↓
Return Statement Check (has_return)
    ↓
Operator Validation (operator names, parameters)
    ↓
✅ Valid / ⚠️ Warning / ❌ Invalid
```

### 4. AFlow Executor (`src/aflow_executor.py`)

**Purpose**: Execute generated workflows with standardized interface

**Key Operators**:

| Operator | Input | Output | Use Case |
|----------|-------|--------|----------|
| **Custom** | input, instruction | {"response": str} | Flexible task handling |
| **AnswerGenerate** | problem | {"answer": str, "thought": str} | Step-by-step reasoning |
| **Programmer** | problem, analysis | {"code": str, "output": str} | Code generation & execution |
| **Test** | problem, solution, entry_point | {"result": bool, "solution": str} | HumanEval testing |
| **Review** | problem, solution | {"feedback": str} | Solution evaluation |
| **Revise** | problem, solution, feedback | {"solution": str} | Iterative improvement |
| **ScEnsemble** | solutions, problem | {"response": str} | Self-consistency voting |

**Response Standardization**:
```python
{
    "success": bool,           # New: execution success flag
    "code": str,               # For Programmer
    "output": str,             # For Programmer (execution result)
    "answer": str,             # For AnswerGenerate
    "solution": str,           # For Test/Revise
    "feedback": str,           # For Review
    "result": bool,            # For Test
    "test_passed": bool,       # For Test
    "error": Optional[str]     # Error message if failed
}
```

**Timeout Configuration** (NEW - Phase 5):
- Programmer operator: configurable timeout (default 60s, was hardcoded 30s)
- Usage: `programmer = Programmer(llm, timeout=180)`

### 5. Training Loop (`src/grpo_trainer.py`)

**GRPO Framework**:
- Online learning: sample → execute → reward → optimize
- Gradient-based policy optimization
- Advantage-based value weighting

**Key Metrics Tracked**:
- `reward/mean`: Average reward per step
- `reward/std`: Reward distribution variance
- `reward/tier_{1-5}_pct`: Distribution across tiers
- `train_accuracy`: % samples with reward ≥ 0.7
- `problem_type_accuracy/{math,code,qa}`: Per-domain accuracy

**Phase 3 Optimizations**:

| Config | Old | New | Reason |
|--------|-----|-----|--------|
| K (sequences per sample) | 6 | 4 | 更多workflows提高质量 |
| B (batch size) | 5 | 4 | 平衡样本多样性 |
| KL coefficient | 0.1 | 0.02 | 稳定策略更新 |
| Gradient accumulation | 1 | 4 | ↓ OOM, ↓ variance |
| Temperature schedule | Disabled | Enabled (0.5→0.15) | Better exploration-exploitation |

**Training Flow**:
```
Step Loop (max_steps=500)
  ├─ Sample batch from mixed dataset
  ├─ Generate K workflows per sample
  ├─ Execute workflows with AFlow
  ├─ Compute 5-tier rewards
  ├─ Compute GRPO advantages
  ├─ Update LoRA weights
  └─ Log metrics to W&B
```

### 6. Experience Buffer (`src/experience_buffer.py`)

**Purpose**: Collect high-quality examples for few-shot learning

**Configuration** (from training.yaml):
```yaml
experience_buffer:
  buffer_size: 100                    # Max samples per problem type
  reward_threshold: 0.7               # Only samples with tier ≥ 4
  persistence_dir: "data/experience_buffer"
```

**Integration with PromptOptimizer**:
- Retrieves top-K similar examples by cosine similarity
- Embeds into few-shot prompt section
- Helps model learn from successful patterns

**Metrics**:
- Accumulation rate: samples/step
- Per-domain distribution
- Reward distribution within buffer

## Data Flow Diagrams

### Training Loop
```
Input Sample (problem, type, ground_truth)
            ↓
[PromptOptimizer] (with few-shot from ExperienceBuffer)
            ↓
Generated Workflow Code
            ↓
[WorkflowValidator] → Syntax/structure check
            ↓
Generated Code ✅
            ↓
[WorkflowCodeBuilder] (build & auto-fix)
            ↓
[AFlow Executor] → Execute with operators
            ↓
Execution Result (code output, operator responses)
            ↓
[AnswerExtractor_v2] (6-level fallback)
            ↓
Extracted Answer
            ↓
[RewardComputer_v2] (5-tier evaluation)
            ↓
Reward [0.0, 0.2, 0.4, 0.7, 1.0]
            ↓
[GRPO Optimizer] (advantage-based update)
            ↓
Updated LoRA Weights
            ↓
If reward ≥ 0.7:
  → Add to ExperienceBuffer
```

### Evaluation Path (offline)
```
Test Sample
        ↓
Trained Workflow Generator
        ↓
Generated Code
        ↓
AFlow Execution
        ↓
Answer Extraction
        ↓
5-Tier Reward Computation
        ↓
Metrics Report
```

## Key Improvements from Comprehensive Fix

### Phase 1: 5-Tier Reward System
- **Before**: Binary (0/1) or 10-point rewards → no gradient signal
- **After**: [0.0, 0.2, 0.4, 0.7, 1.0] → fine-grained learning

### Phase 2: Data Standardization
- **Before**: Inconsistent field names (question vs problem, etc.)
- **After**: Unified schema with validation (100% pass rate)

### Phase 3: Training Configuration
- **Before**: K=6, B=4, KL=0.1 → 24 concurrent, high timeout, slow learning
- **Current**: K=4, B=4, KL=0.02 → 16 concurrent, balanced quality and efficiency

### Phase 4: Code Generation Enhancement
- **Before**: Poor quality generation, many failed workflows
- **After**: Better prompts, fixer patterns, semantic validation

### Phase 5: AFlow Compatibility
- **Before**: Inconsistent operator returns, hardcoded timeouts
- **After**: Standardized responses, configurable timeouts, unified error handling

### Phase 6: Experience Buffer & Documentation
- **Before**: No few-shot learning, minimal documentation
- **After**: Few-shot integration, comprehensive docs, reproducible system

## Training Expected Metrics

### First 50 steps (before learning)
- `reward/mean`: 0.1-0.3
- `reward/tier_distribution`: Spans all 5 tiers
- `reward/std`: High variance

### After 100-200 steps
- `reward/mean`: 0.4-0.5+
- `reward/tier_5_pct`: Gradually increases
- `train_accuracy`: ~20% → 40%+

### After 300-500 steps (convergence)
- `reward/mean`: 0.6-0.8+
- Best models consistently in tier 4-5
- Experience buffer contains 50+ high-quality examples

## Performance Metrics

**Model Size**: Qwen2.5-7B-Instruct
**LoRA Config**: Rank=64, Alpha=64, Dropout=0.05
**Training Hardware**: 1x GPU (A100/V100 equivalent)
**Training Time**: ~2 hours for 500 steps

**Success Criteria**:
- ✅ Minimum: Training runs 50 steps without crash
- ✅ Target: Avg reward > 0.5 after 200 steps
- ✅ Excellent: Generated workflows > 80% valid syntax
- ✅ Success: Qwen outputs beat gpt-4o-mini baseline

## Documentation References

- `OPERATOR_SPECIFICATIONS.md`: Detailed operator API
- `REWARD_SYSTEM.md`: 5-tier system explanation
- `DATA_FORMAT.md`: Schema definition and validation
- `CONFIGURATION.md`: Training hyperparameter tuning
- `TROUBLESHOOTING.md`: Common issues and fixes

---

**Last Updated**: 2025-12-04
**Maintained By**: LLM-as-Judge Team
**Status**: Production Ready
