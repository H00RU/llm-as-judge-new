# Project Diagnosis and Comprehensive Fix Plan

**Date**: 2025-12-04
**Status**: CRITICAL - Multiple issues causing training accuracy to remain at 0
**Goal**: Achieve Qwen model learning to generate workflows that outperform pure GPT-4o-mini baseline

---

## Executive Summary

### Current Situation
- **Training accuracy stuck at 0%** across multiple training runs
- **Multiple design approaches coexisting** (soft learning, meta marking, hard constraints)
- **Potential code generation errors** from Qwen model
- **Possible operator call conflicts** with /root/AFlow
- **Reward mechanism issues** preventing learning signal
- **Incomplete modifications** from previous attempts due to disconnections
- **Prompt incompatibilities** after recent changes

### Root Causes Identified

1. **Generated Workflow Code Errors**
   - Qwen may be generating syntactically valid but semantically incorrect workflows
   - Operator parameter mismatches
   - Async/await issues
   - Variable initialization problems

2. **Operator Integration Conflicts**
   - AFlow operator interface expectations vs. generated code
   - Import path conflicts between /root/AFlow and project
   - Config file inconsistencies

3. **Reward System Problems**
   - Reward computation may not be capturing correctness properly
   - Answer extraction failures
   - Evaluation metric misalignment with training objective

4. **Training Configuration Issues**
   - Learning rate may be too low/high
   - Temperature scheduling conflicts
   - GRPO group size mismatches
   - Batch size causing memory/timeout issues

5. **Data Processing Misalignment**
   - Dataset format inconsistencies
   - Mixed dataset processing errors
   - Ground truth extraction failures

---

## Detailed Analysis by Component

### 1. WORKFLOW GENERATION (Qwen Code Quality)

#### Current State
- Qwen2.5-7B generates Python workflow code with operator chaining
- Uses prompts from `src/prompt_optimizer.py` and `src/operator_prompt_enhancer.py`
- Validation via `src/workflow_validator.py`
- Auto-fixing via `src/workflow_code_fixer.py`

#### Identified Issues

**Issue 1.1: Operator Interface Mismatches**
- **Symptom**: Generated code calls operators with wrong parameter names
- **Root Cause**: Prompts don't have exact operator signatures from AFlow
- **Evidence**: Need to compare operator descriptions in prompts vs. actual AFlow operator signatures

**Issue 1.2: Return Value Handling**
- **Symptom**: Generated code expects wrong dict keys from operator responses
- **Root Cause**: Operator response format not standardized/documented in prompts
- **Evidence**: Check if `response_standardizer.py` is actually being used

**Issue 1.3: Async/Await Correctness**
- **Symptom**: Workflows may have improper async handling
- **Root Cause**: Generated code missing awaits or using wrong async patterns
- **Evidence**: Check validation logs for async errors

**Issue 1.4: Variable Initialization**
- **Symptom**: UnboundLocalError during execution
- **Root Cause**: Variables used in conditionals not initialized
- **Evidence**: Common in code fixer logs

#### Required Fixes

1. **Update prompts with exact operator signatures** (High Priority)
   - Extract ALL operator signatures from `/root/AFlow/scripts/operators.py`
   - Create comprehensive operator specification document
   - Update `prompt_optimizer.py` to include exact signatures
   - Add parameter type information
   - Include return value structure for each operator

2. **Strengthen validation pipeline** (High Priority)
   - Add semantic validation (not just syntax)
   - Validate operator parameter names match exactly
   - Check all operator calls have await
   - Verify return value unpacking matches operator response format

3. **Improve code fixer** (Medium Priority)
   - Add more auto-fix patterns based on common errors
   - Log all auto-fixes for analysis
   - Create fallback safe workflow for critical failures

---

### 2. OPERATOR EXECUTION (AFlow Integration)

#### Current State
- `src/aflow_executor.py` executes generated workflows
- Imports operators from `/root/AFlow/scripts/operators.py`
- Creates LLM instance with config from `config/aflow_llm.yaml`

#### Identified Issues

**Issue 2.1: Import Path Conflicts**
- **Symptom**: ImportError or wrong module loaded
- **Root Cause**: Both projects have `scripts/` directories
- **Evidence**: PYTHONPATH resolution issues

**Issue 2.2: LLM Config Inconsistencies**
- **Symptom**: Operators fail to initialize or make API calls
- **Root Cause**: Config format mismatch between AFlow expectations and provided config
- **Evidence**: Check if `config/aflow_llm.yaml` matches AFlow's `config/config2.yaml` format

**Issue 2.3: Response Standardization Failures**
- **Symptom**: Workflows can't extract answers from operator responses
- **Root Cause**: `response_standardizer.py` not handling all operator response formats
- **Evidence**: Check if standardizer is actually called in executor

**Issue 2.4: Execution Timeouts**
- **Symptom**: Workflows timeout, causing zero rewards
- **Root Cause**: Timeout settings too aggressive or operator calls hanging
- **Evidence**: Check execution logs for timeout errors

#### Required Fixes

1. **Resolve import conflicts** (Critical Priority)
   - Set explicit PYTHONPATH in training script
   - Use absolute imports for AFlow components
   - Document required PYTHONPATH setup
   - Test import resolution in isolated environment

2. **Standardize LLM configuration** (Critical Priority)
   - Validate `config/aflow_llm.yaml` format
   - Ensure API keys and endpoints are correct
   - Add config validation at startup
   - Create unified config loader

3. **Enforce response standardization** (High Priority)
   - Ensure ALL operator calls go through standardizer
   - Add logging for pre/post standardization
   - Handle edge cases (empty responses, errors)

4. **Optimize timeout settings** (Medium Priority)
   - Analyze actual operator execution times
   - Set per-operator timeouts
   - Add graceful timeout handling
   - Log timeout occurrences

---

### 3. REWARD COMPUTATION (Learning Signal Quality)

#### Current State
- `src/reward_computer.py` computes rewards from execution results
- `src/answer_extractor.py` extracts final answers
- `src/unified_evaluator.py` does domain-specific evaluation
- Reward scale: [0, 1] normalized from 10-point correctness score

#### Identified Issues

**Issue 3.1: Answer Extraction Failures**
- **Symptom**: Cannot extract answer from workflow output
- **Root Cause**: Extraction patterns don't match actual output format
- **Evidence**: Check if answer_extractor logs show frequent failures

**Issue 3.2: Evaluation Metric Misalignment**
- **Symptom**: Workflows that seem correct get low rewards
- **Root Cause**: Evaluation is too strict or wrong metric
- **Evidence**: Compare reference /content/llm-as-judge 5-tier reward system

**Issue 3.3: Constraint Penalty Over-Application**
- **Symptom**: All workflows get massive penalties
- **Root Cause**: Constraint violation detection too aggressive
- **Evidence**: Check metadata for constraint_violation flags

**Issue 3.4: Binary Reward Problem**
- **Symptom**: Rewards are only 0 or 1, no gradation
- **Root Cause**: No partial credit for near-correct answers
- **Evidence**: Current system may not implement 5-tier rewards

#### Required Fixes

1. **Implement 5-tier reward system** (Critical Priority)
   - Study /content/llm-as-judge reward computation
   - Implement fine-grained reward levels:
     - 1.0: Perfect correctness
     - 0.7: Close/partial correctness
     - 0.4: Right direction, wrong answer
     - 0.2: Valid output, incorrect
     - 0.0: Execution failure
   - Use domain-specific metrics:
     - Math: Numeric tolerance, relative error
     - Code: Test pass rate (not just all-or-nothing)
     - QA: F1 score, not just exact match

2. **Enhance answer extraction** (High Priority)
   - Add multi-stage extraction pipeline (see /content/llm-as-judge)
   - Support multiple answer formats
   - Add LLM-based fallback extraction
   - Log extraction attempts and results

3. **Calibrate constraint penalties** (Medium Priority)
   - Review all constraint violations
   - Adjust penalty magnitudes (currently -0.6 to -5.0)
   - Separate hard violations (immediate fail) from soft (penalty)
   - Track penalty distribution to ensure not over-penalizing

4. **Add LLM Judge integration** (Medium Priority)
   - Use LLM to compare semantic equivalence
   - Especially important for QA tasks
   - Fallback when rule-based comparison uncertain

---

### 4. TRAINING CONFIGURATION

#### Current State (from `config/training.yaml`)
```yaml
max_steps: 500
rollout_batch_size: 4
num_return_sequences_in_group: 6
learning_rate: 2.0e-5
temperature: 0.2 (fixed)
use_lora: true
lora_rank: 64
```

#### Identified Issues

**Issue 4.1: Learning Rate Too High/Low**
- **Symptom**: No learning progress
- **Root Cause**: Learning rate not tuned for signal strength
- **Evidence**: Compare to /content/llm-as-judge (uses 2e-5)

**Issue 4.2: GRPO Group Size Mismatch**
- **Symptom**: GRPO advantage computation incorrect
- **Root Cause**: K=6 may be too large, causing variance issues
- **Evidence**: /content/llm-as-judge uses K=2

**Issue 4.3: Temperature Scheduling Disabled**
- **Symptom**: No exploration-exploitation balance
- **Root Cause**: Fixed temperature may be too low for early training
- **Evidence**: /content/llm-as-judge uses dynamic 0.5→0.15

**Issue 4.4: Batch Size Issues**
- **Symptom**: Training instability or OOM errors
- **Root Cause**: Batch size 4 × K=6 = 24 concurrent workflows may overwhelm system
- **Evidence**: Check for timeout errors in logs

#### Required Fixes

1. **Optimize GRPO configuration** (Critical Priority)
   - Reduce K from 6 to 2 (match /content/llm-as-judge)
   - Adjust batch size if needed (start with 4, may increase to 5-8)
   - This reduces concurrent load: 4×2=8 workflows vs. current 4×6=24

2. **Enable temperature scheduling** (High Priority)
   ```yaml
   temperature_schedule:
     enabled: true
     initial: 0.5
     final: 0.15
     warmup_steps: 100
   ```

3. **Validate learning rate** (High Priority)
   - Current 2e-5 matches reference, keep it
   - Add learning rate warmup (100 steps)
   - Consider cosine decay after warmup

4. **Add gradient clipping** (Medium Priority)
   - Prevent gradient explosions
   - Typical value: 1.0 or 0.5

---

### 5. DATA PROCESSING

#### Current State
- Mixed dataset: `data/mixed/train_mixed.jsonl`, `data/mixed/test_mixed.jsonl`
- Domain ratios: math 40%, qa 30%, code 30%
- Processing script: `scripts/process_datasets.py`

#### Identified Issues

**Issue 5.1: Dataset Format Inconsistencies**
- **Symptom**: Data loading errors or missing fields
- **Root Cause**: Different source datasets have different field names
- **Evidence**: Check data_manager logs for KeyError

**Issue 5.2: Ground Truth Extraction**
- **Symptom**: Can't compare predictions to ground truth
- **Root Cause**: Ground truth format varies by dataset (GSM8K uses ####, MATH uses \boxed, etc.)
- **Evidence**: Check if answer_extractor handles all formats

**Issue 5.3: Mixed Dataset Quality**
- **Symptom**: Model sees inconsistent data quality
- **Root Cause**: Some datasets may be malformed or not properly processed
- **Evidence**: Manually inspect train_mixed.jsonl

**Issue 5.4: Train/Test Leakage**
- **Symptom**: Overfitting or unrealistic performance
- **Root Cause**: eval_every > 0 may cause test set exposure
- **Evidence**: Current config has eval_every: 0 (good!)

#### Required Fixes

1. **Validate all datasets** (Critical Priority)
   - Run comprehensive data validation script
   - Check for:
     - Missing required fields
     - Malformed JSON
     - Empty or null values
     - Inconsistent formats
   - Fix or remove bad samples

2. **Standardize dataset schema** (High Priority)
   - Create unified schema:
     ```json
     {
       "id": "unique_id",
       "dataset": "gsm8k|humaneval|hotpotqa|...",
       "domain": "math|code|qa",
       "problem": "problem text",
       "reference_answer": "ground truth",
       "answer_type": "numeric|code|text",
       "entry_point": "function_name (code only)",
       "test": "test cases (code only)",
       "metadata": {}
     }
     ```
   - Update process_datasets.py to enforce schema
   - Re-process all datasets

3. **Enhance ground truth extraction** (High Priority)
   - Support all formats:
     - GSM8K: `#### <number>`
     - MATH: `\boxed{answer}`
     - HumanEval: code in ground_truth field
     - HotpotQA: direct answer string
   - Add extraction tests for each format

4. **Verify data split integrity** (Medium Priority)
   - Ensure no train/test overlap
   - Check domain balance in both splits
   - Verify sufficient samples per domain

---

### 6. CONFLICTING DESIGN APPROACHES

#### Current State
Three approaches coexist:
1. **Hard Constraints**: Immediate rejection of invalid workflows
2. **Soft Learning**: Penalty-based RL learning
3. **Meta Marking**: Metadata tracking without enforcement

#### Identified Issues

**Issue 6.1: Inconsistent Constraint Handling**
- **Symptom**: Same violation handled differently in different code paths
- **Root Cause**: Multiple constraint checking locations without coordination
- **Evidence**: workflow_validator.py, reward_computer.py, aflow_executor.py all check constraints

**Issue 6.2: Redundant Code**
- **Symptom**: Duplicate logic for constraint checking
- **Root Cause**: Evolution of design without cleanup
- **Evidence**: Multiple files implementing similar validation

**Issue 6.3: Unclear Design Intent**
- **Symptom**: Hard to understand what should happen for violations
- **Root Cause**: Documentation doesn't specify which approach is active
- **Evidence**: Config doesn't clearly enable/disable approaches

#### Required Fixes

1. **Choose unified approach** (Critical Priority)
   - **Recommended**: Soft Learning + Meta Marking
   - Rationale:
     - Hard constraints prevent RL from learning
     - Soft penalties provide learning signal
     - Meta marking enables analysis
   - Remove pure hard constraint enforcement (except syntax errors)

2. **Centralize constraint handling** (High Priority)
   - Create single `constraint_checker.py` module
   - Define all constraints with:
     - Name
     - Check function
     - Penalty value
     - Metadata key
   - Call from single location in reward computation

3. **Document active design** (High Priority)
   - Update all docs to reflect soft learning approach
   - Remove mentions of pure hard constraints
   - Clarify meta marking is for analysis only
   - Add config flags to enable/disable constraint types

4. **Clean up redundant code** (Medium Priority)
   - Remove duplicate validation logic
   - Keep validation pipeline for syntax/safety only
   - Move semantic checks to reward computation

---

### 7. AFLOW PROJECT MODIFICATIONS

#### Required Changes in /root/AFlow

**Change 7.1: Response Format Standardization**
- **File**: `/root/AFlow/scripts/operators.py`
- **Issue**: Operators return inconsistent dict formats
- **Fix**: Ensure all operators return standardized format:
  ```python
  return {"response": main_output, "metadata": {...}}
  ```
- **Note**: May need to modify base Operator class

**Change 7.2: Error Handling**
- **File**: `/root/AFlow/scripts/operators.py`
- **Issue**: Exceptions crash workflow execution
- **Fix**: Add try-catch in each operator:
  ```python
  async def __call__(self, **kwargs):
      try:
          result = await self._execute(**kwargs)
          return {"response": result}
      except Exception as e:
          return {"error": str(e), "response": ""}
  ```

**Change 7.3: Timeout Configuration**
- **File**: `/root/AFlow/scripts/operators.py` (Programmer operator)
- **Issue**: Fixed 30s timeout may be too short
- **Fix**: Make timeout configurable via operator init:
  ```python
  def __init__(self, llm, timeout=60):
      self.timeout = timeout
  ```

**Change 7.4: Import Path Robustness**
- **File**: `/root/AFlow/scripts/async_llm.py`
- **Issue**: Relative imports assume AFlow is in PYTHONPATH
- **Fix**: Add path setup at module level:
  ```python
  import sys
  from pathlib import Path
  aflow_root = Path(__file__).parent.parent
  if str(aflow_root) not in sys.path:
      sys.path.insert(0, str(aflow_root))
  ```

---

## Comprehensive Fix Plan

### Phase 1: Critical Path Fixes (Week 1)

**Goal**: Get training to show non-zero accuracy

#### Day 1: Data Validation & Operator Specification
- [ ] Task 1.1: Validate all datasets in data/mixed/
  - Run data validation script
  - Fix or remove corrupted samples
  - Verify schema consistency
  - Document any anomalies

- [ ] Task 1.2: Extract exact operator signatures from AFlow
  - Parse `/root/AFlow/scripts/operators.py`
  - Document each operator's:
    - Full signature with parameter types
    - Return value structure
    - Dataset compatibility
    - Example usage
  - Save to `docs/OPERATOR_SPECIFICATIONS.md`

- [ ] Task 1.3: Update prompts with exact signatures
  - Modify `src/prompt_optimizer.py`
  - Include exact operator APIs in system prompt
  - Add parameter type hints
  - Add return value format specifications

#### Day 2: Reward System Overhaul
- [ ] Task 2.1: Implement 5-tier reward system
  - Study /content/llm-as-judge reward logic
  - Rewrite `src/reward_computer.py` to support:
    - Fine-grained correctness levels (0.0, 0.2, 0.4, 0.7, 1.0)
    - Domain-specific metrics (numeric tolerance, F1 score, test pass rate)
    - Constraint penalties as separate component
  - Add comprehensive logging

- [ ] Task 2.2: Enhance answer extraction
  - Add multi-stage extraction pipeline
  - Support all answer formats (LaTeX, GSM8K, code, text)
  - Add extraction tests
  - Log extraction success rate

- [ ] Task 2.3: Calibrate constraint penalties
  - Review all constraint types
  - Adjust penalty magnitudes (start with -0.1 to -0.3, not -5.0)
  - Create constraint configuration file
  - Add penalty justification comments

#### Day 3: Training Configuration & GRPO Tuning
- [ ] Task 3.1: Update training config
  - Set K=2 (reduce from 6)
  - Enable temperature scheduling (0.5→0.15)
  - Keep learning_rate=2e-5
  - Add gradient clipping (1.0)
  - Increase warmup_steps to 100
  - Update `config/training.yaml`

- [ ] Task 3.2: Optimize execution settings
  - Increase operator timeout to 60s
  - Add per-operator timeout config
  - Set workflow timeout to 180s
  - Add graceful timeout handling

- [ ] Task 3.3: Fix import paths
  - Set PYTHONPATH explicitly in train.py:
    ```python
    sys.path.insert(0, "/root/AFlow")
    sys.path.insert(0, "/root/llm-as-judge-new")
    ```
  - Test imports in isolation
  - Document required environment setup

#### Day 4: Validation & Code Quality
- [ ] Task 4.1: Strengthen workflow validator
  - Add semantic validation (not just syntax)
  - Check operator parameter names
  - Verify all async calls have await
  - Validate return value unpacking
  - Add detailed error messages

- [ ] Task 4.2: Enhance code fixer
  - Add common error patterns from logs
  - Log all applied fixes
  - Create fallback safe workflow template
  - Test auto-fix on known bad workflows

- [ ] Task 4.3: Centralize constraint checking
  - Create `src/constraint_checker.py`
  - Define all constraints with penalties
  - Remove duplicate validation code
  - Call from reward_computer only

#### Day 5: Integration Testing
- [ ] Task 5.1: End-to-end test
  - Run single training step
  - Verify workflow generation
  - Check operator execution
  - Validate reward computation
  - Inspect all logs

- [ ] Task 5.2: Fix any issues found
  - Document errors
  - Apply fixes
  - Retest until clean run

- [ ] Task 5.3: Small-scale training run
  - Train for 50 steps
  - Monitor:
    - Average reward trend
    - Workflow diversity
    - Execution success rate
    - Constraint violation rate
  - Adjust config based on results

#### Day 6-7: AFlow Modifications (if needed)
- [ ] Task 6.1: Standardize operator responses
  - Modify `/root/AFlow/scripts/operators.py`
  - Ensure all return {"response": value}
  - Add error handling
  - Test each operator

- [ ] Task 6.2: Make timeouts configurable
  - Add timeout parameter to operator init
  - Update llm-as-judge config to set timeouts
  - Test timeout behavior

- [ ] Task 6.3: Improve import robustness
  - Add path setup in AFlow modules
  - Test imports from llm-as-judge context
  - Document required PYTHONPATH

### Phase 2: Performance Optimization (Week 2)

#### Task 7: Experience Buffer Tuning
- [ ] Enable experience buffer with threshold=8.0
- [ ] Add few-shot examples to prompts
- [ ] Monitor buffer fill rate
- [ ] Adjust threshold if needed

#### Task 8: Prompt Engineering
- [ ] Add more diverse workflow examples
- [ ] Include failure case examples with corrections
- [ ] Add problem-type-specific guidance
- [ ] Test prompt variations

#### Task 9: Evaluation Pipeline
- [ ] Create separate evaluation script (no train/test leakage)
- [ ] Run on held-out test set
- [ ] Compare to GPT-4o-mini baseline
- [ ] Analyze failure modes

#### Task 10: Hyperparameter Search
- [ ] Try different learning rates (1e-5, 2e-5, 5e-5)
- [ ] Vary temperature schedule
- [ ] Adjust constraint penalty magnitudes
- [ ] Test different K values (2, 3, 4)

### Phase 3: Scaling & Robustness (Week 3)

#### Task 11: Full Training Run
- [ ] Train for full 500 steps
- [ ] Monitor W&B metrics continuously
- [ ] Save checkpoints every 50 steps
- [ ] Run evaluation every 100 steps (on separate eval set)

#### Task 12: Multi-Dataset Testing
- [ ] Test on each dataset individually
- [ ] Compare cross-dataset generalization
- [ ] Identify dataset-specific issues
- [ ] Adjust domain ratios if needed

#### Task 13: Production Readiness
- [ ] Add comprehensive error recovery
- [ ] Improve logging and monitoring
- [ ] Create deployment documentation
- [ ] Write user guide

---

## Success Metrics

### Phase 1 Success Criteria (Week 1)
- [ ] Training runs without crashes for 50 steps
- [ ] Average reward > 0.1 (not stuck at 0)
- [ ] Workflow execution success rate > 50%
- [ ] At least 20% of workflows get reward > 0.5
- [ ] Constraint violation rate < 30%

### Phase 2 Success Criteria (Week 2)
- [ ] Average reward reaches 0.5+
- [ ] Best workflows score 0.8+ reward
- [ ] Experience buffer has 50+ high-quality samples
- [ ] Workflow diversity (unique structures) > 10

### Phase 3 Success Criteria (Week 3)
- [ ] **PRIMARY GOAL**: Qwen-generated workflows outperform pure GPT-4o-mini on test set
  - Qwen + AFlow accuracy > GPT-4o-mini baseline accuracy
  - Even 1-2% improvement counts as success
- [ ] Stable training convergence (reward not oscillating wildly)
- [ ] Reproducible results across runs

---

## Risk Mitigation

### High-Risk Items

**Risk 1: AFlow modifications break existing functionality**
- Mitigation: Create AFlow branch, test thoroughly before merging
- Rollback plan: Keep original AFlow code backed up

**Risk 2: Data issues cause persistent training failures**
- Mitigation: Validate data before training, fix all issues upfront
- Rollback plan: Use single high-quality dataset (GSM8K) first

**Risk 3: Reward system still doesn't provide learning signal**
- Mitigation: Start with simple rule-based rewards, add LLM judge later
- Rollback plan: Use binary rewards with high tolerance (e.g., 10% error = success)

**Risk 4: GRPO configuration causes instability**
- Mitigation: Test multiple K values, use proven config from /content/llm-as-judge
- Rollback plan: Switch to simpler RL algorithm (REINFORCE)

### Backup Plans

**Backup Plan A: Simplify to Single Domain**
- If mixed training fails, focus on math only (GSM8K)
- Once working, expand to other domains

**Backup Plan B: Use Smaller Model**
- If Qwen2.5-7B too large, try Qwen2.5-1.5B
- Faster iteration, less resource intensive

**Backup Plan C: Supervised Learning First**
- Collect high-quality workflows manually
- Do supervised fine-tuning first
- Then add RL on top

---

## Documentation Updates Required

### Files to Create/Update

1. **`docs/OPERATOR_SPECIFICATIONS.md`** (NEW)
   - Complete operator API reference
   - Parameter types and return formats
   - Usage examples per operator
   - Dataset compatibility matrix

2. **`docs/REWARD_SYSTEM.md`** (NEW)
   - 5-tier reward structure
   - Domain-specific metrics
   - Constraint penalties
   - Evaluation examples

3. **`docs/TRAINING_GUIDE.md`** (UPDATE)
   - Complete training configuration explanation
   - Hyperparameter tuning guide
   - Troubleshooting common issues
   - Performance optimization tips

4. **`docs/DATA_FORMAT.md`** (UPDATE)
   - Standardized dataset schema
   - Ground truth format specifications
   - Processing pipeline documentation
   - Validation procedures

5. **`docs/ARCHITECTURE.md`** (UPDATE)
   - Remove conflicting design mentions
   - Document soft learning + meta marking approach
   - Update component diagrams
   - Clarify AFlow integration

6. **`README.md`** (UPDATE)
   - Add troubleshooting section
   - Update installation instructions (PYTHONPATH)
   - Document environment setup
   - Add quick start guide

7. **`config/training.yaml`** (UPDATE)
   - Add inline comments for all parameters
   - Document recommended values
   - Mark deprecated options

---

## Verification Checklist

### Before Starting Training
- [ ] All datasets validated and error-free
- [ ] Operator specifications documented
- [ ] Prompts updated with exact APIs
- [ ] Reward system implements 5-tier structure
- [ ] Training config matches recommended values (K=2, temp schedule enabled)
- [ ] Import paths tested and working
- [ ] AFlow modifications applied (if any)
- [ ] All documentation updated
- [ ] Backup of current code created

### During Training (Every 50 Steps)
- [ ] Check average reward trend (should increase or stabilize)
- [ ] Inspect sample generated workflows (quality improving?)
- [ ] Review execution logs (errors decreasing?)
- [ ] Monitor constraint violations (rate decreasing?)
- [ ] Check W&B dashboard (metrics reasonable?)

### After Training
- [ ] Final model evaluation on test set
- [ ] Compare to GPT-4o-mini baseline
- [ ] Analyze failure cases
- [ ] Document lessons learned
- [ ] Save best checkpoint

---

## Conclusion

This plan addresses all identified issues systematically:

1. **Workflow Generation**: Fixed through exact operator specs and better prompts
2. **Operator Integration**: Resolved via import path fixes and config standardization
3. **Reward System**: Overhauled to 5-tier fine-grained rewards
4. **Training Config**: Optimized based on proven /content/llm-as-judge setup
5. **Data Quality**: Validated and standardized
6. **Design Conflicts**: Unified to soft learning + meta marking
7. **AFlow Issues**: Documented required modifications

**Expected Outcome**: Training should show non-zero accuracy within first 50 steps, with steady improvement to 0.5+ average reward by 200 steps. Final model should outperform GPT-4o-mini baseline by end of 500-step training.

**Next Steps**: Begin Phase 1, Day 1 tasks immediately. Track progress daily. Adjust plan based on results.
