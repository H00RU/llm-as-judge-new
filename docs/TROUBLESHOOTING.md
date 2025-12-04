# Troubleshooting Guide

**Last Updated**: 2025-12-04 (Phase 6)
**Status**: Complete fixes documented

## Quick Diagnosis

### Issue: Training doesn't start

**Error**: `ModuleNotFoundError`, `ImportError`, etc.

**Solution**:
```bash
# 1. Verify environment setup
echo $PYTHONPATH  # Should include /root/llm-as-judge-new

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Verify models downloaded
ls -la models/Qwen2.5-7B-Instruct/

# 4. Check data files
ls -la data/mixed/train_mixed.jsonl data/mixed/test_mixed.jsonl
```

### Issue: Training runs but metrics are all 0%

**Common Cause**: 5-tier reward system not integrated properly

**Diagnosis**:
```python
# Check if using new reward system
from src.reward_computer_v2 import RewardComputer

# Should work without errors
rc = RewardComputer(use_answer_extractor=True)
result = rc.compute_reward(
    problem="2+2=?",
    prediction="The answer is 4",
    ground_truth="4",
    problem_type="math"
)
print(f"Reward: {result['reward']}")  # Should be 1.0
```

**Solution**:
1. Verify `grpo_trainer.py` imports `reward_computer_v2`
2. Check reward extraction returns 5-tier values, not binary
3. Verify experience buffer threshold is 0.7 (not old 8.0)

---

## Training Issues

### 1. Out of Memory (OOM) Error

**Symptom**:
```
CUDA out of memory. Tried to allocate 2.5GB...
RuntimeError: CUDA out of memory
```

**Causes**:
- Batch size too large
- Gradient accumulation not working
- Too many K (sequences per sample)
- Model not using LoRA

**Solutions** (in order of preference):

```yaml
# 1. Reduce K (num_return_sequences_in_group)
num_return_sequences_in_group: 1  # from 2

# 2. Increase gradient accumulation
gradient_accumulation_steps: 8  # from 4

# 3. Reduce batch size
rollout_batch_size: 3  # from 5

# 4. Verify LoRA is enabled
use_lora: true

# 5. Verify device mapping
device_mapping: [0]  # Single GPU
physical_gpus: [0]
```

**Check if working**:
```bash
# Monitor GPU memory during training
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

---

### 2. Training Accuracy Stuck at 0%

**Symptom**:
```
Step 100: reward/mean=0.05, train_accuracy=0.0%
Step 200: reward/mean=0.08, train_accuracy=0.0%
Step 300: reward/mean=0.11, train_accuracy=0.0%
```

**Root Cause**: Reward system misconfiguration

**Diagnosis**:
```python
# Check correctness threshold
from src.grpo_trainer import GRPO_Trainer

# In grpo_trainer.py around line 515
if reward_score >= 0.7:  # Should be 0.7 for new 5-tier system
    correct_count += 1
```

**Solution**:
```python
# In grpo_trainer.py:

# ✅ CORRECT (5-tier system)
if reward_score >= 0.7:  # Tier 4 and above
    correct_count += 1

# ❌ WRONG (old 10-point system)
if reward_score >= 5.0:
    correct_count += 1
```

**Verify**:
```bash
# Look at W&B logs
# reward/mean should be > 0.2 after 50 steps
# train_accuracy should increase after 100 steps
```

---

### 3. High Timeout Rate

**Symptom**:
```
Step 50: 12/50 workflows timed out (24%)
Step 100: 10/50 workflows timed out (20%)
```

**Root Cause**: Too many concurrent workflows (K value too high)

**Solution**:
```yaml
# Reduce K value from 6 to 2
num_return_sequences_in_group: 2  # was 6

# Optional: Increase timeout
execution_timeout: 300  # from 180 (if only occasional timeouts)
```

**Check actual concurrency**:
```python
# In training loop:
total_concurrent = rollout_batch_size * num_return_sequences_in_group
print(f"Total concurrent workflows: {total_concurrent}")
# Should be <= 10 to be safe (5 * 2)
```

---

### 4. NaN or Inf in Loss

**Symptom**:
```
Step 25: loss=nan
Step 26: loss=inf
Training crashed
```

**Possible Causes**:
1. Learning rate too high
2. KL coefficient too large
3. Numerical instability

**Solution**:
```yaml
# Reduce learning rate
learning_rate: 1e-5  # from 2e-5

# Reduce KL coefficient
kl_loss_coef: 0.001  # from 0.005

# Increase warmup steps
warmup_steps: 200  # from 100

# Enable gradient clipping (should be on)
max_grad_norm: 0.5  # from 1.0
```

---

## Workflow Generation Issues

### 1. Generated Code Has Syntax Errors

**Symptom**: Most workflows rejected by validator

**Diagnosis**:
```python
from src.workflow_validator import WorkflowValidator

validator = WorkflowValidator()
is_valid, error_msg, details = validator.validate_workflow_code(generated_code)

print(f"Valid: {is_valid}")
print(f"Error: {error_msg}")
print(f"Details: {details}")
```

**Solutions**:

1. **Check prompt quality**:
```python
from src.prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer()
prompt = optimizer.build_dynamic_prompt(
    problem="2+2=?",
    problem_type="math"
)
# Verify operator specifications are complete
assert "PascalCase" in prompt
assert "Custom" in prompt
```

2. **Enhanced code builder**:
```python
from src.workflow_code_builder import WorkflowCodeBuilder

builder = WorkflowCodeBuilder()
built_code, fixes = builder.build_workflow(generated_code, problem_type="math")

print(f"Fixes applied: {len(fixes)}")
for fix in fixes:
    print(f"  - {fix}")
```

3. **Improve generation**:
- Add more few-shot examples to prompt
- Emphasize operator initialization rules
- Use Review-Revise cycle for code generation

---

### 2. Operators Called with Wrong Parameters

**Symptom**:
```
TypeError: Custom.__call__() missing 1 required positional argument: 'instruction'
```

**Cause**: Generated code missing required parameters

**Solution**:

1. **Verify operator signatures in prompt**:
```python
# In prompt_optimizer.py, check operator_templates
assert operator_templates['Test']['required_params'] == ['problem', 'solution', 'entry_point']
```

2. **Add parameter verification to builder**:
```python
# In workflow_code_builder.py, add method:
def _check_operator_parameters(self, code: str) -> List[str]:
    """Verify all operator calls have required parameters"""
    issues = []
    # Parse and check parameters
    return issues
```

3. **Enhance prompt with explicit examples**:
```
CRITICAL OPERATOR CALL RULES:
✅ CORRECT:
  result = await self.test(problem=problem, solution=code, entry_point=entry_point)

❌ WRONG:
  result = await self.test(problem=problem)  # Missing parameters!
```

---

### 3. Workflow Execution Fails

**Symptom**: Valid code but execution crashes

**Diagnosis**:
```python
# Check AFlow executor logs
from src.aflow_executor import AFlowExecutor

executor = AFlowExecutor()
# Check logs in /root/AFlow/workspace/logs/
```

**Common Issues**:

1. **Operator timeout**:
```
asyncio.TimeoutError: Code execution timed out
```
Solution: Increase timeout in config.yaml:
```yaml
execution_timeout: 300  # from 180
```

2. **Operator response format mismatch**:
```
KeyError: 'response' or KeyError: 'answer'
```
Solution: Verify ResponseStandardizer handles all formats:
```python
from src.response_standardizer import ResponseStandardizer

response = {...}  # Raw operator response
standardized = ResponseStandardizer.standardize(response, 'Custom')
# Should have 'success' and 'content' fields
```

3. **Async issues**:
```
TypeError: object NoneType can't be used in 'await' expression
```
Solution: Ensure all operator calls have await:
```python
# ✅ CORRECT
result = await self.operator(...)

# ❌ WRONG
result = self.operator(...)  # Missing await!
```

---

## Data Issues

### 1. Schema Validation Failures

**Symptom**:
```
Schema validation failed: 50 samples invalid out of 2071
```

**Diagnosis**:
```bash
# Run validation script
python scripts/validate_data_schema.py

# Check output for specific issues
```

**Solution**:

1. Check field mapping in data_manager.py:
```python
FIELD_MAPPING = {
    "question": "problem",
    "reference_answer": "ground_truth",
    "domain": "problem_type"
}
# Ensure all source files use these source field names
```

2. Regenerate data:
```bash
python scripts/process_datasets.py
```

3. Verify data integrity:
```python
import json
from pathlib import Path

with open("data/mixed/train_mixed.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 10: break
        sample = json.loads(line)
        assert "problem" in sample
        assert "ground_truth" in sample
        assert "problem_type" in sample
        assert sample["problem_type"] in ["math", "code", "qa"]
```

---

### 2. Unbalanced Domain Distribution

**Symptom**:
```
Domain distribution: math=60%, code=20%, qa=20%
(Expected: 40%, 30%, 30%)
```

**Solution**:
```bash
# Regenerate data with proper ratios
python scripts/process_datasets.py

# Verify distribution
python scripts/validate_data_schema.py | grep "域分布"
```

---

## Performance Issues

### 1. Training is Too Slow

**Symptom**: <5 steps per minute

**Causes**:
1. Workflow execution too slow (timeouts)
2. AFlow operations slow
3. Reward computation expensive

**Solutions**:

1. **Profile execution time**:
```python
import time

start = time.time()
result = await aflow_executor.execute(workflow)
elapsed = time.time() - start

print(f"Execution time: {elapsed:.2f}s")
# Should be < 30s per workflow (with 180s timeout)
```

2. **Optimize AFlow**:
- Use faster LLM (gpt-4o-mini vs gpt-4)
- Reduce operator count in workflows
- Enable response caching

3. **Optimize reward computation**:
- Skip LLM Judge for obvious cases
- Cache similarity computations
- Use vectorized operations for QA F1

---

### 2. W&B Logging Slowing Training

**Symptom**: Significant slowdown with logging enabled

**Solution**:
```yaml
# Reduce logging frequency
log_every: 10  # from 5

# Reduce save frequency
save_every: 50  # from 20

# In W&B, use sampling
wandb.log(metrics, step=step, commit=False)  # Batch commits
```

---

## Integration Issues

### 1. AFlow Not Found

**Symptom**:
```
ModuleNotFoundError: No module named 'scripts'
```

**Solution**:
```bash
# Set AFLOW_PATH environment variable
export AFLOW_PATH=/root/AFlow

# Or modify aflow_executor.py:
aflow_path = os.getenv("AFLOW_PATH", "/root/AFlow")
sys.path.insert(0, aflow_path)
```

---

### 2. OpenAI API Not Working

**Symptom**:
```
AuthenticationError: Incorrect API key provided
```

**Solution**:
```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Verify in Python
import os
assert os.getenv("OPENAI_API_KEY"), "API key not set!"
```

---

## Debugging Utilities

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In grpo_trainer.py:
logger.debug(f"Step {step}: reward={reward}, accuracy={accuracy}")
```

### Save Problematic Workflows

```python
# In aflow_executor.py:
if not execution_success:
    with open(f"debug/failed_{step}_{i}.py", "w") as f:
        f.write(workflow_code)
    logger.warning(f"Saved failed workflow to debug/failed_{step}_{i}.py")
```

### Profile Reward Computation

```python
import time

start = time.time()
result = reward_computer.compute_reward(...)
elapsed = time.time() - start

logger.info(f"Reward computation time: {elapsed:.3f}s")
```

---

## Health Check Checklist

Before starting training, verify:

```bash
□ Python 3.8+ installed
□ CUDA available: python -c "import torch; print(torch.cuda.is_available())"
□ Models downloaded: ls models/Qwen2.5-7B-Instruct/
□ Data prepared: ls data/mixed/train_mixed.jsonl
□ Dependencies installed: pip list | grep -E "transformers|peft|torch"
□ OpenAI API key set: echo $OPENAI_API_KEY
□ AFlow path set: echo $AFLOW_PATH
□ Validation passes: python scripts/validate_data_schema.py
□ Test workflow runs: python -c "from src.grpo_trainer import GRPO_Trainer"
```

---

## Getting Help

### Check Logs

```bash
# W&B logs: https://wandb.ai/yourproject/llm-as-judge
# Local logs:
ls -la logs/

# Latest run:
tail -100 logs/training_latest.log
```

### Minimal Reproduction

```python
# Test single reward computation
from src.reward_computer_v2 import RewardComputer

rc = RewardComputer()
result = rc.compute_reward(
    problem="2+2=?",
    prediction="The answer is 4",
    ground_truth="4",
    problem_type="math"
)
print(result)  # Should show reward: 1.0
```

### Contact Information

- **Issue Tracking**: GitHub Issues
- **Documentation**: `/root/llm-as-judge-new/docs/`
- **Configuration**: `/root/llm-as-judge-new/config/training.yaml`

---

**Version**: 1.0 (Phase 6 Complete)
**Last Updated**: 2025-12-04
