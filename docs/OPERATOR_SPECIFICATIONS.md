# AFlow Operator Specifications

**Last Updated**: 2025-12-04 (Phase 5 - Enhanced with standardized returns)

## Overview

This document provides complete API specifications for all 8 operators used in the workflow generation system.

## Standardized Return Format (Phase 5 Enhancement)

All operators now return a standardized format with success flag:

```python
{
    "success": bool,           # Execution succeeded
    "error": Optional[str],    # Error message if failed
    # operator-specific fields below:
    ...
}
```

---

## Operator Specifications

### 1. Custom

**Purpose**: Flexible custom instruction execution

**Interface**:
```python
async def __call__(self, input: str, instruction: str) -> Dict:
```

**Parameters**:
- `input` (str): Task input text
- `instruction` (str): Custom instruction for the task

**Returns**:
```python
{
    "success": bool,
    "response": str,          # Main response text
    "error": Optional[str],
    "metadata": Dict          # Additional metadata
}
```

**Example**:
```python
self.custom = operator.Custom(self.llm)
result = await self.custom(
    input="Solve 2+2",
    instruction="Calculate and explain"
)
answer = result.get('response', '')
```

**Use Cases**:
- Custom prompts for specific tasks
- Flexible problem-solving strategies
- When other operators don't fit

---

### 2. AnswerGenerate

**Purpose**: Step-by-step reasoning and answer generation

**Interface**:
```python
async def __call__(self, input: str) -> Dict:
```

**Parameters**:
- `input` (str): Problem description

**Returns**:
```python
{
    "success": bool,
    "answer": str,            # Final answer
    "thought": str,           # Step-by-step reasoning
    "error": Optional[str]
}
```

**Example**:
```python
self.answer_generate = operator.AnswerGenerate(self.llm)
result = await self.answer_generate(input=problem)
answer = result.get('answer', '')
reasoning = result.get('thought', '')
```

**⚠️ CRITICAL**: Always use `.get()` to safely access fields:
```python
# ✅ CORRECT - Safe
answer = result.get('answer', '')

# ❌ WRONG - May crash
answer = result['answer']  # KeyError if missing!
```

**Use Cases**:
- Math reasoning problems
- QA with reasoning steps
- Complex logic problems

---

### 3. Programmer

**Purpose**: Code generation and execution

**Interface**:
```python
async def __call__(self, problem: str, analysis: str = "None") -> Dict:
```

**Parameters**:
- `problem` (str): Problem description
- `analysis` (str): Analysis guidance for code generation

**Returns**:
```python
{
    "success": bool,          # Code executed successfully (Phase 5 NEW)
    "code": str,              # Generated code
    "output": str,            # Execution result/output
    "error": Optional[str]    # Execution error if failed
}
```

**⚠️ CRITICAL for Math/QA**:
- Use `output` field (execution result), NOT `code`
- `code` is the source code (for Code problems only)
- `output` is what you want for answers

**Example**:
```python
self.programmer = operator.Programmer(self.llm, timeout=60)  # NEW: configurable timeout
result = await self.programmer(problem=problem, analysis="Analyze and solve")

if result.get('success', False):
    answer = result.get('output', '')  # Use output for math/qa!
    code = result.get('code', '')      # Only for code problems
else:
    error = result.get('error', 'Unknown error')
```

**New in Phase 5**:
- Configurable `timeout` parameter (default 60s)
- `success` flag indicates execution success
- `error` field contains failure reason

**Use Cases**:
- Computational math problems
- Calculation-heavy tasks
- Code generation

---

### 4. Test

**Purpose**: Test code against HumanEval test cases

**Interface**:
```python
async def __call__(
    self,
    problem: str,
    solution: str,
    entry_point: str,
    test_loop: int = 3
) -> Dict:
```

**Parameters**:
- `problem` (str): Problem description
- `solution` (str): Code solution to test
- `entry_point` (str): Function name to test (e.g., "has_close_elements")
- `test_loop` (int): Number of refinement attempts (default 3)

**Returns**:
```python
{
    "success": bool,          # All tests passed (Phase 5 NEW)
    "result": bool,           # Test passed (backward compat)
    "test_passed": bool,      # Test passed indicator (Phase 5 NEW)
    "solution": str,          # Final tested solution
    "error": Optional[str]    # Test error message
}
```

**⚠️⚠️⚠️ CRITICAL - ALL 3 PARAMETERS REQUIRED**:

```python
# ✅ CORRECT - Will work
result = await self.test(
    problem=problem,
    solution=code,
    entry_point=entry_point
)

# ❌ WRONG - Will fail with TypeError
await self.test(problem=problem)  # Missing solution & entry_point!
await self.test(problem=problem, solution=code)  # Missing entry_point!
```

**Example**:
```python
self.test = operator.Test(self.llm)
result = await self.test(
    problem=problem,
    solution=code,
    entry_point="solve"
)

if result.get('success', result.get('result', False)):
    final_code = result.get('solution', code)
    print("✅ Tests passed!")
else:
    print(f"❌ Test failed: {result.get('error', 'Unknown')}")
```

**Use Cases**:
- Code problem validation
- HumanEval benchmark testing
- Solution verification

---

### 5. Review

**Purpose**: Review and evaluate solution quality

**Interface**:
```python
async def __call__(self, problem: str, solution: str) -> Dict:
```

**Parameters**:
- `problem` (str): Problem description
- `solution` (str): Solution to review

**Returns**:
```python
{
    "success": bool,
    "feedback": str,          # Review feedback
    "review_result": str,     # May also return as review_result
    "error": Optional[str]
}
```

**⚠️ Multiple field names possible**:
```python
# Use .get() with fallback chain
feedback = result.get('feedback', result.get('review_result', 'No feedback'))
```

**Example**:
```python
self.review = operator.Review(self.llm)
result = await self.review(problem=problem, solution=code)

if result.get('success', True):
    feedback = result.get('feedback', result.get('review_result', ''))
    print(f"Review: {feedback}")
```

**Use Cases**:
- Solution quality evaluation
- Error detection
- Improvement suggestions
- Before/after Review-Revise cycle

---

### 6. Revise

**Purpose**: Improve solution based on feedback

**Interface**:
```python
async def __call__(
    self,
    problem: str,
    solution: str,
    feedback: str
) -> Dict:
```

**Parameters**:
- `problem` (str): Problem description
- `solution` (str): Current solution
- `feedback` (str): Feedback or improvement hints

**Returns**:
```python
{
    "success": bool,
    "solution": str,          # Revised solution
    "error": Optional[str]    # Revision error
}
```

**⚠️ ALL 3 PARAMETERS REQUIRED**:

```python
# ✅ CORRECT
result = await self.revise(
    problem=problem,
    solution=code,
    feedback=feedback
)

# ❌ WRONG - Missing feedback parameter!
await self.revise(problem=problem, solution=code)
```

**Fallback on Error**:
```python
result = await self.revise(problem=problem, solution=code, feedback=feedback)
revised = result.get('solution', code)  # Use original if revision failed
```

**Example**:
```python
self.revise = operator.Revise(self.llm)

# Review cycle
review = await self.review(problem=problem, solution=code)
feedback = review.get('feedback', 'Improve solution')

# Revise based on feedback
revised = await self.revise(problem=problem, solution=code, feedback=feedback)
final_code = revised.get('solution', code)  # Fallback to original
```

**Use Cases**:
- Iterative improvement
- Error correction
- Quality enhancement
- Review-Revise cycles

---

### 7. ScEnsemble (Self-Consistency Ensemble)

**Purpose**: Self-consistency voting across multiple solutions

**Interface**:
```python
async def __call__(self, solutions: List[str], problem: str) -> Dict:
```

**Parameters**:
- `solutions` (List[str]): List of candidate solutions
- `problem` (str): Problem description

**Returns**:
```python
{
    "success": bool,
    "response": str,          # Best voted solution
    "error": Optional[str]
}
```

**Example**:
```python
self.sc_ensemble = operator.ScEnsemble(self.llm)

# Generate multiple solutions
solutions = [
    "solution1_code",
    "solution2_code",
    "solution3_code"
]

result = await self.sc_ensemble(solutions=solutions, problem=problem)
best_solution = result.get('response', solutions[0])
```

**Use Cases**:
- Consensus among multiple attempts
- Uncertainty reduction
- Better solution selection

---

### 8. MdEnsemble (Multi-Domain Ensemble)

**Purpose**: Voting across multiple solutions with domain knowledge

**Interface**:
```python
async def __call__(
    self,
    solutions: List[str],
    problem: str,
    vote_count: int = 5
) -> Dict:
```

**Parameters**:
- `solutions` (List[str]): Candidate solutions
- `problem` (str): Problem description
- `vote_count` (int): Number of votes to aggregate

**Returns**:
```python
{
    "success": bool,
    "response": str,          # Best voted solution
    "error": Optional[str]
}
```

**Use Cases**:
- Multi-expert ensemble
- Diverse reasoning paths
- Robust solution selection

---

## Response Format Reference

### Safe Access Patterns

```python
# Single field with default
content = result.get('response', '')

# Fallback chain - try multiple fields
feedback = result.get('feedback', result.get('review_result', 'No feedback'))

# For nested structures
success = result.get('success', result.get('result', False))
```

### Common Patterns

**Success Check**:
```python
if result.get('success', result.get('result', False)):
    # Success path
    content = result.get('response', result.get('answer', ''))
else:
    # Error path
    error = result.get('error', 'Unknown error')
```

**Chaining Operators**:
```python
# Generate → Test → Revise pattern
prog_result = await self.programmer(problem=problem, analysis="analyze")
code = prog_result.get('code', '')

if code:
    test_result = await self.test(problem=problem, solution=code, entry_point="solve")
    if not test_result.get('success', test_result.get('result', False)):
        review = await self.review(problem=problem, solution=code)
        feedback = review.get('feedback', '')
        revised = await self.revise(problem=problem, solution=code, feedback=feedback)
        code = revised.get('solution', code)
```

---

## Phase 5 Enhancements Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Return Format** | Inconsistent | Standardized `success` flag |
| **Programmer Timeout** | Hardcoded 30s | Configurable (default 60s) |
| **Error Handling** | Sparse | Comprehensive with `error` field |
| **Field Consistency** | Varies | Unified field names |
| **Fallback Strategy** | None | Safe `.get()` with defaults |

---

**Version**: 2.0 (Phase 5 Updated)
**Last Updated**: 2025-12-04
**Maintainer**: AFlow Integration Team
