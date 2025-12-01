# 改进后的 Prompt 版本（治本）

这是一个为 Qwen2.5-7B 重新设计的 Prompt，使用强制性语言、教育性解释和完整的代码示例。

---

## 改进的关键点

1. **强制性语言**: MUST, MUST NOT, ERROR（而不是 RECOMMENDED, AVOID）
2. **教育性**: WHY 每个约束，WHAT_HAPPENS 如果违反
3. **完整示例**: 为每个问题类型展示正确和错误的代码
4. **设计逻辑**: 解释为什么签名要这样设计
5. **参数说明**: 明确什么时候使用什么参数

---

## 改进后的完整 Prompt

```python
"""
Generate a Python Workflow class that solves problems using AFlow operators.

================================================================================
⚠️ CRITICAL DESIGN PRINCIPLE - READ THIS FIRST
================================================================================

The Workflow.__call__ method signature is FIXED and MUST NOT be changed:

    async def __call__(self, problem: str, entry_point: str = None)

WHY this signature is fixed:
  - Provides a UNIFIED interface for all problem types
  - Different problem types are distinguished by CONTENT, not by parameters
  - Allows the system to call the Workflow consistently: workflow(problem, entry_point)

WHAT HAPPENS if you add extra parameters:
  - TypeError: missing positional arguments
  - The workflow cannot be executed
  - System crashes, penalty: FAILURE (reward = -10.0)
  - Example of WRONG signature:
      async def __call__(self, problem, code, entry_point=None, test=None):  # ❌ WRONG!
  - Example of CORRECT signature:
      async def __call__(self, problem: str, entry_point: str = None):  # ✅ CORRECT!

================================================================================
OPERATORS - Use EXACTLY these signatures
================================================================================

The following operators are available. Use them with EXACT parameters only:

1. Custom(llm)
   Signature: await self.custom(input=str, instruction=str)
   Returns: {'response': str}
   Use: When you need custom, flexible task execution

2. AnswerGenerate(llm)
   Signature: await self.answer_generate(input=str)
   Returns: {'thought': str, 'answer': str}
   Use: To generate step-by-step reasoning and final answer
   IMPORTANT: NO 'instruction' parameter!

3. Programmer(llm)
   Signature: await self.programmer(problem=str, analysis=str)
   Returns: {'code': str, 'output': str}
   Use: To generate and execute Python code

4. Test(llm)
   Signature: await self.test(problem=str, solution=str, entry_point=str)
   Returns: {'result': bool, 'solution': str}
   Use: To test code solutions with automated test cases
   IMPORTANT:
     - entry_point is a REQUIRED parameter (function name to test)
     - Test finds test cases automatically using entry_point
     - Use ONLY for code problems that have test cases!
     - DO NOT pass a 'test' parameter - Test operator finds it automatically!

5. Review(llm)
   Signature: await self.review(problem=str, solution=str)
   Returns: {'review_result': bool, 'feedback': str}
   Use: To review and validate a solution

6. Revise(llm)
   Signature: await self.revise(problem=str, solution=str, feedback=str)
   Returns: {'solution': str}
   Use: To improve a solution based on feedback

7. ScEnsemble(llm)
   Signature: await self.sc_ensemble(solutions=list, problem=str)
   Returns: {'response': str}
   Use: To ensemble multiple candidate solutions

================================================================================
PROBLEM-TYPE SPECIFIC RULES
================================================================================

Your workflow MUST follow the rules for its problem type.
Violating these rules causes system errors and penalties.

---

📊 MATH PROBLEMS
================================================================================

WHAT: Math problems require step-by-step mathematical reasoning
      (e.g., "Find the derivative of x^2", "How many hours in a year?")

MUST DO:
  ✅ Use AnswerGenerate to generate reasoning and answer
  ✅ Optionally use Review to verify correctness
  ✅ Optionally use Revise to improve based on feedback

MUST NOT DO:
  ❌ Use Test operator
     WHY: MATH problems have NO automated test cases
          Test operator will try to look up test cases using entry_point
          But entry_point is NONE for MATH problems
     WHAT_HAPPENS: Test tries to access test_cases[entry_point]
                   entry_point is None
                   Result: TypeError - 'NoneType' object is not subscriptable
                   Workflow crashes with NoneType error (penalty: -5.0)

  ❌ Use Programmer operator
     WHY: MATH is not code-related, Programmer is inefficient
          It adds unnecessary complexity and cost

  ❌ Add entry_point parameter to __call__
     WHY: MATH problems don't have entry_point
     WHAT_HAPPENS: entry_point will always be None
                   If you try to use it, it causes parameter errors

✅ CORRECT MATH WORKFLOW:

    async def __call__(self, problem: str, entry_point: str = None):
        # Step 1: Generate step-by-step reasoning and answer
        answer_result = await self.answer_generate(input=problem)
        answer = answer_result.get('answer', '')

        # Step 2: (Optional) Review the answer
        review_result = await self.review(problem=problem, solution=answer)

        # Step 3: (Optional) Revise if needed
        if not review_result.get('review_result', True):
            revise_result = await self.revise(
                problem=problem,
                solution=answer,
                feedback=review_result.get('feedback', '')
            )
            answer = revise_result.get('solution', answer)

        return answer, self.llm.get_usage_summary()["total_cost"]

❌ WRONG MATH WORKFLOW (Example 1):

    async def __call__(self, problem):  # Missing entry_point parameter
        code = await self.programmer(problem=problem)  # ❌ WRONG operator for MATH
        test_result = await self.test(problem=problem, solution=code)  # ❌ NO TEST CASES for MATH
        return code, cost

❌ WRONG MATH WORKFLOW (Example 2):

    async def __call__(self, problem, code, entry_point=None, test=None):  # ❌ WRONG SIGNATURE
        answer = await self.answer_generate(input=problem)
        return answer['answer'], cost

---

💻 CODE PROBLEMS
================================================================================

WHAT: Code problems require implementing a function and testing it
      (e.g., "Implement has_close_elements(numbers, threshold)")

entry_point REQUIREMENT:
  - entry_point is the function name you're implementing
  - It is ALWAYS provided for code problems (never None)
  - Example: entry_point = "has_close_elements"
  - You MUST use it when calling Test operator

MUST DO:
  ✅ Use Programmer to generate Python code
  ✅ Use Test to verify code with test cases
  ✅ Test operator MUST use entry_point to find test cases

MUST NOT DO:
  ❌ Skip Test operator
     WHY: Test is the only way to verify code correctness
     WHAT_HAPPENS: Code might have bugs, wrong answers
                   No automated verification

  ❌ Use Test without entry_point
     WHY: Test needs entry_point to look up test cases
     WHAT_HAPPENS: Test cannot find test cases
                   Result: TypeError - entry_point is required
                   Workflow crashes (penalty: -10.0)

  ❌ Create extra parameters
     WHY: __call__ signature is fixed for all problem types
     WHAT_HAPPENS: TypeError - extra parameters not expected

✅ CORRECT CODE WORKFLOW:

    async def __call__(self, problem: str, entry_point: str = None):
        # Step 1: Generate Python code
        prog_result = await self.programmer(
            problem=problem,
            analysis='Generate clean, efficient code'
        )
        code = prog_result.get('code', '')

        # Step 2: Test the code
        test_result = await self.test(
            problem=problem,
            solution=code,
            entry_point=entry_point  # ✅ MUST USE entry_point
        )

        # Step 3: If test passed, return code
        if test_result.get('result', False):
            return code, self.llm.get_usage_summary()["total_cost"]

        # Step 4: If test failed, optionally revise
        review_result = await self.review(problem=problem, solution=code)
        feedback = review_result.get('feedback', '')
        revise_result = await self.revise(
            problem=problem,
            solution=code,
            feedback=feedback
        )
        revised_code = revise_result.get('solution', code)

        return revised_code, self.llm.get_usage_summary()["total_cost"]

❌ WRONG CODE WORKFLOW (Example 1):

    async def __call__(self, problem: str, entry_point: str = None):
        code = await self.programmer(problem=problem)
        # Missing Test - code is not verified!
        return code, cost

❌ WRONG CODE WORKFLOW (Example 2):

    async def __call__(self, problem: str, entry_point: str = None):
        code = await self.programmer(problem=problem)
        # ❌ WRONG: Calling Test without entry_point
        test_result = await self.test(problem=problem, solution=code)
        return code, cost

---

📋 QA PROBLEMS
================================================================================

WHAT: QA problems are question-answering based on knowledge
      (e.g., "What is the capital of France?", "Explain the CAP theorem")

MUST DO:
  ✅ Use AnswerGenerate to generate reasoning and answer
  ✅ Optionally use Review to validate answer quality
  ✅ Optionally use Revise to improve based on feedback

MUST NOT DO:
  ❌ Use Test operator
     WHY: QA problems have NO automated test cases
     WHAT_HAPPENS: Same as MATH - TypeError with NoneType
                   Workflow crashes (penalty: -5.0)

  ❌ Use Programmer operator
     WHY: QA is text-based, not code-related
     WHAT_HAPPENS: Generates wrong solution type
                   Inefficient and incorrect

  ❌ Use entry_point
     WHY: QA problems don't have entry_point
     WHAT_HAPPENS: entry_point is None, causes errors

✅ CORRECT QA WORKFLOW:

    async def __call__(self, problem: str, entry_point: str = None):
        # Generate answer with reasoning
        answer_result = await self.answer_generate(input=problem)
        answer = answer_result.get('answer', '')

        # Optional: Review and improve
        review_result = await self.review(problem=problem, solution=answer)
        if not review_result.get('review_result', True):
            revise_result = await self.revise(
                problem=problem,
                solution=answer,
                feedback=review_result.get('feedback', '')
            )
            answer = revise_result.get('solution', answer)

        return answer, self.llm.get_usage_summary()["total_cost"]

================================================================================
GENERAL RULES FOR ALL WORKFLOWS
================================================================================

1. Signature MUST be: async def __call__(self, problem: str, entry_point: str = None)
   - NEVER add extra parameters
   - NEVER change parameter names
   - NEVER change parameter types

2. Return value MUST be: (solution_string, cost_float)
   - First element: the solution as a string
   - Second element: cost from self.llm.get_usage_summary()["total_cost"]
   - Example: return answer['answer'], self.llm.get_usage_summary()["total_cost"]

3. Initialize variables BEFORE if-blocks
   - ✅ CORRECT: answer = ''; if condition: answer = ...
   - ❌ WRONG: if condition: answer = ...; return answer

4. Always check return values are dicts before calling .get()
   - ✅ CORRECT: result = await self.operator(...); val = result.get('key', default)
   - ❌ WRONG: val = await self.operator(...)['key']

5. Use await for all operator calls
   - ✅ CORRECT: result = await self.answer_generate(...)
   - ❌ WRONG: result = self.answer_generate(...)  # Missing await!

================================================================================
TEMPLATE (Complete the __call__ method only)
================================================================================

import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # Initialize ONLY the operators you will use:
        # self.custom = operator.Custom(self.llm)
        # self.answer_generate = operator.AnswerGenerate(self.llm)
        # self.programmer = operator.Programmer(self.llm)
        # self.test = operator.Test(self.llm)
        # self.review = operator.Review(self.llm)
        # self.revise = operator.Revise(self.llm)
        # self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str = None):
        # Implement the workflow here
        # Follow the problem-type specific rules above
        # Return: (solution_string, cost_float)
        pass
"""
```

---

## 如何集成这个改进的 Prompt

修改 `src/rl_workflow_generator.py` 的 `_build_generation_prompt()` 方法，用上面的新 Prompt 替换旧的。

主要变化：
1. 第一部分强制性地解释为什么签名必须固定
2. Operators 部分添加了"IMPORTANT"说明
3. 问题类型部分使用了 MUST/MUST NOT（而不是 RECOMMENDED/AVOID）
4. 为每个约束添加了 WHY 和 WHAT_HAPPENS
5. 为每个问题类型添加了完整的正确和错误示例

---

## 预期改进

使用改进后的 Prompt：

1. **立即改进** (Step 1-2)
   - 签名错误会减少（因为更强的强制语言）
   - Operator 选择错误会减少（因为明确的反例）

2. **逐步改进** (Step 3-5)
   - Qwen 开始理解约束背后的逻辑
   - 生成的代码结构更正确

3. **长期改进** (Step 6-10)
   - 成功率显著提高
   - LoRA 学到了有用的模式
   - 模型真正学会了

---

## 为什么这个 Prompt 更好

```
旧 Prompt:
  ❌ 混合强制和建议语气
  ❌ 缺少解释（WHY）
  ❌ 缺少后果描述（WHAT_HAPPENS）
  ❌ 缺少完整的代码示例
  ❌ 对 7B 模型来说太抽象

新 Prompt:
  ✅ 统一的强制语言（MUST, MUST NOT）
  ✅ 清晰的解释（WHY）
  ✅ 具体的后果（WHAT_HAPPENS）
  ✅ 完整的代码示例（正确和错误）
  ✅ 对 7B 模型来说更具体、更清晰
```

这样 Qwen 才能真正学会，而不仅仅是运气好。

---

*版本*: 治本改进
*时间*: 2025-12-01 16:52:00
