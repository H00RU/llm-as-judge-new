# 治本修复实施指南

**目标**: 用改进的 Prompt 替换旧的，让 Qwen 真正学会生成正确的代码

**预期结果**:
- 短期（Step 1-2）：签名和 Operator 错误减少
- 中期（Step 3-5）：理解约束的逻辑
- 长期（Step 6-10）：模型有效学习，失败率显著下降

---

## 第1步：备份原始代码

```bash
cp /root/llm-as-judge-new/src/rl_workflow_generator.py \
   /root/llm-as-judge-new/src/rl_workflow_generator.py.backup
```

---

## 第2步：修改 `_build_generation_prompt()` 方法

打开文件：
```bash
vim /root/llm-as-judge-new/src/rl_workflow_generator.py
```

找到 `_build_generation_prompt()` 方法（第 113 行）。

**替换内容**：

从：
```python
def _build_generation_prompt(self, problem: str, problem_type: str) -> str:
    """构建提示词，明确算子 API"""

    prompt = f"""Generate a Python Workflow class. Follow the exact template and API signatures.

CRITICAL: Only use operators listed below with their EXACT parameters!
...
"""
```

改为下面的新实现。为了简洁，我提供一个代码框架：

```python
def _build_generation_prompt(self, problem: str, problem_type: str) -> str:
    """构建改进的提示词 - 强制性、教育性、完整示例"""

    prompt = """Generate a Python Workflow class that solves problems using AFlow operators.

================================================================================
⚠️ CRITICAL DESIGN PRINCIPLE - READ THIS FIRST
================================================================================

The Workflow.__call__ method signature is FIXED and MUST NOT be changed:

    async def __call__(self, problem: str, entry_point: str = None)

WHY this signature is fixed:
  - Provides a UNIFIED interface for all problem types
  - Different problem types are distinguished by CONTENT, not by parameters
  - Allows the system to call the Workflow consistently

WHAT HAPPENS if you add extra parameters (WRONG):
  - ERROR: TypeError: missing positional arguments
  - System cannot execute your workflow
  - Workflow crashes with error
  - EXAMPLE OF WRONG SIGNATURE:
      async def __call__(self, problem, code, entry_point=None, test=None):  # ❌ WRONG!
  - EXAMPLE OF CORRECT SIGNATURE:
      async def __call__(self, problem: str, entry_point: str = None):  # ✅ CORRECT!

================================================================================
OPERATORS - Use EXACTLY these signatures
================================================================================

1. AnswerGenerate(llm) - Generate reasoning + answer
   await self.answer_generate(input=str) → {'thought': str, 'answer': str}
   IMPORTANT: NO 'instruction' parameter!

2. Programmer(llm) - Generate and execute Python code
   await self.programmer(problem=str, analysis=str) → {'code': str, 'output': str}

3. Test(llm) - Test code with test cases (CODE ONLY)
   await self.test(problem=str, solution=str, entry_point=str) → {'result': bool, 'solution': str}
   CRITICAL: Use ONLY for CODE problems!
   CRITICAL: entry_point is REQUIRED!

4. Review(llm) - Review solution
   await self.review(problem=str, solution=str) → {'review_result': bool, 'feedback': str}

5. Revise(llm) - Revise based on feedback
   await self.revise(problem=str, solution=str, feedback=str) → {'solution': str}

6. Custom(llm) - Custom task
   await self.custom(input=str, instruction=str) → {'response': str}

7. ScEnsemble(llm) - Ensemble voting
   await self.sc_ensemble(solutions=list, problem=str) → {'response': str}

================================================================================
PROBLEM-TYPE SPECIFIC RULES - MUST FOLLOW
================================================================================
"""

    # Add problem-type specific rules
    if problem_type == "math":
        prompt += """
📊 MATH PROBLEMS
================================================================================

MUST DO:
  ✅ Use AnswerGenerate to generate reasoning and answer
  ✅ Optionally use Review to verify
  ✅ Optionally use Revise to improve

MUST NOT DO (VIOLATION = ERROR + PENALTY):
  ❌ Use Test operator
     WHY: Math problems have NO test cases. Test will crash.
     WHAT_HAPPENS: TypeError - 'NoneType' object is not subscriptable
     PENALTY: -5.0 reward

  ❌ Use Programmer operator
     WHY: Math is not code. This is inefficient and wrong.
     PENALTY: -5.0 reward

  ❌ Use entry_point parameter
     WHY: Math doesn't have entry_point
     PENALTY: Parameter error

✅ CORRECT MATH WORKFLOW:
    async def __call__(self, problem: str, entry_point: str = None):
        answer_result = await self.answer_generate(input=problem)
        answer = answer_result.get('answer', '')

        review_result = await self.review(problem=problem, solution=answer)
        if not review_result.get('review_result', True):
            revise_result = await self.revise(problem=problem, solution=answer,
                                              feedback=review_result.get('feedback', ''))
            answer = revise_result.get('solution', answer)

        return answer, self.llm.get_usage_summary()["total_cost"]

❌ WRONG MATH WORKFLOW:
    async def __call__(self, problem, code, entry_point=None, test=None):  # ❌ Wrong signature!
        code = await self.programmer(problem=problem)  # ❌ Wrong operator!
        result = await self.test(problem=problem, solution=code)  # ❌ Wrong operator!
        return code, cost

================================================================================
"""

    elif problem_type == "code":
        prompt += """
💻 CODE PROBLEMS
================================================================================

CRITICAL: entry_point is ALWAYS provided for code problems.
Use it in Test operator: await self.test(..., entry_point=entry_point)

MUST DO:
  ✅ Use Programmer to generate Python code
  ✅ Use Test to verify code with test cases
  ✅ Test MUST use entry_point

MUST NOT DO (VIOLATION = ERROR + PENALTY):
  ❌ Skip Test operator
     WHY: Code must be verified! Otherwise wrong answers.
     PENALTY: -10.0 reward

  ❌ Call Test without entry_point
     WHY: Test needs entry_point to find test cases
     WHAT_HAPPENS: TypeError - entry_point not found
     PENALTY: -10.0 reward

  ❌ Add extra parameters to __call__
     WHY: Signature is fixed for all problem types
     PENALTY: Workflow crashes

✅ CORRECT CODE WORKFLOW:
    async def __call__(self, problem: str, entry_point: str = None):
        prog_result = await self.programmer(problem=problem, analysis='')
        code = prog_result.get('code', '')

        test_result = await self.test(problem=problem, solution=code,
                                      entry_point=entry_point)  # ✅ Use entry_point!
        if test_result.get('result', False):
            return code, self.llm.get_usage_summary()["total_cost"]

        # Revise if test failed
        review_result = await self.review(problem=problem, solution=code)
        revise_result = await self.revise(problem=problem, solution=code,
                                         feedback=review_result.get('feedback', ''))
        return revise_result.get('solution', code), self.llm.get_usage_summary()["total_cost"]

❌ WRONG CODE WORKFLOW (missing Test):
    async def __call__(self, problem: str, entry_point: str = None):
        code = await self.programmer(problem=problem)  # Missing Test!
        return code, cost

================================================================================
"""

    elif problem_type == "qa":
        prompt += """
📋 QA PROBLEMS
================================================================================

MUST DO:
  ✅ Use AnswerGenerate to generate reasoning and answer
  ✅ Optionally use Review to validate
  ✅ Optionally use Revise to improve

MUST NOT DO (VIOLATION = ERROR + PENALTY):
  ❌ Use Test operator
     WHY: QA problems have NO test cases
     WHAT_HAPPENS: TypeError - 'NoneType' object is not subscriptable
     PENALTY: -5.0 reward

  ❌ Use Programmer operator
     WHY: QA is text-based, not code. This is inefficient.
     PENALTY: -5.0 reward

✅ CORRECT QA WORKFLOW:
    async def __call__(self, problem: str, entry_point: str = None):
        answer_result = await self.answer_generate(input=problem)
        answer = answer_result.get('answer', '')

        review_result = await self.review(problem=problem, solution=answer)
        if not review_result.get('review_result', True):
            revise_result = await self.revise(problem=problem, solution=answer,
                                             feedback=review_result.get('feedback', ''))
            answer = revise_result.get('solution', answer)

        return answer, self.llm.get_usage_summary()["total_cost"]

================================================================================
"""

    # Common rules
    prompt += """
================================================================================
GENERAL RULES (ALL WORKFLOWS)
================================================================================

1. SIGNATURE: async def __call__(self, problem: str, entry_point: str = None)
   - NEVER add extra parameters
   - NEVER change parameter names or types

2. RETURN: (solution_string, cost_float)
   - return answer, self.llm.get_usage_summary()["total_cost"]

3. Initialize variables BEFORE if-blocks
   - answer = ''; if condition: answer = ...
   - NOT: if condition: answer = ...; return answer

4. Always use await for operator calls
   - result = await self.operator_name(...)

5. Check return values are dicts before .get()
   - value = result.get('key', default)

================================================================================
TEMPLATE (Complete __call__ method only)
================================================================================

import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        # Initialize operators you need:
        # self.answer_generate = operator.AnswerGenerate(self.llm)
        # self.programmer = operator.Programmer(self.llm)
        # self.test = operator.Test(self.llm)
        # self.review = operator.Review(self.llm)
        # self.revise = operator.Revise(self.llm)

    async def __call__(self, problem: str, entry_point: str = None):
        # Follow the {problem_type} problem rules above
        # Return: (solution_string, cost_float)
        pass
"""

    return prompt
```

---

## 第3步：测试新 Prompt

生成一个样本来验证新 Prompt 是否工作：

```bash
python3 << 'EOF'
from src.rl_workflow_generator import RLWorkflowGenerator

generator = RLWorkflowGenerator(
    base_model="/root/llm-as-judge-new/models"
)

# 测试 MATH Prompt
math_prompt = generator._build_generation_prompt(
    problem="A problem",
    problem_type="math"
)

print("MATH Prompt length:", len(math_prompt))
print("Contains 'MUST NOT':", "MUST NOT" in math_prompt)
print("Contains correct example:", "✅ CORRECT MATH WORKFLOW" in math_prompt)
print("Contains wrong example:", "❌ WRONG MATH WORKFLOW" in math_prompt)

# 测试 CODE Prompt
code_prompt = generator._build_generation_prompt(
    problem="A problem",
    problem_type="code"
)

print("\nCODE Prompt length:", len(code_prompt))
print("Contains 'entry_point is ALWAYS provided':", "entry_point is ALWAYS provided" in code_prompt)

# 测试 QA Prompt
qa_prompt = generator._build_generation_prompt(
    problem="A problem",
    problem_type="qa"
)

print("\nQA Prompt length:", len(qa_prompt))
print("All Prompts generated successfully!")
EOF
```

---

## 第4步：重新启动训练

### 4a. 杀死当前训练

```bash
kill 42317
```

### 4b. 检查训练日志的最后部分

```bash
tail -50 /root/llm-as-judge-new/nohup_training.log
```

### 4c. 重新启动训练

```bash
cd /root/llm-as-judge-new
nohup python train.py --config config/minimal_training.yaml > nohup_training.log 2>&1 &
echo $! > .minimal_training_pid
tail -f nohup_training.log
```

---

## 预期观察

### 立即（Step 1-2）

```
旧 Prompt 下（已完成）:
  ✅ Fallback成功: 9/9 (100%)
  ✅ 正确评分: 1/9 (11%)
  ✅ 失败评分: 8/9 (89%) 平均 -2.75/10.0

新 Prompt 下（预期）:
  ✅ 签名错误减少（更强的强制语言）
  ✅ Operator 错误减少（更清晰的约束）
  ✅ Fallback 需求减少
```

### 中期（Step 3-5）

```
预期改进:
  - 如果改进有效，Fallback 频率应该降低
  - 通过 Fallback 执行的工作流比例应该下降
  - GRPO 学习信号应该变强
```

### 长期（Step 6-10）

```
目标状态:
  - 生成的代码质量明显提高
  - 不需要 Fallback 的工作流增加
  - 模型真正学到了约束和设计原则
```

---

## 如何监控改进

```bash
# 查看新生成的代码是否有改进
grep -E "async def __call__|TypeError|missing.*positional" nohup_training.log | tail -20

# 统计 Fallback 次数
grep "🔄 执行Fallback" nohup_training.log | wc -l

# 查看评分变化
grep "正确性评分" nohup_training.log | tail -20
```

---

## 总结

这个改进通过以下方式让 Qwen 真正学会：

1. **强制性语言** - MUST, MUST NOT（不是建议）
2. **教育性内容** - WHY, WHAT_HAPPENS（不是仅仅规则）
3. **完整示例** - 正确和错误的代码（不是抽象说明）
4. **逻辑解释** - 为什么签名要固定（设计原理）

这样 Qwen 会逐步理解约束的本质，而不仅仅是遵守规则。

---

*版本*: 治本实施指南
*时间*: 2025-12-01 16:52:00
