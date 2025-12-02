#!/usr/bin/env python3
"""
RL工作流生成器 - 使用RL训练的Qwen2.5-7B生成优化的工作流
"""
import torch
import json
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

class RLWorkflowGenerator:
    """使用RL训练的Qwen2.5-7B生成优化的工作流"""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_checkpoint: Optional[str] = None,
        device_ids: List[int] = [2, 3],
        operator_descriptions_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            base_model: 基座模型路径
            lora_checkpoint: LoRA检查点路径（None表示使用基座模型）
            device_ids: 使用的GPU ID列表
            operator_descriptions_path: AFlow算子描述文件路径
            config: 额外配置
        """
        self.base_model = base_model
        self.lora_checkpoint = lora_checkpoint
        self.device_ids = device_ids
        self.device = f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
        self.config = config or {}

        # 设置CUDA设备
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))

        print(f"🔧 初始化RL工作流生成器")
        print(f"  设备: {self.device}")
        print(f"  GPU: {device_ids}")

        # 加载tokenizer
        print(f"📥 加载tokenizer: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        print(f"📥 加载基座模型: {base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True
        )

        # 加载LoRA权重（如果有）
        if lora_checkpoint:
            print(f"📥 加载LoRA检查点: {lora_checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
            self.model.eval()

        # 加载算子描述
        self.operator_descriptions = self._load_operator_descriptions(operator_descriptions_path)

        print(f"✅ RL工作流生成器初始化完成")

    def _load_operator_descriptions(self, descriptions_path: Optional[str]) -> Dict:
        """加载AFlow算子描述"""
        if descriptions_path and Path(descriptions_path).exists():
            with open(descriptions_path, 'r') as f:
                return json.load(f)

        # 默认算子描述
        return {
            "Custom": {
                "description": "Generates anything based on customized input and instruction.",
                "interface": "custom(input: str, instruction: str) -> dict with key 'response'"
            },
            "AnswerGenerate": {
                "description": "Generates step-by-step reasoning and final answer.",
                "interface": "answer_generate(input: str) -> dict with keys 'thought' and 'answer'"
            },
            "Programmer": {
                "description": "Automatically writes and executes Python code.",
                "interface": "programmer(problem: str, analysis: str = 'None') -> dict with keys 'code' and 'output'"
            },
            "ScEnsemble": {
                "description": "Uses self-consistency to select the most frequent solution.",
                "interface": "sc_ensemble(solutions: List[str], problem: str) -> dict with key 'response'"
            },
            "Review": {
                "description": "Reviews and provides feedback on a solution.",
                "interface": "review(problem: str, solution: str) -> dict with keys 'review_result' and 'feedback'"
            },
            "Revise": {
                "description": "Revises solution based on feedback.",
                "interface": "revise(problem: str, solution: str, feedback: str) -> dict with key 'solution'"
            }
        }

    def _build_generation_prompt(self, problem: str, problem_type: str) -> str:
        """构建提示词，明确算子 API（增强版 - 含Few-shot示例）"""

        # Few-shot正确示例（3个示例覆盖不同场景）
        few_shot_example = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ EXAMPLE 1: SIMPLE QA WORKFLOW (MOST COMMON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
import workspace.qa.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.model = create_llm_instance(llm_config)  # ✓ CORRECT: 'model'
        self.answer_generate = operator.AnswerGenerate(self.model)  # ✓ CORRECT

    async def __call__(self, problem: str, entry_point: str = None):
        result = await self.answer_generate(input=problem)
        answer = result.get('answer', '') if isinstance(result, dict) else str(result)
        cost = self.model.get_usage_summary()["total_cost"]
        return answer, cost
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ EXAMPLE 2: CODE WORKFLOW WITH TEST
━━━━━━━━━━━━━━━━━━━━━━━━━��━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
import workspace.code.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.model = create_llm_instance(llm_config)  # ✓ CORRECT: 'model'
        self.programmer = operator.Programmer(self.model)  # ✓ CORRECT
        self.test = operator.Test(self.model)  # ✓ CORRECT

    async def __call__(self, problem: str, entry_point: str = None):
        # Generate code
        prog_result = await self.programmer(problem=problem, analysis='')
        code = prog_result.get('code', '') if isinstance(prog_result, dict) else str(prog_result)

        # Test code if entry_point available
        if entry_point:
            test_result = await self.test(problem=problem, solution=code, entry_point=entry_point)
            if isinstance(test_result, dict) and test_result.get('result', False):
                code = test_result.get('solution', code)

        cost = self.model.get_usage_summary()["total_cost"]
        return code, cost
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ EXAMPLE 3: MATH WITH REVIEW-REVISE LOOP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
import workspace.math.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.model = create_llm_instance(llm_config)  # ✓ CORRECT: 'model'
        self.answer_generate = operator.AnswerGenerate(self.model)
        self.review = operator.Review(self.model)
        self.revise = operator.Revise(self.model)  # ✓ All three initialized

    async def __call__(self, problem: str, entry_point: str = None):
        # Generate initial answer
        result = await self.answer_generate(input=problem)
        answer = result.get('answer', '') if isinstance(result, dict) else str(result)

        # Review and potentially revise
        review_result = await self.review(problem=problem, solution=answer)
        if isinstance(review_result, dict) and not review_result.get('review_result', True):
            feedback = review_result.get('feedback', '')
            revise_result = await self.revise(problem=problem, solution=answer, feedback=feedback)
            answer = revise_result.get('solution', answer) if isinstance(revise_result, dict) else str(revise_result)

        cost = self.model.get_usage_summary()["total_cost"]
        return answer, cost
```

🚫 COMMON MISTAKES - NEVER DO THESE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ MISTAKE 1: Wrong variable name
   IMPORTANT: The LLM instance variable MUST be named 'model' (single token)
   ✅ CORRECT:   self.model = create_llm_instance(llm_config)
   ❌ WRONG:     self.llm = ...  (causes tokenizer issues)
   ❌ WRONG:     self.language_model = ...

❌ MISTAKE 2: Using undefined variables
   if cond: code = ...
   return code  # ❌ code undefined if cond is False!
   → ✅ CORRECT:
   code = None  # Initialize first!
   if cond: code = ...
   return code

❌ MISTAKE 3: Calling .get() on non-dict (causes NoneType errors)
   result = await operator()  # might return str!
   value = result.get('key')  # ❌ AttributeError if result is str
   → ✅ CORRECT:
   value = result.get('key') if isinstance(result, dict) else result

❌ MISTAKE 4: Confusing Review vs Revise operators
   self.revise = operator.Revise(self.model)  # ❌ Revise not initialized
   await self.revise(...)  # ❌ AttributeError: 'Workflow' has no 'revise'
   → ✅ CORRECT:
   # In __init__: Initialize what you use
   self.review = operator.Review(self.model)  # ✓
   # In __call__:
   await self.review(problem=problem, solution=solution)  # ✓

   # If you need Revise, initialize it too:
   self.revise_op = operator.Revise(self.model)  # ✓ Different name
   await self.revise_op(problem=problem, solution=sol, feedback=fb)  # ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

        prompt = few_shot_example + f"""Now generate YOUR Workflow for the following problem.

CRITICAL RULES:
1. Use EXACT variable name: 'model' NOT 'llm', 'll_m', or 'language_model'
2. Initialize ALL variables before if-blocks
3. Always check isinstance(result, dict) before calling .get()
4. __call__ signature: async def __call__(self, problem: str, entry_point: str = None)
5. Always return (solution_string, cost_float) tuple

Available Operators:

1. Custom(model) - Most flexible, for any custom task
   Call: await self.custom(input=str, instruction=str)
   Returns: {{'response': str}}

2. AnswerGenerate(model) - Step-by-step reasoning
   Call: await self.answer_generate(input=str)  ← NO instruction parameter!
   Returns: {{'thought': str, 'answer': str}}

3. Programmer(model) - Auto-generate and execute Python code
   Call: await self.programmer(problem=str, analysis=str)
   Returns: {{'code': str, 'output': str}}

4. Test(model) - Test code with test cases (uses entry_point to look up test cases automatically)
   Call: await self.test(problem=str, solution=str, entry_point=str)  ← NO 'test' parameter!
   Returns: {{'result': bool, 'solution': str}}

5. Review(model) - Review and validate solution
   Call: await self.review(problem=str, solution=str)
   Returns: {{'review_result': bool, 'feedback': str}}

6. Revise(model) - Revise solution based on feedback
   Call: await self.revise(problem=str, solution=str, feedback=str)
   Returns: {{'solution': str}}

7. ScEnsemble(model) - Self-consistency ensemble voting
   Call: await self.sc_ensemble(solutions=list, problem=str)
   Returns: {{'response': str}}

"""

        # L2.1: 添加问题类型特定的约束（方案B：软建议而非硬命令）
        if problem_type == "qa":
            problem_specific = """
📋 RECOMMENDED: QA PROBLEMS (problem_type="qa")
================================================================================
⚠️  CONSTRAINTS (violation penalty: -5.0 reward):
  ❌ Avoid Test operator - QA typically has no automated test cases
     Using Test will likely cause NoneType errors (penalty: -5.0)
  ❌ Avoid Programmer operator - QA is text-based, not code-related
     Using Programmer is inefficient (penalty: -5.0)
  ❌ Avoid entry_point parameter - QA problems don't have entry_point
     Using entry_point will cause parameter errors (penalty: -5.0)

✅ PREFERRED operators for QA:
  ✅ Custom(model) - Most flexible for text-based tasks
  ✅ AnswerGenerate(model) - Generate reasoning and answers (RECOMMENDED)
  ✅ Review(model) - Validate answer quality
  ✅ Revise(model) - Improve answers based on feedback
  ✅ ScEnsemble(model) - Ensemble multiple candidates

Example workflow structure for QA:
  answer = await self.answer_generate(input=problem)
  # ... optionally review and revise ...
  return answer['answer'], cost

Note: You can try other operators, but they will receive penalty in reward.
================================================================================
"""
        elif problem_type == "code":
            problem_specific = """
✅ CRITICAL: CODE PROBLEMS (problem_type="code") - REQUIRE Test OPERATOR!
================================================================================
MUST use these operators with CODE problems:
  ✅ Programmer(model) - Generate and improve Python code
  ✅ Test(model) - Validate code with entry_point (CRITICAL!)

Test operator MUST be used to verify code correctness:
  - Test signature: await self.test(problem=str, solution=str, entry_point=str)
  - entry_point is the function name you're implementing (e.g., "has_close_elements")
  - Test operator finds test cases automatically using entry_point
  - DO NOT pass 'test' parameter - Test finds it automatically!

Example workflow for CODE:
  code_result = await self.programmer(problem=problem, analysis='')
  code = code_result['code']
  test_result = await self.test(problem=problem, solution=code, entry_point=entry_point)
  if test_result['result']:
      return code, cost
  else:
      # Optionally revise based on test failure
      ...

CRITICAL: entry_point will NOT be None/empty for code problems!
================================================================================
"""
        elif problem_type == "math":
            problem_specific = """
📊 RECOMMENDED: MATH PROBLEMS (problem_type="math")
================================================================================
⚠️  CONSTRAINTS (violation penalty: -5.0 reward):
  ❌ Avoid Test operator - Math has no automated test cases
     Using Test will cause NoneType errors (penalty: -5.0)
  ❌ Avoid Programmer operator - Math is not code-related
     Using Programmer is inefficient (penalty: -5.0)
  ❌ Avoid entry_point parameter - Math problems don't have entry_point
     Using entry_point will cause parameter errors (penalty: -5.0)

✅ PREFERRED operators for MATH:
  ✅ Custom(model) - Flexible mathematical reasoning
  ✅ AnswerGenerate(model) - Step-by-step mathematical reasoning (RECOMMENDED)
  ✅ Review(model) - Verify mathematical correctness
  ✅ Revise(model) - Improve solution based on feedback

Example workflow for MATH:
  answer = await self.answer_generate(input=problem)
  return answer['answer'], cost

Note: You can try other operators, but they will receive penalty in reward.
================================================================================
"""
        else:
            problem_specific = ""

        prompt += problem_specific + """
Template (complete the __call__ method):

import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.model = create_llm_instance(llm_config)
        # Initialize operators you need (ONLY the ones you will use):
        # self.custom = operator.Custom(self.model)
        # self.answer_generate = operator.AnswerGenerate(self.model)
        # self.programmer = operator.Programmer(self.model)
        # self.test = operator.Test(self.model)
        # self.review = operator.Review(self.model)
        # self.revise = operator.Revise(self.model)
        # self.sc_ensemble = operator.ScEnsemble(self.model)

    async def __call__(self, problem: str, entry_point: str = None):
        # Solve: {problem}
        # MUST return (solution, cost) tuple
        # Example: return solution['response'], self.model.get_usage_summary()["total_cost"]
        # Note: entry_point is optional, used for code problems (ignored for other types)

        # IMPORTANT: Initialize solution variable before any if-blocks!
        # Good example:
        #   solution = await self.answer_generate(input=problem)
        #   answer = solution.get('answer', '')
        #   if some_condition:
        #       answer = improved_answer  # Modify existing variable
        #   return answer, cost  # Always defined
        #
        # Bad example (NEVER do this):
        #   if some_condition:
        #       answer = ...  # Only defined in if-block
        #   return answer, cost  # ERROR: answer may be undefined!

        pass
"""

        return prompt

    def generate_workflow(
        self,
        problem: str,
        problem_type: str = "math",
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        return_full_output: bool = False,
        custom_prompt: Optional[str] = None
    ) -> Dict:
        """
        生成优化的工作流

        Args:
            problem: 问题文本
            problem_type: 问题类型 (math/code/qa)
            temperature: 采样温度
            max_new_tokens: 最大生成token数
            return_full_output: 是否返回完整输出
            custom_prompt: 自定义提示词（如果提供，将覆盖默认提示词）

        Returns:
            {
                "workflow_code": "Python代码",
                "valid": bool,
                "error": Optional[str],
                "metadata": {...}
            }
        """

        # 构建提示词（支持动态注入）
        if custom_prompt is not None:
            prompt = custom_prompt
        else:
            prompt = self._build_generation_prompt(problem, problem_type)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.config.get('top_p', 0.95),
                top_k=self.config.get('top_k', 50),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # 解析输出
        workflow_code, is_valid, error = self._parse_workflow_code(generated_text, problem_type)

        result = {
            "workflow_code": workflow_code,
            "valid": is_valid,
            "error": error,
            "metadata": {
                "problem": problem,
                "problem_type": problem_type,
                "temperature": temperature,
                "tokens_generated": outputs.shape[1] - inputs['input_ids'].shape[1]
            }
        }

        if return_full_output:
            result["full_output"] = generated_text
            result["prompt"] = prompt

        return result

    def _parse_workflow_code(self, generated_text: str, problem_type: str) -> Tuple[str, bool, Optional[str]]:
        """解析生成的文本，提取并验证工作流代码"""

        # DEBUG: 打印 Qwen 生成的原始文本
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG: Qwen 生成的原始文本 (完整):")
        print(f"{'='*60}")
        print(generated_text)  # 打印完整文本
        print(f"{'='*60}\n")

        # 提取代码块
        code_start = generated_text.find("```python")
        if code_start == -1:
            # 没有markdown代码块，尝试直接查找class定义
            code_start = generated_text.find("class Workflow:")
            if code_start == -1:
                print(f"⚠️  未找到 'class Workflow:'，使用默认工作流")
                return self._get_default_workflow(problem_type), False, "No Workflow class found in output"

            code = generated_text[code_start:]
        else:
            code_start += len("```python\n")
            code_end = generated_text.find("```", code_start)

            if code_end == -1:
                code = generated_text[code_start:]
            else:
                code = generated_text[code_start:code_end]

        # 去除首尾空白
        code = code.strip()

        # ===== 增强的语法和拼写验证 =====
        # Step 1: AST语法验证
        try:
            tree = ast.parse(code)
            is_valid = True
            error = None
        except SyntaxError as e:
            is_valid = False
            error = f"Syntax error: {str(e)}"
            print(f"⚠️  语法错误: {error}")
            return self._get_default_workflow(problem_type), False, error

        # Step 2: 变量名检查（确保使用'model'而非'llm'）
        # 由于tokenizer将'llm'分为['ll', 'm']两个token，导致生成'll_m'错误
        # 解决方案：强制使用'model'（单token）
        typo_patterns = [
            ('self.llm', 'self.model'),  # 检测旧的self.llm并修复
            ('.llm', '.model'),           # 检测任何.llm并修复
        ]

        found_typos = []
        for typo, correct in typo_patterns:
            # 使用正则避免匹配llm_config
            import re
            pattern = re.escape(typo) + r'(?![a-z_])'  # 确保后面不是字母或下划线
            if re.search(pattern, code):
                found_typos.append(f"{typo} (should be {correct})")

        if found_typos:
            error = f"Variable name issues detected: {', '.join(found_typos)}"
            print(f"⚠️  变量名问题: {error}")
            # 自动修复：将self.llm替换为self.model
            for typo, correct in typo_patterns:
                pattern = re.escape(typo) + r'(?![a-z_])'
                code = re.sub(pattern, correct, code)
            print(f"✅ 已自动修复变量名（llm→model）")
            error = None

        # Step 3: 检查是否定义了必要的方法
        if 'async def __call__' not in code:
            is_valid = False
            error = "Missing '__call__' method"
            print(f"⚠️  缺少__call__方法: {error}")
            return self._get_default_workflow(problem_type), False, error

        if 'def __init__' not in code:
            is_valid = False
            error = "Missing '__init__' method"
            print(f"⚠️  缺少__init__方法: {error}")
            return self._get_default_workflow(problem_type), False, error

        return code, is_valid, error

    def _get_default_workflow(self, problem_type: str = "math") -> str:
        """默认工作流（当生成失败时）"""
        return f"""import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.model = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.model)

    async def __call__(self, problem: str, entry_point: str = None):
        # entry_point is optional, used for code problems
        solution = await self.custom(input=problem, instruction="Solve this problem step by step.")
        response = solution.get('response', '') if isinstance(solution, dict) else str(solution)
        return response, self.model.get_usage_summary()["total_cost"]
"""


def test_generator():
    """测试生成器"""
    print("\n" + "=" * 60)
    print("🧪 测试RL工作流生成器")
    print("=" * 60)

    # 注意：这需要Qwen模型，如果没有下载会很慢
    generator = RLWorkflowGenerator(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        device_ids=[2, 3],
        operator_descriptions_path=os.path.join(os.getenv("AFLOW_PATH", "./AFlow"), "workspace/MATH/workflows/template/operator.json")
    )

    # 测试问题
    test_problem = "What is 15 + 27?"

    print(f"\n📝 测试问题: {test_problem}")

    # 生成工作流
    result = generator.generate_workflow(
        problem=test_problem,
        problem_type="math",
        temperature=0.7,
        max_new_tokens=1024
    )

    print(f"\n✅ 生成结果:")
    print(f"  有效性: {result['valid']}")
    if result['error']:
        print(f"  错误: {result['error']}")

    print(f"\n📄 生成的工作流代码:")
    print(result['workflow_code'])


if __name__ == "__main__":
    test_generator()
