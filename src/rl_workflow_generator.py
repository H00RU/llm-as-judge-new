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

from src.workflow_validator import WorkflowValidator

class RLWorkflowGenerator:
    """使用RL训练的Qwen2.5-7B生成优化的工作流"""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device=None,
        base_model: Optional[str] = None,
        lora_checkpoint: Optional[str] = None,
        device_ids: Optional[List[int]] = None,
        operator_descriptions_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            model: 共享的模型实例（优先使用，若提供则不加载新模型）
            tokenizer: 共享的tokenizer实例（优先使用）
            device: 共享的设备（优先使用）
            base_model: 基座模型路径（仅在model=None时使用）
            lora_checkpoint: LoRA检查点路径（None表示使用基座模型）
            device_ids: 使用的GPU ID列表（仅在device=None时使用）
            operator_descriptions_path: AFlow算子描述文件路径
            config: 额外配置
        """
        self.config = config or {}

        # ✨ NEW: Support model sharing from GRPO Trainer
        if model is not None:
            print(f"🔧 初始化RL工作流生成器（使用共享模型）")
            print(f"  🔗 共享模型ID: {id(model)}")
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.base_model = None  # Not needed when sharing
            self.lora_checkpoint = None
            self.device_ids = None
            print(f"  设备: {self.device}")
            print(f"  ✅ 模型共享成功 - 节省 ~15GB GPU内存")
        else:
            # Legacy path: Load own model (for standalone usage)
            print(f"🔧 初始化RL工作流生成器（独立模式）")

            if base_model is None:
                base_model = "Qwen/Qwen2.5-7B-Instruct"
            if device_ids is None:
                device_ids = [2, 3]

            self.base_model = base_model
            self.lora_checkpoint = lora_checkpoint
            self.device_ids = device_ids
            self.device = f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"

            # 设置CUDA设备
            if torch.cuda.is_available():
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))

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

        # 初始化统一验证器（合并了代码构建器和一致性检查器功能）
        self.validator = WorkflowValidator()

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
        """
        生成简化Prompt - 使用自包含架构（无继承）

        关键改进：
        - 移除继承复杂性，使用自包含类
        - 模型显式初始化需要的operators
        - 更容易学习和auto-fix
        - 明确的约束和负面示例确保操作符正确选择
        """

        if problem_type == "math":
            return f"""================================================================================
🎯 TASK: Generate COMPLETE self-contained class for MATH problem workflow
================================================================================

*** CRITICAL: PROBLEM TYPE = MATH ***
Your problem is a MATH problem. Follow ALL constraints below strictly.

================================================================================
⚠️  CRITICAL CODE STRUCTURE (MUST FOLLOW EXACTLY):
================================================================================

You MUST generate a COMPLETE Python class with:

1. ✅ Import statements (REQUIRED):
   from scripts.operators import AnswerGenerate, Review, Revise, ScEnsemble, Custom
   from scripts.async_llm import create_llm_instance
   from scripts.evaluator import DatasetType

2. ✅ Class definition (REQUIRED - NO inheritance):
   class Workflow:

3. ✅ __init__ method (REQUIRED):
   def __init__(self, name: str, llm_config, dataset: DatasetType):
       self.name = name
       self.dataset = dataset
       self.llm = create_llm_instance(llm_config)
       # Initialize operators you will use
       self.answer_generate = AnswerGenerate(self.llm)
       self.review = Review(self.llm)

4. ✅ __call__ method (REQUIRED):
   async def __call__(self, problem: str) -> Tuple[str, float]:
       # Your workflow logic here

================================================================================
✅ COMPLETE CORRECT EXAMPLE (follow exactly):
================================================================================
```python
from scripts.operators import AnswerGenerate, Review, Revise
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # Initialize operators you will use
        self.answer_generate = AnswerGenerate(self.llm)
        self.review = Review(self.llm)
        self.revise = Revise(self.llm)

    async def __call__(self, problem: str) -> Tuple[str, float]:
        # Step 1: Generate initial answer
        ans = await self.answer_generate(input=problem)
        answer = ans.get('answer', '')

        # Step 2: Review the answer
        review = await self.review(problem=problem, solution=answer)

        # Step 3: If feedback suggests revision, revise
        if not review.get('review_result', True):
            revised = await self.revise(
                problem=problem,
                solution=answer,
                feedback=review.get('feedback', '')
            )
            answer = revised.get('solution', answer)

        # Step 4: Return answer and cost
        return answer, self.llm.get_usage_summary().get("total_cost", 0.0)
```

================================================================================
⚙️  OPERATOR INTERFACE REFERENCE (call operators EXACTLY like this):
================================================================================

1. self.answer_generate(input: str) -> dict with keys 'thought', 'answer'
   ❌ WRONG: await self.answer_generate(problem=problem)
   ✅ RIGHT: await self.answer_generate(input=problem)

2. self.review(problem: str, solution: str) -> dict with keys 'review_result', 'feedback'
   ❌ WRONG: await self.review(solution=answer)  # Missing 'problem'
   ✅ RIGHT: await self.review(problem=problem, solution=answer)

3. self.revise(problem: str, solution: str, feedback: str) -> dict with key 'solution'
   ❌ WRONG: await self.revise(solution=answer, feedback=feedback)
   ✅ RIGHT: await self.revise(problem=problem, solution=answer, feedback=feedback)

4. self.scensemble(solutions: List[str], problem: str) -> dict with key 'response'
   ✅ RIGHT: await self.scensemble(solutions=[answer1, answer2], problem=problem)

5. self.custom(input: str, instruction: str) -> dict with key 'response'
   ✅ RIGHT: await self.custom(input=problem, instruction="custom instruction")

================================================================================
✅ OPERATORS YOU CAN USE (for MATH only):
================================================================================
- AnswerGenerate: Generate step-by-step solution
- Review: Review and validate answer
- Revise: Revise solution based on feedback
- ScEnsemble: Self-consistency ensemble (for multiple solutions)
- Custom: Custom prompting (for special cases only)

================================================================================
❌ OPERATORS YOU MUST NOT USE (for MATH problems):
================================================================================
- Programmer: This is for CODE problems, NOT MATH!
- Test: This is for CODE problems, NOT MATH!

================================================================================
📋 REQUIRED SIGNATURE:
================================================================================
Your __call__ method MUST have exactly this signature:
    async def __call__(self, problem: str) -> Tuple[str, float]:

Parameters: only 'problem: str'
Returns: (answer_string, cost_float)

================================================================================
❌ WRONG EXAMPLES (DO NOT DO THIS):
================================================================================
WRONG #1: Only method body, no class definition
```python
async def __call__(self, problem: str):  # ❌ Missing class definition!
    ans = await self.answer_generate(input=problem)
```

WRONG #2: Missing operator initialization
```python
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        # ❌ Missing: self.answer_generate = AnswerGenerate(self.llm)
```

WRONG #3: Using Programmer/Test operators
```python
code = await self.programmer(problem=problem)  # ❌ WRONG! Use answer_generate instead
```

WRONG #4: Incorrect operator call parameters
```python
await self.answer_generate(problem=problem)  # ❌ WRONG! Parameter should be 'input', not 'problem'
await self.review(solution=answer)           # ❌ WRONG! Must include 'problem' parameter
```

================================================================================
🎯 PROBLEM TO SOLVE:
================================================================================
{problem}

================================================================================
📝 INSTRUCTIONS:
================================================================================
1. Generate a COMPLETE class with imports, class definition, __init__, and __call__
2. Use class Workflow: (NO inheritance)
3. Initialize self.llm = create_llm_instance(llm_config) in __init__
4. Initialize ONLY the operators you will use (e.g., self.answer_generate = AnswerGenerate(self.llm))
5. Follow the CORRECT EXAMPLE pattern above EXACTLY
6. Use ONLY the 5 allowed operators
7. Never use Programmer or Test operators
8. Call operators with the EXACT parameter names shown in "OPERATOR INTERFACE REFERENCE"
9. Ensure the method returns (answer, cost) tuple

BEGIN CODE GENERATION:
"""

        elif problem_type == "code":
            return f"""================================================================================
🎯 TASK: Generate COMPLETE self-contained class for CODE problem workflow
================================================================================

*** CRITICAL: PROBLEM TYPE = CODE ***
Your problem is a CODE problem. Follow ALL constraints below strictly.

================================================================================
⚠️  CRITICAL CODE STRUCTURE (MUST FOLLOW EXACTLY):
================================================================================

You MUST generate a COMPLETE Python class with:

1. ✅ Import statements (REQUIRED):
   from scripts.operators import Programmer, Test, Review, Revise, Custom
   from scripts.async_llm import create_llm_instance
   from scripts.evaluator import DatasetType

2. ✅ Class definition (REQUIRED - NO inheritance):
   class Workflow:

3. ✅ __init__ method (REQUIRED):
   def __init__(self, name: str, llm_config, dataset: DatasetType):
       self.name = name
       self.dataset = dataset
       self.llm = create_llm_instance(llm_config)
       # Initialize operators you will use
       self.programmer = Programmer(self.llm)
       self.test = Test(self.llm)

4. ✅ __call__ method (REQUIRED):
   async def __call__(self, problem: str, entry_point: str, test: str) -> Tuple[str, float]:
       # Your workflow logic here

================================================================================
✅ COMPLETE CORRECT EXAMPLE (follow exactly):
================================================================================
```python
from scripts.operators import Programmer, Test, Review, Revise
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # Initialize operators you will use
        self.programmer = Programmer(self.llm)
        self.test = Test(self.llm)
        self.review = Review(self.llm)
        self.revise = Revise(self.llm)

    async def __call__(self, problem: str, entry_point: str, test: str) -> Tuple[str, float]:
        # Save test parameter to instance variable (framework will use this automatically)
        self._test_input = test

        # Step 1: Generate code using Programmer
        code_result = await self.programmer(problem=problem, analysis='')
        code = code_result.get('code', '')

        # Step 2: Test the code with provided test cases
        test_result = await self.test(
            problem=problem,
            solution=code,
            entry_point=entry_point,
            test_loop=3
        )

        # Step 3: If tests pass, return the solution; otherwise review and revise
        if test_result.get('result', False):
            return test_result.get('solution', code), self.llm.get_usage_summary().get("total_cost", 0.0)
        else:
            # Optionally review and revise
            review = await self.review(problem=problem, solution=code)
            if not review.get('review_result', True):
                revised = await self.revise(problem=problem, solution=code, feedback=review.get('feedback', ''))
                code = revised.get('solution', code)
            return code, self.llm.get_usage_summary().get("total_cost", 0.0)
```

================================================================================
⚙️  OPERATOR INTERFACE REFERENCE (call operators EXACTLY like this):
================================================================================

1. self.programmer(problem: str, analysis: str) -> dict with keys 'code', 'output'
   ✅ RIGHT: await self.programmer(problem=problem, analysis='')

2. self.test(problem: str, solution: str, entry_point: str, test_loop: int) -> dict with 'result', 'solution'
   ✅ RIGHT: await self.test(problem=problem, solution=code, entry_point=entry_point, test_loop=3)

3. self.review(problem: str, solution: str) -> dict with keys 'review_result', 'feedback'
   ❌ WRONG: await self.review(solution=code)  # Missing 'problem'
   ✅ RIGHT: await self.review(problem=problem, solution=code)

4. self.revise(problem: str, solution: str, feedback: str) -> dict with key 'solution'
   ❌ WRONG: await self.revise(solution=code, feedback=feedback)
   ✅ RIGHT: await self.revise(problem=problem, solution=code, feedback=feedback)

5. self.custom(input: str, instruction: str) -> dict with key 'response'
   ✅ RIGHT: await self.custom(input=problem, instruction="custom instruction")

================================================================================
✅ OPERATORS YOU CAN USE (for CODE only):
================================================================================
- Programmer: Generate and execute Python code
- Test: Test code with test cases
- Review: Review code quality
- Revise: Revise code based on feedback
- Custom: Custom prompting (for special cases only)

================================================================================
❌ OPERATORS YOU MUST NOT USE (for CODE problems):
================================================================================
- AnswerGenerate: This is for MATH/QA problems, NOT CODE!
- ScEnsemble: This is for MATH/QA problems, NOT CODE!

================================================================================
📋 REQUIRED SIGNATURE:
================================================================================
Your __call__ method MUST have exactly this signature:
    async def __call__(self, problem: str, entry_point: str, test: str) -> Tuple[str, float]:

Parameters: problem: str, entry_point: str, test: str (EXACTLY 3 parameters)
Returns: (result_string, cost_float)

================================================================================
❌ WRONG EXAMPLES (DO NOT DO THIS):
================================================================================
WRONG #1: Only method body, no class definition
```python
async def __call__(self, problem: str, entry_point: str, test: str):  # ❌ Missing class definition!
    code_result = await self.programmer(problem=problem, analysis='')
```

WRONG #2: Missing operator initialization
```python
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        # ❌ Missing: self.programmer = Programmer(self.llm)
```

WRONG #3: Using AnswerGenerate/ScEnsemble operators
```python
ans = await self.answer_generate(input=problem)  # ❌ WRONG! Use programmer instead
```

WRONG #4: Missing test parameters or wrong order
```python
await self.test(problem=problem, solution=code)  # ❌ WRONG! Missing entry_point and test_loop
```

================================================================================
🎯 PROBLEM TO SOLVE:
================================================================================
{problem}

================================================================================
📝 INSTRUCTIONS:
================================================================================
1. Generate a COMPLETE class with imports, class definition, __init__, and __call__
2. Use class Workflow: (NO inheritance)
3. Initialize self.llm = create_llm_instance(llm_config) in __init__
4. Initialize ONLY the operators you will use (e.g., self.programmer = Programmer(self.llm))
5. Follow the CORRECT EXAMPLE pattern above EXACTLY
6. Use ONLY the 5 allowed operators
7. Never use AnswerGenerate or ScEnsemble operators
8. Call operators with the EXACT parameter names shown in "OPERATOR INTERFACE REFERENCE"
9. Ensure __call__ accepts exactly 3 parameters: problem, entry_point, test
10. Ensure the method returns (result, cost) tuple

BEGIN CODE GENERATION:
"""

        elif problem_type == "qa":
            return f"""================================================================================
🎯 TASK: Generate COMPLETE self-contained class for QA problem workflow
================================================================================

*** CRITICAL: PROBLEM TYPE = QA ***
Your problem is a QA (Question Answering) problem. Follow ALL constraints below strictly.

================================================================================
⚠️  CRITICAL CODE STRUCTURE (MUST FOLLOW EXACTLY):
================================================================================

You MUST generate a COMPLETE Python class with:

1. ✅ Import statements (REQUIRED):
   from scripts.operators import AnswerGenerate, Review, Revise, ScEnsemble, Custom
   from scripts.async_llm import create_llm_instance
   from scripts.evaluator import DatasetType

2. ✅ Class definition (REQUIRED - NO inheritance):
   class Workflow:

3. ✅ __init__ method (REQUIRED):
   def __init__(self, name: str, llm_config, dataset: DatasetType):
       self.name = name
       self.dataset = dataset
       self.llm = create_llm_instance(llm_config)
       # Initialize operators you will use
       self.answer_generate = AnswerGenerate(self.llm)
       self.review = Review(self.llm)

4. ✅ __call__ method (REQUIRED):
   async def __call__(self, problem: str) -> Tuple[str, float]:
       # Your workflow logic here

================================================================================
✅ COMPLETE CORRECT EXAMPLE (follow exactly):
================================================================================
```python
from scripts.operators import AnswerGenerate, Review, Revise
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # Initialize operators you will use
        self.answer_generate = AnswerGenerate(self.llm)
        self.review = Review(self.llm)
        self.revise = Revise(self.llm)

    async def __call__(self, problem: str) -> Tuple[str, float]:
        # Step 1: Generate answer with reasoning
        ans = await self.answer_generate(input=problem)
        answer = ans.get('answer', '')

        # Step 2: Optionally review the answer
        review = await self.review(problem=problem, solution=answer)

        # Step 3: If feedback suggests revision, revise
        if not review.get('review_result', True):
            revised = await self.revise(
                problem=problem,
                solution=answer,
                feedback=review.get('feedback', '')
            )
            answer = revised.get('solution', answer)

        # Step 4: Return answer and cost
        return answer, self.llm.get_usage_summary().get("total_cost", 0.0)
```

================================================================================
⚙️  OPERATOR INTERFACE REFERENCE (call operators EXACTLY like this):
================================================================================

1. self.answer_generate(input: str) -> dict with keys 'thought', 'answer'
   ❌ WRONG: await self.answer_generate(problem=problem)
   ✅ RIGHT: await self.answer_generate(input=problem)

2. self.review(problem: str, solution: str) -> dict with keys 'review_result', 'feedback'
   ❌ WRONG: await self.review(solution=answer)  # Missing 'problem'
   ✅ RIGHT: await self.review(problem=problem, solution=answer)

3. self.revise(problem: str, solution: str, feedback: str) -> dict with key 'solution'
   ❌ WRONG: await self.revise(solution=answer, feedback=feedback)
   ✅ RIGHT: await self.revise(problem=problem, solution=answer, feedback=feedback)

4. self.scensemble(solutions: List[str], problem: str) -> dict with key 'response'
   ✅ RIGHT: await self.scensemble(solutions=[answer1, answer2], problem=problem)

5. self.custom(input: str, instruction: str) -> dict with key 'response'
   ✅ RIGHT: await self.custom(input=problem, instruction="custom instruction")

================================================================================
✅ OPERATORS YOU CAN USE (for QA only):
================================================================================
- AnswerGenerate: Generate answer with reasoning
- Review: Review and validate answer
- Revise: Revise answer based on feedback
- ScEnsemble: Self-consistency ensemble (for multiple candidate answers)
- Custom: Custom prompting (for special cases only)

================================================================================
❌ OPERATORS YOU MUST NOT USE (for QA problems):
================================================================================
- Programmer: This is for CODE problems, NOT QA!
- Test: This is for CODE problems, NOT QA!

================================================================================
📋 REQUIRED SIGNATURE:
================================================================================
Your __call__ method MUST have exactly this signature:
    async def __call__(self, problem: str) -> Tuple[str, float]:

Parameters: only 'problem: str'
Returns: (answer_string, cost_float)

================================================================================
❌ WRONG EXAMPLES (DO NOT DO THIS):
================================================================================
WRONG #1: Only method body, no class definition
```python
async def __call__(self, problem: str):  # ❌ Missing class definition!
    ans = await self.answer_generate(input=problem)
```

WRONG #2: Missing operator initialization
```python
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        # ❌ Missing: self.answer_generate = AnswerGenerate(self.llm)
```

WRONG #3: Using Programmer/Test operators
```python
code = await self.programmer(problem=problem)  # ❌ WRONG! Use answer_generate instead
```

WRONG #4: Incorrect operator call parameters
```python
await self.answer_generate(problem=problem)  # ❌ WRONG! Parameter should be 'input', not 'problem'
await self.review(solution=answer)           # ❌ WRONG! Must include 'problem' parameter
```

================================================================================
🎯 PROBLEM TO SOLVE:
================================================================================
{problem}

================================================================================
📝 INSTRUCTIONS:
================================================================================
1. Generate a COMPLETE class with imports, class definition, __init__, and __call__
2. Use class Workflow: (NO inheritance)
3. Initialize self.llm = create_llm_instance(llm_config) in __init__
4. Initialize ONLY the operators you will use (e.g., self.answer_generate = AnswerGenerate(self.llm))
5. Follow the CORRECT EXAMPLE pattern above EXACTLY
6. Use ONLY the 5 allowed operators
7. Never use Programmer or Test operators
8. Call operators with the EXACT parameter names shown in "OPERATOR INTERFACE REFERENCE"
9. Ensure the method returns (answer, cost) tuple

BEGIN CODE GENERATION:
"""

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
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True   # 🚀 Performance Fix: Enable caching for 10-20x speedup
            )

        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # 解析输出（包含深度质量检查）
        workflow_code, is_valid, error, quality_check = self._parse_workflow_code(generated_text, problem_type)

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

    def generate_workflows_batch(
        self,
        problems: List[Dict],
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> List[Dict]:
        """
        批量生成工作流（8倍加速）

        用途：在GRPO训练中批量生成多个问题的工作流，显著加速

        Args:
            problems: [
                {'text': str, 'type': str},  # problem_type: 'math'/'code'/'qa'
                ...
            ]
            temperature: 采样温度
            max_new_tokens: 最大生成token数

        Returns:
            [{
                "workflow_code": str,
                "valid": bool,
                "error": Optional[str],
                "metadata": {...}
            }, ...]

        性能对比：
        - Sequential: N problems × 100ms/problem = 100N ms
        - Batch: 1 forward pass + N decode ≈ 300-500 ms (vs 10000-15000 ms)
        - 加速: 20-30x（取决于问题复杂度）
        """
        print(f"🚀 批量生成 {len(problems)} 个工作流...")

        # 1. 构建所有提示词
        prompts = []
        for problem in problems:
            prompt = self._build_generation_prompt(
                problem['text'],
                problem['type']
            )
            prompts.append(prompt)

        # 2. 批量tokenize（带padding）
        print(f"  📝 Tokenizing {len(prompts)} prompts...")
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,  # 自动padding到最长长度
            truncation=True,
            max_length=2048
        ).to(self.device)

        # 3. 批量生成
        print(f"  🔨 Generating...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.config.get('top_p', 0.95),
                top_k=self.config.get('top_k', 50),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # 关键：使用num_beams=1避免beam search的额外开销
                num_beams=1,
                use_cache=True   # 🚀 Performance Fix: Enable caching for 10-20x speedup
            )

        # 4. 批量解码和解析
        print(f"  🔍 Decoding and parsing...")
        results = []
        for i, (problem, output_seq) in enumerate(zip(problems, outputs)):
            try:
                # 解码：跳过输入部分，只取生成的部分
                input_length = inputs['input_ids'][i].shape[0]
                generated_text = self.tokenizer.decode(
                    output_seq[input_length:],
                    skip_special_tokens=True
                )

                # 解析工作流代码
                workflow_code, is_valid, error, quality_check = self._parse_workflow_code(
                    generated_text,
                    problem['type']
                )

                results.append({
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "problem": problem['text'],
                        "problem_type": problem['type'],
                        "temperature": temperature,
                        "tokens_generated": output_seq.shape[0] - input_length
                    }
                })

            except Exception as e:
                print(f"    ⚠️  Problem {i} 解析失败: {str(e)}")
                results.append({
                    "workflow_code": self._get_default_workflow(problem['type']),
                    "valid": False,
                    "error": str(e),
                    "metadata": {
                        "problem": problem['text'],
                        "problem_type": problem['type'],
                        "error_type": "parsing"
                    }
                })

        print(f"✅ 批量生成完成: {len(results)} 个工作流")
        return results

    def _parse_workflow_code(self, generated_text: str, problem_type: str) -> Tuple[str, bool, Optional[str], Dict]:
        """
        解析生成的文本，进行多层验证

        流程：
        1. 提取代码块
        2. 进行深度质量检查
        3. 使用WorkflowValidator进行验证
        4. 返回代码和详细的质量信息

        返回：(code, is_valid, error_msg, quality_check_result)
        """

        # DEBUG: 打印 Qwen 生成的原始文本
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG: Qwen 生成的原始文本 (完整):")
        print(f"{'='*60}")
        print(generated_text)  # 打印完整文本
        print(f"{'='*60}\n")

        quality_check = {'operators_used': [], 'issues': []}

        try:
            # 1. 提取代码块（支持markdown和纯代码格式）
            code = self._extract_code_block(generated_text)
            if not code:
                print(f"❌ 无法从生成文本中提取代码块")
                return self._get_default_workflow(problem_type), False, "No code block found", quality_check

            # 2. 进行深度质量检查（新增）
            print(f"\n📋 进行深度代码质量检查...")
            quality_check = self._validate_workflow_code(code, problem_type)

            # 打印质量检查结果
            print(f"  Syntax Error: {quality_check['has_syntax_error']}")
            print(f"  Has __call__: {quality_check['has_call_method']}")
            print(f"  Signature Correct: {quality_check['signature_correct']}")
            print(f"  Operators Valid: {quality_check['operators_valid']}")
            print(f"  Has Return: {quality_check['has_return_statement']}")
            if quality_check['operators_used']:
                print(f"  Operators Used: {quality_check['operators_used']}")
            if quality_check['issues']:
                print(f"  Issues Detected:")
                for issue in quality_check['issues']:
                    print(f"    - {issue}")

            # Phase 1 修复（根本性修复 - 在执行前自动修复代码结构问题）
            print(f"\n🔧 应用根本性代码修复 (Phase 1)...")
            code = self._enforce_correct_structure(code, problem_type)
            code = self._fix_operator_calls(code, problem_type)
            print(f"✅ 代码修复完成，现在验证...")

            # 3. 使用WorkflowValidator进行验证
            print(f"\n🔧 使用WorkflowValidator进行验证...")
            fixed_code, is_valid, error_msg, fixes = self.validator.validate_and_fix_workflow(
                code=code,
                problem_type=problem_type
            )

            if is_valid:
                print(f"✅ 验证成功")
                if fixes:
                    print(f"   应用了以下修复: {fixes}")
                return fixed_code, True, None, quality_check
            else:
                print(f"❌ 验证失败: {error_msg}")

                # 检查是否为严重错误（语法错误）
                if quality_check['has_syntax_error']:
                    print(f"❌ 严重问题：语法错误，无法执行")
                    return self._get_default_workflow(problem_type), False, error_msg, quality_check

                # 其他错误：保留原始代码，让模型从执行错误中学习
                print(f"🎯 策略：保留原始代码，通过执行错误反馈让模型学习")
                try:
                    compile(code, '<string>', 'exec')
                    print(f"✅ 原始代码可编译，将执行并从错误中学习")
                    return code, False, error_msg, quality_check
                except SyntaxError as syntax_error:
                    print(f"❌ 原始代码有语法错误: {syntax_error}")
                    return self._get_default_workflow(problem_type), False, f"Syntax error: {syntax_error}", quality_check
                except:
                    print(f"❌ 编译失败，使用默认工作流")
                    return self._get_default_workflow(problem_type), False, "Compilation failed", quality_check

        except Exception as e:
            print(f"❌ 异常捕获: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_default_workflow(problem_type), False, str(e), quality_check

    def _get_default_workflow(self, problem_type: str = "math") -> str:
        """默认工作流（当生成失败时）"""
        return f"""from scripts.operators import Custom, AnswerGenerate, Programmer, Test, Review, Revise, ScEnsemble
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.custom(input=problem, instruction="Solve this problem step by step.")
        return solution['response'], self.llm.get_usage_summary().get("total_cost", 0.0)
"""

    def _validate_workflow_code(self, code: str, problem_type: str) -> Dict[str, bool]:
        """
        深度代码质量检查 - 在执行前检测问题

        返回字典包含：
        {
            'has_syntax_error': bool,           # 有语法错误？
            'has_call_method': bool,            # 有async def __call__？
            'signature_correct': bool,          # 签名正确？
            'operators_used': [list],           # 使用了哪些operators
            'operators_valid': bool,            # operators 对问题类型有效？
            'operator_calls_valid': bool,       # operator 调用参数合理？
            'has_return_statement': bool,       # 有return语句？
            'issues': [list]                    # 发现的所有问题列表
        }
        """
        import re
        import ast

        issues = []
        result = {
            'has_syntax_error': False,
            'has_call_method': False,
            'signature_correct': False,
            'operators_used': [],
            'operators_valid': False,
            'operator_calls_valid': True,
            'has_return_statement': False,
            'issues': issues
        }

        # ===== Check 1: Syntax Error =====
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            result['has_syntax_error'] = True
            issues.append(f"语法错误: {str(e)}")
            return result

        # ===== Check 2: Has async def __call__ =====
        if 'async def __call__' not in code and 'def __call__' not in code:
            issues.append("缺少 async def __call__ 方法")
            return result

        result['has_call_method'] = True

        # ===== Check 3: Correct Signature =====
        if problem_type == "math":
            pattern = r'async\s+def\s+__call__\s*\(\s*self\s*,\s*problem\s*:\s*str\s*\)'
            if re.search(pattern, code):
                result['signature_correct'] = True
            else:
                issues.append(f"MATH问题的签名错误。应该是: async def __call__(self, problem: str)")
        elif problem_type == "code":
            pattern = r'async\s+def\s+__call__\s*\(\s*self\s*,\s*problem\s*:\s*str\s*,\s*entry_point\s*:\s*str\s*,\s*test\s*:\s*str\s*\)'
            if re.search(pattern, code):
                result['signature_correct'] = True
            else:
                issues.append(f"CODE问题的签名错误。应该是: async def __call__(self, problem: str, entry_point: str, test: str)")
        elif problem_type == "qa":
            pattern = r'async\s+def\s+__call__\s*\(\s*self\s*,\s*problem\s*:\s*str\s*\)'
            if re.search(pattern, code):
                result['signature_correct'] = True
            else:
                issues.append(f"QA问题的签名错误。应该是: async def __call__(self, problem: str)")

        # ===== Check 3.5: Verify operator initialization =====
        # 生成的代码应该从基类继承所有operators，但需要验证它们确实被使用
        # 如果生成的code中出现 `self.llm`, `self.review` 等属性访问，说明基类初始化工作正常
        init_keywords = {
            'llm': r'self\.llm',  # 所有问题类型都需要 llm
            'review': r'self\.review',  # MATH 和 QA 需要 review
            'revise': r'self\.revise',  # MATH 和 QA 需要 revise
            'programmer': r'self\.programmer',  # CODE 需要 programmer
            'test': r'self\.test',  # CODE 需要 test
            'answer_generate': r'self\.answer_generate',  # MATH 和 QA 需要
        }

        # 记录初始化的operators
        initialized_operators = []
        for op_name, op_pattern in init_keywords.items():
            if re.search(op_pattern, code):
                initialized_operators.append(op_name)

        # 验证问题类型所需的operators是否都被初始化了
        required_operators = {
            'math': ['llm', 'review', 'revise'],
            'code': ['llm', 'programmer', 'test'],
            'qa': ['llm', 'review', 'revise'],
        }

        missing_operators = []
        for req_op in required_operators.get(problem_type, []):
            if req_op not in initialized_operators:
                missing_operators.append(req_op)

        if missing_operators:
            issues.append(f"⚠️  缺少必需的operators初始化: {', '.join(missing_operators)}")

        # ===== Check 4: Extract operators used =====
        operator_keywords = {
            'answer_generate': r'await\s+self\.answer_generate\s*\(',
            'programmer': r'await\s+self\.programmer\s*\(',
            'test': r'await\s+self\.test\s*\(',
            'review': r'await\s+self\.review\s*\(',
            'revise': r'await\s+self\.revise\s*\(',
            'scensemble': r'await\s+self\.scensemble\s*\(',
            'custom': r'await\s+self\.custom\s*\(',
        }

        for op_name, op_pattern in operator_keywords.items():
            if re.search(op_pattern, code):
                result['operators_used'].append(op_name)

        # ===== Check 5: Operators valid for problem type =====
        valid_operators = {
            'math': ['answer_generate', 'review', 'revise', 'scensemble', 'custom'],
            'code': ['programmer', 'test', 'review', 'revise', 'custom'],
            'qa': ['answer_generate', 'review', 'revise', 'scensemble', 'custom'],
        }

        invalid_ops = [op for op in result['operators_used'] if op not in valid_operators.get(problem_type, [])]
        if invalid_ops:
            result['operators_valid'] = False
            for op in invalid_ops:
                issues.append(f"❌ Operator '{op}' 不适合 {problem_type} 问题")
        else:
            if result['operators_used']:  # 有operators且都有效
                result['operators_valid'] = True

        # ===== Check 6: Operator call parameters =====
        # 检查常见的参数错误
        param_checks = [
            (r'answer_generate\s*\(\s*problem\s*=', "answer_generate: 应该用 'input' 参数，不是 'problem'"),
            (r'review\s*\(\s*solution\s*=(?![^)]*problem)', "review: 缺少 'problem' 参数"),
            (r'revise\s*\(\s*(?!.*problem)(?!.*solution)(?!.*feedback)', "revise: 缺少必要参数（problem/solution/feedback）"),
            (r'test\s*\(\s*(?!.*entry_point)', "test: 缺少 'entry_point' 参数"),
        ]

        for pattern, error_msg in param_checks:
            # 这是简化的检查，避免复杂的正则
            if re.search(pattern, code):
                issues.append(f"⚠️  {error_msg}")
                result['operator_calls_valid'] = False

        # ===== Check 7: Has return statement =====
        # 检查是否有return语句返回元组
        if re.search(r'return\s+\w+\s*,\s*self\.llm', code) or re.search(r'return\s+\(.*?,.*?\)', code):
            result['has_return_statement'] = True
        else:
            issues.append("⚠️  缺少或错误的 return 语句（应返回 (result, cost) 元组）")

        return result

    def _extract_code_block(self, generated_text: str) -> str:
        """
        从生成的文本中提取Python代码块 - 多策略提取，确保鲁棒性

        支持格式：
        1. Markdown: ```python ... ```
        2. Markdown: ``` ... ``` (flexible newlines)
        3. 纯代码（没有包裹）
        4. class Workflow 定义
        5. def __call__ 方法体

        策略：尝试多种模式，逐步降低严格性，确保总能提取到代码
        """
        import re

        # ===== Strategy 1: Markdown ```python...``` with flexible spacing =====
        # 支持 ```python\n...``` 和 ```python...``` 两种格式
        patterns_markdown_python = [
            r'```python\s*\n(.*?)\n```',  # ```python\n...code...\n```
            r'```python\s*(.*?)\n```',    # ```python...code...\n```
            r'```python\s*\n(.*?)```',    # ```python\n...code...```
            r'```python\s*(.*?)```',      # ```python...code...```
        ]

        for pattern in patterns_markdown_python:
            match = re.search(pattern, generated_text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if code:
                    return code

        # ===== Strategy 2: Markdown ```...``` with flexible spacing =====
        patterns_markdown_general = [
            r'```\s*\n(.*?)\n```',        # ```\n...code...\n```
            r'```\s*(.*?)\n```',          # ```...code...\n```
            r'```\s*\n(.*?)```',          # ```\n...code...```
            r'```\s*(.*?)```',            # ```...code...```
        ]

        for pattern in patterns_markdown_general:
            match = re.search(pattern, generated_text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                # 过滤掉明显的非代码文本
                if not code.startswith(('Here', 'This', 'The', 'For', 'In', 'We')):
                    if code and any(kw in code for kw in ['def', 'class', 'await', 'async', 'return']):
                        return code

        # ===== Strategy 3: Look for class Workflow definition =====
        class_pattern = r'class\s+Workflow\s*:.*?(?=\n(?:class|def\s+\w+\s*\(|\Z))'
        match = re.search(class_pattern, generated_text, re.DOTALL)
        if match:
            code = match.group(0).strip()
            if code:
                return code

        # ===== Strategy 4: Look for async def __call__ =====
        call_pattern = r'async\s+def\s+__call__\s*\(.*?\):\s*(?:->.*?)?\n(.*?)(?=\n(?:async\s+def|def\s+\w+\s*\(|\Z))'
        match = re.search(call_pattern, generated_text, re.DOTALL)
        if match:
            # 只提取方法体
            method_body = match.group(1).strip()
            # 需要返回完整的async def...，所以重新构建
            match_full = re.search(r'(async\s+def\s+__call__\s*\(.*?\):.*?)(?=\n(?:async\s+def|def\s+\w+\s*\(|\Z))',
                                  generated_text, re.DOTALL)
            if match_full:
                code = match_full.group(1).strip()
                if code:
                    return code

        # ===== Strategy 5: Extract lines containing code keywords =====
        lines = generated_text.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # 检查是否进入代码区域
            if any(kw in line for kw in ['class Workflow', 'async def __call__', 'def __call__']):
                in_code = True

            if in_code:
                code_lines.append(line)
                # 简单的启发式：连续空行表示代码结束
                if len(code_lines) > 10 and line.strip() == '' and code_lines[-2].strip() == '':
                    code_lines.pop()  # 移除最后的空行
                    break

        if code_lines:
            code = '\n'.join(code_lines).strip()
            if code and len(code) > 50:  # 确保提取的代码有合理的长度
                return code

        # ===== Strategy 6: Fallback - return all text if it looks like code =====
        if any(keyword in generated_text for keyword in ['class Workflow', 'def __call__', 'async def', 'await', 'return']):
            return generated_text.strip()

        # ===== Strategy 7: Last resort - empty string =====
        return ""

    def _enforce_correct_structure(self, code: str, problem_type: str) -> str:
        """
        强制修复代码结构缺陷（自包含架构），确保：
        1. 有 class Workflow: 定义（无继承）
        2. 有正确的 __init__ 初始化 self.llm
        3. 有 async def __call__() 方法
        4. 自动初始化缺失的operators

        这是根本性修复，不是补丁
        """
        import re

        # Step 1: 检查是否完全缺少class定义
        if not re.search(r'class\s+Workflow', code):
            print(f"⚠️  代码缺少class定义，进行修复...")
            # 提取 __call__ 方法体
            call_match = re.search(r'async\s+def\s+__call__\s*\([^)]*\)\s*(?:->\s*[^\:]+)?\s*:', code)

            if call_match:
                # 找到方法体的开始位置
                method_start = call_match.end()
                call_body = code[method_start:].strip()
                # 重新构建为完整的自包含class
                fixed_code = self._wrap_in_selfcontained_class(call_body, problem_type)
                return fixed_code
            else:
                # 如果连 __call__ 都找不到，返回原始代码并标记
                return code

        # Step 2: 移除旧的继承模式（如果存在）
        inheritance_pattern = r'class\s+Workflow\s*\([^)]*\)'
        if re.search(inheritance_pattern, code):
            print(f"⚠️  检测到旧的继承模式，转换为自包含架构...")
            code = re.sub(
                inheritance_pattern,
                'class Workflow',
                code
            )
            # 移除super()调用
            code = re.sub(
                r'\s*super\(\).__init__\([^)]*\)\s*\n',
                '',
                code
            )

        # Step 3: 确保 __init__ 初始化了 self.llm
        if 'def __init__' in code:
            if 'self.llm = create_llm_instance' not in code:
                print(f"⚠️  __init__ 缺少self.llm初始化，进行修复...")
                # 在 __init__ 方法体的开始处添加基本初始化
                code = re.sub(
                    r'(def __init__\s*\([^)]*\)\s*:\s*\n)',
                    r'\1        self.name = name\n        self.dataset = dataset\n        self.llm = create_llm_instance(llm_config)\n',
                    code
                )

        # Step 4: 自动初始化缺失的operators
        code = self._auto_initialize_operators(code, problem_type)

        return code

    def _auto_initialize_operators(self, code: str, problem_type: str) -> str:
        """
        自动添加缺失的operator初始化

        检测在__call__中使用但在__init__中未初始化的operators，
        并自动添加初始化代码
        """
        import re

        # 查找使用的operators
        used = self._find_used_operators(code)

        # 查找已初始化的operators
        initialized = self._find_initialized_operators(code)

        # 找出缺失的operators
        missing = set(used) - set(initialized)

        if not missing:
            return code

        print(f"✅ Auto-initializing missing operators: {', '.join(missing)}")

        # Operator初始化映射
        operator_map = {
            'answer_generate': 'self.answer_generate = AnswerGenerate(self.llm)',
            'programmer': 'self.programmer = Programmer(self.llm)',
            'test': 'self.test = Test(self.llm)',
            'review': 'self.review = Review(self.llm)',
            'revise': 'self.revise = Revise(self.llm)',
            'scensemble': 'self.scensemble = ScEnsemble(self.llm)',
            'custom': 'self.custom = Custom(self.llm)',
        }

        # 找到__init__方法并添加初始化
        init_pattern = r'(def __init__\s*\([^)]*\)\s*:.*?)((?=\n    async def)|(?=\n    def)|$)'

        def add_inits(match):
            init_body = match.group(1)
            rest = match.group(2)

            # 添加缺失的operator初始化
            for op in sorted(missing):  # 排序以保证一致性
                if op in operator_map:
                    init_body += f"\n        {operator_map[op]}"

            return init_body + rest

        code = re.sub(init_pattern, add_inits, code, flags=re.DOTALL)

        return code

    def _find_used_operators(self, code: str) -> list:
        """查找在__call__中使用的operators"""
        import re

        operator_keywords = {
            'answer_generate': r'self\.answer_generate\s*\(',
            'programmer': r'self\.programmer\s*\(',
            'test': r'self\.test\s*\(',
            'review': r'self\.review\s*\(',
            'revise': r'self\.revise\s*\(',
            'scensemble': r'self\.scensemble\s*\(',
            'custom': r'self\.custom\s*\(',
        }

        used = []
        for op_name, pattern in operator_keywords.items():
            if re.search(pattern, code):
                used.append(op_name)

        return used

    def _find_initialized_operators(self, code: str) -> list:
        """查找在__init__中初始化的operators"""
        import re

        patterns = {
            'answer_generate': r'self\.answer_generate\s*=\s*AnswerGenerate',
            'programmer': r'self\.programmer\s*=\s*Programmer',
            'test': r'self\.test\s*=\s*Test',
            'review': r'self\.review\s*=\s*Review',
            'revise': r'self\.revise\s*=\s*Revise',
            'scensemble': r'self\.scensemble\s*=\s*ScEnsemble',
            'custom': r'self\.custom\s*=\s*Custom',
        }

        initialized = []
        for op_name, pattern in patterns.items():
            if re.search(pattern, code):
                initialized.append(op_name)

        return initialized

    def _wrap_in_selfcontained_class(self, call_body: str, problem_type: str) -> str:
        """将__call__方法体包装成完整的自包含class"""
        # 根据问题类型确定需要的operators
        if problem_type == "code":
            imports = "from scripts.operators import Programmer, Test, Review, Revise, Custom"
            signature = "async def __call__(self, problem: str, entry_point: str, test: str)"
            operators_init = """
        self.programmer = Programmer(self.llm)
        self.test = Test(self.llm)"""
        else:  # math, qa
            imports = "from scripts.operators import AnswerGenerate, Review, Revise, ScEnsemble, Custom"
            signature = "async def __call__(self, problem: str)"
            operators_init = """
        self.answer_generate = AnswerGenerate(self.llm)
        self.review = Review(self.llm)"""

        return f"""{imports}
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config){operators_init}

    {signature}:
{call_body}
"""

    def _fix_operator_calls(self, code: str, problem_type: str) -> str:
        """
        自动修复常见的operator调用参数错误

        Examples:
        - answer_generate(problem=...) -> answer_generate(input=...)
        - review(solution=...) without problem -> review(problem=..., solution=...)
        """
        import re

        # Fix 1: answer_generate 参数 - 所有问题类型都可能用到
        if problem_type in ['math', 'qa']:
            # answer_generate(problem=...) 应该是 answer_generate(input=...)
            code = re.sub(
                r'answer_generate\s*\(\s*problem\s*=',
                'answer_generate(input=',
                code
            )
            # answer_generate(x) 应该改为 answer_generate(input=x)
            code = re.sub(
                r'answer_generate\s*\(\s*([a-zA-Z_]\w*)\s*\)(?![=\w])',
                r'answer_generate(input=\1)',
                code
            )

        # Fix 2: review 参数 - 必须有 problem 和 solution
        # review(solution=...) -> review(problem=..., solution=...)
        code = re.sub(
            r'review\s*\(\s*solution\s*=\s*([^,\)]+)\s*\)(?![=\w])',
            r'review(problem=problem, solution=\1)',
            code
        )

        # review(x) -> review(problem=problem, solution=x)
        code = re.sub(
            r'review\s*\(\s*([a-zA-Z_]\w*)\s*\)(?![=\w])',
            r'review(problem=problem, solution=\1)',
            code
        )

        # Fix 3: revise 参数 - 必须有 problem, solution, feedback
        # revise(solution=..., feedback=...) -> revise(problem=..., solution=..., feedback=...)
        code = re.sub(
            r'revise\s*\(\s*solution\s*=',
            r'revise(problem=problem, solution=',
            code
        )

        # Fix 4: test 参数（CODE问题） - 必须有 problem, solution, entry_point
        if problem_type == 'code':
            # test(solution=..., entry_point=...) -> test(problem=..., solution=..., entry_point=...)
            code = re.sub(
                r'test\s*\(\s*solution\s*=',
                r'test(problem=problem, solution=',
                code
            )

        # Fix 5: scensemble 参数（MATH/QA）- 必须有 solutions 和 problem
        if problem_type in ['math', 'qa']:
            # scensemble(x) -> scensemble(solutions=x, problem=problem)
            code = re.sub(
                r'scensemble\s*\(\s*([a-zA-Z_]\w*)\s*\)(?![=\w])',
                r'scensemble(solutions=\1, problem=problem)',
                code
            )

        return code

    # 旧的继承相关方法已移除，使用_wrap_in_selfcontained_class代替

    def _get_call_signature(self, problem_type: str) -> str:
        """获取问题类型对应的 __call__ 签名"""
        if problem_type == "code":
            return "__call__(self, problem: str, entry_point: str, test: str) -> Tuple[str, float]"
        else:  # math, qa
            return "__call__(self, problem: str) -> Tuple[str, float]"

    def _indent_code(self, code: str, spaces: int) -> str:
        """为代码块添加缩进"""
        indent = ' ' * spaces
        lines = code.split('\n')
        indented_lines = [indent + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)


def test_generator():
    """测试生成器"""
    print("\n" + "=" * 60)
    print("🧪 测试RL工作流生成器")
    print("=" * 60)

    # 注意：这需要Qwen模型，如果没有下载会很慢
    generator = RLWorkflowGenerator(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        device_ids=[2, 3],
        operator_descriptions_path="/home/yijia/.claude/11/AFlow/workspace/MATH/workflows/template/operator.json"
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
