#!/usr/bin/env python3
"""
RL工作流生成器 - 使用RL训练的Qwen2.5-7B生成优化的工作流

设计原则：
1. 支持模型共享（高效，用于GRPO训练）和独立模式
2. Prompt设计清晰简洁（问题类型特定）
3. 验证和修复逻辑统一而不是分散
4. 方法职责清晰，避免过度复杂化
"""
import torch
import json
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys
import os
import re


class RLWorkflowGenerator:
    """使用RL训练的Qwen2.5-7B生成优化的工作流"""

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_checkpoint: Optional[str] = None,
        device_ids: List[int] = None,
        operator_descriptions_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        初始化工作流生成器

        Args:
            model: 共享的模型实例（优先使用）
            tokenizer: 共享的tokenizer实例（优先使用）
            device: 共享的设备（优先使用）
            base_model: 基座模型路径（仅在model=None时使用）
            lora_checkpoint: LoRA检查点路径
            device_ids: GPU ID列表（仅在device=None时使用）
            operator_descriptions_path: AFlow算子描述文件路径
            config: 额外配置
        """
        self.config = config or {}

        # 模型初始化：优先使用共享模型
        if model is not None:
            self._init_shared_model(model, tokenizer, device)
        else:
            self._init_standalone_model(base_model, lora_checkpoint, device_ids or [2, 3])

        # 加载算子描述
        self.operator_descriptions = self._load_operator_descriptions(operator_descriptions_path)
        print(f"✅ RL工作流生成器初始化完成")

    def _init_shared_model(self, model: Any, tokenizer: Any, device: str) -> None:
        """初始化共享模型模式（用于GRPO训练）"""
        print(f"🔧 初始化工作流生成器（模型共享模式）")
        print(f"   共享模型ID: {id(model)}")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        print(f"   设备: {self.device}")
        print(f"   ✅ 节省~15GB GPU内存")

    def _init_standalone_model(self, base_model: str, lora_checkpoint: Optional[str], device_ids: List[int]) -> None:
        """初始化独立模型模式"""
        print(f"🔧 初始化工作流生成器（独立模式）")

        # 设置CUDA设备
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))

        self.device = f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
        print(f"   设备: {self.device}")
        print(f"   GPU: {device_ids}")

        # 加载tokenizer和模型
        print(f"📥 加载tokenizer: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"📥 加载模型: {base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True
        )

        # 加载LoRA权重
        if lora_checkpoint:
            print(f"📥 加载LoRA检查点: {lora_checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
            self.model.eval()

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
                "interface": "programmer(problem: str, analysis: str) -> dict with keys 'code' and 'output'"
            },
            "Test": {
                "description": "Tests code with test cases.",
                "interface": "test(problem: str, solution: str, entry_point: str) -> dict with 'result' and 'solution'"
            },
            "Review": {
                "description": "Reviews and provides feedback on a solution.",
                "interface": "review(problem: str, solution: str) -> dict with keys 'review_result' and 'feedback'"
            },
            "Revise": {
                "description": "Revises solution based on feedback.",
                "interface": "revise(problem: str, solution: str, feedback: str) -> dict with key 'solution'"
            },
            "ScEnsemble": {
                "description": "Uses self-consistency to select the most frequent solution.",
                "interface": "scensemble(solutions: List[str], problem: str) -> dict with key 'response'"
            }
        }

    def _build_generation_prompt(self, problem: str, problem_type: str) -> str:
        """构建问题特定的生成Prompt"""

        if problem_type == "code":
            return self._build_code_prompt(problem)
        else:  # math, qa
            return self._build_math_qa_prompt(problem)

    def _build_code_prompt(self, problem: str) -> str:
        """CODE问题专用Prompt"""
        return f"""Generate a Python Workflow class to solve this CODE problem.

CRITICAL STRUCTURE (MUST follow exactly):
- class Workflow must inherit from CodeWorkflowBase
- __init__ must call super().__init__(name, llm_config, dataset)
- __call__ signature: async def __call__(self, problem: str, entry_point: str, test: str) -> Tuple[str, float]

Available Operators:
1. self.programmer(problem: str, analysis: str) -> {{'code': str, 'output': str}}
2. self.test(problem: str, solution: str, entry_point: str, test_loop: int) -> {{'result': bool, 'solution': str}}
3. self.review(problem: str, solution: str) -> {{'review_result': bool, 'feedback': str}}
4. self.revise(problem: str, solution: str, feedback: str) -> {{'solution': str}}
5. self.custom(input: str, instruction: str) -> {{'response': str}}

✅ CORRECT Example:
```python
class Workflow(CodeWorkflowBase):
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        super().__init__(name, llm_config, dataset)

    async def __call__(self, problem: str, entry_point: str, test: str) -> Tuple[str, float]:
        self._test_input = test
        code_result = await self.programmer(problem=problem, analysis='')
        code = code_result.get('code', '')

        test_result = await self.test(problem=problem, solution=code, entry_point=entry_point, test_loop=3)
        if test_result.get('result', False):
            return test_result.get('solution', code), self.llm.get_usage_summary()["total_cost"]

        review = await self.review(problem=problem, solution=code)
        if not review.get('review_result', True):
            revised = await self.revise(problem=problem, solution=code, feedback=review.get('feedback', ''))
            code = revised.get('solution', code)

        return code, self.llm.get_usage_summary()["total_cost"]
```

PROBLEM:
{problem}

Generate the complete class now:
"""

    def _build_math_qa_prompt(self, problem: str) -> str:
        """MATH/QA问题专用Prompt"""
        return f"""Generate a Python Workflow class to solve this problem.

CRITICAL STRUCTURE (MUST follow exactly):
- class Workflow must inherit from MathWorkflowBase (for MATH) or QAWorkflowBase (for QA)
- __init__ must call super().__init__(name, llm_config, dataset)
- __call__ signature: async def __call__(self, problem: str) -> Tuple[str, float]

Available Operators:
1. self.answer_generate(input: str) -> {{'thought': str, 'answer': str}}
2. self.review(problem: str, solution: str) -> {{'review_result': bool, 'feedback': str}}
3. self.revise(problem: str, solution: str, feedback: str) -> {{'solution': str}}
4. self.scensemble(solutions: List[str], problem: str) -> {{'response': str}}
5. self.custom(input: str, instruction: str) -> {{'response': str}}

✅ CORRECT Example:
```python
class Workflow(MathWorkflowBase):
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        super().__init__(name, llm_config, dataset)

    async def __call__(self, problem: str) -> Tuple[str, float]:
        ans = await self.answer_generate(input=problem)
        answer = ans.get('answer', '')

        review = await self.review(problem=problem, solution=answer)
        if not review.get('review_result', True):
            revised = await self.revise(problem=problem, solution=answer, feedback=review.get('feedback', ''))
            answer = revised.get('solution', answer)

        return answer, self.llm.get_usage_summary()["total_cost"]
```

PROBLEM:
{problem}

Generate the complete class now:
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
        """生成优化的工作流"""

        # 构建或使用自定义Prompt
        prompt = custom_prompt or self._build_generation_prompt(problem, problem_type)

        # 生成
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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

        # 解析和验证
        workflow_code, is_valid, error_msg = self._parse_workflow_code(generated_text, problem_type)

        result = {
            "workflow_code": workflow_code,
            "valid": is_valid,
            "error": error_msg,
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

        # 打印原始输出（DEBUG）
        print(f"\n{'='*60}")
        print(f"🔍 生成的原始文本:")
        print(f"{'='*60}")
        print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
        print(f"{'='*60}\n")

        # 提取代码块
        code = self._extract_code_block(generated_text)
        if not code:
            print(f"❌ 无法提取代码块，使用默认工作流")
            return self._get_default_workflow(problem_type), False, "No code block found"

        # 验证和修复
        code = self._validate_and_fix_workflow(code, problem_type)

        # 验证语法
        try:
            ast.parse(code)
            print(f"✅ 代码验证成功")
            return code, True, None
        except SyntaxError as e:
            print(f"❌ 语法错误: {str(e)}")
            return self._get_default_workflow(problem_type), False, f"Syntax error: {str(e)}"

    def _extract_code_block(self, text: str) -> str:
        """从文本中提取Python代码块"""

        # 策略1：查找```python```标记
        match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 策略2：查找class Workflow定义
        match = re.search(r'(class Workflow.*?)(?=\n(?:class|def|$)|\Z)', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 策略3：如果代码看起来完整就返回
        if 'class Workflow' in text and 'async def __call__' in text:
            # 从class开始到文本末尾
            start = text.find('class Workflow')
            return text[start:].strip()

        return ""

    def _validate_and_fix_workflow(self, code: str, problem_type: str) -> str:
        """验证并自动修复workflow代码"""

        # 修复1：确保有完整的class定义
        code = self._enforce_correct_structure(code, problem_type)

        # 修复2：修复常见的operator参数错误
        code = self._fix_operator_parameters(code, problem_type)

        return code

    def _enforce_correct_structure(self, code: str, problem_type: str) -> str:
        """强制修复代码结构缺陷"""

        base_class = {
            'math': 'MathWorkflowBase',
            'code': 'CodeWorkflowBase',
            'qa': 'QAWorkflowBase'
        }.get(problem_type, 'MathWorkflowBase')

        # 检查是否有class定义
        if not re.search(r'class\s+Workflow', code):
            # 只有__call__方法体，需要包装
            if 'async def __call__' in code:
                call_match = re.search(r'(async def __call__.*?:)(.*)', code, re.DOTALL)
                if call_match:
                    call_body = call_match.group(2)
                    code = f"""class Workflow({base_class}):
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        super().__init__(name, llm_config, dataset)

    async def __call__(self, problem: str) -> Tuple[str, float]:
{self._indent(call_body, 8)}
"""
                    print(f"✅ 已修复：添加class定义和__init__")

        # 检查继承
        if not re.search(rf'class\s+Workflow\s*\(\s*{base_class}\s*\)', code):
            code = re.sub(
                r'class\s+Workflow\s*\(\s*[^)]*\s*\)',
                f'class Workflow({base_class})',
                code
            )
            print(f"✅ 已修复：class继承")

        # 检查super()调用
        if 'def __init__' in code and 'super().__init__' not in code:
            code = re.sub(
                r'(def __init__\s*\([^)]*\)\s*:\s*\n)',
                r'\1        super().__init__(name, llm_config, dataset)\n',
                code
            )
            print(f"✅ 已修复：super().__init__() 调用")

        return code

    def _fix_operator_parameters(self, code: str, problem_type: str) -> str:
        """修复常见的operator参数错误"""

        # 修复answer_generate参数
        if problem_type in ['math', 'qa']:
            code = re.sub(r'answer_generate\s*\(\s*problem\s*=', 'answer_generate(input=', code)

        # 修复review缺少problem参数
        code = re.sub(
            r'review\s*\(\s*solution\s*=\s*([^,\)]+)\s*\)',
            r'review(problem=problem, solution=\1)',
            code
        )

        # 修复revise缺少problem参数
        code = re.sub(
            r'revise\s*\(\s*solution\s*=',
            r'revise(problem=problem, solution=',
            code
        )

        # 修复test缺少problem参数
        if problem_type == 'code':
            code = re.sub(
                r'test\s*\(\s*solution\s*=',
                r'test(problem=problem, solution=',
                code
            )

        return code

    @staticmethod
    def _indent(text: str, spaces: int) -> str:
        """为代码块添加缩进"""
        indent = ' ' * spaces
        lines = text.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)

    def _get_default_workflow(self, problem_type: str = "math") -> str:
        """默认工作流"""
        base_class = {
            'math': 'MathWorkflowBase',
            'code': 'CodeWorkflowBase',
            'qa': 'QAWorkflowBase'
        }.get(problem_type, 'MathWorkflowBase')

        if problem_type == 'code':
            return f"""class Workflow(CodeWorkflowBase):
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        super().__init__(name, llm_config, dataset)

    async def __call__(self, problem: str, entry_point: str, test: str) -> Tuple[str, float]:
        self._test_input = test
        code_result = await self.programmer(problem=problem, analysis='')
        return code_result.get('code', ''), self.llm.get_usage_summary()["total_cost"]
"""
        else:
            return f"""class Workflow({base_class}):
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        super().__init__(name, llm_config, dataset)

    async def __call__(self, problem: str) -> Tuple[str, float]:
        ans = await self.answer_generate(input=problem)
        return ans.get('answer', ''), self.llm.get_usage_summary()["total_cost"]
"""


def test_generator():
    """测试生成器"""
    print("\n" + "=" * 60)
    print("🧪 测试RL工作流生成器")
    print("=" * 60)

    generator = RLWorkflowGenerator(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        device_ids=[2, 3]
    )

    test_problem = "What is 15 + 27?"
    print(f"\n📝 测试问题: {test_problem}")

    result = generator.generate_workflow(
        problem=test_problem,
        problem_type="math",
        temperature=0.7,
        max_new_tokens=1024
    )

    print(f"\n✅ 结果:")
    print(f"   有效: {result['valid']}")
    if result['error']:
        print(f"   错误: {result['error']}")
    print(f"\n📄 代码:")
    print(result['workflow_code'])


if __name__ == "__main__":
    test_generator()
