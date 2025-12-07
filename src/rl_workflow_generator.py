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
        """Generate workflow prompt in clean XML format"""

        # Base prompt with XML structure and operator definitions
        base_prompt = f"""You are building a Workflow to solve {problem_type} problems.

You MUST output in the following XML format:

```xml
<workflow>
  <graph>
    class Workflow:
        def __init__(self, name: str, llm_config, dataset):
            self.name = name
            self.dataset = dataset
            self.llm = create_llm_instance(llm_config)
            # Initialize operators you will use

        async def __call__(self, problem: str, entry_point: str = "solve"):
            # Chain operators and return (answer, cost) tuple
            result = await self.operator(input, instruction)
            return final_answer, self.llm.get_usage_summary()["total_cost"]
  </graph>
  <prompt>
    TASK_PROMPT = '''Task-specific prompt here'''
  </prompt>
</workflow>
```

## Available Operators

Custom(input: str, instruction: str) -> {{'response': str}}
AnswerGenerate(input: str) -> {{'thought': str, 'answer': str}}
Programmer(problem: str, analysis: str) -> {{'code': str, 'output': str}}
Test(problem: str, solution: str, entry_point: str) -> {{'result': bool, 'solution': str}}
Review(problem: str, solution: str) -> {{'review_result': bool, 'feedback': str}}
Revise(problem: str, solution: str, feedback: str) -> {{'solution': str}}
ScEnsemble(solutions: List[str], problem: str) -> {{'response': str}}

## Core Rules
- Use .get('key', default) for safe dictionary access
- Always return (answer, cost) tuple
- Initialize all operators before using
- Import only operators you actually use"""

        # Problem-type-specific constraints
        if problem_type == "code":
            type_constraint = """

## CODE Problem Specific Rules
- async def __call__(self, problem: str, entry_point: str, test: str)
- MUST use Programmer to generate code
- MUST use Test to execute with test cases
- Return (solution, cost) tuple

Required Pattern:
from scripts.operators import Programmer, Test
class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        self.programmer = Programmer(self.llm)
        self.test = Test(self.llm)

    async def __call__(self, problem: str, entry_point: str, test: str):
        code_result = await self.programmer(problem=problem, analysis="")
        test_result = await self.test(problem=problem, solution=code_result['code'], entry_point=entry_point)
        return test_result['solution'] if test_result['result'] else code_result['output'], self.llm.get_usage_summary()["total_cost"]"""

        elif problem_type == "math":
            type_constraint = """

## MATH Problem Specific Rules
- async def __call__(self, problem: str)
- Use AnswerGenerate for step-by-step reasoning
- DO NOT use Programmer or Test
- Return (answer, cost) tuple

Required Pattern:
from scripts.operators import AnswerGenerate
class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        self.answer_generate = AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        solution = await self.answer_generate(input=problem)
        return solution['answer'], self.llm.get_usage_summary()["total_cost"]"""

        elif problem_type == "qa":
            type_constraint = """

## QA Problem Specific Rules
- async def __call__(self, problem: str)
- Use AnswerGenerate for text answers
- DO NOT use Programmer or Test
- Return (answer, cost) tuple

Required Pattern:
from scripts.operators import AnswerGenerate
class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        self.answer_generate = AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        solution = await self.answer_generate(input=problem)
        return solution['answer'], self.llm.get_usage_summary()["total_cost"]"""

        else:
            type_constraint = ""

        return base_prompt + type_constraint

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
                use_cache=False  # ✅ Fix: Disable cache when gradient checkpointing is enabled
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
                use_cache=False  # ✅ Fix: Disable cache when gradient checkpointing is enabled
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
                workflow_code, is_valid, error = self._parse_workflow_code(
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

    def _parse_workflow_code(self, generated_text: str, problem_type: str) -> Tuple[str, bool, Optional[str]]:
        """
        解析生成的文本，提取并使用reactive patching进行修复

        新策略：使用WorkflowValidator的reactive patching模式（参考项目验证过）
        - 只修复实际问题，不做完整重构
        - 更快、更可靠、更少副作用
        """

        # DEBUG: 打印 Qwen 生成的原始文本
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG: Qwen 生成的原始文本 (完整):")
        print(f"{'='*60}")
        print(generated_text)  # 打印完整文本
        print(f"{'='*60}\n")

        try:
            # 1. 提取代码块（支持markdown和纯代码格式）
            code = self._extract_code_block(generated_text)
            if not code:
                print(f"❌ 无法从生成文本中提取代码块")
                return self._get_default_workflow(problem_type), False, "No code block found"

            # 2. 使用WorkflowValidator进行reactive patching验证
            print(f"🔧 使用reactive patching进行验证和修复...")
            fixed_code, is_valid, error_msg, fixes = self.validator.validate_and_fix_workflow(
                code=code,
                problem_type=problem_type
            )

            if is_valid:
                print(f"✅ 验证成功")
                if fixes:
                    print(f"   应用了以下修复: {fixes}")
                return fixed_code, True, None
            else:
                print(f"❌ 验证失败: {error_msg}")
                if fixes:
                    print(f"   尝试修复: {fixes}")
                # 如果修复后通过了基本语法检查，仍返回修复后的代码
                # 否则使用默认工作流
                try:
                    compile(fixed_code, '<string>', 'exec')
                    print(f"⚠️ 代码可编译，使用修复版本")
                    return fixed_code, False, error_msg
                except:
                    print(f"❌ 修复后仍无法编译，使用默认工作流")
                    return self._get_default_workflow(problem_type), False, error_msg

        except Exception as e:
            print(f"❌ 异常捕获: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_default_workflow(problem_type), False, str(e)

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
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
"""

    def _extract_code_block(self, generated_text: str) -> str:
        """
        从生成的文本中提取Python代码块

        支持格式：
        1. Markdown代码块：```python ... ```
        2. 简单代码块：``` ... ```
        3. 纯代码（没有包裹）
        """
        import re

        # 尝试提取markdown代码块
        # Pattern 1: ```python ... ```
        python_pattern = r'```python\s*\n(.*?)\n```'
        match = re.search(python_pattern, generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Pattern 2: ``` ... ```
        general_pattern = r'```\s*\n(.*?)\n```'
        match = re.search(general_pattern, generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Pattern 3: 查找class Workflow定义
        class_pattern = r'(class\s+Workflow\s*:.*?(?=\n\n|\Z))'
        match = re.search(class_pattern, generated_text, re.DOTALL)
        if match:
            # 找到class开始位置
            start_pos = match.start()
            # 获取class之后的所有内容
            code_after_class = generated_text[start_pos:]

            # 尝试找到合适的结束点
            lines = code_after_class.split('\n')
            code_lines = []
            indent_level = None

            for line in lines:
                # 如果是空行，继续
                if not line.strip():
                    code_lines.append(line)
                    continue

                # 获取当前行的缩进
                current_indent = len(line) - len(line.lstrip())

                # 如果这是第一行代码，记录缩进级别
                if indent_level is None and line.strip().startswith(('class', 'def', 'import', 'from')):
                    indent_level = current_indent

                # 如果遇到同级或更小缩进（且不是空行），可能结束了
                if indent_level is not None and current_indent <= indent_level - 4:
                    break

                code_lines.append(line)

            return '\n'.join(code_lines)

        # 如果都找不到，返回原文本（但去除前后的解释文字）
        lines = generated_text.split('\n')
        code_start = -1
        code_end = len(lines)

        for i, line in enumerate(lines):
            if 'class Workflow' in line:
                code_start = i
                break

        if code_start >= 0:
            # 从class Workflow开始
            return '\n'.join(lines[code_start:code_end])

        # 最后尝试：如果文本包含Python代码特征，返回整个文本
        if any(keyword in generated_text for keyword in ['class Workflow', 'def __call__', 'import', 'from']):
            return generated_text.strip()

        return ""


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


def _extract_code_block(self, generated_text: str) -> str:
        """
        从生成的文本中提取Python代码块

        支持格式：
        1. Markdown代码块：```python ... ```
        2. 简单代码块：``` ... ```
        3. 纯代码（没有包裹）
        """
        import re

        # 尝试提取markdown代码块
        # Pattern 1: ```python ... ```
        python_pattern = r'```python\s*\n(.*?)\n```'
        match = re.search(python_pattern, generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Pattern 2: ``` ... ```
        general_pattern = r'```\s*\n(.*?)\n```'
        match = re.search(general_pattern, generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Pattern 3: 查找class Workflow定义
        class_pattern = r'(class\s+Workflow\s*:.*?(?=\n\n|\Z))'
        match = re.search(class_pattern, generated_text, re.DOTALL)
        if match:
            # 找到class开始位置
            start_pos = match.start()
            # 获取class之后的所有内容
            code_after_class = generated_text[start_pos:]

            # 尝试找到合适的结束点
            lines = code_after_class.split('\n')
            code_lines = []
            indent_level = None

            for line in lines:
                # 如果是空行，继续
                if not line.strip():
                    code_lines.append(line)
                    continue

                # 获取当前行的缩进
                current_indent = len(line) - len(line.lstrip())

                # 如果这是第一行代码，记录缩进级别
                if indent_level is None and line.strip().startswith(('class', 'def', 'import', 'from')):
                    indent_level = current_indent

                # 如果遇到同级或更小缩进（且不是空行），可能结束了
                if indent_level is not None and current_indent <= indent_level - 4:
                    break

                code_lines.append(line)

            return '\n'.join(code_lines)

        # 如果都找不到，返回原文本（但去除前后的解释文字）
        lines = generated_text.split('\n')
        code_start = -1
        code_end = len(lines)

        for i, line in enumerate(lines):
            if 'class Workflow' in line:
                code_start = i
                break

        if code_start >= 0:
            # 从class Workflow开始
            return '\n'.join(lines[code_start:code_end])

        # 最后尝试：如果文本包含Python代码特征，返回整个文本
        if any(keyword in generated_text for keyword in ['class Workflow', 'def __call__', 'import', 'from']):
            return generated_text.strip()

        return ""


if __name__ == "__main__":
    test_generator()
