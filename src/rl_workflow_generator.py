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

from src.workflow_code_builder import WorkflowCodeBuilder
from src.workflow_consistency_checker import WorkflowConsistencyChecker

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

        # 初始化代码构建器和一致性检查器
        self.code_builder = WorkflowCodeBuilder()
        self.consistency_checker = WorkflowConsistencyChecker()

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
        """构建提示词，明确算子 API，让模型自主学习选择"""

        # 代码题专用模板
        if problem_type == "code":
            prompt = f"""Generate a Python Workflow class to solve the CODE problem.

CRITICAL for CODE problems:
- Your __call__ method MUST accept THREE parameters: (problem: str, entry_point: str, test: str)
- MUST use Programmer to generate code
- MUST use Test to execute the code with test cases
- MUST return the execution result, NOT the code string

Available Operators:

1. Programmer(llm) - Auto-generate and execute Python code
   Call: await self.programmer(problem=str, analysis=str)
   Returns: {{'code': str, 'output': str}}

2. Test(llm) - Test code with test cases
   Call: await self.test(problem=str, solution=str, entry_point=str)
   Returns: {{'result': bool, 'solution': str}}

3. Review(llm) - Review and validate solution
   Call: await self.review(problem=str, solution=str)
   Returns: {{'review_result': bool, 'feedback': str}}

Template:

from scripts.operators import Custom, AnswerGenerate, Programmer, Test, Review, Revise, ScEnsemble
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        # Initialize Programmer and Test (required for code problems)
        self.programmer = Programmer(self.llm)
        self.test = Test(self.llm)

    async def __call__(self, problem: str, entry_point: str, test: str):
        # Solve: {problem}
        # Generate code using Programmer
        code_result = await self.programmer(problem=problem, analysis='')

        # Test the code (this returns execution result, not code string)
        test_result = await self.test(
            problem=problem,
            solution=code_result['code'],
            entry_point=entry_point
        )

        # CRITICAL: Return execution result and cost
        # test_result['solution'] contains the final code
        # Return the execution output, not the code
        return code_result['output'], self.llm.get_usage_summary()["total_cost"]
"""
            return prompt

        # 通用模板（数学题和QA题）
        prompt = f"""Generate a Python Workflow class to solve the given problem.

IMPORTANT: Consider the problem's difficulty and complexity when designing your workflow.
- Some problems are simple and straightforward
- Some problems are complex and require careful handling
- Choose your strategy based on what you observe about the problem

CRITICAL RULES:
- Only use operators listed below with their EXACT parameters
- Initialize ALL variables before using them - never return undefined variables
- If a variable is defined inside an if-block, either initialize it before the if-block OR handle both branches
- Design your workflow freely - you decide which operators to use and how to combine them

Available Operators:

1. Custom(llm) - Most flexible, for any custom task
   Call: await self.custom(input=str, instruction=str)
   Returns: {{'response': str}}

2. AnswerGenerate(llm) - Step-by-step reasoning
   Call: await self.answer_generate(input=str)  ← NO instruction parameter!
   Returns: {{'thought': str, 'answer': str}}

3. Programmer(llm) - Auto-generate and execute Python code
   Call: await self.programmer(problem=str, analysis=str)
   Returns: {{'code': str, 'output': str}}

4. Test(llm) - Test code with test cases
   Call: await self.test(problem=str, solution=str, entry_point=str)
   Returns: {{'result': bool, 'solution': str}}

5. Review(llm) - Review and validate solution
   Call: await self.review(problem=str, solution=str)
   Returns: {{'review_result': bool, 'feedback': str}}

6. Revise(llm) - Revise solution based on feedback
   Call: await self.revise(problem=str, solution=str, feedback=str)
   Returns: {{'solution': str}}

7. ScEnsemble(llm) - Self-consistency ensemble voting
   Call: await self.sc_ensemble(solutions=list, problem=str)
   Returns: {{'response': str}}

Template (complete the __call__ method):

from scripts.operators import Custom, AnswerGenerate, Programmer, Test, Review, Revise, ScEnsemble
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # ⚠️ CRITICAL: Initialize ALL operators you will use in __call__!
        # Example 1: If you only need answer_generate:
        # self.answer_generate = AnswerGenerate(self.llm)

        # Example 2: If you need review:
        # self.answer_generate = AnswerGenerate(self.llm)
        # self.review = Review(self.llm)

        # Example 3: Full workflow with programmer and test:
        # self.programmer = Programmer(self.llm)
        # self.test = Test(self.llm)
        # self.review = Review(self.llm)

        # Available operators (initialize only what you need):
        # self.custom = Custom(self.llm)
        # self.answer_generate = AnswerGenerate(self.llm)
        # self.programmer = Programmer(self.llm)
        # self.test = Test(self.llm)
        # self.review = Review(self.llm)
        # self.sc_ensemble = ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        # Solve: {problem}
        # CRITICAL: MUST return (answer_string, cost_float) tuple
        # - First value MUST be the final answer (string)
        # - Second value MUST be the cost (float, from self.llm.get_usage_summary()["total_cost"])
        #
        # WRONG: NEVER return (code, answer) - this will cause type errors
        # CORRECT: ALWAYS return (answer, cost)

        # Example 1 - Simple workflow:
        # solution = await self.answer_generate(input=problem)
        # return solution['answer'], self.llm.get_usage_summary()["total_cost"]

        # Example 2 - Review loop:
        # solution = await self.answer_generate(input=problem)
        # review = await self.review(problem=problem, solution=solution['answer'])
        # if not review['review_result']:
        #     # Regenerate or use feedback to guide next attempt
        #     solution = await self.answer_generate(input=problem + "\n" + review['feedback'])
        # return solution['answer'], self.llm.get_usage_summary()["total_cost"]

        # Example 3 - Code problem workflow:
        # code_result = await self.programmer(problem=problem, analysis='None')
        # test_result = await self.test(problem=problem, solution=code_result['code'], entry_point='solution')
        # if test_result['result']:
        #     return test_result['solution'], self.llm.get_usage_summary()["total_cost"]
        # return code_result['output'], self.llm.get_usage_summary()["total_cost"]

        # IMPORTANT: Always initialize variables before any if-blocks!
        # Good:
        #   answer = await self.answer_generate(input=problem)
        #   final = answer['answer']  # Initialize
        #   if condition:
        #       final = modified  # Modify
        #   return final, cost  # Always defined
        #
        # Bad (NEVER):
        #   if condition:
        #       answer = ...  # Only in if-block
        #   return answer, cost  # ERROR if condition is False!

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
        """
        解析生成的文本，提取并完整重构工作流代码

        新策略：使用WorkflowCodeBuilder完整重构而非逐层补救
        """

        # DEBUG: 打印 Qwen 生成的原始文本
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG: Qwen 生成的原始文本 (完整):")
        print(f"{'='*60}")
        print(generated_text)  # 打印完整文本
        print(f"{'='*60}\n")

        # 使用 WorkflowCodeBuilder 进行完整重构
        try:
            print(f"🔨 使用 WorkflowCodeBuilder 完整重构工作流代码...")
            code, success, error = self.code_builder.build_from_qwen_output(
                qwen_text=generated_text,
                problem_type=problem_type,
                strict=False  # 不严格模式，失败时返回error而非抛异常
            )

            if success:
                print(f"✅ 代码重构成功")
                # 验证一致性
                result = self.consistency_checker.check_consistency(code)
                if result['consistent']:
                    print(f"✅ 一致性检查通过")
                    return code, True, None
                else:
                    print(f"⚠️ 一致性检查警告: {result['issues']}")
                    # 仍然返回代码，但标记为有问题
                    return code, False, f"Consistency check: {result['issues']}"
            else:
                print(f"❌ 代码重构失败: {error}")
                # 使用默认工作流
                return self._get_default_workflow(problem_type), False, error

        except Exception as e:
            print(f"❌ 异常捕获: {str(e)}")
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
