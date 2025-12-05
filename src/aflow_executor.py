#!/usr/bin/env python3
"""
AFlow执行适配器 - 执行RL生成的工作流
"""
import sys
import os
import tempfile
import importlib.util
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import asyncio
import time

# 导入工作流验证器、响应标准化器和SymPy修复器
try:
    from .workflow_validator_v2 import WorkflowValidatorV2
    from .response_standardizer import ResponseStandardizer
    from .sympy_code_fixer import SymPyCodeFixer
except ImportError:
    from workflow_validator_v2 import WorkflowValidatorV2
    from response_standardizer import ResponseStandardizer
    from sympy_code_fixer import SymPyCodeFixer

# 添加AFlow到路径（添加多个可能需要的路径）
aflow_path = os.getenv("AFLOW_PATH", "../AFlow")
sys.path.insert(0, aflow_path)
sys.path.insert(0, os.path.join(aflow_path, 'workspace'))

# 导入AFlow组件
from scripts.async_llm import create_llm_instance, LLMsConfig
from scripts import operators as operator_module


class AsyncOpenAILLMWrapper:
    """
    OpenAI 异步LLM包装器 - 作为Fallback备用LLM

    实现AsyncLLM接口，与主LLM兼容的异步接口
    当主LLM初始化失败时使用作为Tier 2备用方案
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1",
                 temperature: float = 0.0, top_p: float = 1.0):
        """
        初始化OpenAI异步客户端

        Args:
            api_key: OpenAI API密钥
            model: 使用的模型名称
            base_url: API基础URL
            temperature: 温度参数
            top_p: top_p参数
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("需要安装openai库: pip install openai")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p

        # 初始化OpenAI异步客户端
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # 跟踪使用统计
        self._total_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0

    async def __call__(self, prompt: str, **kwargs) -> str:
        """
        调用OpenAI API生成响应

        Args:
            prompt: 输入提示词
            **kwargs: 其他参数（被忽略以保持接口兼容）

        Returns:
            生成的响应文本
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=2048
            )

            # 提取响应
            answer = response.choices[0].message.content

            # 跟踪使用统计
            if hasattr(response, 'usage') and response.usage:
                self._total_input_tokens += response.usage.prompt_tokens
                self._total_output_tokens += response.usage.completion_tokens
                self._total_tokens += response.usage.total_tokens

                # 估算成本（gpt-4o-mini: $0.15/M input, $0.60/M output）
                input_cost = (response.usage.prompt_tokens / 1_000_000) * 0.15
                output_cost = (response.usage.completion_tokens / 1_000_000) * 0.60
                cost = input_cost + output_cost
                self._total_cost += cost

            self._call_count += 1

            return answer

        except Exception as e:
            print(f"❌ OpenAI API调用失败: {e}")
            raise

    async def call_with_format(self, prompt: str, formatter=None, **kwargs) -> str:
        """
        带格式化的调用（兼容性方法）

        Args:
            prompt: 输入提示词
            formatter: 格式化器（可选）
            **kwargs: 其他参数

        Returns:
            生成的响应文本
        """
        response = await self(prompt, **kwargs)

        if formatter and callable(formatter):
            try:
                return formatter(response)
            except Exception as e:
                print(f"⚠️ 格式化失败: {e}")
                return response

        return response

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        获取使用统计摘要

        Returns:
            包含token和成本信息的字典
        """
        return {
            "total_tokens": self._total_tokens,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost": self._total_cost,
            "call_count": self._call_count
        }

    def reset_usage(self):
        """重置使用统计"""
        self._total_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0


class AFlowExecutor:
    """执行RL生成的工作流，使用AFlow的算子"""

    def __init__(
        self,
        llm_config_path: str = "config/aflow_llm.yaml",
        llm_model_name: str = "gpt-4o-mini",  # 使用OpenAI官方gpt-4o-mini
        timeout: int = 300,
        operator_enhancer: Optional[Any] = None,
        enable_fallback: bool = True  # 启用Fallback机制
    ):
        """
        Args:
            llm_config_path: AFlow LLM配置文件路径
            llm_model_name: 使用的LLM模型名称
            timeout: 执行超时时间（秒）
            operator_enhancer: Layer 2 operator提示词增强器（可选）
            enable_fallback: 是否启用Fallback机制
        """
        self.llm_config_path = Path(llm_config_path)
        self.llm_model_name = llm_model_name
        self.timeout = timeout
        self.operator_enhancer = operator_enhancer
        self.enable_fallback = enable_fallback
        self.validator_v2 = WorkflowValidatorV2()  # 统一验证器（支持reactive patching + TASK_PROMPT提取）
        self.standardizer = ResponseStandardizer()  # 响应标准化器
        self.sympy_fixer = SymPyCodeFixer()  # SymPy修复器

        # 加载LLM配置
        self._load_llm_config()

        print(f"✅ AFlow执行器初始化完成")
        print(f"  LLM模型: {llm_model_name}")
        print(f"  超时: {timeout}秒")
        if operator_enhancer is not None:
            print(f"  Layer 2增强: 启用")

    def _load_llm_config(self):
        """加载LLM配置"""
        try:
            # 设置配置路径
            abs_config_path = self.llm_config_path.absolute()

            # 读取YAML配置文件
            import yaml
            with open(abs_config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # LLMsConfig期望的是models字典
            models_config = yaml_data.get('models', {})

            # 为本地LLM服务禁用代理（如果使用本地服务）
            import os
            model_config = models_config.get(self.llm_model_name, {})
            if 'localhost' in str(model_config.get('base_url', '')) or \
               '127.0.0.1' in str(model_config.get('base_url', '')):
                os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
                os.environ['no_proxy'] = 'localhost,127.0.0.1'
                print("  📌 设置 NO_PROXY=localhost,127.0.0.1 (本地LLM服务无需代理)")

            # 直接加载配置
            from scripts.async_llm import LLMsConfig
            self.llm_configs = LLMsConfig(models_config)

            print(f"✅ 加载LLM配置: {abs_config_path}")

        except Exception as e:
            print(f"⚠️  加载LLM配置失败: {e}")
            print(f"  将使用 LLMsConfig.default()")
            # 使用默认配置而不是 None
            from scripts.async_llm import LLMsConfig
            try:
                self.llm_configs = LLMsConfig.default()
                print(f"✅ 成功加载默认LLM配置")
            except Exception as e2:
                print(f"  默认配置也加载失败: {e2}")
                # 最后的降级方案：设为 None，后续用字符串
                self.llm_configs = None

    def _detect_code_leakage(self, answer: str, problem_type: str) -> bool:
        """
        检测 Programmer 算子是否返回了源代码而不是执行结果（来自参考项目）

        某些情况下，Programmer 算子可能返回源代码而非执行结果，如：
        - def function_name(...):
        - class ClassName:
        - import module_name
        - async def function_name

        对于代码问题，这种情况意味着算子没有正确执行代码。

        Args:
            answer: 返回的答案
            problem_type: 问题类型（'code', 'math', 'qa'）

        Returns:
            如果检测到代码泄露返回 True，否则返回 False
        """
        if not isinstance(answer, str) or problem_type != "code":
            return False

        # 代码泄露的典型模式
        code_patterns = [
            "def ",
            "class ",
            "import ",
            "from ",
            "async def ",
            "@",  # 装饰器
            "try:",
            "except",
            "while ",
            "for ",
        ]

        answer_stripped = answer.strip()

        # 检查答案是否以代码模式开头
        for pattern in code_patterns:
            if answer_stripped.startswith(pattern):
                return True

        # 检查答案是否包含多行代码（缩进）
        lines = answer_stripped.split('\n')
        if len(lines) > 1:
            # 计算有缩进的行数（表示代码块）
            indented_lines = sum(1 for line in lines if line and line[0] in (' ', '\t'))
            if indented_lines > len(lines) * 0.3:  # 超过 30% 的行有缩进
                return True

        return False

    def _clean_answer(self, answer: str) -> str:
        """
        清理答案中的无效模式（来自参考项目）

        某些LLM可能在答案前添加解释性文本，如：
        - "Based on the feedback, ..."
        - "Revised Solution: ..."
        - "Here's the solution: ..."

        这些模式会污染答案，影响评估准确性。

        Args:
            answer: 原始答案字符串

        Returns:
            清理后的答案
        """
        if not isinstance(answer, str):
            return answer

        invalid_patterns = [
            "Based on the feedback",
            "Based on the previous",
            "Revised Solution:",
            "Here's the solution:",
            "Here is the solution:",
            "The solution is:",
            "Here's the revised",
            "Here is the revised",
            "Following the feedback",
            "According to the feedback",
            "Taking the feedback",
            "Let me revise:",
            "Let me reconsider:",
        ]

        for pattern in invalid_patterns:
            if answer.startswith(pattern):
                # 找到模式后的内容
                idx = answer.find(pattern)
                # 跳过模式和可能的冒号/换行
                rest = answer[idx + len(pattern):].lstrip(':').strip()
                if rest:
                    print(f"  🧹 清理答案中的无效前缀: '{pattern}'")
                    return rest

        return answer

    def _check_operator_problem_type_mismatch(
        self,
        workflow_code: str,
        problem_type: str
    ) -> Optional[str]:
        """
        检查工作流中使用的操作符是否与 problem_type 匹配

        Returns:
            如果存在不匹配，返回错误消息；否则返回 None
        """
        code_lower = workflow_code.lower()

        # 检查 problem_type 与操作符的匹配
        if problem_type == "math":
            # Math 问题不应该使用 Test 或 Programmer
            if "self.test(" in code_lower or "await self.test(" in code_lower:
                return (
                    "❌ MATH problem uses Test operator!\n"
                    "   Math problems don't have automated test cases.\n"
                    "   This will cause NoneType errors when Test tries to look up test cases.\n"
                    "   Use only: Custom, AnswerGenerate, Review, Revise, ScEnsemble"
                )
            if "self.programmer(" in code_lower or "await self.programmer(" in code_lower:
                return (
                    "❌ MATH problem uses Programmer operator!\n"
                    "   Math is not code-related, don't use Programmer.\n"
                    "   Use only: Custom, AnswerGenerate, Review, Revise, ScEnsemble"
                )

        elif problem_type == "qa":
            # QA 问题不应该使用 Test 或 Programmer
            if "self.test(" in code_lower or "await self.test(" in code_lower:
                return (
                    "❌ QA problem uses Test operator!\n"
                    "   QA problems don't have automated test cases.\n"
                    "   This will cause NoneType errors when Test tries to look up test cases.\n"
                    "   Use only: Custom, AnswerGenerate, Review, Revise, ScEnsemble"
                )
            if "self.programmer(" in code_lower or "await self.programmer(" in code_lower:
                return (
                    "❌ QA problem uses Programmer operator!\n"
                    "   QA is not code-related, don't use Programmer.\n"
                    "   Use only: Custom, AnswerGenerate, Review, Revise, ScEnsemble"
                )

        elif problem_type == "code":
            # Code问题：不强制要求Test operator
            # 原因：Test operator虽然推荐，但不是必需的（Custom也可以生成代码）
            pass

        return None

    def validate_operator_output(self, output: Any, operator_name: str) -> Dict:
        """
        验证并标准化算子输出格式（使用ResponseStandardizer）

        Args:
            output: 算子的原始输出
            operator_name: 算子名称

        Returns:
            标准化后的输出字典
        """
        # 使用ResponseStandardizer进行标准化
        standardized = self.standardizer.standardize(output, operator_name)

        # 保持向后兼容，同时返回原始字段和标准化字段
        if isinstance(output, dict):
            result = output.copy()
            result.update({
                '__standardized__': standardized,
                # 确保关键字段存在
                'response': standardized['content'],
                'success': standardized['success'],
                'error': standardized.get('error')
            })
            return result
        else:
            return standardized

    async def execute_workflow(
        self,
        workflow_code: str,
        problem: str,
        problem_type: str = "math",
        **kwargs
    ) -> Tuple[Any, float, Dict]:
        """
        执行工作流

        Args:
            workflow_code: RL模型生成的Workflow类代码
            problem: 问题文本
            problem_type: 问题类型
            **kwargs: 其他参数（如entry_point for code）

        Returns:
            (answer, cost, metadata)
        """

        start_time = time.time()

        # 0. 检查 operator-problem type 匹配（方案B：软学习而非硬阻止）
        # CHANGE: 不再硬拒绝，而是标记为metadata，让RL通过reward学习
        operator_type_mismatch = self._check_operator_problem_type_mismatch(
            workflow_code, problem_type
        )
        mismatch_detected = operator_type_mismatch is not None
        mismatch_details = operator_type_mismatch if operator_type_mismatch else None

        if mismatch_detected:
            # 记录警告但继续执行（允许模型探索）
            print(f"⚠️  Operator-problem type mismatch detected:")
            print(f"   {mismatch_details}")
            print(f"   → Will mark in metadata and apply penalty in reward")
            # 不raise异常 - 继续执行workflow，稍后在metadata中标记

        # 1. 验证工作流代码
        # 使用WorkflowValidatorV2进行reactive patching验证和修复
        print(f"  1️⃣ 验证和修复工作流代码...")
        fixed_code, is_valid, error_msg, fixes_applied = \
            self.validator_v2.validate_and_fix_workflow(workflow_code, problem_type)

        workflow_code = fixed_code

        # 记录修复和错误到元数据（给GRPO学习）
        metadata = kwargs.get('metadata', {})
        if fixes_applied:
            metadata['auto_fixes_applied'] = fixes_applied
            print(f"  ✅ 应用了以下修复: {fixes_applied}")

        if not is_valid:
            # 修复后仍然无效，才考虑降级
            print(f"  ⚠️  工作流代码修复后仍然无效: {error_msg}")

            if self.enable_fallback:
                print(f"  使用Fallback工作流")
                # 标记这是因为验证失败而执行的 Fallback
                answer, cost, fb_metadata = await self._execute_fallback_workflow(problem, problem_type, **kwargs)

                # 合并元数据
                metadata['validation_failed'] = True
                metadata['validation_error'] = msg
                metadata['needed_fallback'] = True  # 🔧 修复：统一key名称（从'fallback_used'改为'needed_fallback'）
                metadata.update(fb_metadata)

                return answer, cost, metadata
            else:
                # Fallback禁用，抛出异常
                raise ValueError(f"工作流代码无效且Fallback已禁用: {msg}")

        # 修复后有效，继续执行（不降级！）
        print(f"  ✅ 代码验证通过（{len(fixes_applied)}个修复）")

        # 2. 修复SymPy兼容性问题（针对Code类型）
        if problem_type == "code" or 'sympy' in workflow_code.lower():
            fixed_code, was_modified, fixes = self.sympy_fixer.fix_code(workflow_code)
            if was_modified:
                print(f"🔧 SymPy代码修复: {', '.join(fixes)}")
                workflow_code = fixed_code

        try:
            # 创建临时工作流模块
            workflow_class = self._create_workflow_class(workflow_code, problem_type)

            # 实例化工作流
            llm_config = self._get_llm_config()

            # 确保 llm_config 不是 None
            if llm_config is None:
                print(f"⚠️  llm_config 为 None，降级为字符串: {self.llm_model_name}")
                llm_config = self.llm_model_name

            try:
                workflow = workflow_class(
                    name="rl_generated_workflow",
                    llm_config=llm_config,
                    dataset=problem_type
                )
            except Exception as e:
                # 工作流实例化失败，使用fallback
                print(f"⚠️  工作流实例化失败: {e}")
                import traceback
                traceback.print_exc()
                print(f"  使用fallback工作流")
                fallback_class = self._get_fallback_workflow_class(problem_type)
                workflow = fallback_class(
                    name="fallback_workflow",
                    llm_config=llm_config,
                    dataset=problem_type
                )
                # 🔧 修复：记录实例化失败标记，后续如果成功需要记录needed_fallback
                metadata['had_instantiation_error'] = True

            # 执行（带超时）
            # 根本性修复：智能3级参数降级策略（参考项目方案）
            try:
                if problem_type == "code":
                    # 策略1: 尝试传入所有3个参数 (problem, entry_point, test)
                    if "entry_point" in kwargs and "test" in kwargs:
                        try:
                            print(f"  📋 尝试3参数模式: (problem, entry_point, test)")
                            result = await asyncio.wait_for(
                                workflow(problem, kwargs["entry_point"], kwargs["test"]),
                                timeout=self.timeout
                            )
                            print(f"  ✅ 3参数模式成功")
                        except TypeError as e:
                            # 策略2: 降级到2参数 (problem, entry_point)
                            if "positional argument" in str(e) or "missing" in str(e).lower():
                                print(f"  ⚠️  3参数失败，尝试2参数模式: (problem, entry_point)")
                                try:
                                    result = await asyncio.wait_for(
                                        workflow(problem, kwargs["entry_point"]),
                                        timeout=self.timeout
                                    )
                                    print(f"  ✅ 2参数模式成功")
                                except TypeError as e2:
                                    # 策略3: 降级到1参数 (problem only)
                                    if "positional argument" in str(e2) or "missing" in str(e2).lower():
                                        print(f"  ⚠️  2参数失败，降级到1参数模式: (problem)")
                                        result = await asyncio.wait_for(
                                            workflow(problem),
                                            timeout=self.timeout
                                        )
                                        print(f"  ✅ 1参数模式成功")
                                    else:
                                        raise
                            else:
                                raise
                    elif "entry_point" in kwargs:
                        # 只有entry_point，没有test
                        try:
                            print(f"  📋 尝试2参数模式: (problem, entry_point)")
                            result = await asyncio.wait_for(
                                workflow(problem, kwargs["entry_point"]),
                                timeout=self.timeout
                            )
                            print(f"  ✅ 2参数模式成功")
                        except TypeError as e:
                            if "positional argument" in str(e) or "missing" in str(e).lower():
                                print(f"  ⚠️  2参数失败，降级到1参数模式: (problem)")
                                result = await asyncio.wait_for(
                                    workflow(problem),
                                    timeout=self.timeout
                                )
                                print(f"  ✅ 1参数模式成功")
                            else:
                                raise
                    else:
                        # 没有entry_point，直接用1参数
                        print(f"  📋 使用1参数模式: (problem)")
                        result = await asyncio.wait_for(
                            workflow(problem),
                            timeout=self.timeout
                        )
                        print(f"  ✅ 1参数模式成功")
                else:
                    # Non-code problems (Math/QA) - 仅传problem参数
                    print(f"  📋 {problem_type.upper()}问题使用1参数模式: (problem)")
                    result = await asyncio.wait_for(
                        workflow(problem),
                        timeout=self.timeout
                    )
                    print(f"  ✅ 执行成功")
            except Exception as e:
                # 捕获所有异常（operator执行失败）
                print(f"  ❌ Workflow执行异常: {type(e).__name__}")
                print(f"     异常信息: {str(e)}")

                # 快速处理Test算子的已知问题
                if "'NoneType' object is not iterable" in str(e) and "test_cases" in str(e):
                    print(f"  🚀 检测到Test算子None问题，快速切换到Fallback")
                    import traceback
                else:
                    import traceback
                    print(f"  完整堆栈:")
                    traceback.print_exc()

                # 检查是否启用Fallback
                if self.enable_fallback:
                    print(f"  🔄 尝试使用Fallback机制")
                    # 🔧 修复：合并fallback metadata并记录needed_fallback标记
                    answer, cost, fb_metadata = await self._execute_fallback_workflow(problem, problem_type, **kwargs)
                    metadata['needed_fallback'] = True
                    metadata['fallback_type'] = 'operator_error'
                    metadata.update(fb_metadata)
                    return answer, cost, metadata
                else:
                    print(f"  ⚠️  Fallback已禁用，直接抛出异常")
                    # 直接抛出异常而不是使用fallback
                    raise

            # 安全地解包结果（可能返回2个或更多值）
            if isinstance(result, tuple):
                if len(result) >= 2:
                    answer, cost = result[0], result[1]
                elif len(result) == 1:
                    answer, cost = result[0], 0.0
                else:
                    answer, cost = None, 0.0
            else:
                answer, cost = result, 0.0

            # ✨ FIX 1: Cost 类型验证与颠倒检测（来自参考项目）
            # 问题：某些格式错误的workflow可能返回 (cost, answer) 而非 (answer, cost)
            # 或者 cost 可能是字符串而不是数字，导致奖励计算失败
            if not isinstance(cost, (int, float)):
                # 检测是否 answer 和 cost 被颠倒了
                if isinstance(answer, (int, float)) and isinstance(cost, str):
                    print(f"  🔄 检测到answer/cost颠倒，已交换")
                    answer, cost = cost, answer
                else:
                    print(f"  ⚠️  无效的cost类型: {type(cost).__name__}，使用默认值 0.0")
                    cost = 0.0

            execution_time = time.time() - start_time

            # ✨ FIX 2: 空答案检测与Fallback触发（来自参考项目）
            # 问题：Workflow可能返回None或空字符串，导致训练污染
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                print(f"  🚨 检测到空答案（None或空字符串）")

                if self.enable_fallback:
                    print(f"  🔄 触发Fallback机制以处理空答案")
                    # 🔧 修复：合并fallback metadata并记录needed_fallback标记
                    answer, cost, fb_metadata = await self._execute_fallback_workflow(problem, problem_type, **kwargs)
                    metadata['needed_fallback'] = True
                    metadata['fallback_type'] = 'empty_answer'
                    metadata.update(fb_metadata)
                    return answer, cost, metadata
                else:
                    print(f"  ⚠️  Fallback已禁用，返回空答案")
                    metadata = {
                        "success": False,
                        "error": "empty_answer",
                        "error_type": "empty_answer",  # 新增（方案B）：明确错误类型
                        "execution_time": execution_time,
                        "cost": cost,
                        "problem_type": problem_type,
                        "validation_failed": False,
                        "fallback_executed": False
                    }
                    return None, 0.0, metadata

            # ✨ FIX 3: 答案模式清理（来自参考项目）
            # 问题：某些LLM可能在答案前添加解释性文本，如"Based on feedback..."、"Revised Solution:"等
            # 这些模式会污染答案，需要清理
            if isinstance(answer, str):
                answer = self._clean_answer(answer)

            # ✨ FIX 4: 代码泄露检测（来自参考项目）
            # 问题：Programmer 算子可能返回源代码而非执行结果（特别是对代码问题）
            # 此时应触发 fallback 而不是返回源代码
            if self._detect_code_leakage(answer, problem_type):
                print(f"  🚨 检测到代码泄露（Programmer返回了源代码而非执行结果）")

                if self.enable_fallback:
                    print(f"  🔄 触发Fallback机制以处理代码泄露")
                    # 🔧 修复：合并fallback metadata并记录needed_fallback标记
                    answer, cost, fb_metadata = await self._execute_fallback_workflow(problem, problem_type, **kwargs)
                    metadata['needed_fallback'] = True
                    metadata['fallback_type'] = 'code_leakage'
                    metadata.update(fb_metadata)
                    return answer, cost, metadata
                else:
                    print(f"  ⚠️  Fallback已禁用，返回源代码")
                    metadata = {
                        "success": False,
                        "error": "code_leakage",
                        "error_type": "code_leakage",  # 新增（方案B）：明确错误类型
                        "execution_time": execution_time,
                        "cost": cost,
                        "problem_type": problem_type,
                        "validation_failed": False,
                        "fallback_executed": False
                    }
                    return None, 0.0, metadata

            # 元数据（方案B：添加operator_problem_type_mismatch标记用于soft learning）
            # 🔧 修复：检查是否存在had_instantiation_error标记，如果有则添加needed_fallback
            if not metadata.get('had_instantiation_error', False):
                # 正常流程：更新metadata（保留之前的had_signature_error等标志！）
                metadata.update({
                    "success": True,
                    "execution_time": execution_time,
                    "cost": cost,
                    "problem_type": problem_type,
                    "validation_failed": False,
                    "fallback_executed": False,
                    # 新增（方案B）：标记operator-problem type匹配情况
                    "operator_problem_type_mismatch": mismatch_detected,
                    "mismatch_type": mismatch_details.split('\n')[0] if mismatch_details else None
                })
            else:
                # 实例化失败但最终成功的流程：保留had_instantiation_error，添加needed_fallback
                metadata['success'] = True
                metadata['needed_fallback'] = True  # 🔧 标记：虽然最终成功，但生成的代码无法实例化
                metadata['fallback_type'] = 'instantiation_error'
                metadata['execution_time'] = execution_time
                metadata['operator_problem_type_mismatch'] = mismatch_detected
                metadata['mismatch_type'] = mismatch_details.split('\n')[0] if mismatch_details else None

            if mismatch_detected:
                print(f"  ⚠️  Workflow violates operator-problem constraint")
                print(f"     This will be penalized (-5.0) in training reward")

            return answer, cost, metadata

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            print(f"⏱️  执行超时 ({self.timeout}秒)")

            metadata = {
                "success": False,
                "error": "timeout",
                "execution_time": execution_time,
                "cost": 0.0,
                "problem_type": problem_type,
                "validation_failed": False,  # 🔴 新增：工作流通过验证，但执行超时了
                "fallback_executed": False
            }

            return None, 0.0, metadata

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ 执行错误: {str(e)}")

            import traceback
            traceback.print_exc()

            metadata = {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "cost": 0.0,
                "problem_type": problem_type,
                "validation_failed": False,  # 🔴 新增：工作流通过验证，但执行失败了
                "fallback_executed": False
            }

            return None, 0.0, metadata

    def _create_workflow_class(self, workflow_code: str, problem_type: str):
        """
        从工作流代码动态创建Workflow类，支持TASK_PROMPT注入

        设计：
        1. 提取TASK_PROMPT变量（如果存在）
        2. 创建基础工作流类
        3. 如果有TASK_PROMPT，创建EnhancedWorkflow包装器自动注入
        """

        # 1. 提取TASK_PROMPT（可选）
        task_prompt = self.validator_v2.extract_task_prompt(workflow_code)
        if task_prompt:
            print(f"📝 检测到TASK_PROMPT，将在执行时注入")

        # 准备命名空间
        namespace = {
            "operator": operator_module,
            "create_llm_instance": create_llm_instance,
            "DatasetType": str
        }

        # 替换import路径（使workspace路径可用）
        # 这里简化处理，直接使用scripts中的operator
        modified_code = workflow_code.replace(
            f"import workspace.{problem_type}.workflows.template.operator as operator",
            "# operator already imported"
        )

        # 修复常见typo（RL模型可能产生的错误）
        modified_code = modified_code.replace("async_lll", "async_llm")
        modified_code = modified_code.replace("create_lll_instance", "create_llm_instance")

        try:
            # 2. 执行代码创建基础类
            exec(modified_code, namespace)

            # 返回Workflow类
            if "Workflow" not in namespace:
                raise ValueError("No Workflow class found in generated code")

            base_class = namespace["Workflow"]

            # 3. 如果有TASK_PROMPT，创建EnhancedWorkflow包装器
            if task_prompt:
                class EnhancedWorkflow:
                    """自动注入TASK_PROMPT的包装器"""
                    def __init__(self, name: str, llm_config, dataset):
                        self.base_workflow = base_class(name, llm_config, dataset)
                        self.task_prompt = task_prompt
                        self.llm = self.base_workflow.llm

                    async def __call__(self, problem: str, *args, **kwargs):
                        """
                        自动在问题前注入TASK_PROMPT

                        Args:
                            problem: 原始问题文本
                            *args, **kwargs: 传递给基础工作流的其他参数（如entry_point, test）

                        Returns:
                            (answer, cost) 元组
                        """
                        # 注入TASK_PROMPT到问题前面
                        enhanced_problem = f"{self.task_prompt}\n\n{problem}"
                        return await self.base_workflow(enhanced_problem, *args, **kwargs)

                return EnhancedWorkflow
            else:
                return base_class

        except Exception as e:
            print(f"⚠️  生成的工作流代码有错误: {e}")
            print(f"  使用默认fallback工作流")
            import traceback
            traceback.print_exc()

            # 使用简单的默认工作流作为fallback
            return self._get_fallback_workflow_class(problem_type)

    def _get_llm_config(self):
        """获取LLM配置（确保返回正确类型）"""
        from scripts.async_llm import LLMsConfig, LLMConfig

        try:
            if self.llm_configs:
                result = self.llm_configs.get(self.llm_model_name)
            else:
                # 尝试使用默认配置
                result = LLMsConfig.default().get(self.llm_model_name)

            # 类型验证（关键！）
            if isinstance(result, LLMConfig):
                return result
            elif isinstance(result, dict):
                # 如果意外返回了 dict，转换为 LLMConfig
                print(f"⚠️  警告：get() 返回了 dict，正在转换为 LLMConfig")
                return LLMConfig(result)
            elif isinstance(result, str):
                return result
            else:
                print(f"⚠️  未知类型: {type(result)}，降级为字符串")
                return self.llm_model_name

        except Exception as e:
            print(f"⚠️  获取LLM配置失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回字符串模型名，让 create_llm_instance 自动处理
            print(f"  降级为字符串模式: {self.llm_model_name}")
            return self.llm_model_name

    def _create_qa_fallback_workflow(self) -> str:
        """
        创建 QA 专用 Fallback 工作流代码

        特点：
        - 仅使用 Custom 操作符，不使用 Test
        - 特别针对 QA 问题的指令
        - 不处理 entry_point 参数（QA 不需要）
        """
        return '''
import asyncio

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.model = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.model)

    async def __call__(self, problem, entry_point=None, test=None):
        """QA Fallback 工作流：使用 Custom 操作符生成答案，不使用 Test"""
        instruction = "Answer this question comprehensively. Provide the final answer clearly."
        result = await self.custom(input=problem, instruction=instruction)

        # 安全提取响应
        if isinstance(result, dict):
            response = result.get("response", "")
        else:
            response = str(result)

        # 获取成本
        try:
            cost = self.model.get_usage_summary().get("total_cost", 0.0)
        except:
            cost = 0.0

        return response, cost
'''

    async def _execute_fallback_workflow(
        self,
        problem: str,
        problem_type: str,
        **kwargs
    ) -> Tuple[Any, float, Dict]:
        """
        执行Fallback工作流

        使用最简单但可靠的方式执行
        """
        print(f"🔄 执行Fallback工作流（类型: {problem_type}）")
        start_time = time.time()

        try:
            # 根据问题类型选择 Fallback 工作流
            if problem_type == "qa":
                # QA 专用 Fallback：避免 Test 操作符
                simple_workflow_code = self._create_qa_fallback_workflow()
                print(f"  ℹ️  使用 QA 专用 Fallback（不包含 Test 操作符）")
            else:
                # 通用 Fallback（用于 code 和 math）
                if problem_type == "code":
                    func_signature = ", entry_point"
                else:
                    func_signature = ""

                simple_workflow_code = f'''
import asyncio

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.model = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.model)

    async def __call__(self, problem{func_signature}):
        """Simple fallback workflow using only Custom operator"""

        # Use Custom operator with appropriate instruction
        if self.dataset == "code":
            instruction = "Solve this coding problem. Provide a complete Python solution."
        elif self.dataset == "math":
            instruction = "Solve this math problem step by step. Show your work and provide the final answer."
        else:
            instruction = "Answer this question comprehensively."

        result = await self.custom(input=problem, instruction=instruction)

        # Validate and extract response
        if isinstance(result, dict):
            response = result.get("response", "")
        else:
            response = str(result)

        # Get cost
        try:
            cost = self.model.get_usage_summary().get("total_cost", 0.0)
        except:
            cost = 0.0

        return response, cost
'''

            # 创建工作流类
            workflow_class = self._create_workflow_class(simple_workflow_code, problem_type)

            # 实例化
            llm_config = self._get_llm_config()
            workflow = workflow_class(
                name="fallback_workflow",
                llm_config=llm_config,
                dataset=problem_type
            )

            # 执行
            if problem_type == "code" and "entry_point" in kwargs:
                result = await asyncio.wait_for(
                    workflow(problem, kwargs["entry_point"]),
                    timeout=self.timeout
                )
            else:
                result = await asyncio.wait_for(
                    workflow(problem),
                    timeout=self.timeout
                )

            # 解包结果
            if isinstance(result, tuple) and len(result) >= 2:
                answer, cost = result[0], result[1]
            else:
                answer, cost = result, 0.0

            execution_time = time.time() - start_time

            metadata = {
                "success": True,
                "needed_fallback": True,  # 🔧 修复：统一key名称（从'fallback_used'改为'needed_fallback'）
                "execution_time": execution_time,
                "cost": cost,
                "problem_type": problem_type
            }

            print(f"✅ Fallback成功 (耗时: {execution_time:.2f}秒)")
            return answer, cost, metadata

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Fallback也失败了: {e}")

            metadata = {
                "success": False,
                "needed_fallback": True,  # 🔧 修复：统一key名称（从'fallback_used'改为'needed_fallback'）
                "error": str(e),
                "execution_time": execution_time,
                "cost": 0.0,
                "problem_type": problem_type
            }

            # 返回空结果而不是抛出异常
            return "", 0.0, metadata

    def _get_fallback_workflow_class(self, problem_type: str):
        """返回一个简单的默认工作流类（用于生成失败时）

        改进的fallback策略：
        1. 先尝试直接调用LLM生成解决方案
        2. 如果失败，返回占位符而不是None
        3. 避免依赖可能失败的Test operator
        """
        # 保存llm_config_path供FallbackWorkflow使用
        llm_config_path = self.llm_config_path

        class FallbackWorkflow:
            def __init__(self, name: str, llm_config, dataset):
                self.name = name
                self.dataset = dataset

                # L1.2: 3-tier LLM 初始化降级机制（增强可靠性）
                try:
                    # Tier 1: 尝试主 LLM 初始化
                    self.model = create_llm_instance(llm_config)
                    print(f"✅ LLM 初始化成功（主 LLM）")
                except Exception as e:
                    print(f"⚠️  主 LLM 初始化失败: {e}")

                    # Tier 2: 使用 OpenAI 备用 LLM
                    try:
                        print(f"  尝试使用 OpenAI 备用 LLM...")
                        import os
                        import yaml

                        api_key = None

                        # 策略1: 从环境变量获取
                        api_key = os.getenv("OPENAI_API_KEY")

                        # 策略2: 从YAML配置文件读取
                        if not api_key:
                            try:
                                config_path = Path(llm_config_path).absolute()
                                if config_path.exists():
                                    with open(config_path, 'r') as f:
                                        config_data = yaml.safe_load(f)
                                        model_config = config_data.get('models', {}).get('gpt-4o-mini', {})
                                        api_key = model_config.get('api_key')

                                        # 如果是环境变量引用（如 ${OPENAI_API_KEY}），解析它
                                        if api_key and api_key.startswith('${') and api_key.endswith('}'):
                                            env_var_name = api_key[2:-1]
                                            api_key = os.getenv(env_var_name)
                                        elif api_key and api_key.startswith('$'):
                                            env_var_name = api_key[1:]
                                            api_key = os.getenv(env_var_name)
                            except Exception as e_yaml:
                                print(f"    ⚠️  无法读取YAML配置: {e_yaml}")

                        # 策略3: 如果llm_config是dict，尝试从中提取
                        if not api_key and isinstance(llm_config, dict):
                            api_key = llm_config.get('api_key')

                        if api_key and not api_key.startswith('$'):
                            # API Key 可用，使用 OpenAI 备用
                            self.model = AsyncOpenAILLMWrapper(api_key=api_key)
                            print(f"✅ OpenAI 备用 LLM 初始化成功")
                        else:
                            # 没有有效的 API Key，进入 Tier 3
                            raise ValueError(f"无有效的 OpenAI API Key (api_key={api_key})")

                    except Exception as e2:
                        print(f"⚠️  OpenAI 备用 LLM 初始化失败: {e2}")

                        # Tier 3: 最后降级为 None
                        self.model = None
                        print(f"⚠️  LLM 初始化完全失败，将使用占位符返回")

            @staticmethod
            def _safe_extract_response(result):
                """
                L1.3: 安全提取响应，处理多种返回格式

                支持的格式：
                - dict: 查找 'response' / 'answer' / 'solution' 键
                - tuple: 取第一个元素
                - str: 直接返回
                - None: 返回空字符串
                """
                if result is None:
                    return ""

                # 处理字典格式
                if isinstance(result, dict):
                    # 尝试多个可能的键
                    response = (result.get('response') or
                               result.get('answer') or
                               result.get('solution') or
                               str(result))
                    return response if response else ""

                # 处理元组格式
                elif isinstance(result, tuple):
                    return str(result[0]) if result and result[0] is not None else ""

                # 处理字符串格式
                elif isinstance(result, str):
                    return result

                # 其他格式：转为字符串
                else:
                    return str(result) if result else ""

            async def __call__(self, problem: str, *args, **kwargs):
                """改进的fallback：不依赖Test operator"""

                # 策略1: 直接调用LLM生成，不经过任何operator
                if self.model is not None:
                    try:
                        print(f"  📝 Fallback: 直接调用LLM生成解决方案")

                        # 根据问题类型选择合适的prompt
                        if self.dataset == "code":
                            prompt = f"""Given the following coding problem, provide a Python solution.

Problem:
{problem}

Provide ONLY the Python function code, no explanations."""
                        else:
                            prompt = f"""Solve the following problem step by step and provide the final answer.

Problem:
{problem}

Provide the final answer clearly."""

                        # 🔴 修复: 使用正确的 AsyncLLM 接口
                        # AsyncLLM 的方法是 __call__(prompt) 而不是 agenerate(messages=[...])
                        response = await self.model(prompt)

                        if response:
                            usage = self.model.get_usage_summary()
                            if isinstance(usage, dict) and "total_cost" in usage:
                                cost = usage["total_cost"]
                            else:
                                cost = 0.0

                            # L1.3: 使用安全提取方法获取响应
                            # response 可能是字符串或字典，需要处理
                            if isinstance(response, dict):
                                answer = response.get('response', str(response))
                            else:
                                answer = str(response)
                            return answer, cost

                    except Exception as e:
                        print(f"  ⚠️  Fallback直接调用LLM失败: {e}")

                # 策略2: 如果LLM调用也失败，使用Custom operator但不依赖Test
                # 🔴 修复: 只在 self.model 不是 None 时才尝试
                if self.model is not None:
                    try:
                        print(f"  📝 Fallback: 尝试使用Custom operator")
                        custom = operator_module.Custom(self.model)
                        result = await custom(
                            input=problem,
                            instruction="Generate a solution without requiring test validation."
                        )

                        if result:
                            # L1.3: 使用安全提取方法获取响应
                            response_text = self._safe_extract_response(result)
                            if response_text:
                                usage = self.model.get_usage_summary()
                                if isinstance(usage, dict) and "total_cost" in usage:
                                    cost = usage["total_cost"]
                                else:
                                    cost = 0.0
                                return response_text, cost

                    except Exception as e:
                        print(f"  ⚠️  Fallback Custom operator失败: {e}")

                # 策略3: 所有策略都失败，返回占位符而不是None
                print(f"  ⚠️  所有fallback策略都失败，返回占位符")
                placeholder = f"[Fallback placeholder for problem: {problem[:80]}...]"
                return placeholder, 0.0

        return FallbackWorkflow


async def test_executor():
    """测试AFlow执行器"""
    print("\n" + "=" * 60)
    print("🧪 测试AFlow执行器")
    print("=" * 60)

    # 创建执行器
    executor = AFlowExecutor(
        llm_config_path="config/aflow_llm.yaml",
        llm_model_name="gpt-4o-mini",
        timeout=60
    )

    # 测试工作流代码（简单示例）
    test_workflow_code = """
import workspace.math.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.model = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.model)

    async def __call__(self, problem: str):
        solution = await self.custom(input=problem, instruction="Solve this problem step by step and provide the final answer.")
        return solution['response'], self.model.get_usage_summary()["total_cost"]
"""

    # 测试问题
    test_problem = "What is 15 + 27?"

    print(f"\n📝 测试问题: {test_problem}")

    # 执行工作流
    answer, cost, metadata = await executor.execute_workflow(
        workflow_code=test_workflow_code,
        problem=test_problem,
        problem_type="math"
    )

    print(f"\n✅ 执行结果:")
    print(f"  成功: {metadata['success']}")
    print(f"  答案: {answer}")
    print(f"  成本: ${cost:.6f}")
    print(f"  时间: {metadata['execution_time']:.2f}秒")


if __name__ == "__main__":
    asyncio.run(test_executor())
