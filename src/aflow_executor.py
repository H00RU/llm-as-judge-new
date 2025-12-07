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

# 导入响应标准化器
try:
    from .response_standardizer import ResponseStandardizer
except ImportError:
    from response_standardizer import ResponseStandardizer

# 添加AFlow到路径（添加多个可能需要的路径）
aflow_path = os.getenv("AFLOW_PATH", "../AFlow")
sys.path.insert(0, aflow_path)
sys.path.insert(0, os.path.join(aflow_path, 'workspace'))

# 导入AFlow组件
from scripts.async_llm import create_llm_instance, LLMsConfig
from scripts import operators as operator_module


class AFlowExecutor:
    """执行RL生成的工作流，使用AFlow的算子"""

    def __init__(
        self,
        llm_config_path: str = "config/aflow_llm.yaml",
        llm_model_name: str = "gpt-4o-mini",
        timeout: int = 300,
        operator_enhancer: Optional[Any] = None,
        enable_fallback: bool = True
    ):
        """
        Args:
            llm_config_path: AFlow LLM配置文件路径
            llm_model_name: 使用的LLM模型名称
            timeout: 执行超时时间（秒）
            operator_enhancer: Layer 2 operator提示词增强器（可选）
            enable_fallback: 是否启用Fallback机制（安全网）
        """
        self.llm_config_path = Path(llm_config_path)
        self.llm_model_name = llm_model_name
        self.timeout = timeout
        self.operator_enhancer = operator_enhancer
        self.enable_fallback = enable_fallback
        self.standardizer = ResponseStandardizer()  # 响应标准化器

        # 初始化工作流验证器（增强版，包含一致性检查）
        from .workflow_validator import WorkflowValidator
        self.validator = WorkflowValidator()

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

    def _basic_syntax_check(self, workflow_code: str) -> Tuple[bool, str]:
        """
        基础语法检查 - 只检查，不修复

        检查内容：
        - Python语法有效性
        - 包含Workflow类定义
        - 包含__call__方法

        Args:
            workflow_code: 工作流代码

        Returns:
            (is_valid, error_msg) 元组
        """
        try:
            # 1. 检查Python语法
            compile(workflow_code, '<string>', 'exec')

            # 2. 检查必需结构
            if 'class Workflow' not in workflow_code:
                return False, "Missing 'class Workflow' definition"

            if 'def __call__' not in workflow_code and 'async def __call__' not in workflow_code:
                return False, "Missing '__call__' method in Workflow class"

            return True, ""

        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

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

    def _get_learning_point(self, error: Exception) -> str:
        """根据错误类型提供学习点"""
        error_str = str(error).lower()
        error_type = type(error).__name__

        if error_type == 'AttributeError':
            if 'has no attribute' in error_str:
                return 'Operator导入-初始化-使用不一致：确保使用的operator已导入并初始化'
            else:
                return '检查operator属性访问是否正确'
        elif error_type == 'ImportError':
            return '导入错误：检查operator导入语句是否正确'
        elif error_type == 'NameError':
            return '名称错误：检查变量名是否定义'
        elif error_type == 'TypeError':
            return '类型错误：检查operator参数和调用方式'
        elif 'timeout' in error_str:
            return '执行超时：可能需要优化workflow逻辑'
        else:
            return f'执行错误：{error_type} - 需要检查workflow代码逻辑'

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

        # 1. 增强验证检查（包含operator一致性）
        print(f"  1️⃣ 验证工作流代码和operator一致性...")
        validated_code, is_valid, error_msg, fixes_applied = self.validator.validate_and_fix_workflow(
            workflow_code, problem_type
        )

        # 初始化元数据
        metadata = kwargs.get('metadata', {})

        # 设置validation_metadata到metadata中，供reward_computer使用
        validation_metadata = {
            'is_consistent': is_valid,
            'consistency_errors': [error_msg] if error_msg else [],
            'original_code': workflow_code,
            'validated_code': validated_code,
            'was_fixed': len(fixes_applied) > 0 if fixes_applied else False
        }
        metadata['validation_metadata'] = validation_metadata

        if not is_valid:
            # 代码不一致，记录错误但仍执行原始代码
            print(f"  ❌ 工作流验证失败: {error_msg}")
            print(f"  ⚠️ 将执行原始代码，Qwen需从错误中学习")
            workflow_code = workflow_code  # 使用Qwen生成的原始代码
        else:
            print(f"  ✅ 代码验证和一致性检查通过")
            workflow_code = validated_code  # 使用验证后的代码

        try:
            # 创建临时工作流模块
            workflow_class = self._create_workflow_class(workflow_code, problem_type)

            # 实例化工作流
            llm_config = self._get_llm_config()

            # 确保 llm_config 不是 None
            if llm_config is None:
                print(f"⚠️  llm_config 为 None，降级为字符串: {self.llm_model_name}")
                llm_config = self.llm_model_name

            # 实例化工作流
            workflow = workflow_class(
                name="rl_generated_workflow",
                llm_config=llm_config,
                dataset=problem_type
            )

            # 执行（带超时）- 简化版，不降级参数
            try:
                if problem_type == "code" and "entry_point" in kwargs:
                    print(f"  📋 执行CODE workflow: (problem, entry_point)")
                    result = await asyncio.wait_for(
                        workflow(problem, kwargs["entry_point"]),
                        timeout=self.timeout
                    )
                else:
                    # Math/QA problems or code without entry_point
                    print(f"  📋 执行{problem_type.upper()} workflow: (problem)")
                    result = await asyncio.wait_for(
                        workflow(problem),
                        timeout=self.timeout
                    )
                print(f"  ✅ 执行成功")
            except Exception as e:
                # 捕获所有异常（operator执行失败）- 记录真实错误
                print(f"  ❌ Workflow执行异常: {type(e).__name__}")
                print(f"     异常信息: {str(e)}")

                # 记录真实的执行错误到metadata，用于reward计算
                execution_error = {
                    'type': type(e).__name__,
                    'message': str(e),
                    'learning_point': self._get_learning_point(e)
                }

                # 检查是否是AttributeError（operator一致性问题）
                if isinstance(e, AttributeError):
                    print(f"  🔍 检测到AttributeError：可能是operator导入-初始化-使用不一致")
                    execution_error['is_consistency_error'] = True
                    execution_error['learning_point'] = 'Operator导入-初始化-使用必须一致'

                # 将执行错误信息添加到metadata中
                metadata['execution_error'] = execution_error

                import traceback
                print(f"  完整堆栈:")
                traceback.print_exc()

                # 触发执行级fallback（如果启用）
                if self.enable_fallback:
                    print(f"  🔄 触发执行级fallback安全网")
                    try:
                        return await self._execute_fallback_workflow(
                            workflow_code, problem, problem_type, **kwargs
                        )
                    except Exception as fallback_error:
                        print(f"  ❌ Fallback也失败了: {fallback_error}")
                        metadata['fallback_failed'] = True
                        metadata['fallback_error'] = str(fallback_error)

                # 如果没有fallback或fallback失败，抛出异常
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

            # 空答案检测 - 直接返回0奖励，不fallback
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                print(f"  🚨 检测到空答案（None或空字符串）- 直接失败")
                metadata = {
                    "success": False,
                    "error": "empty_answer",
                    "error_type": "empty_answer",
                    "execution_time": execution_time,
                    "cost": cost,
                    "problem_type": problem_type
                }
                return None, 0.0, metadata

            # ✨ FIX 3: 答案模式清理（来自参考项目）
            # 问题：某些LLM可能在答案前添加解释性文本，如"Based on feedback..."、"Revised Solution:"等
            # 这些模式会污染答案，需要清理
            if isinstance(answer, str):
                answer = self._clean_answer(answer)

            # 代码泄露检测 - 直接返回0奖励，不fallback
            if self._detect_code_leakage(answer, problem_type):
                print(f"  🚨 检测到代码泄露（Programmer返回了源代码而非执行结果）- 直接失败")
                metadata = {
                    "success": False,
                    "error": "code_leakage",
                    "error_type": "code_leakage",
                    "execution_time": execution_time,
                    "cost": cost,
                    "problem_type": problem_type
                }
                return None, 0.0, metadata

            # 元数据
            metadata.update({
                "success": True,
                "execution_time": execution_time,
                "cost": cost,
                "problem_type": problem_type,
                "operator_problem_type_mismatch": mismatch_detected,
                "mismatch_type": mismatch_details.split('\n')[0] if mismatch_details else None
            })

            if mismatch_detected:
                print(f"  ⚠️  Workflow violates operator-problem constraint")
                print(f"     This will be penalized in training reward")

            return answer, cost, metadata

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            print(f"⏱️  执行超时 ({self.timeout}秒)")

            metadata = {
                "success": False,
                "error": "timeout",
                "execution_time": execution_time,
                "cost": 0.0,
                "problem_type": problem_type
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
                "problem_type": problem_type
            }

            return None, 0.0, metadata

    def _create_workflow_class(self, workflow_code: str, problem_type: str):
        """
        从工作流代码动态创建Workflow类

        直接执行代码创建类，不进行任何修复或fallback
        """
        # 准备命名空间
        namespace = {
            "operator": operator_module,
            "create_llm_instance": create_llm_instance,
            "DatasetType": str
        }

        # 替换import路径（使workspace路径可用）
        modified_code = workflow_code.replace(
            f"import workspace.{problem_type}.workflows.template.operator as operator",
            "# operator already imported"
        )

        # 修复常见typo（RL模型可能产生的错误）
        modified_code = modified_code.replace("async_lll", "async_llm")
        modified_code = modified_code.replace("create_lll_instance", "create_llm_instance")

        # 执行代码创建类
        exec(modified_code, namespace)

        # 返回Workflow类
        if "Workflow" not in namespace:
            raise ValueError("No Workflow class found in generated code")

        return namespace["Workflow"]

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
