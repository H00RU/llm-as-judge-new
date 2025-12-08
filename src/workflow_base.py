#!/usr/bin/env python3
"""
Workflow Base Classes - 预初始化算子架构

核心设计理念：
- 根据问题类型预先初始化所有可能需要的operators
- 模型只需编写__call__方法的调用逻辑，无需处理imports和初始化
- 完全消除operator初始化错误
"""
import sys
import os
from typing import Tuple

# 添加AFlow到路径
aflow_path = os.getenv("AFLOW_PATH", "../AFlow")
sys.path.insert(0, aflow_path)

from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType
from scripts.operators import (
    Custom,
    AnswerGenerate,
    Programmer,
    Test,
    Review,
    Revise,
    ScEnsemble
)


class MathWorkflowBase:
    """
    数学问题的Workflow基类

    预初始化MATH类型的所有operators：
    - answer_generate: 生成步骤推理
    - review: 审查答案
    - revise: 修订答案
    - scensemble: 自一致性集成
    - custom: 自定义prompting
    """

    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # 预初始化所有MATH类型的operators
        self.answer_generate = AnswerGenerate(self.llm)
        self.review = Review(self.llm)
        self.revise = Revise(self.llm)
        self.scensemble = ScEnsemble(self.llm)
        self.custom = Custom(self.llm)

        print(f"✅ MathWorkflowBase initialized with 5 operators")

    async def __call__(self, problem: str) -> Tuple[str, float]:
        """
        子类必须实现此方法

        Args:
            problem: 数学问题

        Returns:
            (answer, cost) tuple
        """
        raise NotImplementedError("Subclass must implement __call__ method")


class CodeWorkflowBase:
    """
    代码问题的Workflow基类

    预初始化CODE类型的所有operators：
    - programmer: 生成和执行代码
    - test: 测试代码
    - review: 审查代码
    - revise: 修订代码
    - custom: 自定义prompting
    """

    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # 预初始化所有CODE类型的operators
        self.programmer = Programmer(self.llm)
        self.test = Test(self.llm)
        self.review = Review(self.llm)
        self.revise = Revise(self.llm)
        self.custom = Custom(self.llm)

        print(f"✅ CodeWorkflowBase initialized with 5 operators")

    async def __call__(self, problem: str, entry_point: str, test: str) -> Tuple[str, float]:
        """
        子类必须实现此方法

        Args:
            problem: 代码问题描述
            entry_point: 函数入口点
            test: 测试用例

        Returns:
            (solution, cost) tuple
        """
        raise NotImplementedError("Subclass must implement __call__ method")


class QAWorkflowBase:
    """
    QA问题的Workflow基类

    预初始化QA类型的所有operators：
    - answer_generate: 生成答案
    - review: 审查答案
    - revise: 修订答案
    - scensemble: 自一致性集成
    - custom: 自定义prompting
    """

    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # 预初始化所有QA类型的operators
        self.answer_generate = AnswerGenerate(self.llm)
        self.review = Review(self.llm)
        self.revise = Revise(self.llm)
        self.scensemble = ScEnsemble(self.llm)
        self.custom = Custom(self.llm)

        print(f"✅ QAWorkflowBase initialized with 5 operators")

    async def __call__(self, problem: str) -> Tuple[str, float]:
        """
        子类必须实现此方法

        Args:
            problem: QA问题

        Returns:
            (answer, cost) tuple
        """
        raise NotImplementedError("Subclass must implement __call__ method")


# 工厂函数：根据问题类型创建对应的基类
def create_workflow_base(problem_type: str, name: str, llm_config, dataset: DatasetType):
    """
    工厂函数：创建对应问题类型的Workflow基类实例

    Args:
        problem_type: "math", "code", 或 "qa"
        name: workflow名称
        llm_config: LLM配置
        dataset: 数据集类型

    Returns:
        对应的Workflow基类实例
    """
    if problem_type == "math":
        return MathWorkflowBase(name, llm_config, dataset)
    elif problem_type == "code":
        return CodeWorkflowBase(name, llm_config, dataset)
    elif problem_type == "qa":
        return QAWorkflowBase(name, llm_config, dataset)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}. Must be 'math', 'code', or 'qa'")
