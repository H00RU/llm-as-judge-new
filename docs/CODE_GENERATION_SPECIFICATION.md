# Qwen 代码生成规范

**日期**: 2025-12-04
**主题**: Qwen2.5-7B-Instruct 生成的工作流代码详细规范

---

## 核心问题答案

### ❓ Q1: 生成的是什么格式？XML 还是 Python?

**A: 100% Python 代码**

Qwen 生成的是**完整的 Python 代码**，不是 XML 或其他格式。

```python
# ✅ Qwen 生成的格式
class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        # Qwen 填充这部分 ⬇️
        result = await self.answer_generate(input=problem)
        answer = result.get('answer', '')
        return answer, self.llm.get_usage_summary()["total_cost"]
```

**绝不是**:
- ❌ XML 标签
- ❌ 部分代码片段
- ❌ 伪代码
- ❌ JSON 格式

---

### ❓ Q2: 生成全部代码还是部分代码？

**A: 部分代码 - 只生成 `__call__` 方法体**

Qwen **不生成**:
- ✅ 类定义框架（已提供）
- ✅ `__init__` 初始化（已提供）
- ✅ 导入语句（已提供）

Qwen **生成**:
- ❌ `async def __call__(self, problem: str):` 内部的实现逻辑
- ❌ Operator 调用
- ❌ 返回语句

### 具体例子

**框架（提供给Qwen）**:
```python
from scripts.operators import Custom, AnswerGenerate, Programmer, Test, Review, Revise, ScEnsemble
from scripts.async_llm import create_llm_instance

class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        # Operator 初始化由 Qwen 决定
        # (注释掉的选项)

    async def __call__(self, problem: str):
        # ⬇️ Qwen 生成这里
        pass
```

**Qwen 生成的补充**:
```python
    async def __call__(self, problem: str):
        # Step 1: Generate answer with reasoning
        result = await self.answer_generate(input=problem)
        answer = result.get('answer', '')

        # Step 2: Review for quality
        review = await self.review(problem=problem, solution=answer)
        feedback = review.get('feedback', '')

        # Step 3: Revise if needed
        if feedback:
            revised = await self.revise(problem=problem, solution=answer, feedback=feedback)
            answer = revised.get('solution', answer)

        return answer, self.llm.get_usage_summary()["total_cost"]
```

---

### ❓ Q3: Operator 随意调用还是只能调用初始化的 Operator?

**A: 严格限制 - 只能调用在 `__init__` 中初始化的 Operator**

#### 规则 1: 必须在 `__init__` 中初始化

```python
class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)

        # ✅ 初始化要使用的 operator
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.programmer = operator.Programmer(self.llm)
        # self.test 没有初始化
```

#### 规则 2: 只能在 `__call__` 中使用已初始化的 operator

```python
    async def __call__(self, problem: str):
        # ✅ CORRECT - 已在 __init__ 初始化
        result = await self.answer_generate(input=problem)

        # ✅ CORRECT - 已在 __init__ 初始化
        code = await self.programmer(problem=problem, analysis="solve")

        # ❌ WRONG - test 未初始化!
        # await self.test(problem=problem, solution=code, entry_point="solve")
        # AttributeError: 'Workflow' object has no attribute 'test'
```

#### 验证过程

工作流验证器会检查:
```
1. 语法正确 ✓
2. 有 Workflow 类 ✓
3. 有 __call__ 方法 ✓
4. Operator 名字正确 (PascalCase) ✓
5. ❌ 调用的 operator 是否在 __init__ 初始化 (不检查)
    ← 这个留给运行时发现
```

**如果违规**:
```
Runtime Error:
AttributeError: 'Workflow' object has no attribute 'undefined_op'
```

---

## 代码生成流程

```
问题输入
    ↓
[PromptOptimizer] 构建完整提示词
    ├─ 框架代码 (class, __init__, import)
    ├─ Operator 规范 (8个 API)
    ├─ 问题类型指导 (math/code/qa)
    └─ Few-shot 示例 (来自 ExperienceBuffer)
    ↓
[Qwen2.5-7B-Instruct] 生成 __call__ 逻辑
    ├─ 分析问题类型
    ├─ 选择合适的 operator
    ├─ 组织调用流程
    ├─ 决定初始化哪些 operator
    └─ 返回完整 __call__ 方法体
    ↓
生成的 Python 代码
    ├─ 保证 async def __call__ 格式
    ├─ 包含 operator 初始化 (如需要)
    ├─ 包含 operator 调用
    └─ 返回 (solution, cost) 元组
    ↓
[WorkflowCodeFixer] 自动修复常见错误
    ├─ 修复拼写错误
    ├─ 添加变量初始化
    ├─ 修复 dict 访问
    └─ 确保返回值正确
    ↓
[WorkflowValidator] 验证代码质量
    ├─ 语法检查 ✓
    ├─ 结构检查 ✓
    ├─ Operator 规范检查 (部分)
    └─ 返回值检查 ✓
    ↓
✅ 有效 Python 代码 或 ⚠️ 警告
```

---

## 生成示例

### 示例 1: 数学问题

**输入问题**:
```
"What is 2+2?"
```

**Qwen 生成**:
```python
class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        # Use Programmer for calculation
        result = await self.programmer(problem=problem, analysis="Calculate the sum")
        answer = result.get('output', '')  # ← 关键: output not code for math!

        return answer, self.llm.get_usage_summary()["total_cost"]
```

**关键点**:
- ✅ 使用 Programmer (计算)
- ✅ 提取 `output` 字段 (执行结果)
- ✅ 返回 (answer, cost) 元组

---

### 示例 2: 代码问题

**输入问题**:
```
"Write Python function to check if number is even"
```

**Qwen 生成**:
```python
class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.programmer = operator.Programmer(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str, test: str):
        # Step 1: Generate code
        prog_result = await self.programmer(
            problem=problem,
            analysis="Write a function to check if number is even"
        )
        code = prog_result.get('code', '')

        # Step 2: Test solution
        test_result = await self.test(
            problem=problem,
            solution=code,
            entry_point=entry_point
        )

        # Return best solution
        final_code = test_result.get('solution', code)
        return final_code, self.llm.get_usage_summary()["total_cost"]
```

**关键点**:
- ✅ `__call__` 签名必须是 `(problem, entry_point, test)`
- ✅ Programmer: 提取 `code` 字段
- ✅ Test: 必须提供全部 3 个参数

---

### 示例 3: QA 问题 (复杂)

**输入问题**:
```
"What is the capital of France?"
```

**Qwen 生成**:
```python
class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.review = operator.Review(self.llm)
        self.revise = operator.Revise(self.llm)

    async def __call__(self, problem: str):
        # Step 1: Generate initial answer
        ans_result = await self.answer_generate(input=problem)
        answer = ans_result.get('answer', '')

        # Step 2: Review answer quality
        review_result = await self.review(problem=problem, solution=answer)
        feedback = review_result.get('feedback', review_result.get('review_result', ''))

        # Step 3: Revise if feedback suggests errors
        if "incorrect" in feedback.lower() or "wrong" in feedback.lower():
            revised = await self.revise(
                problem=problem,
                solution=answer,
                feedback=feedback
            )
            answer = revised.get('solution', answer)

        return answer, self.llm.get_usage_summary()["total_cost"]
```

**关键点**:
- ✅ Review-Revise 循环
- ✅ 条件判断修改
- ✅ Fallback 链 (`.get()` with defaults)

---

## Operator 初始化规则

### 规则总结

| 情况 | 规则 | 示例 |
|------|------|------|
| **使用简单问题** | 只初始化 1 个 operator | `self.answer_generate = ...` |
| **使用复杂问题** | 初始化 2-3 个 operator | `self.programmer`, `self.review`, `self.revise` |
| **使用集成方案** | 初始化 4+ operator | 罕见 |
| **不使用的 operator** | 不初始化 | 注释掉或完全省略 |

### 初始化位置

```python
class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # ⬇️ Qwen 决定这里初始化哪些
        self.custom = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        # 其他的可以注释掉
        # self.programmer = operator.Programmer(self.llm)
```

---

## 常见错误示例

### ❌ 错误 1: 未初始化就调用

```python
class Workflow:
    def __init__(self, ...):
        self.llm = create_llm_instance(llm_config)
        # ❌ 没有初始化 self.test

    async def __call__(self, problem: str, entry_point: str, test: str):
        code = "..."
        # ❌ AttributeError: no attribute 'test'
        result = await self.test(problem=problem, solution=code, entry_point=entry_point)
```

**修复**:
```python
def __init__(self, ...):
    self.llm = create_llm_instance(llm_config)
    self.test = operator.Test(self.llm)  # ✅ 先初始化
```

---

### ❌ 错误 2: 初始化但不调用

```python
class Workflow:
    def __init__(self, ...):
        self.llm = create_llm_instance(llm_config)
        self.programmer = operator.Programmer(self.llm)
        self.test = operator.Test(self.llm)  # ❌ 初始化了但用不了

    async def __call__(self, problem: str):
        result = await self.programmer(problem=problem, analysis="solve")
        # ❌ 没用 self.test，浪费资源
```

**优化**:
```python
def __init__(self, ...):
    self.llm = create_llm_instance(llm_config)
    self.programmer = operator.Programmer(self.llm)
    # ✅ 删除不用的初始化
```

---

### ❌ 错误 3: 访问字段出错

```python
async def __call__(self, problem: str):
    result = await self.programmer(problem=problem, analysis="solve")
    # ❌ result 可能没有 'code' 字段
    code = result['code']  # KeyError!
```

**修复**:
```python
async def __call__(self, problem: str):
    result = await self.programmer(problem=problem, analysis="solve")
    # ✅ 安全访问
    code = result.get('code', '')
```

---

## 输出格式

### 返回值规范

所有 `__call__` 方法必须返回:

```python
(solution: str, cost: float)
```

**示例**:
```python
async def __call__(self, problem: str):
    # ... 工作流逻辑 ...
    final_answer = "42"
    cost = self.llm.get_usage_summary()["total_cost"]

    # ✅ 返回元组
    return final_answer, cost
```

### 答案格式

不同问题类型的答案格式:

| 类型 | 格式 | 示例 |
|------|------|------|
| **Math** | `\boxed{answer}` | `\boxed{42}` |
| **Code** | Python 代码块 | `def is_even(n):\n    return n % 2 == 0` |
| **QA** | 纯文本 | `The capital of France is Paris` |

---

## 验证检查清单

生成的代码必须满足:

```
☐ 格式: 100% Python 代码
☐ 结构: class Workflow with __init__ and async def __call__
☐ 导入: 正确导入所有 operator
☐ 初始化: Operator 在 __init__ 初始化
☐ 调用: __call__ 中只调用已初始化的 operator
☐ 参数: operator 调用包含全部必需参数
☐ 返回值: 返回 (solution, cost) 元组
☐ 安全访问: 使用 .get() 而非直接索引
☐ 异步: 使用 await 调用 operator
☐ 语法: 通过 AST 解析
```

---

## 总结表

```
┌─────────────────────────────────────────────────────┐
│ Qwen 代码生成关键特性                               │
├─────────────────────────────────────────────────────┤
│ 格式:        Python (100%)                         │
│ 生成范围:    __call__ 方法体 (部分代码)            │
│ Operator:   只能调用初始化的                       │
│ 初始化位:   __init__ 中                            │
│ 返回值:      (solution: str, cost: float)          │
│ 验证方式:    AST 解析 + 语法检查                    │
│ 调试工具:    WorkflowValidator + WorkflowCodeFixer │
└─────────────────────────────────────────────────────┘
```

---

**版本**: 1.0
**最后更新**: 2025-12-04
**状态**: 生产就绪
