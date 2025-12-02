# 深度问题调查文档

**宗旨**: 不简化训练流程，不绕过问题而是解决问题的本质（不要打补丁）

## 调查清单

### 问题1: LLM Judge执行与奖励计算合理性
**状态**: ✅ 调查完成

#### 1.1 LLM Judge是否由gpt-4o-mini执行？
- [x] 检查reward_computer.py中LLM Judge初始化
- [x] 确认是否使用gpt-4o-mini还是其他模型
- [x] 查看训练日志中的实际调用记录

#### 1.2 奖励计算是否合理？
- [x] 分析训练日志中的奖励分布
- [x] 检查奖励计算逻辑 (reward_computer.py)
- [x] 评估-10到10分制转换是否合理
- [x] 查看三级惩罚系统的实际触发情况

#### 1.3 LLM Judge评估标准是否合理？
- [x] 检查prompt设计
- [x] 分析QA任务19-21%低准确率是否与Judge过严有关
- [x] 对比规则匹配vs语义匹配的差异

**关键发现**:

**✅ 1.1 LLM Judge执行确认:**
- **确认**: LLM Judge使用gpt-4o-mini执行
- **证据**: `logs/training_20251201_213033.log` 行137,143,145,150
- **配置**: 从`config/aflow_llm.yaml`读取配置
- **调用方式**: OpenAI API (temperature=0.0, max_tokens=200)

**❌ 1.2 奖励计算存在严重问题:**

**问题A: 双重奖励系统混乱**
```python
# reward_computer.py 计算:
答案质量奖励: +10.0 (正确) 或 -5.0 (错误)
生成质量奖励: -3.0到+3.0 (取决于代码质量)
总奖励 = 答案质量 + 生成质量

# 但是grpo_trainer.py中又有三级惩罚系统:
Level 1: operator_problem_type_mismatch → -5.0
Level 2: validation_failed → -3.0
Level 3: 执行失败 → -8.0/-7.0/-10.0
```

**冲突**: 两套独立的奖励系统在不同情况下触发，导致：
1. `metadata['success'] = True` 时使用reward_computer（答案+生成质量）
2. 有约束违规时使用固定惩罚（-5/-3/-8/-10）
3. 没有统一的奖励框架

**问题B: 归一化不一致**
```python
# reward_computer.py 归一化:
normalized_reward = total_score / 20.0  # 范围 [-0.5, 1.0]

# 但三级惩罚是固定值，未归一化
reward = -5.0  # 直接使用原始值
```

**问题C: 训练日志显示所有样本都是错误答案**
```
日志模式（重复出现）:
│ 答案质量奖励:       -5.0  ❌ 错误
│ 生成质量奖励:       +2.0
│ 总奖励:            -3.0

预测: # Code generation failed after 3 attempts
```

**核心问题**: 绝大部分样本预测都是"Code generation failed"，说明:
1. Programmer算子在训练时频繁失败
2. 即使workflow生成正确，执行阶段也失败
3. 模型几乎学不到正确的信号（全是-5.0）

**❓ 1.3 LLM Judge本身设计合理，但使用有问题:**

**Prompt设计**: ✅ 优秀
- 4步骤：提取答案 → 提取真值 → 归一化 → 比较等价性
- 支持多种格式（LaTeX, boxed, 数字等）
- 宽容格式差异但严格事实差异

**BUT**: LLM Judge从未有机会判断正确答案
- 因为几乎所有预测都是"Code generation failed"
- LLM Judge只能判断"失败消息 vs 正确答案" → 永远False
- 所以答案质量奖励永远是-5.0

**结论**: LLM Judge没问题，问题在于workflow执行总是失败

---

### 问题2: 治标不治本的修改排查
**状态**: 🔍 调查中

#### 2.1 检查所有try-except块
- [ ] 搜索所有异常捕获逻辑
- [ ] 识别是否有"静默失败"隐藏真实问题
- [ ] 确认错误是否被正确传播到训练信号

#### 2.2 检查错误检测机制
- [ ] workflow_validator.py是否只发warning不阻止执行
- [ ] aflow_executor.py是否所有错误都设置了元数据标志
- [ ] 是否有错误被掩盖导致模型学不到

#### 2.3 审查AFlow项目的修改
- [ ] 列出对AFlow项目的所有修改
- [ ] 评估每个修改是否治标不治本
- [ ] 确认修改是否从根源解决问题

**发现**: 待补充

---

### 问题3: Fallback频繁执行与AFlow调用完整性
**状态**: 🔍 调查中

#### 3.1 Fallback触发频率分析
- [ ] 从训练日志中统计Fallback触发次数
- [ ] 分析Fallback触发的具体原因
- [ ] 确认Plan B是否真的消除了Fallback

#### 3.2 Fallback执行失败率
- [ ] 统计Fallback执行失败的案例
- [ ] 分析失败原因（超时？API错误？逻辑错误？）
- [ ] 评估Fallback本身是否有bug

#### 3.3 AFlow调用完整性检查
- [ ] 列出aflow_executor.py对AFlow的所有调用点
- [ ] 检查是否遗漏必要的初始化步骤
- [ ] 确认算子API调用是否完整正确

#### 3.4 对AFlow项目的修改完整性
- [ ] 列出所有修改的AFlow文件
- [ ] 评估修改的必要性和充分性
- [ ] 检查是否需要修改但遗漏的部分

**发现**: 待补充

---

### 问题4: Workflow生成质量根本原因
**状态**: 🔍 调查中

#### 4.1 当前优化措施评估
- [ ] 检查rl_workflow_generator.py中的示例质量
- [ ] 评估soft learning提示是否足够清晰
- [ ] 分析Few-shot示例是否覆盖常见错误

#### 4.2 用词错误原因排查
- [ ] 统计训练日志中常见的用词错误（如'll_m'而非'llm'）
- [ ] 检查tokenizer是否正确处理变量名
- [ ] 分析是否是temperature设置导致的随机性过高

#### 4.3 NoneType错误根源
- [ ] 统计NoneType错误的具体位置
- [ ] 检查是否是生成的代码缺少初始化
- [ ] 评估是否是Few-shot示例不够明确

#### 4.4 排除模型能力不足之外的原因
- [ ] 检查生成配置（temperature, top_p, top_k）
- [ ] 分析prompt是否过于复杂导致模型困惑
- [ ] 评估是否是训练数据质量问题
- [ ] 检查是否是生成长度限制导致截断

**发现**: 待补充

---

### 问题5: 评估脚本与执行流程正确性
**状态**: ✅ 调查完成

#### 5.1 评估流程澄清
- [x] 确认评估时谁生成workflow（Qwen还是其他）
- [x] 确认评估时谁执行workflow（gpt-4o-mini还是Qwen）
- [x] 理清训练时vs评估时的流程差异

#### 5.2 代码任务100%准确率异常分析
- [x] 检查eval_6datasets.py的实现
- [x] 确认是否是测试集泄露到训练集
- [x] 分析HumanEval和MBPP的评估标准是否过松
- [x] 检查是否是执行引擎(gpt-4o-mini)而非Qwen的功劳

#### 5.3 正确的评估流程应该是什么
- [x] 明确训练阶段的职责分工
- [x] 明确评估阶段的职责分工
- [x] 对比参考项目的评估方式

**关键发现**:

**❌ 5.1-5.2 评估脚本完全错误!**

**根本问题**: `eval_6datasets.py`评估的是**Qwen直接生成答案**，而非**Qwen生成workflow → gpt-4o-mini执行workflow**

**证据A: eval_6datasets.py实现**
```python
# 行89-104: generate方法直接让Qwen生成答案
def generate(self, prompt: str, max_tokens: int = 512) -> str:
    inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
    with torch.no_grad():
        outputs = self.model.generate(  # ❌ Qwen直接生成
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.95,
            do_sample=False,
        )
    response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()

# 行151-176: 评估循环
for sample in test_samples:
    prediction = self.generate(question, max_tokens=512)  # ❌ 直接生成答案
    is_correct = self._check_correctness(dataset_name, prediction, reference_answer)
```

**证据B: 代码任务100%的虚假原因**
```python
# 行192-194: 代码任务的"准确性"检查
def _check_correctness(self, dataset_name: str, prediction: str, reference: str) -> bool:
    if dataset_name in ["humaneval", "mbpp"]:
        # ❌ 只要包含"def"或"return"就算正确！
        return "def " in prediction or "return" in prediction
```

**这意味着**: 任何包含函数定义或return语句的输出都被认为"正确"，完全不检查:
- 代码是否正确实现了功能
- 代码是否能通过测试用例
- 代码逻辑是否正确

**✅ 5.3 正确的评估流程应该是:**

**训练阶段** (已正确):
```
问题 → Qwen生成workflow代码 → AFlow执行workflow → gpt-4o-mini运行算子 → 答案 → LLM Judge评估
      (RL策略模型)           (工作流引擎)       (执行引擎)            (评估器)
```

**评估阶段** (当前错误 vs 应该):
```
❌ 当前错误:
问题 → Qwen直接生成答案 → 简单字符串匹配 → "准确率"
      (错误用法)          (启发式检查)

✅ 应该:
问题 → Qwen生成workflow代码 → AFlow执行workflow → gpt-4o-mini运行算子 → 答案 → 准确性评估
      (RL策略模型)           (工作流引擎)       (执行引擎)      (精确匹配/LLM Judge)
```

**核心错误**:
1. **评估的不是训练的东西**: 训练的是"workflow生成能力"，评估的是"直接答题能力"
2. **混淆了模型角色**: Qwen是workflow生成器，不是答题器
3. **代码100%是假象**: 只检查有没有"def"关键词，不检查代码正确性
4. **数学92%也可疑**: 简单的数字提取匹配，可能误判
5. **QA 19-21%**: 词汇重叠≥3个词就算对，标准过松

**结论**: 评估脚本需要完全重写，使用与训练一致的流程

---

## 调查方法论

### 原则
1. **追溯源码**: 不依赖文档，直接阅读实现
2. **数据证据**: 从训练日志提取实际数据
3. **对比参考**: 与参考项目(如有)进行对比
4. **根因分析**: 不止于表象，深挖底层原因

### 工具
- Grep: 搜索关键代码模式
- Read: 深度阅读核心文件
- Bash: 统计日志数据
- Task: 探索复杂问题

---

## 进度追踪

- [x] 问题1调查完成 - **严重问题**: 双重奖励系统 + workflow执行频繁失败
- [ ] 问题2调查完成
- [x] 问题3调查完成 - **中等问题**: Fallback触发166次（非频繁）
- [ ] 问题4调查完成
- [x] 问题5调查完成 - **严重问题**: 评估脚本完全错误，评估!=训练
- [ ] 整合发现，制定解决方案
- [ ] 实施修改
- [ ] 验证修改效果
