# 深度调查总结报告

**宗旨**: 不简化训练流程，不绕过问题而是解决问题的本质（不要打补丁）

---

## 🚨 核心发现总结

### 严重问题（影响训练效果）

**问题1: 双重奖励系统混乱 + Workflow执行频繁失败**
- **影响**: 模型无法学习，几乎所有样本都是-5.0惩罚
- **证据**: 训练日志显示绝大部分预测是"Code generation failed after 3 attempts"
- **根因**: reward_computer.py和grpo_trainer.py有两套独立的奖励系统，且Programmer算子频繁失败

**问题5: 评估脚本完全错误**
- **影响**: 评估结果毫无意义（代码100%是假象）
- **证据**: eval_6datasets.py让Qwen直接生成答案，而非生成workflow
- **根因**: 评估流程与训练流程完全不一致

**问题4: Workflow生成质量差**
- **影响**: 大量拼写错误（ll_m, lll, ll_config）
- **证据**: 训练日志行52169显示`'ll_m'`错误，应为`'llm'`
- **根因**: Few-shot示例不足以纠正变量名拼写

### 中等问题

**问题3: Fallback触发166次**
- **影响**: 中等（非频繁，但增加开销）
- **证据**: 日志中grep Fallback显示166次触发
- **根因**: Workflow执行失败时触发，与问题1相关

**问题2: 潜在的静默失败**
- **影响**: 低（需进一步验证）
- **证据**: 发现3个文件有`except: pass`模式
- **根因**: 某些异常被静默处理

---

## 📊 问题详细分析

### 问题1: 奖励计算系统混乱

**问题A: 双重奖励系统**
```python
# 系统1: reward_computer.py (metadata['success'] = True时使用)
答案质量: +10.0 (正确) 或 -5.0 (错误)
生成质量: -3.0 到 +3.0
归一化: total_score / 20.0

# 系统2: grpo_trainer.py (约束违规时使用)
Level 1: operator_mismatch → -5.0 (未归一化)
Level 2: validation_failed → -3.0 (未归一化)
Level 3: execution_error → -8.0/-7.0/-10.0 (未归一化)
```

**冲突**:
- 系统1归一化到[-0.5, 1.0]，系统2直接使用原始值
- 两个系统在不同条件下触发，没有统一框架
- GRPO优化器接收的奖励尺度不一致

**问题B: Workflow执行频繁失败**

训练日志模式（重复数千次）:
```
│ 答案质量奖励:       -5.0  ❌ 错误
│ 生成质量奖励:       +2.0
│ 总奖励:            -3.0
预测: # Code generation failed after 3 attempts
```

**影响**: 模型接收到的奖励信号几乎全是负数，无法学到正确行为

---

### 问题4: Workflow生成质量差

**证据**: 训练日志行52169-52281
```python
AttributeError: 'Workflow' object has no attribute 'll_m'. Did you mean: 'llm'?

生成的错误代码:
self.review = operator.Review(self.ll_m)  # ❌ 错误拼写
self.revise = operator.Revise(self.ll_m)  # ❌ 错误拼写
self.llm = create_ll_m_instance(llm_config)  # ❌ create_ll_m_instance
self.answer_generate = operator.AnswerGenerate(self.ll_m)  # ❌
return final_answer, self.ll_m.get_usage_summary()  # ❌
```

**根本原因分析**:

1. **不是模型能力问题**: Qwen2.5-7B完全有能力写对`llm`这个变量名
2. **是Few-shot示例不足**: 当前示例只有1个正确示例，不足以形成稳定模式
3. **是tokenizer分词问题**: `llm`可能被分为`ll` + `m`两个token，生成时容易插入下划线
4. **温度0.4过高**: 允许了过多随机性

---

### 问题5: 评估脚本根本错误

**训练流程** (正确):
```
问题 → Qwen生成workflow Python代码 → AFlow执行 → gpt-4o-mini运行算子 → 答案
```

**评估流程** (错误):
```
问题 → Qwen直接生成答案 → 字符串匹配检查
```

**代码100%的真相**:
```python
def _check_correctness(self, dataset_name: str, prediction: str, reference: str):
    if dataset_name in ["humaneval", "mbpp"]:
        return "def " in prediction or "return" in prediction  # ❌
```

任何包含`def`关键词的输出都算"正确"，即使:
- 函数逻辑完全错误
- 代码无法运行
- 根本没解决问题

---

### 问题2: 静默失败检查

**发现的`except: pass`位置**:
1. src/reward_computer.py
2. src/unified_evaluator.py
3. src/aflow_executor.py

**需要审查**: 这些异常处理是否掩盖了重要错误

---

### 问题3: Fallback触发分析

**统计**: 166次触发
**频率**: 非频繁（第一步就有Fallback，说明工作流生成失败导致）
**类型**:
- QA专用Fallback（不含Test算子）
- MATH Fallback
- 空答案触发Fallback

**结论**: Fallback本身不是主要问题，是workflow执行失败的下游症状

---

## 🎯 根本原因链

```
根本原因: Workflow生成质量差（拼写错误、语法错误）
    ↓
导致: Workflow执行失败（AttributeError, NoneType等）
    ↓
触发: Fallback机制 (166次) 或返回失败消息
    ↓
结果: 答案几乎都是"Code generation failed"
    ↓
反馈: LLM Judge判定全错误 → 奖励-5.0
    ↓
学习: 模型收到全负反馈 → 无法学习正确行为
    ↓
循环: 下一轮继续生成低质量workflow
```

**核心问题**: 不是某个单一bug，而是一个恶性循环

---

## ✅ 解决方案框架

### 优先级1: 修复Workflow生成质量（问题4）

**方案**:
1. **增加Few-shot示例数量**: 从1个增加到3-5个
2. **降低temperature**: 从0.4降低到0.1-0.2（减少随机性）
3. **强化变量名约束**: 在prompt中明确列出禁止的拼写
4. **添加语法检查**: 生成后立即用ast.parse验证

### 优先级2: 统一奖励系统（问题1）

**方案**:
1. **合并两套系统**: 只保留reward_computer.py
2. **统一归一化**: 所有奖励值归一化到同一尺度
3. **清晰的奖励层级**:
   - 正确答案: +1.0
   - 错误答案但执行成功: -0.2 到 -0.5
   - 约束违规: -0.6 到 -0.8
   - 完全失败: -1.0

### 优先级3: 重写评估脚本（问题5）

**方案**:
1. **使用训练流程**: Qwen生成workflow → AFlow执行
2. **精确的准确性检查**:
   - 代码: 运行测试用例
   - 数学: 数值比较或LLM Judge
   - QA: F1分数或LLM Judge

### 优先级4: 审查静默失败（问题2）

**方案**:
1. 检查3个文件的except块
2. 确保关键错误被记录
3. 不掩盖训练信号

---

## 📋 行动清单

**阶段1: 紧急修复（必须）**
- [ ] 降低temperature到0.1-0.2
- [ ] 增加Few-shot示例到3个
- [ ] 添加workflow语法验证
- [ ] 统一奖励归一化

**阶段2: 重构（重要）**
- [ ] 重写评估脚本使用正确流程
- [ ] 合并双重奖励系统
- [ ] 审查并修复静默失败

**阶段3: 验证（必要）**
- [ ] 重新训练20步
- [ ] 检查workflow生成质量
- [ ] 检查奖励分布
- [ ] 使用正确评估脚本测试

---

## 🔬 验证指标

**成功标准**:
1. Workflow生成拼写错误率 < 5%
2. Workflow执行成功率 > 70%
3. 奖励分布: 正负样本比例接近实际准确率
4. 评估准确率下降（因为之前是虚假高分）

**失败信号**:
1. 仍然大量"Code generation failed"
2. 奖励仍然全是-5.0
3. Workflow仍有ll_m拼写错误

---

**调查完成日期**: 2025-12-02
**调查者**: Claude (Sonnet 4.5 Thinking)
**调查文件**: INVESTIGATION_ISSUES.md
