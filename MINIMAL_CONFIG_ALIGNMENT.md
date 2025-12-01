# Minimal Training Config 对齐总结

**日期**: 2025-12-01
**任务**: 对齐 minimal_training.yaml 到 training.yaml
**状态**: ✅ 完成

---

## 修改摘要

### 目标
- ✅ 将 minimal_training.yaml 的训练参数对齐到 training.yaml
- ✅ 保留 max_steps（改为10步，用户要求）
- ✅ 保留 save_every（保留5步的checkpoint间隔）
- ✅ 其他所有参数对齐到 training.yaml

### 修改结果

| 参数 | training.yaml | minimal_training.yaml (修改前) | minimal_training.yaml (修改后) | 状态 |
|------|--------------|------------------------------|-------------------------------|------|
| **max_steps** | 500 | 15 | **10** | ✅ 改为10（用户要求） |
| **rollout_batch_size** | 4 | 8 | **4** | ✅ 对齐 |
| **learning_rate** | 2.0e-5 | 0.00005 | **2.0e-5** | ✅ 对齐 |
| **warmup_steps** | 100 | 3 | **2** | ✅ 对齐（按比例：10/500*100=2） |
| **lora_rank** | 64 | 32 | **64** | ✅ 对齐 |
| **lora_alpha** | 64 | 16 | **64** | ✅ 对齐 |
| **lora_dropout** | 0.05 | 0.1 | **0.05** | ✅ 对齐 |
| **lora_target_modules** | q_proj,k_proj,v_proj,o_proj | q_proj,v_proj,up_proj,down_proj | **q_proj,k_proj,v_proj,o_proj** | ✅ 对齐 |
| **temperature** | 0.4 | 0.2 | **0.4** | ✅ 对齐 |
| **max_tokens** | 4096 | 2048 | **4096** | ✅ 对齐 |
| **top_k** | 50 | 40 | **50** | ✅ 对齐 |
| **log_every** | 5 | 1 | **5** | ✅ 对齐 |
| **val_samples** | 50 | 10 | **50** | ✅ 对齐 |
| **save_every** | 25 | 5 | **5** | ✅ 保留（适合快速测试） |
| **experience_buffer.buffer_size** | 100 | 20 | **100** | ✅ 对齐 |
| **prompt_optimizer.enabled** | true | false | **true** | ✅ 对齐 |
| **operator_prompt_enhancer.enabled** | true | false | **true** | ✅ 对齐 |
| **wandb.project** | aflow-roll-integration | aflow-roll-minimal-test | **aflow-roll-integration** | ✅ 对齐 |
| **wandb.run_name** | grpo-500steps-4batch-6workflows-reference-restored | grpo-15steps-8batch-6workflows-tuned-planD | **grpo-10steps-4batch-6workflows-minimal-aligned** | ✅ 更新 |

---

## 参数对比详解

### 1. 核心训练参数

**修改前**:
```yaml
max_steps: 15                      # 15步测试
rollout_batch_size: 8              # 8个样本/批
learning_rate: 0.00005             # 0.00005
warmup_steps: 3                    # 3步预热
```

**修改后**:
```yaml
max_steps: 10                      # ✅ 改为10步（用户要求）
rollout_batch_size: 4              # ✅ 对齐training.yaml
learning_rate: 2.0e-5              # ✅ 对齐training.yaml（更高效率）
warmup_steps: 2                    # ✅ 对齐比例（10步的20%）
```

**说明**:
- batch_size 从 8 → 4: 减少显存需求，与标准配置一致
- learning_rate 从 5e-5 → 2e-5: 实际上是降低（5e-5 > 2e-5），更温和的权重更新
- warmup_steps 按 10/500 * 100 = 2 步计算，保持相同的预热比例

### 2. 生成配置

**修改前**:
```yaml
temperature: 0.2
max_tokens: 2048
top_k: 40
```

**修改后**:
```yaml
temperature: 0.4           # ✅ 对齐training.yaml（更高的采样多样性）
max_tokens: 4096           # ✅ 对齐training.yaml（防止截断）
top_k: 50                  # ✅ 对齐training.yaml
```

**说明**:
- temperature 0.4 提高模型探索能力
- max_tokens 4096 足以覆盖所有代码和测试用例
- top_k 50 增加采样多样性

### 3. LoRA 配置

**修改前**:
```yaml
lora_rank: 32
lora_alpha: 16
lora_dropout: 0.1
lora_target_modules: "q_proj,v_proj,up_proj,down_proj"
```

**修改后**:
```yaml
lora_rank: 64                      # ✅ 对齐training.yaml（更强表达能力）
lora_alpha: 64                     # ✅ 对齐（维持 alpha/rank = 1.0）
lora_dropout: 0.05                 # ✅ 对齐（减少过度正则化）
lora_target_modules: "q_proj,k_proj,v_proj,o_proj"  # ✅ 标准modules
```

**说明**:
- rank 64 提供充分的适配空间（之前32可能不足）
- alpha/rank = 1.0 是标准LoRA缩放比例
- dropout 0.05 比 0.1 的正则化更温和

### 4. 评估和记录

**修改前**:
```yaml
log_every: 1              # 每步记录
val_samples: 10           # 仅10个验证样本
save_every: 5             # 每5步保存
```

**修改后**:
```yaml
log_every: 5              # ✅ 对齐training.yaml（减少日志噪声）
val_samples: 50           # ✅ 对齐training.yaml（更充分的验证）
save_every: 5             # ✅ 保留（适合快速测试）
```

**说明**:
- log_every 5 既能监控进度，又不过度日志输出
- val_samples 50 与training.yaml一致
- save_every 5 在10步测试中是合理的（每2步一次checkpoint）

### 5. 功能开关

**修改前**:
```yaml
prompt_optimizer.enabled: false
operator_prompt_enhancer.enabled: false
```

**修改后**:
```yaml
prompt_optimizer.enabled: true        # ✅ 对齐training.yaml
operator_prompt_enhancer.enabled: true # ✅ 对齐training.yaml
```

**说明**:
- 在minimal test中也应启用这些优化
- 可以测试这些功能是否引入问题

---

## 总样本数计算

### 修改前
```
总样本数 = max_steps × batch_size × num_return_sequences
         = 15 × 8 × 6
         = 720 个工作流
```

### 修改后
```
总样本数 = max_steps × batch_size × num_return_sequences
         = 10 × 4 × 6
         = 240 个工作流（快速测试）
```

---

## 关键配置对齐清单

### ✅ 完全对齐参数（与training.yaml相同）

| 参数 | 值 | 用途 |
|------|-----|------|
| rollout_batch_size | 4 | 标准批大小 |
| learning_rate | 2.0e-5 | 学习效率 |
| lora_rank | 64 | LoRA表达能力 |
| lora_alpha | 64 | LoRA缩放 |
| lora_dropout | 0.05 | 正则化 |
| lora_target_modules | q_proj,k_proj,v_proj,o_proj | 适配层 |
| temperature | 0.4 | 采样多样性 |
| max_tokens | 4096 | 防止截断 |
| top_k | 50 | 采样范围 |
| top_p | 0.95 | 采样阈值 |
| eval_every | 0 | 评估策略 |
| val_samples | 50 | 验证样本数 |
| log_every | 5 | 日志频率 |
| domain_ratios | math:0.4, qa:0.3, code:0.3 | 采样比例 |
| reward_weights | correctness:0.7, efficiency:0.2, code_quality:0.1 | 奖励权重 |
| experience_buffer.buffer_size | 100 | Buffer大小 |
| prompt_optimizer.enabled | true | 提示词优化 |
| operator_prompt_enhancer.enabled | true | Operator增强 |

### ⚠️ 保留不同的参数（用于快速测试）

| 参数 | training.yaml | minimal_training.yaml | 原因 |
|------|--------------|---------------------|------|
| max_steps | 500 | 10 | 用户要求（快速测试） |
| save_every | 25 | 5 | 适合10步测试（5步间隔） |

---

## 验证

✅ 所有参数已验证
✅ 配置文件语法正确
✅ 所有对齐参数值匹配 training.yaml
✅ max_steps 正确改为 10
✅ save_every 保留为 5

---

## 用途

**minimal_training.yaml** 适用于：
- ✅ 快速功能测试（10步，~10分钟）
- ✅ 检查Plan B实现是否有问题
- ✅ 验证新的计算图是否正确
- ✅ 测试AFlow执行器集成
- ✅ 验证数据加载和处理流程

**参数对齐意义**：
- ✅ 确保minimal test中使用的是标准配置
- ✅ 任何在minimal test中出现的问题也会在full training中出现
- ✅ 避免minimal test通过但full training失败的情况

---

## 快速参考

```bash
# 快速测试（10步）
python train.py --config config/minimal_training.yaml

# 完整训练（500步）
python train.py --config config/training.yaml
```

### 参数对比速查

```
minimal (10步快速测试)  vs  training (500步完整训练)
┌────────────────────┬───────────────┬──────────────┐
│ 参数               │ minimal       │ training     │
├────────────────────┼───────────────┼──────────────┤
│ max_steps          │ 10            │ 500          │
│ 总样本数           │ 240           │ 12,000       │
│ rollout_batch_size │ 4             │ 4            │
│ learning_rate      │ 2.0e-5        │ 2.0e-5       │
│ warmup_steps       │ 2             │ 100          │
│ lora_rank          │ 64            │ 64           │
│ temperature        │ 0.4           │ 0.4          │
│ max_tokens         │ 4096          │ 4096         │
│ save_every         │ 5             │ 25           │
└────────────────────┴───────────────┴──────────────┘
```

---

**对齐完成** ✅
**配置文件**: /root/llm-as-judge-new/config/minimal_training.yaml
**参考配置**: /root/llm-as-judge-new/config/training.yaml

