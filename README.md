# LLM-as-Judge: GRPO Training Framework for Mixed-Domain LLM Evaluation

A production-ready framework for training language models using **Group Relative Policy Optimization (GRPO)** on balanced mixed-domain datasets (Math, QA, Code).

## 🎯 Current Status

✅ **Plan B Implementation Complete** (Soft Learning with Metadata Flags)
- Constraint violations → metadata flags + GRPO penalties (instead of hard blocks)
- Three-level penalty hierarchy: -5.0 (mismatch), -3.0 (validation), -8.0/-7.0/-10.0 (errors)
- 100% test pass rate (26/26 tests)
- [Details](PLAN_B_SESSION_SUMMARY.md)

✅ **Configuration Restored** (Reference Project Alignment)
- LoRA rank: 64 | Batch size: 4 | Learning rate: 2.0e-5 | Temperature: 0.4
- Max tokens increased to 4096 (prevent truncation)
- [Config Details](CONFIG_RESTORATION_SUMMARY.md)

## 🚀 Quick Start

```bash
# Start training with Plan B configuration
python train.py --config config/training.yaml
```

**New to the project?**
👉 [Read the SETUP guide](docs/SETUP.md) | 📚 [Full documentation](docs/README.md) | 📋 [Plan B Summary](PLAN_B_SESSION_SUMMARY.md)

---

## 📋 Documentation

### Core Documentation (docs/)

| Document | Purpose |
|----------|---------|
| [README.md](docs/README.md) | 📖 Complete project overview and architecture |
| [SETUP.md](docs/SETUP.md) | 🔧 Installation and environment setup |
| [INSTALLATION.md](docs/INSTALLATION.md) | 📥 Detailed installation steps |
| [DATA.md](docs/DATA.md) | 📊 Data mixing strategy (5:1 split, domain balance) |
| [TRAINING.md](docs/TRAINING.md) | 🎓 Training configuration and modes |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | 🤝 How to contribute |

### Implementation Documentation (Root)

| Document | Purpose |
|----------|---------|
| [PLAN_B_SESSION_SUMMARY.md](PLAN_B_SESSION_SUMMARY.md) | 🎯 Plan B soft learning approach overview |
| [PLAN_B_IMPLEMENTATION_VERIFICATION.md](PLAN_B_IMPLEMENTATION_VERIFICATION.md) | ✅ Plan B test results (26/26 tests pass) |
| [CONFIG_RESTORATION_SUMMARY.md](CONFIG_RESTORATION_SUMMARY.md) | ⚙️ Configuration parameter recovery details |
| [CONFIG_QUICK_REFERENCE.txt](CONFIG_QUICK_REFERENCE.txt) | 📌 Quick parameter reference card |
| [IMPLEMENTATION_COMPLETE.txt](IMPLEMENTATION_COMPLETE.txt) | 📝 Complete implementation status report |

---

## 🎯 Key Features

✅ **Plan B Soft Learning**: Operator constraints via metadata flags + GRPO penalties (not hard blocks)
✅ **6-Dataset Mixed Training**: GSM8K, MATH, SQuAD2.0, HotpotQA, HumanEval, MBPP
✅ **Production-Ready GRPO**: Online learning with three-tier penalty hierarchy
✅ **Multi-Model Support**: Qwen2.5-7B with LoRA (rank=64, alpha=64)
✅ **Domain-Balanced Sampling**: 5:1 train/test split, 4:3:3 cross-domain ratio
✅ **LLM Judge Integration**: gpt-4o-mini for semantic evaluation & AFlow execution
✅ **Optimized Configuration**: LoRA rank=64, batch_size=4, learning_rate=2.0e-5, temperature=0.4

---

## 🏗️ Project Structure

```
llm-as-judge/
├── config/                    # Configuration files
│   ├── training.yaml          # GRPO training (Plan B optimized)
│   ├── aflow_llm.yaml         # AFlow executor config
│   └── aflow_operators.yaml   # Operator definitions
├── docs/                      # Core documentation
├── src/                       # Core training code (15 modules)
│   ├── aflow_executor.py      # Plan B: soft constraint detection
│   ├── grpo_trainer.py        # Plan B: three-tier penalty hierarchy
│   ├── rl_workflow_generator.py # Plan B: soft generation guidance
│   ├── workflow_validator.py  # Plan B: warning mode validation
│   └── ... (10 more modules)
├── tests/                     # Test suite
├── scripts/                   # Data processing and evaluation
├── train.py                   # Training entry point
├── test_plan_b_changes.py     # Plan B verification (26/26 tests)
└── requirements.txt           # Dependencies
```

---

## 🚦 Training Workflow

```
1. Data Preparation
   └─ Download datasets (GSM8K, MATH, HumanEval, etc.)
   └─ Process and mix (5:1 train/test, 4:3:3 domains)

2. Model Training (Plan B)
   └─ Generate workflows (RL policy)
   └─ Execute with AFlow (gpt-4o-mini executor)
   └─ Compute rewards with three-tier penalty system
   └─ Update weights via GRPO gradients

3. Constraint Learning
   └─ Operator-problem type mismatch → -5.0 penalty
   └─ Validation failures → -3.0 penalty
   └─ Execution errors → -8.0 to -10.0 penalties
   └─ RL model learns constraints naturally

4. Monitoring & Evaluation
   └─ W&B tracking (metrics by domain and error type)
   └─ Checkpoint saving every 25 steps
   └─ No Fallback overhead (Plan B removes hard blocks)
```

## 📞 Getting Help

- 📖 **Full Docs**: [docs/README.md](docs/README.md)
- 🎯 **Quick Setup**: [docs/SETUP.md](docs/SETUP.md)
- 📊 **Training Guide**: [docs/TRAINING.md](docs/TRAINING.md)
- 💡 **Plan B Details**: [PLAN_B_SESSION_SUMMARY.md](PLAN_B_SESSION_SUMMARY.md)
- ⚙️ **Config Help**: [CONFIG_QUICK_REFERENCE.txt](CONFIG_QUICK_REFERENCE.txt)

---

**Implementation Status**: ✅ Complete (Plan B + Configuration Restored)
**Test Coverage**: ✅ 100% (26/26 tests pass)
**Generated with Claude Code** 🤖
