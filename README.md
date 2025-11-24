# LLM-as-Judge: GRPO Training Framework for Mixed-Domain LLM Evaluation

A production-ready framework for training language models using **Group Relative Policy Optimization (GRPO)** on balanced mixed-domain datasets (Math, QA, Code).

## 🚀 Quick Start

```bash
# Full automation
./scripts/run_full_pipeline.sh --model qwen25-7b --device cuda:0
```

**New to the project?**
👉 [Read the SETUP guide](docs/SETUP.md) | 📚 [Full documentation](docs/README.md)

---

## 📋 Documentation

All documentation has been organized in the `docs/` directory:

| Document | Purpose |
|----------|---------|
| [README.md](docs/README.md) | 📖 Complete project overview and architecture |
| [SETUP.md](docs/SETUP.md) | 🔧 Installation and environment setup |
| [INSTALLATION.md](docs/INSTALLATION.md) | 📥 Detailed installation steps |
| [DATA.md](docs/DATA.md) | 📊 Data mixing strategy (5:1 split, domain balance) |
| [TRAINING.md](docs/TRAINING.md) | 🎓 Training configuration and modes |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | 🤝 How to contribute |

---

## 🎯 Key Features

✅ **6-Dataset Mixed Training**: GSM8K, MATH, SQuAD2.0, HotpotQA, HumanEval, MBPP
✅ **Production-Ready GRPO**: Online learning without replay buffer
✅ **Multi-Model Support**: Qwen2.5-7B and Qwen-3-8B with LoRA (rank=64)
✅ **Domain-Balanced Sampling**: 5:1 train/test split, 4:3:3 cross-domain ratio
✅ **LLM Judge Integration**: gpt-4o for semantic evaluation
✅ **Complete Automation**: Download → Process → Train → Evaluate

---

## 🏗️ Project Structure

```
llm-as-judge/
├── config/                    # Configuration files (models, datasets, training)
├── docs/                      # Full documentation
├── src/                       # Core training code (15 modules)
├── tests/                     # Test suite (unit, integration, e2e)
├── scripts/                   # Data processing and evaluation scripts
├── train.py                   # Training entry point
└── requirements.txt           # Dependencies
```

---

## 📞 Support

- 🐛 [Report Issues](https://github.com/anthropics/claude-code/issues)
- 📖 [View Full Docs](docs/README.md)
- 🔗 [Related Project: AFlow](https://github.com/geekan/MetaGPT)

---

**Generated with Claude Code** 🤖
