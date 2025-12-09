#!/usr/bin/env python3
"""
GRPO训练器 - 在线学习模式的强化学习训练器
"""
import os
import torch
import torch.nn.functional as F
import asyncio
import numpy as np
import gc  # ✅ PHASE 5 FIX: For memory cleanup
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import json
import wandb  # ✨ 新增wandb集成

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_manager import DataManager
from rl_workflow_generator import RLWorkflowGenerator
from aflow_executor import AFlowExecutor
from reward_computer import RewardComputer
from gpu_manager import GPUManager
from experience_buffer import ExperienceBuffer
from prompt_optimizer import PromptOptimizer
from operator_prompt_enhancer import OperatorPromptEnhancer


class GRPOTrainer:
    """GRPO训练器：在线学习模式"""

    def __init__(self, config_path: str = "config/training.yaml",
                 model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Args:
            config_path: 训练配置文件路径
            model_name: 模型名称 (qwen25-7b, qwen3-8b) - 会覆盖config配置
            device: GPU设备 (cuda:0, cuda:1等) - 会覆盖config配置
            output_dir: 检查点输出目录 - 会覆盖config配置
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 处理模型名称覆盖
        if model_name:
            # 模型名称到base_model路径的映射
            model_mapping = {
                "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
                "qwen3-8b": "Qwen/Qwen-3-8B"
            }
            if model_name in model_mapping:
                self.config['base_model'] = model_mapping[model_name]
                print(f"✅ 覆盖base_model: {self.config['base_model']}")

        # 处理设备覆盖
        if device:
            # 从设备字符串解析GPU ID（如 cuda:0 -> [0]）
            if device.startswith("cuda:"):
                gpu_id = int(device.split(":")[-1])
                self.config['device_mapping'] = [gpu_id]
                self.config['physical_gpus'] = [gpu_id]
                print(f"✅ 覆盖设备: {device}")

        # 处理输出目录覆盖
        if output_dir:
            self.config['checkpointing'] = self.config.get('checkpointing', {})
            self.config['checkpointing']['save_dir'] = output_dir
            print(f"✅ 覆盖输出目录: {output_dir}")

        print("=" * 60)
        print("🚀 初始化GRPO训练器")
        print("=" * 60)

        # GPU管理（使用物理GPU ID）
        physical_gpus = self.config.get('physical_gpus', self.config['device_mapping'])
        self.gpu_manager = GPUManager(
            target_gpus=physical_gpus,
            protected_pids=self.config.get('protected_pids', []),
            auto_clean=False  # 禁用自动清理
        )

        # 跳过GPU环境验证，直接使用
        print(f"✅ 使用GPU {physical_gpus}（已禁用清理和验证）")

        # 简化配置：使用固定generation config
        gen_config = self.config.get('generation_config', {})
        self.generation_temperature = gen_config.get('temperature', 0.4)
        print(f"\n🌡️  Generation Config:")
        print(f"  Temperature: {self.generation_temperature} (fixed)")

        # ✨ 初始化wandb
        self._initialize_wandb()

        # 初始化组件
        self._initialize_components()

        print("=" * 60)
        print("✅ GRPO训练器初始化完成")
        print("=" * 60)

    def _initialize_wandb(self):
        """初始化wandb监控"""
        # 从配置或环境变量获取wandb设置
        wandb_config = self.config.get('wandb', {})

        # 设置API key(如果提供的话)
        wandb_api_key = wandb_config.get('api_key', os.getenv('WANDB_API_KEY'))

        # 尝试登录,如果失败则使用offline模式
        try:
            if wandb_api_key and len(wandb_api_key) == 40:
                wandb.login(key=wandb_api_key)
                mode = "online"
            else:
                print("⚠️  wandb API key无效或未提供,使用offline模式")
                mode = "offline"
        except Exception as e:
            print(f"⚠️  wandb登录失败: {e}, 使用offline模式")
            mode = "offline"

        # 初始化wandb run
        wandb.init(
            project=wandb_config.get('project', 'aflow-roll-integration'),
            name=wandb_config.get('run_name', f"grpo-training-{time.strftime('%Y%m%d-%H%M%S')}"),
            mode=mode,  # online或offline
            config={
                # 训练配置
                "base_model": self.config['base_model'],
                "learning_rate": self.config['learning_rate'],
                "batch_size": self.config['rollout_batch_size'],
                "num_sequences": self.config['num_return_sequences_in_group'],
                "max_steps": self.config['max_steps'],
                "lora_rank": self.config['lora_rank'],
                "lora_alpha": self.config['lora_alpha'],
                # 数据配置
                "domain_ratios": self.config['domain_ratios'],
                # 奖励配置
                "reward_weights": self.config.get('reward_weights', {}),
            },
            tags=["grpo", "aflow", "roll", "workflow-generation"],
            notes="GRPO training with improved reward function (ROLL+AgentFlow design)"
        )

        print("\n✅ wandb初始化完成")
        print(f"  模式: {mode}")
        print(f"  项目: {wandb.run.project}")
        print(f"  Run名称: {wandb.run.name}")
        if mode == "online":
            print(f"  Run URL: {wandb.run.url}")
        else:
            print(f"  离线日志: wandb/offline-run-*")

    def _initialize_components(self):
        """初始化所有组件"""

        # 1. 数据管理器
        print("\n📂 初始化数据管理器...")
        self.data_manager = DataManager(
            data_dir=self.config['data_dir'],
            domain_ratios=self.config['domain_ratios']
        )
        self.data_manager.initialize()

        # 2. RL模型（Qwen2.5-7B + LoRA）
        print("\n🤖 加载RL模型...")
        self._load_rl_model()

        # 3. RL工作流生成器（使用共享模型）
        print("\n🔧 初始化工作流生成器...")
        self.generator = RLWorkflowGenerator(
            model=self.model,  # ✨ Pass shared model reference
            tokenizer=self.tokenizer,  # ✨ Pass shared tokenizer
            device=self.model.device,  # ✨ Pass shared device
            operator_descriptions_path=self.config.get('aflow_operator_descriptions_path')
        )
        print(f"  ✅ 模型共享验证:")
        print(f"    Trainer模型ID: {id(self.model)}")
        print(f"    Generator模型ID: {id(self.generator.model)}")
        if id(self.model) == id(self.generator.model):
            print(f"    ✅ 模型共享成功！节省 ~15GB GPU内存")
        else:
            print(f"    ❌ 警告: 模型未共享，存在内存浪费！")

        # 4. ExperienceBuffer - 高质量样本管理（需先初始化，用于后续组件）
        print("\n📚 初始化ExperienceBuffer...")
        experience_config = self.config.get('experience_buffer', {})
        self.experience_buffer = ExperienceBuffer(
            buffer_size=experience_config.get('buffer_size', 100),
            reward_threshold=experience_config.get('reward_threshold', 8.0),
            persistence_dir=experience_config.get('persistence_dir', 'data/experience_buffer'),
            problem_types=["math", "code", "qa"]
        )
        print(f"  Buffer大小: {self.experience_buffer.buffer_size}")
        print(f"  奖励阈值: {self.experience_buffer.reward_threshold}")

        # 5. PromptOptimizer - Layer 1动态提示词优化
        print("\n✨ 初始化PromptOptimizer (Layer 1)...")
        prompt_config = self.config.get('prompt_optimizer', {})
        self.prompt_optimizer = PromptOptimizer()
        self.use_dynamic_prompts = prompt_config.get('enabled', True)
        print(f"  动态提示词: {'启用' if self.use_dynamic_prompts else '禁用'}")

        # 6. OperatorPromptEnhancer - Layer 2 operator提示词增强
        print("\n🔧 初始化OperatorPromptEnhancer (Layer 2)...")
        operator_config = self.config.get('operator_prompt_enhancer', {})
        self.operator_enhancer = OperatorPromptEnhancer(
            enable_enhancement=operator_config.get('enabled', True)
        )
        print(f"  Operator增强: {'启用' if self.operator_enhancer.enable_enhancement else '禁用'}")

        # 7. AFlow执行器（传入operator_enhancer）
        print("\n⚙️  初始化AFlow执行器...")
        timeout = self.config.get('execution_timeout', 180)  # 默认180秒

        # 读取fallback配置
        fallback_enabled = self.config.get('reward_system', {}).get('fallback', True)

        self.executor = AFlowExecutor(
            llm_config_path=self.config['aflow_config_path'],
            timeout=timeout,
            operator_enhancer=self.operator_enhancer,  # 传递Layer 2增强器
            enable_fallback=fallback_enabled  # 传递fallback配置
        )
        print(f"  执行超时: {timeout}秒")
        print(f"  Fallback机制: {'启用' if fallback_enabled else '禁用'}")

        # 8. 奖励计算器 - ✨ PHASE 1: NEW 5-tier reward system
        print("\n🎯 初始化奖励计算器 (5-Tier System V2)...")
        use_llm_judge = False  # Set to True if OpenAI API key available
        if os.getenv("OPENAI_API_KEY"):
            use_llm_judge = True
            print("  ✅ LLM Judge enabled (gpt-4o-mini)")
        else:
            print("  ⚠️  LLM Judge disabled (OPENAI_API_KEY not found)")

        self.reward_computer = RewardComputer(
            use_answer_extractor=True,  # ✨ Use enhanced 6-level extraction
            use_llm_judge=use_llm_judge,
            llm_config={
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model_name": "gpt-4o-mini"
            } if use_llm_judge else None
        )

        # 9. 优化器
        print("\n🔬 初始化优化器...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )

    def _load_rl_model(self):
        """加载RL模型（Qwen2.5-7B + LoRA）"""
        device = f"cuda:{self.config['device_mapping'][0]}"

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            torch_dtype=torch.bfloat16 if self.config.get('bf16') else torch.float16,
            device_map={"": device},
            trust_remote_code=True
        )

        # ✅ FUNDAMENTAL FIX: Enable gradient checkpointing
        # Trade compute for memory - reduces peak memory by ~40-50%
        if self.config.get('use_gradient_checkpointing', True):
            self.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled (trade compute for memory)")

        # 应用LoRA
        if self.config.get('use_lora', True):
            lora_config = LoraConfig(
                r=self.config['lora_rank'],
                lora_alpha=self.config['lora_alpha'],
                target_modules=self.config['lora_target_modules'].split(','),
                lora_dropout=self.config['lora_dropout'],
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)

            print(f"✅ LoRA应用完成")
            self.model.print_trainable_parameters()

        # ✨ Log GPU memory after model loading
        self._log_gpu_memory("Model Loaded")

    async def _process_sample_batch_parallel(self, batch, num_sequences, current_temp):
        """
        🚀 Performance Fix: Parallel processing of workflow generation and execution
        Processes multiple samples concurrently using asyncio.gather
        """
        import asyncio
        from tqdm import tqdm

        # Create a semaphore to limit concurrent API calls (avoid rate limiting)
        semaphore = asyncio.Semaphore(8)  # Max 8 concurrent workflows

        async def process_single_sample_with_semaphore(sample):
            async with semaphore:
                return await self._process_single_sample(sample, num_sequences, current_temp)

        # Process all samples in parallel
        print(f"\n🚀 Parallel processing {len(batch)} samples with {num_sequences} sequences each")
        results = await asyncio.gather(
            *[process_single_sample_with_semaphore(sample) for sample in batch],
            return_exceptions=True
        )

        # Collect successful results and handle exceptions
        all_workflows = []
        all_answers = []
        all_rewards = []
        all_log_probs = []
        all_problem_types = []
        all_ground_truths = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠️ Sample {i} failed: {result}")
                continue

            workflows, answers, rewards, log_probs, problem_types, ground_truths = result
            all_workflows.extend(workflows)
            all_answers.extend(answers)
            all_rewards.extend(rewards)
            all_log_probs.extend(log_probs)
            all_problem_types.extend(problem_types)
            all_ground_truths.extend(ground_truths)

        return all_workflows, all_answers, all_rewards, all_log_probs, all_problem_types, all_ground_truths

    async def _process_single_sample(self, sample, num_sequences, current_temp):
        """Process a single sample and generate multiple workflows"""
        problem = sample['problem']
        ground_truth = sample['ground_truth']
        problem_type = sample['problem_type']

        # Storage for this sample's workflows
        workflows = []
        answers = []
        rewards = []
        log_probs = []
        problem_types = []
        ground_truths = []

        # Process all sequences for this sample in parallel
        async def process_single_sequence(i):
            # Build dynamic prompt if enabled
            custom_prompt = None
            if self.use_dynamic_prompts:
                custom_prompt = self.prompt_optimizer.build_dynamic_prompt(
                    problem=problem,
                    problem_type=problem_type
                )

            # Generate workflow
            result = self.generator.generate_workflow(
                problem=problem,
                problem_type=problem_type,
                temperature=current_temp,
                custom_prompt=custom_prompt
            )

            workflow_code = result['workflow_code']

            # Compute log probability
            log_prob = await self._compute_log_prob(problem, workflow_code, problem_type)

            # Execute workflow
            try:
                answer, cost, metadata = await self.executor.execute_workflow(
                    workflow_code=workflow_code,
                    problem=problem,
                    problem_type=problem_type,
                    entry_point=sample.get('entry_point', ''),
                    test=sample.get('test', '')
                )

                # Compute reward
                reward_result = self.reward_computer.compute_reward(
                    problem=problem,
                    prediction=answer,
                    ground_truth=ground_truth,
                    problem_type=problem_type,
                    execution_metadata=metadata
                )
                # Extract float reward value from dict
                reward = reward_result.get('reward', 0.0) if isinstance(reward_result, dict) else reward_result

                return workflow_code, answer, reward, log_prob

            except Exception as e:
                print(f"⚠️ Workflow execution failed: {e}")
                # Return failure values
                return workflow_code, "", 0.0, log_prob

        # Process all sequences in parallel
        sequence_results = await asyncio.gather(
            *[process_single_sequence(i) for i in range(num_sequences)],
            return_exceptions=True
        )

        # Collect results
        for result in sequence_results:
            if isinstance(result, Exception):
                print(f"⚠️ Sequence failed: {result}")
                continue

            workflow, answer, reward, log_prob = result
            workflows.append(workflow)
            answers.append(answer)
            rewards.append(reward)
            log_probs.append(log_prob)
            problem_types.append(problem_type)
            ground_truths.append(ground_truth)

        return workflows, answers, rewards, log_probs, problem_types, ground_truths

    def _log_gpu_memory(self, stage: str):
        """Log current GPU memory usage

        Args:
            stage: Description of current training stage
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free = total - allocated

            print(f"\n🔍 GPU Memory [{stage}]:")
            print(f"  📊 Allocated: {allocated:.2f} GB")
            print(f"  📦 Reserved: {reserved:.2f} GB")
            print(f"  ✅ Free: {free:.2f} GB")
            print(f"  💾 Total: {total:.2f} GB")
            print(f"  📈 Usage: {(allocated/total)*100:.1f}%")


    async def train_step(self, step: int) -> Dict:
        """
        单步GRPO训练（在线学习）

        Returns:
            metrics: 训练指标
        """

        # 1. 采样batch
        batch = self.data_manager.sample_batch(
            batch_size=self.config['rollout_batch_size'],
            split="train"
        )

        # 统计
        batch_stats = self.data_manager.get_batch_stats(batch)
        print(f"\n📦 Batch {step}: {len(batch)} 样本, 分布: {batch_stats}")

        # 使用固定temperature（简化版）
        current_temp = self.generation_temperature
        print(f"🌡️  Temperature: {current_temp:.3f}")

        # 2. 为每个问题生成K个工作流（GRPO组）
        all_workflows = []
        all_problems = []
        all_answers = []
        all_rewards = []
        all_log_probs = []

        # ✨ 新增：准确率统计
        correctness_scores = []  # 存储所有正确性分数

        num_sequences = self.config['num_return_sequences_in_group']

        # 🚀 Performance Fix: Use parallel processing instead of sequential
        print(f"\n🚀 Using parallel processing for {len(batch)} samples")

        # Call parallel processing method
        all_workflows, all_answers, all_rewards, all_log_probs, all_problem_types, all_ground_truths = \
            await self._process_sample_batch_parallel(batch, num_sequences, current_temp)

        # Create problems list for backward compatibility
        all_problems = [s['problem'] for s in batch for _ in range(num_sequences)]

        # Calculate correctness scores for metrics
        correctness_scores = [reward for reward in all_rewards]

        # Add samples to experience buffer (if they meet threshold)
        for i, (workflow, answer, reward, problem_type, ground_truth) in enumerate(zip(all_workflows, all_answers, all_rewards, all_problem_types, all_ground_truths)):
            if reward >= self.experience_buffer.reward_threshold:
                sample = {
                    'problem': all_problems[i],
                    'workflow_code': workflow,
                    'answer': answer,
                    'ground_truth': ground_truth,
                    'reward': reward,
                    'correctness_score': reward,
                    'metadata': {'step': step}
                }
                self.experience_buffer.add_sample(sample, problem_type)

        # 3. 策略梯度更新
        print(f"\n🔄 更新策略...")
        # ✨ Log memory before policy update
        self._log_gpu_memory("Before Policy Update")

        loss, kl_div = await self._update_policy(
            problems=all_problems,
            workflows=all_workflows,
            old_log_probs=all_log_probs,
            advantages=all_rewards,
            problem_types=[s['problem_type'] for s in batch for _ in range(num_sequences)]
        )

        # ✨ Log memory after policy update
        self._log_gpu_memory("After Policy Update")

        # 4. 指标 - ✨ Updated for 5-tier system
        # ✨ Threshold: tier 4+ (reward >= 0.7) = success
        num_correct = sum(1 for score in correctness_scores if score >= 0.7)
        num_total = len(correctness_scores)
        accuracy = (num_correct / num_total * 100) if num_total > 0 else 0.0
        avg_correctness = np.mean(correctness_scores) if correctness_scores else 0.0

        # ✨ Calculate problem type stats with 5-tier thresholds
        problem_type_stats = {}
        for problem_type in ['math', 'code', 'qa']:
            type_scores = [s for s, p in zip(correctness_scores,
                          [s['problem_type'] for s in batch for _ in range(num_sequences)])
                          if p == problem_type]
            if type_scores:
                # ✨ Tier 4+ (>= 0.7) is considered correct
                type_correct = sum(1 for s in type_scores if s >= 0.7)
                type_accuracy = (type_correct / len(type_scores) * 100)
                type_avg = np.mean(type_scores)
                problem_type_stats[problem_type] = {
                    "accuracy": type_accuracy,
                    "avg_score": type_avg,
                    "count": len(type_scores)
                }

        metrics = {
            "step": step,
            "loss": loss,
            "kl_div": kl_div,
            "avg_reward": np.mean(all_rewards),
            "max_reward": np.max(all_rewards),
            "min_reward": np.min(all_rewards),
            "num_samples": len(all_workflows),
            # ✨ 新增准确率指标
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_total": num_total,
            "avg_correctness_score": avg_correctness
        }

        # ✨ Update logging for 5-tier system
        print(f"\n🎯 准确率统计 (Tier 4+): {num_correct}/{num_total} = {accuracy:.1f}% (平均正确性评分: {avg_correctness:.2f}/1.0)")

        # Calculate 5-tier distribution
        tier_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for score in correctness_scores:
            if score >= 0.95:
                tier_dist[5] += 1
            elif score >= 0.6:
                tier_dist[4] += 1
            elif score >= 0.4:
                tier_dist[3] += 1
            elif score >= 0.2:
                tier_dist[2] += 1
            else:
                tier_dist[1] += 1

        print(f"\n📊 5-Tier分布: ", end="")
        for tier, count in tier_dist.items():
            pct = 100 * count / num_total if num_total > 0 else 0
            print(f"Tier {tier}={count}({pct:.1f}%) ", end="")
        print()

        print(f"\n📊 问题类型分布:")
        for ptype, stats in problem_type_stats.items():
            print(f"  {ptype}: {stats['accuracy']:.1f}% (avg: {stats['avg_score']:.2f}, n={stats['count']})")

        # ✨ 详细 wandb logging - NEW 5-tier metrics
        wandb_log_data = {
            "train/loss": loss,
            "train/kl_div": kl_div,
            "train/avg_reward": np.mean(all_rewards) if all_rewards else 0,
            "train/max_reward": np.max(all_rewards) if all_rewards else 0,
            "train/min_reward": np.min(all_rewards) if all_rewards else 0,
            "train/accuracy": accuracy,
            "train/avg_correctness_score": avg_correctness,
            "train/num_correct": num_correct,
            "train/num_total": num_total,
            "train/temperature": current_temp,
            "train/step": step,
        }

        # ✨ Add 5-tier distribution metrics
        for tier, count in tier_dist.items():
            pct = 100 * count / num_total if num_total > 0 else 0
            wandb_log_data[f"train/tier_{tier}_count"] = count
            wandb_log_data[f"train/tier_{tier}_pct"] = pct

        # 添加问题类型的分布指标
        for ptype, stats in problem_type_stats.items():
            wandb_log_data[f"train/accuracy_{ptype}"] = stats['accuracy']
            wandb_log_data[f"train/avg_score_{ptype}"] = stats['avg_score']
            wandb_log_data[f"train/count_{ptype}"] = stats['count']

        wandb.log(wandb_log_data, step=step)

        return metrics

    async def _compute_log_prob(
        self,
        problem: str,
        workflow_code: str,
        problem_type: str
    ) -> torch.Tensor:
        """计算工作流的log概率（旧策略）"""

        self.model.eval()

        with torch.no_grad():
            # 构建完整文本
            prompt = self.generator._build_generation_prompt(problem, problem_type)
            full_text = prompt + workflow_code

            # Tokenize
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

            # 前向传播
            outputs = self.model(**inputs, labels=inputs["input_ids"])

            # 负对数似然 -> log概率
            log_prob = -outputs.loss

            return log_prob.detach().cpu()

    async def _update_policy(
        self,
        problems: List[str],
        workflows: List[str],
        old_log_probs: List[torch.Tensor],
        advantages: List[float],
        problem_types: List[str]
    ) -> Tuple[float, float]:
        """更新策略（GRPO）"""

        self.model.train()

        total_loss = 0.0
        total_kl = 0.0
        num_updates = 0

        # ✅ FUNDAMENTAL FIX: Add micro-batching for forward passes
        # Process workflows in small micro-batches to prevent computation graph accumulation
        # This is the ROOT CAUSE fix for CUDA OOM during policy update
        microbatch_size = self.config.get('forward_pass_microbatch_size', 1)  # Default: 1 workflow at a time
        grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)

        # 🚀 Performance Fix: Reduced memory cleanup frequency
        # Only cleanup if memory usage is high (>80%) to avoid excessive interruptions
        if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.8:
            torch.cuda.empty_cache()
            gc.collect()

        # Process workflows in micro-batches
        for i in range(0, len(workflows), microbatch_size):
            microbatch_end = min(i + microbatch_size, len(workflows))
            microbatch_loss = 0.0
            microbatch_kl = 0.0

            # Process each workflow in the micro-batch
            for j in range(i, microbatch_end):
                problem = problems[j]
                workflow = workflows[j]
                old_log_prob = old_log_probs[j]
                advantage = advantages[j]
                problem_type = problem_types[j]

                # 🚀 Performance Fix: Reduced cleanup frequency (every 20 samples instead of 5)
                if j % 20 == 0 and torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.8:
                    torch.cuda.empty_cache()

                # 计算新log概率 WITH gradients
                new_log_prob = await self._compute_log_prob_trainable(problem, workflow, problem_type)

                # Compute PPO loss components (all operations keep gradients)
                old_log_prob_device = old_log_prob.to(self.model.device)
                ratio = torch.exp(new_log_prob - old_log_prob_device)

                # PPO裁剪
                clip_range = self.config['clip_range']
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

                # Advantage tensor (constant, no gradients needed)
                advantage_tensor = torch.tensor(
                    advantage,
                    device=self.model.device,
                    dtype=torch.bfloat16,
                    requires_grad=False  # Advantage is constant
                )

                # PPO裁剪损失
                policy_loss = -torch.min(
                    ratio * advantage_tensor,
                    clipped_ratio * advantage_tensor
                )

                # KL正则化
                if self.config.get('use_kl_loss'):
                    kl_loss = self.config['kl_loss_coef'] * (
                        new_log_prob - old_log_prob_device
                    ).pow(2)
                else:
                    kl_loss = 0.0

                # 总损失
                loss = policy_loss + kl_loss

                # 累积到micro-batch
                microbatch_loss += loss
                microbatch_kl += kl_loss if isinstance(kl_loss, torch.Tensor) else 0.0

                # Cleanup
                del old_log_prob_device, advantage_tensor, new_log_prob, ratio, clipped_ratio

            # Normalize loss by micro-batch size
            microbatch_loss = microbatch_loss / (microbatch_end - i)

            # ✅ KEY FIX: Backward IMMEDIATELY after each micro-batch
            # This prevents computation graphs from accumulating
            microbatch_loss.backward()

            # Cleanup AFTER backward
            microbatch_loss_value = microbatch_loss.item()
            microbatch_kl_value = microbatch_kl.item() if isinstance(microbatch_kl, torch.Tensor) else microbatch_kl
            del microbatch_loss, microbatch_kl
            torch.cuda.empty_cache()

            total_loss += microbatch_loss_value
            total_kl += microbatch_kl_value
            num_updates += 1

            # 优化器步骤 (every grad_accum_steps micro-batches)
            if (num_updates % grad_accum_steps == 0) or (microbatch_end >= len(workflows)):
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                self.optimizer.step()
                # Use set_to_none=True to free memory
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_updates, 1)
        avg_kl = total_kl / max(num_updates, 1)

        return avg_loss, avg_kl

    async def _compute_log_prob_trainable(
        self,
        problem: str,
        workflow_code: str,
        problem_type: str
    ) -> torch.Tensor:
        """计算工作流的log概率（新策略，可训练）

        ✅ FUNDAMENTAL FIX: Proper gradient flow without premature tensor deletion
        - Forward pass builds computation graph
        - Returns log_prob WITH gradients (no .detach())
        - NO premature deletion of inputs/outputs (breaks gradient graph)
        - Let PyTorch handle tensor lifecycle automatically
        """

        # 构建完整文本
        prompt = self.generator._build_generation_prompt(problem, problem_type)
        full_text = prompt + workflow_code

        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

        # 前向传播 WITH gradients (needed for backprop)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            # ✅ CRITICAL: Keep gradients! No .detach()
            log_prob = -outputs.loss

        # ✅ FUNDAMENTAL FIX: DO NOT delete inputs/outputs here!
        # They are still needed by the computation graph for backward()
        # PyTorch will automatically release them after backward() completes

        return log_prob  # Returns tensor WITH gradients

    async def evaluate_on_val_set(self, num_samples: int = 50) -> Dict:
        """
        在验证集上评估模型性能

        Args:
            num_samples: 验证样本数量

        Returns:
            验证指标字典
        """
        print(f"\n{'='*60}")
        print(f"🧪 验证集评估 ({num_samples}个样本)")
        print(f"{'='*60}")

        # 采样验证集
        val_batch = self.data_manager.sample_batch(
            batch_size=num_samples,
            split="val"  # 使用验证集
        )

        # 统计
        batch_stats = self.data_manager.get_batch_stats(val_batch)
        print(f"📦 验证集分布: {batch_stats}")

        # 评估每个样本
        correctness_scores = []
        total_cost = 0.0
        successful_executions = 0

        for idx, sample in enumerate(tqdm(val_batch, desc="验证集评估"), 1):
            problem = sample['problem']
            ground_truth = sample['ground_truth']
            problem_type = sample['problem_type']

            try:
                # 使用当前策略生成workflow（使用动态提示词）
                custom_prompt = None
                if self.use_dynamic_prompts:
                    custom_prompt = self.prompt_optimizer.build_dynamic_prompt(
                        problem=problem,
                        problem_type=problem_type
                    )

                result = self.generator.generate_workflow(
                    problem=problem,
                    problem_type=problem_type,
                    temperature=self.config['generation_config']['temperature'],
                    custom_prompt=custom_prompt
                )

                workflow_code = result['workflow_code']

                # 执行workflow
                answer, cost, metadata = await self.executor.execute_workflow(
                    workflow_code=workflow_code,
                    problem=problem,
                    problem_type=problem_type,
                    entry_point=sample.get('entry_point', ''),
                    test=sample.get('test', '')  # NEW: pass test cases for HumanEval
                )

                # 计算正确性
                if metadata['success']:
                    correctness_result = self.reward_computer.compute_reward(
                        problem=problem,
                        prediction=answer,
                        ground_truth=ground_truth,
                        problem_type=problem_type,
                        execution_metadata={'success': True}
                    )
                    correctness = correctness_result.get('reward', 0.0) * 10.0  # Convert [0, 1] to [0, 10]
                    correctness_scores.append(correctness)
                    total_cost += cost
                    successful_executions += 1

                    is_correct = correctness >= 5.0
                    status_icon = "✅" if is_correct else "❌"
                    if idx <= 5:  # 只打印前5个样本的详情
                        print(f"  {status_icon} [{idx}/{num_samples}] 正确性: {correctness:.1f}/10.0")
                else:
                    correctness_scores.append(0.0)
                    if idx <= 5:
                        print(f"  ❌ [{idx}/{num_samples}] 执行失败")

            except Exception as e:
                print(f"  ⚠️  [{idx}/{num_samples}] 错误: {type(e).__name__}")
                correctness_scores.append(0.0)

        # 计算指标
        num_correct = sum(1 for score in correctness_scores if score >= 5.0)
        val_accuracy = (num_correct / num_samples * 100) if num_samples > 0 else 0.0
        avg_correctness = np.mean(correctness_scores) if correctness_scores else 0.0
        avg_cost = total_cost / successful_executions if successful_executions > 0 else 0.0
        success_rate = (successful_executions / num_samples * 100) if num_samples > 0 else 0.0

        metrics = {
            "val_accuracy": val_accuracy,
            "val_num_correct": num_correct,
            "val_num_total": num_samples,
            "val_avg_correctness": avg_correctness,
            "val_avg_cost": avg_cost,
            "val_success_rate": success_rate
        }

        print(f"\n📊 验证集结果:")
        print(f"  准确率: {num_correct}/{num_samples} = {val_accuracy:.1f}%")
        print(f"  平均正确性: {avg_correctness:.2f}/10.0")
        print(f"  执行成功率: {success_rate:.1f}%")
        print(f"  平均成本: ${avg_cost:.4f}")
        print(f"{'='*60}\n")

        return metrics

    async def train(self):
        """完整训练循环"""
        print("\n" + "=" * 60)
        print("🎓 开始GRPO训练")
        print("=" * 60)

        max_steps = self.config['max_steps']
        save_every = self.config.get('save_every', 50)
        log_every = self.config.get('log_every', 5)
        eval_every = self.config.get('eval_every', 10)  # 每10步验证一次
        val_samples = self.config.get('val_samples', 50)  # 验证集样本数

        for step in range(1, max_steps + 1):
            print(f"\n{'=' * 60}")
            print(f"📍 Step {step}/{max_steps}")
            print(f"{'=' * 60}")

            # 训练步骤
            metrics = await self.train_step(step)

            # 日志
            if step % log_every == 0:
                print(f"\n📊 Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

                # 记录到wandb
                wandb.log(metrics, step=step)

            # 🧪 验证集评估（每N步）
            if eval_every > 0 and step % eval_every == 0:
                val_metrics = await self.evaluate_on_val_set(num_samples=val_samples)

                # 合并验证指标到训练指标
                metrics.update(val_metrics)

                # 记录验证指标到wandb
                wandb.log(val_metrics, step=step)

                print(f"✅ 验证集评估完成 (Step {step})")

            # 保存检查点
            if step % save_every == 0:
                self.save_checkpoint(step)

        print("\n" + "=" * 60)
        print("✅ 训练完成")
        print("=" * 60)

    def save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint_dir = Path(self.config['output_dir']) / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存LoRA权重
        self.model.save_pretrained(checkpoint_dir)

        # 💾 保存ExperienceBuffer
        self.experience_buffer.save(step=step)

        # 📊 打印ExperienceBuffer统计信息
        buffer_stats = self.experience_buffer.get_stats()
        print(f"\n📚 ExperienceBuffer统计:")
        for problem_type, stats in buffer_stats.items():
            if stats['count'] > 0:
                print(f"  {problem_type}: {stats['count']}样本, "
                      f"平均奖励={stats['avg_reward']:.2f}, "
                      f"最高奖励={stats['max_reward']:.2f}, "
                      f"平均正确性={stats['avg_correctness']:.2f}")

        print(f"💾 检查点已保存: {checkpoint_dir}")


async def main():
    """主函数"""
    trainer = GRPOTrainer(config_path="config/training.yaml")
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
