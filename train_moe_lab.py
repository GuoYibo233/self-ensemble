#!/usr/bin/env python3
"""
实验室服务器MoE模型训练脚本

在self-ensemble框架中训练和评估MoE模型
"""

from utils import set_seed
from dataset import MyriadLamaDataset
from lab_moe_config import LAB_MOE_MODELS, MOE_TRAINING_CONFIG, get_lab_model_path
import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def setup_training_args():
    """设置训练参数"""
    parser = argparse.ArgumentParser(description="Lab Server MoE Training")

    # 模型配置
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(LAB_MOE_MODELS.keys()),
                        help="MoE model to train")

    # 数据配置
    parser.add_argument("--dataset", type=str, default="myriadlama",
                        help="Dataset to use")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum training samples")

    # 训练配置
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Training batch size (auto-detect if None)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum training steps")
    parser.add_argument("--output_dir", type=str, default="/data/results/moe-training",
                        help="Output directory")

    # 实验配置
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (auto-generated if None)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # GPU配置
    parser.add_argument("--use_multi_gpu", action="store_true",
                        help="Use multiple GPUs if available")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Specific GPU IDs to use (e.g., '0,1,2')")

    return parser.parse_args()


def check_gpu_environment():
    """检查GPU环境"""
    print("🔍 Checking GPU Environment...")

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False

    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} GPU(s)")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

    return True


def load_moe_model(model_name, device_map="auto"):
    """加载MoE模型"""
    print(f"🧠 Loading MoE model: {model_name}")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_config = LAB_MOE_MODELS[model_name]
    model_path = model_config["hf_name"]

    print(f"  Model path: {model_path}")
    print(f"  Active params: {model_config['active_params']}")
    print(f"  Memory requirement: {model_config['memory_requirement']}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.float16,  # 节省显存
            trust_remote_code=True
        )

        print(f"✅ Model loaded successfully")
        print(f"  Model device: {model.device}")
        print(f"  Model dtype: {model.dtype}")

        return tokenizer, model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None


def prepare_dataset(dataset_name, max_samples=1000):
    """准备训练数据集"""
    print(f"📚 Preparing dataset: {dataset_name}")

    if dataset_name == "myriadlama":
        # 使用现有的MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name="moe_placeholder")

        # 限制样本数量以加速实验
        if hasattr(dataset, 'ds') and len(dataset.ds) > max_samples:
            dataset.ds = dataset.ds.select(range(max_samples))
            print(f"  Limited to {max_samples} samples")

        return dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def run_self_ensemble_evaluation(tokenizer, model, dataset, method="avg"):
    """运行self-ensemble评估"""
    print(f"🎯 Running self-ensemble evaluation with method: {method}")

    # 这里可以复用现有的generate.py逻辑
    # 或者实现简化版本的ensemble评估

    from generate import ensemble_generation

    # 获取few-shot示例
    few_shot_context = dataset.get_few_shot_examples()

    # 获取数据加载器
    dataloader = dataset.get_dataloader(batch_size=4, shuffle=False)

    results = []
    correct_count = 0
    total_count = 0

    print("  Processing batches...")
    for i, (uuids, answers, all_paraphrases) in enumerate(dataloader):
        if i >= 10:  # 只处理前10个batch进行快速测试
            break

        # 使用前6个paraphrases
        all_paraphrases = all_paraphrases[:6]
        all_prompts = []

        for paraphrases in all_paraphrases:
            prompts = dataset.construct_prompts(few_shot_context, paraphrases)
            all_prompts.append(prompts)

        # MoE ensemble生成
        try:
            generations = ensemble_generation(
                all_prompts,
                integration_method=method,
                weights=None
            )

            predictions = [gen.strip().split("\n")[0] for gen in generations]

            # 简单准确率计算
            for pred, ans in zip(predictions, answers):
                if pred.lower() in [a.lower() for a in ans]:
                    correct_count += 1
                total_count += 1

            results.extend(list(zip(uuids, predictions, answers)))

        except Exception as e:
            print(f"    Batch {i} failed: {e}")
            continue

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"  Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")

    return accuracy, results


def main():
    """主训练流程"""
    args = setup_training_args()

    # 设置随机种子
    set_seed(args.seed)

    # 生成实验名称
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"moe_{args.model_name}_{timestamp}"

    print(f"🚀 Starting MoE Training Experiment: {args.experiment_name}")
    print("=" * 60)

    # 检查GPU环境
    if not check_gpu_environment():
        return

    # 设置输出目录
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # 保存实验配置
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"📝 Config saved to: {config_path}")

    # 加载模型
    tokenizer, model = load_moe_model(args.model_name)
    if model is None:
        return

    # 准备数据集
    dataset = prepare_dataset(args.dataset, args.max_samples)

    # 运行self-ensemble评估
    print("\n" + "="*60)
    accuracy, results = run_self_ensemble_evaluation(tokenizer, model, dataset)

    # 保存结果
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "total_samples": len(results),
            "results": results[:100]  # 保存前100个结果作为样例
        }, f, indent=2)

    print(f"📊 Results saved to: {results_path}")
    print(f"🎉 Experiment completed! Final accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
