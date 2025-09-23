"""
实验室服务器MoE模型训练配置

针对实验室GPU集群环境优化的MoE模型训练设置
"""

import os
from pathlib import Path

# 实验室服务器路径配置
LAB_SERVER_CONFIG = {
    "project_root": "/home/[your_username]/self-ensemble-moe",
    "model_cache_dir": "/data/models/moe",  # 根据实验室存储结构调整
    "dataset_cache_dir": "/data/datasets/myriadlama",
    "results_dir": "/data/results/self-ensemble-moe",
    "logs_dir": "/data/logs/self-ensemble-moe"
}

# MoE模型配置 - 实验室服务器版本
LAB_MOE_MODELS = {
    # 小规模MoE - 适合初步测试
    "qwen1.5_moe_a2.7b": {
        "hf_name": "Qwen/Qwen1.5-MoE-A2.7B",
        "total_params": "14.3B",
        "active_params": "2.7B", 
        "memory_requirement": "~6GB",
        "recommended_gpus": ["RTX 3090", "RTX 4090", "A100"],
        "batch_size_recommendations": {
            "1x RTX 3090": 4,
            "1x RTX 4090": 6,
            "1x A100": 8
        }
    },
    
    "qwen1.5_moe_a2.7b_chat": {
        "hf_name": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "total_params": "14.3B",
        "active_params": "2.7B",
        "memory_requirement": "~6GB",
        "optimized_for": "QA and dialogue tasks",
        "recommended_gpus": ["RTX 3090", "RTX 4090", "A100"],
        "batch_size_recommendations": {
            "1x RTX 3090": 4,
            "1x RTX 4090": 6, 
            "1x A100": 8
        }
    },
    
    # 中等规模MoE - 如果实验室有充足GPU资源
    "mixtral_7b_instruct": {
        "hf_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "total_params": "45B",
        "active_params": "12.9B",
        "memory_requirement": "~16GB",
        "recommended_gpus": ["A100", "H100"],
        "batch_size_recommendations": {
            "1x A100": 2,
            "2x A100": 4,
            "1x H100": 6
        }
    }
}

# 训练配置
MOE_TRAINING_CONFIG = {
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "max_steps": 1000,
    "save_steps": 200,
    "eval_steps": 100,
    "gradient_accumulation_steps": 4,
    "fp16": True,  # 节省显存
    "dataloader_num_workers": 4,
    "remove_unused_columns": False,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "greater_is_better": True
}

# 分布式训练配置
DISTRIBUTED_CONFIG = {
    "use_multi_gpu": True,
    "strategy": "ddp",  # DistributedDataParallel
    "find_unused_parameters": True,  # MoE可能需要这个设置
}

def get_lab_model_path(model_name):
    """获取实验室服务器上的模型路径"""
    if model_name in LAB_MOE_MODELS:
        cache_dir = LAB_SERVER_CONFIG["model_cache_dir"]
        return os.path.join(cache_dir, model_name)
    return LAB_MOE_MODELS[model_name]["hf_name"]

def get_optimal_batch_size(model_name, gpu_info):
    """根据GPU信息推荐最优batch size"""
    if model_name not in LAB_MOE_MODELS:
        return 4  # 默认值
    
    model_config = LAB_MOE_MODELS[model_name]
    batch_recommendations = model_config.get("batch_size_recommendations", {})
    
    # 简单匹配逻辑，实际使用时可以更智能
    for gpu_config, batch_size in batch_recommendations.items():
        if gpu_info in gpu_config:
            return batch_size
    
    return 4  # 保守默认值

def setup_lab_environment():
    """设置实验室环境"""
    for key, path in LAB_SERVER_CONFIG.items():
        if key.endswith('_dir'):
            os.makedirs(path, exist_ok=True)
            print(f"✅ Created directory: {path}")

if __name__ == "__main__":
    print("🏭 Laboratory Server MoE Configuration")
    print("=" * 50)
    
    setup_lab_environment()
    
    print("\n📋 Available MoE Models:")
    for model_name, config in LAB_MOE_MODELS.items():
        print(f"\n🧠 {model_name}:")
        print(f"  HuggingFace: {config['hf_name']}")
        print(f"  Active Params: {config['active_params']}")
        print(f"  Memory: {config['memory_requirement']}")
        print(f"  Recommended GPUs: {', '.join(config['recommended_gpus'])}")