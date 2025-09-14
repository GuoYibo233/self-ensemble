"""
使用原有generate.py逻辑的250样本快速测试
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import MyriadLamaDataset
from constants import MODEL_PATHs
from generate import ensemble_generation, init_spacy, lemmaize_chunk, append_lemmas
import multiprocessing as mp
import numpy as np


def run_ensemble_250_samples(method="avg", num_samples=250):
    """使用原有逻辑运行250个样本的ensemble测试"""
    
    print(f"🚀 Running {method} method on {num_samples} samples")
    
    # 加载数据集
    dataset = MyriadLamaDataset("qwen2.5_1.5b_it")
    
    # 创建子集 - 取前num_samples个样本
    subset_data = [dataset.ds[i] for i in range(min(num_samples, len(dataset.ds)))]
    print(f"📊 Selected {len(subset_data)} samples from {len(dataset.ds)} total")
    
    # 设置模型
    model_name = "qwen2.5_1.5b_it"
    model_path = MODEL_PATHs.get(model_name, "D:/Codes/Models/qwen2.5_1.5b_it")
    
    print(f"🔧 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 设置全局变量（原代码需要）
    import generate
    generate.tokenizer = tokenizer
    generate.model = model
    
    device = next(model.parameters()).device
    print(f"🖥️ Using device: {device}")
    
    # 创建输出文件路径
    dump_file = f"d:/Codes/Research/SelfE/self-ensemble/datasets/myriadlama/qwen2.5_1.5b_it/ensemble_{method}-6-250samples.feather"
    
    # 按原有逻辑处理数据
    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation"])
    print(f"Ensemble generation with method: {method}")
    
    few_shot_context = dataset.get_few_shot_examples()
    
    # 模拟dataloader - 按batch_size=8分组
    batch_size = 8
    batches = []
    for i in range(0, len(subset_data), batch_size):
        batch_data = subset_data[i:i+batch_size]
        
        uuids = [item["uuid"] for item in batch_data]
        answers = [item["answers"] for item in batch_data]
        
        # 按原有逻辑创建paraphrases结构
        all_paraphrases = []
        for item in batch_data:
            # 5个manual + 5个auto = 10个paraphrases
            import random
            random.seed(item["uuid"])
            auto_paras = random.sample(item["auto_paraphrases"], min(5, len(item["auto_paraphrases"])))
            paraphrases = item["manual_paraphrases"] + auto_paras
            all_paraphrases.append(paraphrases)
        
        # 转置以匹配原有格式 (paraphrase_position, batch)
        all_paraphrases = list(zip(*all_paraphrases))
        
        batches.append((uuids, answers, all_paraphrases))
    
    print(f"📦 Created {len(batches)} batches")
    
    # 处理每个batch
    for uuids, answers, all_paraphrases in tqdm(batches, desc=f"Processing {method}"):
        # 使用前6个paraphrases（与默认num_ensemble=6一致）
        num_ensemble = 6
        all_paraphrases = all_paraphrases[:num_ensemble]
        
        all_paraphrases = [list(paras) for paras in all_paraphrases]
        all_prompts = []
        
        for paraphrases in all_paraphrases:
            prompts = dataset.construct_prompts(few_shot_context, paraphrases)
            all_prompts.append(prompts)
        
        # 使用原有的ensemble_generation函数
        generations = ensemble_generation(
            all_prompts, integration_method=method, weights=None
        )
        predictions = [gen.strip().split("\\n")[0] for gen in generations]
        
        items = {
            "uuid": uuids,
            "paraphrases": list(zip(*all_paraphrases)),
            "prompts": list(zip(*all_prompts)),
            "answers": answers,
            "prediction": predictions,
            "generation": generations,
        }
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)
    
    # 使用原有的lemmatization逻辑
    print("🔤 Applying lemmatization...")
    num_parts = min(4, len(df))  # 减少进程数
    chunks = np.array_split(df, num_parts) if len(df) > 0 else [df]
    
    with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
        results = pool.map(lemmaize_chunk, chunks)
    
    df = append_lemmas(df, results)
    
    # 保存结果
    os.makedirs(os.path.dirname(dump_file), exist_ok=True)
    df.to_feather(dump_file)
    
    print(f"💾 Results saved to: {dump_file}")
    print(f"📊 Processed {len(df)} samples")
    
    return dump_file, df


def main():
    """运行avg和max方法的比较"""
    print("🎯 MyriadLAMA 250-Sample Comparison Test")
    print("=" * 60)
    
    # 运行avg方法
    print("\\n🔄 Running AVG method...")
    avg_file, avg_df = run_ensemble_250_samples("avg", 250)
    
    # 运行max方法
    print("\\n🔄 Running MAX method...")
    max_file, max_df = run_ensemble_250_samples("max", 250)
    
    print("\\n✅ Both methods completed!")
    print(f"📁 AVG results: {avg_file}")
    print(f"📁 MAX results: {max_file}")
    print(f"📊 Sample count: {len(avg_df)} each")
    
    return avg_file, max_file


if __name__ == "__main__":
    avg_file, max_file = main()