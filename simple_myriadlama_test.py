"""
简化版250样本测试 - 不使用spacy lemmatization
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import MyriadLamaDataset
from constants import MODEL_PATHs
from utils import partial_match
import json
from datetime import datetime


def simple_ensemble_avg(model, tokenizer, prompts):
    """简化的平均集成方法"""
    device = next(model.parameters()).device
    all_logits = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # 取最后一个位置的logits
            logits = outputs.logits[:, -1, :]
            all_logits.append(logits)
    
    # 平均logits
    avg_logits = torch.mean(torch.stack(all_logits), dim=0)
    
    # 从平均logits生成
    input_ids = tokenizer(prompts[0], return_tensors="pt").input_ids.to(device)
    generated_ids = []
    
    for _ in range(10):  # 生成10个token
        next_token_logits = avg_logits
        next_token = torch.argmax(next_token_logits, dim=-1)
        generated_ids.append(next_token.item())
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # 为下一步准备（简化处理）
        new_input = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        with torch.no_grad():
            outputs = model(new_input)
            avg_logits = outputs.logits[:, -1, :]
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def simple_ensemble_max(model, tokenizer, prompts):
    """简化的最大集成方法 - 选择最高概率的生成"""
    device = next(model.parameters()).device
    best_generation = ""
    best_score = 0.0
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generation = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 计算平均分数
            if outputs.scores:
                avg_score = torch.mean(torch.stack([torch.max(torch.softmax(score, dim=-1)) for score in outputs.scores]))
                if avg_score > best_score:
                    best_score = avg_score
                    best_generation = generation
    
    return best_generation


def run_simple_test(method="avg", num_samples=250):
    """运行简化的测试"""
    print(f"🚀 Running {method} method on {num_samples} samples")
    
    # 加载数据集
    dataset = MyriadLamaDataset("qwen2.5_1.5b_it")
    subset_data = [dataset.ds[i] for i in range(min(num_samples, len(dataset.ds)))]
    print(f"📊 Selected {len(subset_data)} samples")
    
    # 加载模型
    model_path = "D:/Codes/Models/qwen2.5_1.5b_it"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    device = next(model.parameters()).device
    print(f"🖥️ Using device: {device}")
    
    few_shot_context = dataset.get_few_shot_examples()
    results = []
    
    for item in tqdm(subset_data, desc=f"Processing {method}"):
        uuid = item["uuid"]
        answers = item["answers"]
        
        # 获取6个paraphrases（5个manual + 1个auto）
        import random
        random.seed(uuid)
        manual_paras = item["manual_paraphrases"][:5]  # 取前5个
        auto_paras = random.sample(item["auto_paraphrases"], min(1, len(item["auto_paraphrases"])))
        all_paraphrases = manual_paras + auto_paras
        
        # 构建prompts
        prompts = [f"{few_shot_context}\\n\\nQ: {para}\\nA:" for para in all_paraphrases]
        
        # 根据方法生成预测
        if method == "avg":
            prediction = simple_ensemble_avg(model, tokenizer, prompts)
        elif method == "max":
            prediction = simple_ensemble_max(model, tokenizer, prompts)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 清理预测结果
        prediction = prediction.strip().split("\\n")[0].split(".")[0]
        
        results.append({
            "uuid": uuid,
            "answers": answers,
            "prediction": prediction,
            "paraphrases": all_paraphrases
        })
    
    return results


def calculate_simple_accuracy(results):
    """简化的准确率计算"""
    correct = 0
    total = len(results)
    
    for result in results:
        prediction = result["prediction"].lower().strip()
        answers = [ans.lower().strip() for ans in result["answers"]]
        
        # 简单匹配：检查预测是否在任何答案中，或答案是否在预测中
        match = False
        for ans in answers:
            if prediction in ans or ans in prediction or prediction == ans:
                match = True
                break
        
        if match:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def main():
    """运行完整的比较测试"""
    print("🎯 Simplified MyriadLAMA 250-Sample Test")
    print("=" * 60)
    
    # 运行avg方法
    print("\\n🔄 Running AVG method...")
    avg_results = run_simple_test("avg", 250)
    avg_accuracy = calculate_simple_accuracy(avg_results)
    
    # 运行max方法
    print("\\n🔄 Running MAX method...")
    max_results = run_simple_test("max", 250)
    max_accuracy = calculate_simple_accuracy(max_results)
    
    # 显示结果
    print("\\n🎯 RESULTS SUMMARY")
    print("=" * 60)
    print(f"📈 AVG Method Accuracy: {avg_accuracy:.4f} ({sum([1 for r in avg_results if calculate_simple_accuracy([r]) > 0])}/{len(avg_results)})")
    print(f"📈 MAX Method Accuracy: {max_accuracy:.4f} ({sum([1 for r in max_results if calculate_simple_accuracy([r]) > 0])}/{len(max_results)})")
    print(f"📊 Difference: {abs(avg_accuracy - max_accuracy):.4f}")
    
    if avg_accuracy > max_accuracy:
        print("🏆 AVG method performs better!")
    elif max_accuracy > avg_accuracy:
        print("🏆 MAX method performs better!")
    else:
        print("🤝 Both methods perform equally!")
    
    # 显示示例
    print("\\n📝 Sample Results:")
    for i in range(min(5, len(avg_results))):
        print(f"\\nSample {i+1}:")
        print(f"  Question: {avg_results[i]['paraphrases'][0]}")
        print(f"  Gold: {avg_results[i]['answers']}")
        print(f"  AVG: {avg_results[i]['prediction']}")
        print(f"  MAX: {max_results[i]['prediction']}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simple_myriadlama_comparison_{timestamp}.json"
    
    results_data = {
        "timestamp": timestamp,
        "num_samples": len(avg_results),
        "avg_accuracy": avg_accuracy,
        "max_accuracy": max_accuracy,
        "avg_results": avg_results[:10],  # 保存前10个示例
        "max_results": max_results[:10]   # 保存前10个示例
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\\n💾 Results saved to: {results_file}")
    print("✅ Test completed successfully!")


if __name__ == "__main__":
    main()