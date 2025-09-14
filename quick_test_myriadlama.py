"""
MyriadLAMA 快速测试版本 - 100个样本
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json

from dataset import MyriadLamaDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import partial_match
import torch


def quick_test_myriadlama(model_name="qwen2.5_1.5b_it", num_samples=100):
    """快速测试MyriadLAMA的avg vs max方法"""
    print(f"🎯 Quick MyriadLAMA Test: {num_samples} samples")
    print("=" * 50)

    # 加载数据集
    print("📊 Loading MyriadLAMA dataset...")
    dataset = MyriadLamaDataset(model_name)

    # 随机选择样本
    random.seed(42)
    full_size = len(dataset.ds)
    indices = random.sample(range(full_size), min(num_samples, full_size))
    subset_data = [dataset.ds[i] for i in indices]

    print(f"✅ Selected {len(subset_data)} samples from {full_size} total")

    # 加载模型
    model_path = "D:/Codes/Models/qwen2.5_1.5b_it"
    print(f"🚀 Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"🖥️ Using device: {device}")

    # 获取few-shot示例
    few_shot_context = dataset.get_few_shot_examples()

    avg_results = []
    max_results = []

    print(f"\\n🔄 Processing {len(subset_data)} samples...")

    for i, item in enumerate(tqdm(subset_data)):
        uuid = item["uuid"]
        answers = item["answers"]

        # 获取paraphrases (5个手工 + 5个自动)
        random.seed(uuid)
        auto_paras = random.sample(item["auto_paraphrases"], min(
            5, len(item["auto_paraphrases"])))
        all_paraphrases = item["manual_paraphrases"] + auto_paras

        # 构建prompts
        prompts = [
            f"{few_shot_context}\\n\\nQ: {para}\\nA:" for para in all_paraphrases]

        # AVG方法 - 简化版本：对所有输出取多数投票
        avg_prediction = simple_avg_method(model, tokenizer, prompts, device)
        avg_results.append({
            "uuid": uuid,
            "answers": answers,
            "prediction": avg_prediction
        })

        # MAX方法 - 选择置信度最高的
        max_prediction = simple_max_method(model, tokenizer, prompts, device)
        max_results.append({
            "uuid": uuid,
            "answers": answers,
            "prediction": max_prediction
        })

    # 计算准确率
    print("\\n📊 Calculating accuracy...")

    avg_accuracy = calculate_accuracy(avg_results)
    max_accuracy = calculate_accuracy(max_results)

    # 显示结果
    print("\\n🎯 QUICK TEST RESULTS")
    print("=" * 50)
    print(f"📈 AVG Method Accuracy: {avg_accuracy:.4f}")
    print(f"📈 MAX Method Accuracy: {max_accuracy:.4f}")
    print(f"📊 Difference: {abs(avg_accuracy - max_accuracy):.4f}")

    if avg_accuracy > max_accuracy:
        print("🏆 AVG method performs better!")
    elif max_accuracy > avg_accuracy:
        print("🏆 MAX method performs better!")
    else:
        print("🤝 Both methods perform equally!")

    # 显示一些示例
    print("\\n📝 Sample Results:")
    for i in range(min(5, len(avg_results))):
        print(f"\\nSample {i+1}:")
        print(f"  Question: {subset_data[i]['manual_paraphrases'][0]}")
        print(f"  Gold: {avg_results[i]['answers']}")
        print(f"  AVG: {avg_results[i]['prediction']}")
        print(f"  MAX: {max_results[i]['prediction']}")

    return {
        "avg_accuracy": avg_accuracy,
        "max_accuracy": max_accuracy,
        "num_samples": len(subset_data)
    }


def simple_avg_method(model, tokenizer, prompts, device):
    """简化的AVG方法：生成多个回答后取多数投票"""
    predictions = []

    for prompt in prompts[:6]:  # 只用前6个paraphrase节省时间
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        prediction = tokenizer.decode(
            generated_ids, skip_special_tokens=True).strip()
        predictions.append(prediction.split()[0] if prediction.split() else "")

    # 简单多数投票
    from collections import Counter
    if predictions:
        most_common = Counter(predictions).most_common(1)[0][0]
        return most_common
    return ""


def simple_max_method(model, tokenizer, prompts, device):
    """简化的MAX方法：选择概率最高的第一个token"""
    best_prediction = ""
    best_prob = 0.0

    for prompt in prompts[:6]:  # 只用前6个paraphrase节省时间
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(inputs.input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            max_prob, max_token = torch.max(probs, dim=0)

            if max_prob.item() > best_prob:
                best_prob = max_prob.item()
                best_prediction = tokenizer.decode(
                    [max_token.item()], skip_special_tokens=True).strip()

    return best_prediction


def calculate_accuracy(results):
    """计算准确率"""
    correct = 0
    total = len(results)

    for result in results:
        prediction = result["prediction"].lower().strip()
        answers = [ans.lower().strip() for ans in result["answers"]]

        # 简单的包含匹配
        if any(prediction in ans or ans in prediction for ans in answers):
            correct += 1

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    results = quick_test_myriadlama(num_samples=100)
    print(f"\\n✅ Quick test completed!")
    print(
        f"📊 Final Results: AVG={results['avg_accuracy']:.4f}, MAX={results['max_accuracy']:.4f}")
