"""
MyriadLAMA 1000样本子集准备和self-ensemble比较测试
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
from utils import partial_match, partial_match_scores
import torch


def create_subset_dataset(model_name, subset_size=1000, seed=42):
    """从MyriadLAMA数据集创建随机子集"""
    print(f"🔄 Creating subset dataset with {subset_size} samples...")

    # 加载完整的MyriadLAMA数据集
    dataset = MyriadLamaDataset(model_name)

    # 获取数据集大小
    full_size = len(dataset.ds)
    print(f"📊 Full dataset size: {full_size}")

    if subset_size > full_size:
        print(
            f"⚠️ Requested subset size ({subset_size}) is larger than dataset ({full_size})")
        subset_size = full_size

    # 随机选择样本
    random.seed(seed)
    indices = random.sample(range(full_size), subset_size)
    indices.sort()  # 排序便于处理

    # 创建子集
    subset_data = [dataset.ds[i] for i in indices]

    print(f"✅ Created subset with {len(subset_data)} samples")
    return subset_data, dataset


def run_self_ensemble(model_name, subset_data, dataset, method, max_new_tokens=20):
    """运行self-ensemble方法"""
    print(f"🚀 Running {method} ensemble method...")

    # 加载模型
    model_path = f"D:/Codes/Models/{model_name}"
    if model_name == "qwen2.5_1.5b_it":
        model_path = "D:/Codes/Models/qwen2.5_1.5b_it"

    print(f"📁 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"🖥️ Using device: {device}")

    results = []
    few_shot_context = dataset.get_few_shot_examples()

    # 处理每个样本
    for item in tqdm(subset_data, desc=f"Processing {method}"):
        uuid = item["uuid"]
        answers = item["answers"]

        # 获取手工模板 + 5个随机自动模板（总共10个paraphrases）
        random.seed(uuid)  # 确保一致性
        auto_paras = random.sample(item["auto_paraphrases"], 5)
        all_paraphrases = item["manual_paraphrases"] + auto_paras

        # 构建prompts
        prompts = [
            f"{few_shot_context}\\n\\nQ: {para}\\nA:" for para in all_paraphrases]

        if method == "avg":
            # 平均集成方法
            prediction = ensemble_avg_generation(
                model, tokenizer, prompts, max_new_tokens)
        elif method == "max":
            # 最大集成方法
            prediction = ensemble_max_generation(
                model, tokenizer, prompts, max_new_tokens)
        else:
            raise ValueError(f"Unsupported method: {method}")

        results.append({
            "uuid": uuid,
            "answers": answers,
            "paraphrases": all_paraphrases,
            "prediction": prediction,
            "method": method
        })

    return results


def ensemble_avg_generation(model, tokenizer, prompts, max_new_tokens=20):
    """平均集成生成"""
    device = next(model.parameters()).device

    # 编码所有prompts
    inputs = tokenizer(prompts, padding=True, truncation=True,
                       return_tensors="pt", padding_side="left")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        # 获取初始logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # 最后一个token的logits

        # 平均logits
        avg_logits = torch.mean(logits, dim=0, keepdim=True)

        # 从平均logits开始生成
        generated_ids = []
        current_logits = avg_logits

        for _ in range(max_new_tokens):
            # 采样下一个token
            probs = torch.softmax(current_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated_ids.append(next_token.item())

            # 如果遇到EOS token就停止
            if next_token.item() == tokenizer.eos_token_id:
                break

            # 为下一步准备输入（简化处理，这里只用第一个prompt的格式）
            new_input = torch.cat(
                [input_ids[0:1], next_token.unsqueeze(0)], dim=1)
            new_attention = torch.ones_like(new_input)

            # 获取下一个token的logits
            next_outputs = model(input_ids=new_input,
                                 attention_mask=new_attention)
            current_logits = next_outputs.logits[:, -1, :]

    # 解码生成的tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()


def ensemble_max_generation(model, tokenizer, prompts, max_new_tokens=20):
    """最大集成生成（选择置信度最高的）"""
    device = next(model.parameters()).device

    # 为每个prompt单独生成
    all_generations = []
    all_scores = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 贪心解码
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # 提取生成的文本
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True)
            all_generations.append(generated_text.strip())

            # 计算平均置信度
            if outputs.scores:
                avg_score = torch.mean(torch.stack(
                    [torch.max(torch.softmax(score, dim=-1)) for score in outputs.scores]))
                all_scores.append(avg_score.item())
            else:
                all_scores.append(0.0)

    # 选择置信度最高的生成结果
    max_idx = np.argmax(all_scores)
    return all_generations[max_idx]


def evaluate_accuracy(results):
    """计算准确率"""
    predictions = []
    gold_answers = []

    for result in results:
        predictions.append([result["prediction"].lower().split()])
        gold_answers.append([[ans.lower().split()
                            for ans in result["answers"]]])

    # 使用partial_match计算准确率
    matches = []
    for pred, gold in zip(predictions, gold_answers):
        # pred是一个词列表，gold是答案选项列表的列表
        score = partial_match(pred[0], gold[0][0], birdirectional=False)
        matches.append(int(score))

    accuracy = sum(matches) / len(matches)
    return accuracy, matches


def main():
    model_name = "qwen2.5_1.5b_it"
    subset_size = 1000

    print("🎯 MyriadLAMA Self-Ensemble Comparison Test")
    print(f"📊 Model: {model_name}")
    print(f"📊 Subset size: {subset_size}")
    print("=" * 60)

    # 1. 创建子集数据
    subset_data, dataset = create_subset_dataset(model_name, subset_size)

    # 2. 运行avg方法
    print("\\n" + "=" * 60)
    avg_results = run_self_ensemble(model_name, subset_data, dataset, "avg")

    # 3. 运行max方法
    print("\\n" + "=" * 60)
    max_results = run_self_ensemble(model_name, subset_data, dataset, "max")

    # 4. 计算准确率
    print("\\n" + "=" * 60)
    print("📊 Calculating Accuracy...")

    avg_accuracy, avg_matches = evaluate_accuracy(avg_results)
    max_accuracy, max_matches = evaluate_accuracy(max_results)

    # 5. 生成报告
    print("\\n🎯 RESULTS SUMMARY")
    print("=" * 60)
    print(
        f"📈 AVG Method Accuracy: {avg_accuracy:.4f} ({sum(avg_matches)}/{len(avg_matches)})")
    print(
        f"📈 MAX Method Accuracy: {max_accuracy:.4f} ({sum(max_matches)}/{len(max_matches)})")
    print(f"📊 Difference: {abs(avg_accuracy - max_accuracy):.4f}")

    if avg_accuracy > max_accuracy:
        print("🏆 AVG method performs better!")
    elif max_accuracy > avg_accuracy:
        print("🏆 MAX method performs better!")
    else:
        print("🤝 Both methods perform equally!")

    # 6. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"myriadlama_comparison_{timestamp}.json"

    comparison_results = {
        "model_name": model_name,
        "subset_size": subset_size,
        "timestamp": timestamp,
        "avg_method": {
            "accuracy": avg_accuracy,
            "matches": sum(avg_matches),
            "total": len(avg_matches),
            "results": avg_results[:10]  # 保存前10个示例
        },
        "max_method": {
            "accuracy": max_accuracy,
            "matches": sum(max_matches),
            "total": len(max_matches),
            "results": max_results[:10]  # 保存前10个示例
        }
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {results_file}")
    print("✅ Comparison test completed successfully!")


if __name__ == "__main__":
    main()
