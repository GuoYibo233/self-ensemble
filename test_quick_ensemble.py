"""
快速测试脚本 - 使用预定义的paraphrases进行self-ensemble测试
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from datetime import datetime
import numpy as np

def test_quick_self_ensemble():
    """使用预定义的5个问题和paraphrases进行快速测试"""
    
    model_path = "D:/Codes/Models/qwen2.5_1.5b_it"
    
    print("🚀 Loading Qwen2.5-1.5B-Instruct model...")
    print(f"📁 Model path: {model_path}")
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    device = next(model.parameters()).device
    print(f"🖥️ Using device: {device}")
    
    # 预定义的测试数据（来自paraphrase.py）
    test_data = [
        {
            "original": "who is the governor of hawaii now?",
            "paraphrases": [
                "as of now, who leads Hawaii as its governor?",
                "who's currently serving as Hawaii's governor?",
                "can you tell me who governs Hawaii right now?",
                "who's in charge of the Hawaii state government these days?",
                "who's the top executive official in Hawaii right now?",
            ]
        },
        {
            "original": "what was nelson mandela's religion?",
            "paraphrases": [
                "what was Mandela's faith tradition",
                "can you tell me Mandela's religion?",
                "what faith did Nelson Mandela practice?",
                "what was the religious affiliation of Nelson Mandela?",
                "what religion did Nelson Mandela follow?",
            ]
        },
        {
            "original": "who played sean in scrubs?",
            "paraphrases": [
                "which actor portrayed Sean in Scrubs?",
                "who took on the role of Sean in Scrubs?",
                "who played the character Sean in the TV show Scrubs?",
                "who was the actor that played Sean in the series Scrubs?",
                "do you know who played the part of Sean in Scrubs?",
            ]
        },
    ]
    
    results = []
    
    for i, item in enumerate(test_data, 1):
        print(f"\n🔍 Test {i}: {item['original']}")
        
        # 收集所有变体（原问题 + 5个paraphrases）
        all_prompts = [item['original']] + item['paraphrases']
        all_responses = []
        all_logits = []
        
        # 对每个变体生成回答
        for j, prompt in enumerate(all_prompts):
            prompt_type = "original" if j == 0 else f"paraphrase_{j}"
            print(f"  📝 {prompt_type}: {prompt}")
            
            # 准备输入
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # 提取回答
                generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                all_responses.append(response)
                
                # 提取logits（用于ensemble）
                # 这里简化处理，只取第一个token的logits作为示例
                if outputs.scores:
                    first_token_logits = outputs.scores[0][0].cpu().numpy()
                    all_logits.append(first_token_logits)
                
                print(f"    💬 Response: {response}")
        
        # Self-ensemble方法
        print(f"\n🎯 Self-Ensemble Results for Question {i}:")
        print(f"  📊 Original: {all_responses[0]}")
        
        # 简单的多数投票（基于前几个词的相似性）
        from collections import Counter
        first_words = [resp.split()[:2] for resp in all_responses if resp.strip()]
        first_words_str = [' '.join(words) for words in first_words if words]
        if first_words_str:
            most_common = Counter(first_words_str).most_common(1)[0]
            print(f"  🏆 Most Common Start: {most_common[0]} (appears {most_common[1]} times)")
        
        # 平均长度
        avg_length = np.mean([len(resp.split()) for resp in all_responses if resp.strip()])
        print(f"  📏 Average Response Length: {avg_length:.1f} words")
        
        results.append({
            "question": item['original'],
            "paraphrases": item['paraphrases'],
            "responses": all_responses,
            "avg_length": avg_length
        })
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quick_self_ensemble_test_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {filename}")
    print("✅ Quick self-ensemble test completed successfully!")

if __name__ == "__main__":
    test_quick_self_ensemble()