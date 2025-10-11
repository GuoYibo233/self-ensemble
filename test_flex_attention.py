#!/usr/bin/env python3
"""
测试FlexAttention ensemble生成的简单脚本
"""

import sys
import os
sys.path.append('/home/y-guo/self-ensemble/self-ensemble')

from flex_attention_generate import flex_attention_generation
from dataset import WebQADataset
from constants import MODEL_PATHs
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_flex_attention():
    print("🧪 Testing FlexAttention Ensemble Generation")
    print("=" * 50)
    
    # 设置模型
    model_name = "llama3.2_3b_it"
    model_path = MODEL_PATHs.get(model_name)
    
    print(f"Loading model: {model_name}")
    print(f"Model path: {model_path}")
    
    # 加载数据集
    print("Loading WebQA dataset...")
    dataset = WebQADataset(model_name=model_name)
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
    
    # 获取第一个样本
    for uuids, answers, all_paraphrases in dataloader:
        print(f"✅ Got sample UUID: {uuids[0]}")
        print(f"✅ Answers: {answers[0]}")
        print(f"✅ Number of paraphrases: {len(all_paraphrases)}")
        
        # 选择前5个paraphrases
        selected_paraphrases = all_paraphrases[:5]
        paraphrases_for_question = [para[0] for para in selected_paraphrases]
        
        print("\n📝 Selected paraphrases:")
        for i, para in enumerate(paraphrases_for_question):
            print(f"   {i+1}. {para}")
            
        # 构建prompts
        few_shot_context = dataset.get_few_shot_examples()
        prompts = []
        for paraphrase in paraphrases_for_question:
            prompt = dataset.construct_prompts(few_shot_context, [paraphrase])
            prompts.append(prompt[0])
            
        print(f"\n🔥 Testing FlexAttention generation with {len(prompts)} prompts...")
        
        # 这里会自动加载模型和tokenizer
        global model, tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype="auto"
        )
        
        # 运行FlexAttention生成
        result = flex_attention_generation(prompts, tokenizer, model, max_new_tokens=10)
        
        print(f"\n✅ Generated result: {result}")
        print("\n🎉 Test completed successfully!")
        break
        
if __name__ == "__main__":
    test_flex_attention()