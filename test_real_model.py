"""
真实模型测试脚本 - 使用下载的Qwen2.5-1.5B模型
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from datetime import datetime

def test_real_model_generation():
    """使用真实模型进行简单的生成测试"""
    
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
    
    # 测试提示
    test_prompts = [
        "What is the capital of France?",
        "Explain machine learning in simple terms.",
        "Write a short story about a robot.",
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt}")
        
        # 准备输入
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        
        print(f"🎯 Answer: {answer}")
        
        results.append({
            "prompt": prompt,
            "response": response,
            "answer": answer
        })
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"real_model_test_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_path": model_path,
            "device": str(device),
            "timestamp": timestamp,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {result_file}")
    print("✅ Real model test completed successfully!")
    
    return results

if __name__ == "__main__":
    test_real_model_generation()