#!/usr/bin/env python3
"""
Small scale test script for MoE model with only 10 Q&A pairs
"""
import os
import argparse
from dataset import WebQADataset
from constants import MODEL_PATHs

def test_small_scale():
    print("=== Small Scale MoE Test ===")
    print("Testing with qwen1.5_moe_a2.7b_chat model")
    print("Limited to 10 Q&A pairs")
    
    # Test model availability
    model_name = "qwen1.5_moe_a2.7b_chat"
    if model_name not in MODEL_PATHs:
        print(f"Error: Model {model_name} not found in MODEL_PATHs")
        return
    
    model_path = MODEL_PATHs[model_name]
    print(f"Model path: {model_path}")
    print(f"Model path exists: {os.path.exists(model_path)}")
    
    # Test basic imports
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ Transformers import successful")
    except Exception as e:
        print(f"✗ Transformers import failed: {e}")
        return
    
    # Test tokenizer loading
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        return
    
    # Test model loading
    try:
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype="auto"
        )
        print(f"✓ Model loaded successfully")
        print(f"Model device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return
    
    # Test basic generation
    try:
        print("Testing basic generation...")
        test_prompt = "Q: What is the capital of France?\nA:"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation test successful")
        print(f"Test output: {response}")
        
    except Exception as e:
        print(f"✗ Generation test failed: {e}")
        return
    
    print("=== All tests passed! Ready for small scale experiment ===")

if __name__ == "__main__":
    import torch
    test_small_scale()