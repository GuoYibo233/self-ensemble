#!/usr/bin/env python3
"""
Small scale experiment with 10 Q&A pairs using MoE model
"""
import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from constants import MODEL_PATHs

def run_small_experiment():
    print("=== Starting Small Scale Experiment ===")
    
    # Configuration
    model_name = "qwen1.5_moe_a2.7b_chat"
    num_samples = 10
    device = "cuda:0"
    
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Device: {device}")
    
    # Load model
    model_path = MODEL_PATHs[model_name]
    print(f"\nLoading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model and tokenizer loaded")
    
    # Load small dataset
    print(f"\nLoading WebQA dataset (first {num_samples} samples)...")
    test_ds = load_dataset("stanfordnlp/web_questions", split=f"test[:{num_samples}]")
    print(f"✓ Loaded {len(test_ds)} samples")
    
    # Prepare results
    results = []
    
    # Process each question
    print(f"\nProcessing {num_samples} questions...")
    for i, item in enumerate(tqdm(test_ds)):
        question = item["question"]
        answers = item["answers"]
        
        # Create prompt
        prompt = f"Answer the question based on general world knowledge. Provide a short and direct answer.\n\nQ: {question}\nA:"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = full_response[len(prompt):].strip().split('\n')[0]
        
        # Store result
        result = {
            'id': i,
            'question': question,
            'prediction': prediction,
            'correct_answers': answers,
            'full_response': full_response
        }
        results.append(result)
        
        print(f"Q{i+1}: {question}")
        print(f"A{i+1}: {prediction}")
        print(f"Expected: {answers}")
        print("---")
    
    # Save results
    os.makedirs("./results/test", exist_ok=True)
    df = pd.DataFrame(results)
    output_file = f"./results/test/small_experiment_{model_name}_{num_samples}samples.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n=== Experiment Completed ===")
    print(f"Results saved to: {output_file}")
    print(f"Total questions processed: {len(results)}")
    
    # Simple accuracy check
    print("\n=== Sample Results ===")
    for i, result in enumerate(results[:5]):  # Show first 5
        print(f"Q{i+1}: {result['question']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Expected: {result['correct_answers']}")
        print()

if __name__ == "__main__":
    run_small_experiment()