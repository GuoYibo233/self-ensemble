import os

MODEL_PATHs = {
    # Local models (原有的xzhao路径保留作为备选)
    "llama3.2_3b_it": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.2_3b_it",
    "llama3.2_1b_it": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.2_1b_it",
    "llama3.1_8b_it": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.1_8b_it",
    "llama3.2_3b": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.2_3b",
    "llama3.2_1b": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.2_1b",
    "llama3.1_8b": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.1_8b",
    
    # Hugging Face models (会自动下载到~/.cache/huggingface/)
    "qwen2.5_7b": "Qwen/Qwen2.5-7B",
    "qwen2.5_7b_it": "Qwen/Qwen2.5-7B-Instruct", 
    "qwen2.5_3b_it": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    
    # Additional HuggingFace models for FlexAttention testing
    "llama3.2_3b_hf": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2_1b_hf": "meta-llama/Llama-3.2-1B-Instruct", 
    "llama3.1_8b_hf": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi3_mini": "microsoft/Phi-3-mini-4k-instruct",
}