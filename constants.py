import os

MODEL_PATHs = {
    "llama3.2_3b_it": "/net/tokyo100-10g/data/str01_01/yguo/models/llama_hf/llama3.2_3b_it",
    "llama3.2_1b_it": "/net/tokyo100-10g/data/str01_01/yguo/models/llama_hf/llama3.2_1b_it",
    "llama3.1_8b_it": "/net/tokyo100-10g/data/str01_01/yguo/models/llama_hf/llama3.1_8b_it",
    "llama3.2_3b": "/net/tokyo100-10g/data/str01_01/yguo/models/llama_hf/llama3.2_3b",
    "llama3.2_1b": "/net/tokyo100-10g/data/str01_01/yguo/models/llama_hf/llama3.2_1b",
    "llama3.1_8b": "/net/tokyo100-10g/data/str01_01/yguo/models/llama_hf/llama3.1_8b",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B",
    "qwen2.5_7b_it": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5_3b_it": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    # MoE Models
    "qwen1.5_moe_a2.7b_chat": "/net/tokyo100-10g/data/str01_01/yguo/models/moe/qwen1.5_moe_a2.7b_chat",
    "deepseek_moe_16b_chat": "/net/tokyo100-10g/data/str01_01/yguo/models/moe/deepseek_moe_16b_chat",
    "llada_moe_7b_instruct": "/net/tokyo100-10g/data/str01_01/yguo/models/moe/llada_moe_7b_instruct",
}