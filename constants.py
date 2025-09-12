"""
常量配置。

本模块集中维护可用模型的标识与其在本地/Hub 的加载路径映射。
根据你的运行环境，可能需要将本地绝对路径替换为可用的本机路径，
或者直接使用 Hugging Face Hub 上的模型名称。
"""

import os

MODEL_PATHs = {
    "llama3.2_3b_it": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.2_3b_it",
    "llama3.2_1b_it": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.2_1b_it",
    "llama3.1_8b_it": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.1_8b_it",
    "llama3.2_3b": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.2_3b",
    "llama3.2_1b": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.2_1b",
    "llama3.1_8b": "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.1_8b",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B",
    "qwen2.5_7b_it": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5_3b_it": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}
