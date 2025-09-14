"""
Qwen3模型专用配置和优化设置

提供Qwen3模型的特殊配置，包括：
- 聊天模板配置
- 生成参数优化
- 特殊token处理
- 模型特定的最佳实践
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Qwen3模型的推荐生成参数（资源友好版）
QWEN3_GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
    "max_new_tokens": 256,  # 减少到256以节省显存
    "pad_token_id": None,  # 会在模型加载时设置
}

# Qwen3聊天模板配置（仅包含支持的模型）
QWEN3_CHAT_TEMPLATES = {
    "qwen3_1.7b_it": "qwen",
    "qwen3_4b_it": "qwen",
    "qwen3_8b_it": "qwen",
}


def setup_qwen3_model_and_tokenizer(model_path, device="auto", load_in_8bit=False):
    """
    为Qwen3模型设置特定的配置（资源优化版）

    Args:
        model_path: 模型路径
        device: 设备配置
        load_in_8bit: 是否使用8bit量化（推荐用于4b及以上模型）

    Returns:
        tuple: (model, tokenizer)
    """

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"  # Qwen3推荐左侧padding
    )

    # 设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 模型加载配置（资源优化）
    model_kwargs = {
        "device_map": device,
        "torch_dtype": torch.float16,  # 使用float16节省显存
        "trust_remote_code": True,
    }

    # 对于较大模型自动启用8bit量化
    if load_in_8bit or "4b" in model_path or "8b" in model_path:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
        print(f"✅ Enabled 8-bit quantization to save VRAM")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )

    # 设置生成配置
    generation_config = QWEN3_GENERATION_CONFIG.copy()
    generation_config["pad_token_id"] = tokenizer.pad_token_id

    for key, value in generation_config.items():
        setattr(model.generation_config, key, value)

    return model, tokenizer


def is_qwen3_model(model_name):
    """检查是否是Qwen3模型"""
    return model_name.startswith("qwen3")


def is_qwen3_instruct_model(model_name):
    """检查是否是Qwen3指令优化模型"""
    return model_name.startswith("qwen3") and model_name.endswith("_it")


def apply_qwen3_chat_template(tokenizer, messages):
    """
    为Qwen3指令模型应用聊天模板

    Args:
        tokenizer: Qwen3 tokenizer
        messages: 消息列表，格式为[{"role": "user", "content": "..."}]

    Returns:
        str: 格式化后的提示文本
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # 如果没有chat_template，使用简单格式
        formatted = ""
        for message in messages:
            if message["role"] == "user":
                formatted += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                formatted += f"Assistant: {message['content']}\n"
        formatted += "Assistant:"
        return formatted


def format_prompt_for_qwen3(prompt_text, model_name):
    """
    根据Qwen3模型类型格式化提示

    Args:
        prompt_text: 原始提示文本
        model_name: 模型名称

    Returns:
        str: 格式化后的提示
    """
    if is_qwen3_instruct_model(model_name):
        # 指令模型使用聊天格式
        messages = [{"role": "user", "content": prompt_text}]
        # 这里需要实际的tokenizer来应用模板，所以返回消息格式
        return {"type": "chat", "messages": messages}
    else:
        # 基础模型直接使用文本
        return {"type": "text", "content": prompt_text}


# Qwen3模型资源优化建议
QWEN3_OPTIMIZATION_TIPS = """
🚀 Qwen3模型资源优化指南（适合有限资源环境）

📊 推荐模型选择：
- qwen3_1.7b_it: ~4GB显存，CPU也可运行
- qwen3_4b_it:   ~8GB显存，8bit量化后~4GB
- qwen3_8b_it:   ~16GB显存，8bit量化后~8GB

💾 内存优化策略：
1. 自动8bit量化：4b及以上模型自动启用
2. 使用 torch.float16 替代 bfloat16
3. 减少 max_new_tokens 到 256
4. 小批量处理: batch_size=1-2

⚡ 速度优化：
1. 优先使用指令模型(_it)，收敛更快
2. 适当调低 temperature (0.3-0.7)
3. 使用 top_k=20 限制候选token数量

🎯 实用建议：
- 开发测试: 使用 qwen3_1.7b_it
- 生产环境: 使用 qwen3_4b_it + 8bit量化
- 高质量需求: 使用 qwen3_8b_it + 8bit量化

💡 显存不足时：
1. 启用8bit量化: --load_in_8bit
2. 减少批量大小: --batch_size 1
3. 使用CPU推理: --device cpu (仅1.7b模型)
"""

if __name__ == "__main__":
    print(QWEN3_OPTIMIZATION_TIPS)
