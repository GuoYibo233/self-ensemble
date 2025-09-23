#!/usr/bin/env python3
"""
下载MoE模型到lab服务器的共享存储
"""

import os
import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import snapshot_download


def download_moe_model():
    # 设置cache目录到共享存储
    cache_dir = '/net/tokyo100-10g/data/str01_01/yguo/cache/huggingface'
    model_dir = '/net/tokyo100-10g/data/str01_01/yguo/models/moe'
    model_name = 'Qwen/Qwen1.5-MoE-A2.7B-Chat'

    print(f'开始下载模型: {model_name}')
    print(f'Cache目录: {cache_dir}')
    print(f'模型保存目录: {model_dir}')

    # 确保目录存在
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 设置环境变量
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir

    try:
        print('=' * 60)
        print('开始下载模型组件...')
        print('=' * 60)

        # 使用snapshot_download获得更好的进度显示
        print('正在下载完整模型...')
        local_model_path = os.path.join(model_dir, 'qwen1.5_moe_a2.7b_chat')

        # 下载模型到cache并复制到本地目录
        downloaded_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f'模型下载到: {downloaded_path}')
        print('验证模型文件...')

        # 验证tokenizer和模型可以正常加载
        print('加载tokenizer进行验证...')
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True
        )

        print('加载模型进行验证...')
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='cpu'
        )

        print('=' * 60)
        print('模型下载和验证完成！')
        print('=' * 60)

        # 计算模型大小
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                total_size += size
                file_count += 1
                print(f'  {file}: {size / (1024**2):.1f} MB')

        print(f'总文件数: {file_count}')
        print(f'模型总大小: {total_size / (1024**3):.2f} GB')

    except Exception as e:
        print(f'下载过程中出现错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    download_moe_model()
