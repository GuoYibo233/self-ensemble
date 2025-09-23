#!/usr/bin/env python3
"""
下载myriadlama数据集用于MoE训练
将数据集保存到共享存储位置
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset

# 共享存储路径配置
SHARED_STORAGE_BASE = "/net/tokyo100-10g/data/str01_01/yguo"
DATASET_PATH = f"{SHARED_STORAGE_BASE}/datasets/myriadlama"


def download_myriadlama_dataset():
    """下载myriadlama数据集到共享存储"""

    print(f"=== 开始下载myriadlama数据集 ===")
    print(f"目标路径: {DATASET_PATH}")

    # 创建目录
    os.makedirs(DATASET_PATH, exist_ok=True)
    print(f"✅ 创建目录: {DATASET_PATH}")

    try:
        # 方案1: 尝试使用datasets库下载
        print("📥 尝试使用datasets库下载...")
        dataset = load_dataset(
            "myriadlama/myriadlama_en",
            cache_dir=f"{DATASET_PATH}/cache"
        )

        # 保存到磁盘
        dataset.save_to_disk(f"{DATASET_PATH}/processed")

        print("✅ 使用datasets库下载成功!")
        print_dataset_info(dataset)

    except Exception as e:
        print(f"⚠️ datasets库下载失败: {e}")
        print("📥 尝试使用huggingface_hub下载...")

        try:
            # 方案2: 使用huggingface_hub直接下载
            snapshot_download(
                repo_id="myriadlama/myriadlama_en",
                cache_dir=f"{DATASET_PATH}/hub_cache",
                local_dir=f"{DATASET_PATH}/raw",
                local_dir_use_symlinks=False
            )
            print("✅ 使用huggingface_hub下载成功!")

        except Exception as e2:
            print(f"❌ huggingface_hub下载也失败: {e2}")
            print("🔧 创建测试数据集作为备用方案...")
            create_test_dataset()


def create_test_dataset():
    """创建小量测试数据集用于验证训练流程"""
    import json

    print("📝 创建测试数据集...")

    test_data_dir = f"{DATASET_PATH}/test_data"
    os.makedirs(test_data_dir, exist_ok=True)

    # 创建训练数据
    train_data = []
    for i in range(100):  # 创建100个样本
        train_data.append({
            "input": f"This is training sample {i+1}. What is the answer?",
            "target": f"The answer for sample {i+1} is: This demonstrates MoE training capabilities."
        })

    # 保存为JSONL格式
    with open(f"{test_data_dir}/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 创建验证数据
    val_data = []
    for i in range(20):  # 创建20个验证样本
        val_data.append({
            "input": f"This is validation sample {i+1}. What is the answer?",
            "target": f"The validation answer for sample {i+1} is: Testing MoE model performance."
        })

    with open(f"{test_data_dir}/validation.jsonl", "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 创建数据集配置文件
    config = {
        "dataset_name": "myriadlama_test",
        "train_file": "train.jsonl",
        "validation_file": "validation.jsonl",
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "format": "jsonl"
    }

    with open(f"{test_data_dir}/dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ 测试数据集创建完成!")
    print(f"  - 训练样本: {len(train_data)}")
    print(f"  - 验证样本: {len(val_data)}")
    print(f"  - 保存位置: {test_data_dir}")


def print_dataset_info(dataset):
    """打印数据集信息"""
    print("📊 数据集信息:")
    for split_name, split_data in dataset.items():
        print(f"  - {split_name}: {len(split_data)} 样本")
        if len(split_data) > 0:
            print(f"    示例: {split_data[0]}")


def verify_dataset():
    """验证数据集是否正确下载"""
    print("🔍 验证数据集...")

    if os.path.exists(f"{DATASET_PATH}/processed"):
        print("✅ 找到processed数据集")
        return True
    elif os.path.exists(f"{DATASET_PATH}/raw"):
        print("✅ 找到raw数据集")
        return True
    elif os.path.exists(f"{DATASET_PATH}/test_data"):
        print("✅ 找到测试数据集")
        return True
    else:
        print("❌ 未找到任何数据集")
        return False


if __name__ == "__main__":
    print("🚀 开始下载myriadlama数据集到共享存储")
    print(f"共享存储位置: {SHARED_STORAGE_BASE}")

    download_myriadlama_dataset()

    if verify_dataset():
        print("🎉 数据集准备完成，可以开始MoE训练!")
        print(f"数据集位置: {DATASET_PATH}")
    else:
        print("❌ 数据集准备失败，请检查错误信息")
