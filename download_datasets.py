#!/usr/bin/env python3
"""
预下载数据集脚本
将 HuggingFace 数据集下载到本地目录
"""
import os
from datasets import load_dataset

# 数据集根目录
DATATASET_ROOT = "/net/tokyo100-10g/data/str01_01/y-guo/datasets"

def download_myriadlama():
    """下载 MyriadLAMA 数据集到本地"""
    print("🔄 开始下载 MyriadLAMA 数据集...")
    
    # 创建目录
    myriad_dir = os.path.join(DATATASET_ROOT, "myriadlama")
    raw_dataset_path = os.path.join(myriad_dir, "raw_dataset")
    
    os.makedirs(myriad_dir, exist_ok=True)
    
    if os.path.exists(raw_dataset_path):
        print(f"✅ 原始数据集已存在: {raw_dataset_path}")
        return True
    
    try:
        print("📥 从 HuggingFace 下载中...")
        ds = load_dataset("iszhaoxin/MyriadLAMA", split="train")
        
        print(f"💾 保存到本地: {raw_dataset_path}")
        ds.save_to_disk(raw_dataset_path)
        
        print(f"✅ MyriadLAMA 数据集下载完成!")
        print(f"   数据量: {len(ds)} 条记录")
        print(f"   保存位置: {raw_dataset_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_webqa():
    """下载 WebQA 数据集到本地"""
    print("\n🔄 开始下载 WebQA 数据集...")
    
    # 创建目录
    webqa_dir = os.path.join(DATATASET_ROOT, "webqa")
    raw_dataset_path = os.path.join(webqa_dir, "raw_dataset")
    
    os.makedirs(webqa_dir, exist_ok=True)
    
    if os.path.exists(raw_dataset_path):
        print(f"✅ WebQA 数据集已存在: {raw_dataset_path}")
        return True
    
    try:
        print("📥 从 HuggingFace 下载 WebQA...")
        ds_train = load_dataset("stanfordnlp/web_questions", split="train")
        ds_test = load_dataset("stanfordnlp/web_questions", split="test")
        
        print(f"💾 保存到本地: {raw_dataset_path}")
        # 保存训练集和测试集
        ds_train.save_to_disk(os.path.join(raw_dataset_path, "train"))
        ds_test.save_to_disk(os.path.join(raw_dataset_path, "test"))
        
        print(f"✅ WebQA 数据集下载完成!")
        print(f"   训练集: {len(ds_train)} 条记录")
        print(f"   测试集: {len(ds_test)} 条记录")
        print(f"   保存位置: {raw_dataset_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ WebQA 下载失败: {e}")
        return False

def main():
    print("🚀 开始预下载数据集...")
    print(f"📁 目标目录: {DATATASET_ROOT}")
    
    # 下载 MyriadLAMA
    success1 = download_myriadlama()
    
    # 下载 WebQA
    success2 = download_webqa()
    
    print("\n" + "="*50)
    if success1 and success2:
        print("🎉 所有数据集下载完成!")
        print(f"📂 数据集位置:")
        print(f"   MyriadLAMA: {DATATASET_ROOT}/myriadlama/raw_dataset")
        print(f"   WebQA: {DATATASET_ROOT}/webqa/raw_dataset")
        print("\n现在可以运行实验，数据将从本地加载 🚀")
    else:
        print("❌ 部分数据集下载失败，请检查网络连接")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())