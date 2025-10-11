#!/usr/bin/env python3
"""
数据集预加载脚本
在正式实验前手动下载和预处理所有需要的数据集
"""
import os
import sys
from datetime import datetime
from dataset import MyriadLamaDataset, WebQADataset

def print_header():
    print("=" * 60)
    print("📥 数据集预加载脚本")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_directory_permissions(path):
    """检查目录权限"""
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
            print(f"✅ 创建目录: {parent_dir}")
        except PermissionError:
            print(f"❌ 权限错误: 无法创建目录 {parent_dir}")
            return False
    
    if not os.access(parent_dir, os.W_OK):
        print(f"❌ 权限错误: 无法写入目录 {parent_dir}")
        return False
    
    print(f"✅ 目录权限检查通过: {parent_dir}")
    return True

def prepare_myriadlama_dataset(model_name="qwen1.5_moe_a2.7b_chat"):
    """准备 MyriadLAMA 数据集"""
    print("📊 准备 MyriadLAMA 数据集...")
    print("-" * 40)
    
    try:
        # 检查权限
        dataset_root = f"/net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama"
        if not check_directory_permissions(dataset_root):
            return False
            
        print(f"模型: {model_name}")
        print("开始下载和预处理...")
        
        # 创建数据集实例（这会触发下载和预处理）
        dataset = MyriadLamaDataset(model_name=model_name)
        
        # 获取数据集信息
        data_size = len(dataset.ds)
        print(f"✅ MyriadLAMA 数据集准备完成!")
        print(f"   - 数据集大小: {data_size:,} 个样本")
        print(f"   - 原始数据路径: {dataset.dataset_path}")
        print(f"   - 输出目录: {dataset.dataset_root}")
        
        # 显示样本示例
        if data_size > 0:
            sample = dataset.ds[0]
            print(f"   - 样本示例:")
            print(f"     UUID: {sample['uuid']}")
            print(f"     答案数量: {len(sample['answers'])}")
            print(f"     答案示例: {sample['answers'][:3]}")
            print(f"     手动paraphrases: {len(sample['manual_paraphrases'])}")
            print(f"     自动paraphrases: {len(sample['auto_paraphrases'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ MyriadLAMA 数据集准备失败: {e}")
        return False

def prepare_webqa_dataset(model_name="qwen1.5_moe_a2.7b_chat"):
    """准备 WebQA 数据集 (可选)"""
    print("📊 准备 WebQA 数据集...")
    print("-" * 40)
    print("⚠️  注意: WebQA 需要加载模型进行paraphrase生成，这将花费较长时间")
    
    choice = input("是否要预处理 WebQA 数据集? (y/N): ").strip().lower()
    if choice != 'y':
        print("⏭️  跳过 WebQA 数据集预处理")
        return True
    
    try:
        # 检查权限
        dataset_root = f"/net/tokyo100-10g/data/str01_01/y-guo/datasets/webqa"
        if not check_directory_permissions(dataset_root):
            return False
            
        print(f"模型: {model_name}")
        print("开始下载、加载模型和生成paraphrases...")
        print("⏰ 这可能需要很长时间...")
        
        # 创建数据集实例
        dataset = WebQADataset(model_name=model_name)
        
        data_size = len(dataset.ds)
        print(f"✅ WebQA 数据集准备完成!")
        print(f"   - 数据集大小: {data_size:,} 个样本")
        print(f"   - 数据路径: {dataset.dataset_path}")
        print(f"   - 输出目录: {dataset.dataset_root}")
        
        return True
        
    except Exception as e:
        print(f"❌ WebQA 数据集准备失败: {e}")
        return False

def check_datasets_status():
    """检查数据集状态"""
    print("🔍 检查数据集状态...")
    print("-" * 40)
    
    datasets_info = [
        {
            "name": "MyriadLAMA (原始)",
            "path": "/net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/paraphrases_dataset",
            "type": "共享原始数据"
        },
        {
            "name": "MyriadLAMA (输出)",
            "path": "/net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/qwen1.5_moe_a2.7b_chat",
            "type": "模型特定输出目录"
        },
        {
            "name": "WebQA (输出)", 
            "path": "/net/tokyo100-10g/data/str01_01/y-guo/datasets/webqa/qwen1.5_moe_a2.7b_chat",
            "type": "模型特定输出目录"
        }
    ]
    
    for info in datasets_info:
        if os.path.exists(info["path"]):
            if os.path.isfile(info["path"]):
                size = os.path.getsize(info["path"]) / (1024*1024)  # MB
                print(f"✅ {info['name']}: 存在 ({size:.1f} MB)")
            else:
                files = len(os.listdir(info["path"]))
                print(f"✅ {info['name']}: 存在 ({files} 个文件)")
        else:
            print(f"❌ {info['name']}: 不存在")
        print(f"   路径: {info['path']}")
        print(f"   类型: {info['type']}")
        print()

def main():
    print_header()
    
    # 检查当前状态
    check_datasets_status()
    
    # 准备数据集
    model_name = "qwen1.5_moe_a2.7b_chat"
    print(f"🎯 使用模型: {model_name}")
    print()
    
    success_count = 0
    total_count = 0
    
    # 1. 准备 MyriadLAMA 数据集（主要的）
    total_count += 1
    if prepare_myriadlama_dataset(model_name):
        success_count += 1
    print()
    
    # 2. 准备 WebQA 数据集（可选的）
    total_count += 1
    if prepare_webqa_dataset(model_name):
        success_count += 1
    print()
    
    # 最终检查
    print("🔍 最终状态检查...")
    print("-" * 40)
    check_datasets_status()
    
    # 总结
    print("=" * 60)
    print("📋 预加载总结")
    print("=" * 60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功: {success_count}/{total_count} 个数据集")
    
    if success_count == total_count:
        print("🎉 所有数据集预加载完成！现在可以开始正式实验了。")
        print("\n建议的下一步:")
        print("  bash scripts/main.sh qwen1.5_moe_a2.7b_chat 0 myriadlama")
        return 0
    else:
        print("⚠️  部分数据集预加载失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    exit(main())