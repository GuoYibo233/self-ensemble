#!/usr/bin/env python3
"""
简单测试：直接运行数据集处理，验证文件保存位置
"""
from dataset import MyriadLamaDataset

def main():
    print("=" * 60)
    print("🧪 测试数据集处理位置")
    print("=" * 60)
    
    # 创建数据集实例
    model_name = "qwen1.5_moe_a2.7b_chat"
    print(f"模型名称: {model_name}")
    
    # 实例化会自动触发数据集下载和处理
    print("\n开始处理数据集...")
    dataset = MyriadLamaDataset(model_name=model_name)
    
    # 显示路径信息
    print("\n📂 路径信息:")
    print(f"数据集根目录: {dataset.dataset_root}")
    print(f"原始数据路径: {dataset.dataset_path}")
    
    # 显示数据集信息
    print(f"\n📊 数据集信息:")
    print(f"数据集大小: {len(dataset.ds):,} 个样本")
    
    # 显示第一个样本
    if len(dataset.ds) > 0:
        sample = dataset.ds[0]
        print(f"\n📝 样本示例:")
        print(f"UUID: {sample['uuid']}")
        print(f"答案: {sample['answers'][:2]}...")
        print(f"手动paraphrases数量: {len(sample['manual_paraphrases'])}")
        print(f"自动paraphrases数量: {len(sample['auto_paraphrases'])}")
    
    print("\n✅ 数据集处理完成！")
    print("\n所有文件都保存在你的目录:")
    print(f"  {dataset.dataset_root}")

if __name__ == "__main__":
    main()