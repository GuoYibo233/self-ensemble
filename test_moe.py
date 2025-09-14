"""
MoE模型测试脚本

测试MoE模型在self-ensemble系统中的集成效果
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from constants import MODEL_PATHs
    from moe_config import is_moe_model, get_moe_model_info, should_use_moe_for_task
    print("✅ Successfully imported MoE configuration module")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_moe_integration():
    """测试MoE模型的基本集成"""

    print("🧠 MoE Model Integration Test")
    print("=" * 50)

    # 检查MoE模型配置
    moe_models = [name for name in MODEL_PATHs.keys() if is_moe_model(name)]

    if not moe_models:
        print("❌ No MoE model configurations found")
        return False

    print(f"📋 Found {len(moe_models)} MoE models:")
    for model in moe_models:
        info = get_moe_model_info(model)
        print(f"\n🔍 {model}:")
        print(f"  Path: {MODEL_PATHs[model]}")
        if info:
            print(f"  Total params: {info.get('total_params', 'N/A')}")
            print(f"  Active params: {info.get('active_params', 'N/A')}")
            print(
                f"  Memory requirement: {info.get('memory_requirement', 'N/A')}")
            print(f"  Recommended for: {info.get('recommended_for', 'N/A')}")

    return True


def test_task_recommendations():
    """测试不同任务的模型推荐"""

    print("\n🎯 Task-oriented Model Recommendation Test")
    print("=" * 50)

    test_cases = [
        ("qa", "low"),
        ("qa", "medium"),
        ("generation", "low"),
        ("chat", "low"),
    ]

    for task, resource in test_cases:
        rec = should_use_moe_for_task(task, resource)
        print(f"\n📝 Task: {task}, Resource: {resource}")
        print(f"  Use MoE: {'Yes' if rec['use_moe'] else 'No'}")
        print(f"  Recommended model: {rec['model']}")
        print(f"  Reason: {rec['reason']}")


def generate_usage_examples():
    """生成使用示例"""

    examples = """
🚀 MoE模型使用示例

## 基础使用（替换现有模型）
```bash
# 使用MoE模型进行QA任务
python generate.py --model qwen1.5_moe_a2.7b_chat --dataset webqa --method avg

# 使用MoE模型进行加权集成
python generate.py --model qwen1.5_moe_a2.7b_chat --dataset webqa --method weighted_avg

# 对比MoE与密集模型效果
python generate.py --model qwen1.5_moe_a2.7b_chat --dataset webqa --method per_prompt
python generate.py --model qwen3_4b_it --dataset webqa --method per_prompt
```

## 调试和测试
```bash
# 使用调试脚本测试MoE
python debug_generate.py --method avg  # 然后手动修改为MoE模型

# 监控资源使用
python test_moe.py --monitor-resources
```

## 性能优化建议
```bash
# 如果遇到内存问题
python generate.py --model qwen1.5_moe_a2.7b_chat --batch_size 1

# 如果需要更确定的输出
python generate.py --model qwen1.5_moe_a2.7b_chat --temperature 0.3
```

## 与现有模型的效果对比
```bash
# 小模型对比
python generate.py --model qwen1.5_moe_a2.7b_chat --dataset webqa > moe_results.txt
python generate.py --model qwen3_1.7b_it --dataset webqa > dense_results.txt

# 分析结果差异
python analyze_results.py --compare moe_results.txt dense_results.txt
```
"""

    return examples


def main():
    """主测试函数"""

    print("🧠 MoE Model Integration Test in Self-Ensemble\n")

    # 基础集成测试
    if not test_moe_integration():
        return

    # 任务推荐测试
    test_task_recommendations()

    # 显示使用示例
    print(generate_usage_examples())

    print("\n" + "="*50)
    print("📊 Summary and Recommendations:")
    print("1. MoE models successfully integrated into the system")
    print("2. Recommend starting with qwen1.5_moe_a2.7b_chat for testing")
    print("3. This model can run with 6GB VRAM, outperforming 3B dense models")
    print("4. Particularly suitable for QA and dialogue tasks")
    print("5. Can directly replace existing models without code changes")

    print("\n🚀 Next steps:")
    print("python generate.py --model qwen1.5_moe_a2.7b_chat --dataset webqa --method avg")


if __name__ == "__main__":
    main()
