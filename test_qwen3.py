"""
Qwen3模型测试脚本

用于测试Qwen3模型在self-ensemble系统中的集成效果
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from constants import MODEL_PATHs
    from qwen3_config import (
        setup_qwen3_model_and_tokenizer,
        is_qwen3_model,
        is_qwen3_instruct_model,
        format_prompt_for_qwen3,
        apply_qwen3_chat_template,
        QWEN3_OPTIMIZATION_TIPS
    )
    print("✅ Successfully imported Qwen3 configuration module")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_qwen3_basic_generation(model_name="qwen3_1.7b"):
    """测试Qwen3基本生成功能"""

    print(f"\n🔄 Testing Qwen3 model: {model_name}")

    if model_name not in MODEL_PATHs:
        print(f"❌ Model {model_name} not in supported list")
        print(f"Supported models: {list(MODEL_PATHs.keys())}")
        return False

    try:
        # 获取模型路径
        model_path = MODEL_PATHs[model_name]
        print(f"📍 Model path: {model_path}")

        # 加载模型（这里只是测试配置，不实际加载大模型）
        print("⚙️ Testing model configuration...")

        # 测试提示格式化
        test_prompt = "What is the capital of France?"
        formatted = format_prompt_for_qwen3(test_prompt, model_name)

        print(f"📝 Original prompt: {test_prompt}")
        print(f"📋 Formatted result: {formatted}")

        # 检查模型类型
        print(f"🔍 Is Qwen3 model: {is_qwen3_model(model_name)}")
        print(f"🎯 Is instruction model: {is_qwen3_instruct_model(model_name)}")

        print("✅ Basic configuration test passed")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_all_qwen3_models():
    """测试所有Qwen3模型的配置"""

    print("🚀 Starting tests for all Qwen3 model configurations...")

    qwen3_models = [name for name in MODEL_PATHs.keys()
                    if name.startswith("qwen3")]

    if not qwen3_models:
        print("❌ No Qwen3 model configurations found")
        return

    print(f"📋 Found {len(qwen3_models)} Qwen3 models:")
    for model in qwen3_models:
        print(f"  - {model}")

    print("\n" + "="*50)

    success_count = 0
    for model_name in qwen3_models:
        if test_qwen3_basic_generation(model_name):
            success_count += 1
        print("-" * 30)

    print(
        f"\n📊 Test results: {success_count}/{len(qwen3_models)} model configurations passed")


def show_integration_guide():
    """显示Qwen3集成指南"""

    guide = """
    🎯 Qwen3集成到self-ensemble的完整指南
    
    1. 📦 安装依赖:
       pip install transformers>=4.37.0
       pip install torch>=2.0.0
       pip install flash-attn  # 可选，用于加速
    
    2. 🔧 使用Qwen3模型:
       # 基础模型
       python generate.py --model qwen3_1.7b --dataset webqa --method avg
       
       # 指令优化模型  
       python generate.py --model qwen3_4b_it --dataset webqa --method weighted_avg
       
       # 大模型（需要更多显存）
       python generate.py --model qwen3_14b_it --dataset webqa --method max
    
    3. ⚡ 性能优化:
       # 使用8bit量化节省显存
       python generate.py --model qwen3_8b_it --load_in_8bit
       
       # 指定GPU设备
       python generate.py --model qwen3_4b_it --device cuda:0
    
    4. 🎛️ 生成参数调优:
       推荐参数在 qwen3_config.py 中已预设：
       - temperature: 0.7 (平衡创造性和准确性)
       - top_p: 0.8 (核采样)
       - repetition_penalty: 1.05 (避免重复)
    
    5. 💡 最佳实践:
       - 使用指令模型(_it)处理QA任务
       - 基础模型适合续写和生成任务
       - 根据GPU显存选择合适的模型大小
       - 批量处理时注意显存管理
    
    6. 🔍 调试建议:
       # 使用调试脚本测试单个模型
       python test_qwen3.py --model qwen3_1.7b_it
       
       # 查看详细输出
       python generate.py --model qwen3_4b_it --verbose
    """

    print(guide)
    print(QWEN3_OPTIMIZATION_TIPS)


def main():
    parser = argparse.ArgumentParser(description="Qwen3模型测试工具")
    parser.add_argument("--model", type=str, help="指定要测试的Qwen3模型")
    parser.add_argument("--test-all", action="store_true", help="测试所有Qwen3模型")
    parser.add_argument("--guide", action="store_true", help="显示集成指南")

    args = parser.parse_args()

    if args.guide:
        show_integration_guide()
    elif args.test_all:
        test_all_qwen3_models()
    elif args.model:
        test_qwen3_basic_generation(args.model)
    else:
        print("Please specify an operation:")
        print("  --model <model_name> : Test specific model")
        print("  --test-all          : Test all Qwen3 models")
        print("  --guide             : Show integration guide")
        print("\nAvailable Qwen3 models:")
        qwen3_models = [name for name in MODEL_PATHs.keys()
                        if name.startswith("qwen3")]
        for model in qwen3_models:
            print(f"  - {model}")


if __name__ == "__main__":
    main()
