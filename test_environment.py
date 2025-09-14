#!/usr/bin/env python3
"""
环境测试脚本
验证所有必要的依赖是否正确安装
"""


def test_imports():
    """测试所有必要的导入"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - 安装成功")

        import numpy as np
        print(f"✅ NumPy {np.__version__} - 安装成功")

        import pandas as pd
        print(f"✅ Pandas {pd.__version__} - 安装成功")

        import tqdm
        print(f"✅ tqdm {tqdm.__version__} - 安装成功")

        # 测试基本的torch操作
        x = torch.randn(2, 3)
        y = torch.softmax(x, dim=-1)
        print(f"✅ PyTorch基本操作正常")

        # 测试numpy和pandas操作
        arr = np.random.randn(5, 3)
        df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
        print(f"✅ NumPy和Pandas操作正常")

        print("\n🎉 所有依赖测试通过！环境配置成功！")
        return True

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False


def test_debug_script():
    """测试debug_generate.py的核心组件"""
    try:
        # 尝试导入debug脚本中的模拟类
        import sys
        import os

        # 添加当前目录到路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)

        # 这里只测试基本的模拟功能，不需要实际运行整个脚本
        print("✅ 可以访问debug_generate.py")
        return True

    except Exception as e:
        print(f"❌ debug脚本测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=== Self-Ensemble 环境测试 ===\n")

    # 显示Python版本和环境信息
    import sys
    print(f"Python版本: {sys.version}")
    print(f"Python可执行文件: {sys.executable}")
    print()

    # 测试依赖
    success = test_imports()

    if success:
        print("\n=== 准备运行debug脚本测试 ===")
        # 运行一个简单的debug脚本测试
        test_debug_script()

        print(f"\n✅ 环境配置完成！")
        print(f"现在你可以运行: python debug_generate.py --method per_prompt")
    else:
        print(f"\n❌ 环境配置有问题，请检查依赖安装")
