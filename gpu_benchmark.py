#!/usr/bin/env python3
"""
GPU vs CPU 性能对比脚本
用于测试RTX 4070 Ti移动版的加速效果
"""

import torch
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def benchmark_device(device_name, vocab_size=10000, batch_size=8, seq_len=512, num_iterations=100):
    """在指定设备上进行性能测试"""
    device = torch.device(device_name)
    logger.info(f"\n=== 测试设备: {device} ===")

    if device.type == "cuda":
        logger.info(f"GPU名称: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(
            f"GPU计算能力: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")

    # 创建测试数据
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device)

    # 模拟语言模型的logits计算
    def simulate_model_forward():
        # 模拟embedding + transformer计算
        logits = torch.randn(batch_size, seq_len, vocab_size,
                             device=device, requires_grad=True)

        # 模拟常见的语言模型操作
        # 1. Softmax计算概率
        probs = torch.softmax(logits, dim=-1)

        # 2. 交叉熵损失计算
        targets = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )

        # 3. 反向传播
        loss.backward()

        # 4. 下一个token预测
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        return loss.item(), next_tokens

    # 预热
    logger.info("预热中...")
    for _ in range(10):
        simulate_model_forward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 正式测试
    logger.info(f"开始性能测试 ({num_iterations} 次迭代)...")
    start_time = time.time()

    losses = []
    for i in range(num_iterations):
        loss, tokens = simulate_model_forward()
        losses.append(loss)

        if (i + 1) % 20 == 0:
            logger.info(f"完成 {i+1}/{num_iterations} 次迭代")

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time

    # 计算性能指标
    avg_time_per_iteration = total_time / num_iterations
    tokens_per_second = (batch_size * seq_len * num_iterations) / total_time

    logger.info(f"总耗时: {total_time:.2f} 秒")
    logger.info(f"平均每次迭代: {avg_time_per_iteration*1000:.2f} 毫秒")
    logger.info(f"处理速度: {tokens_per_second:.0f} tokens/秒")
    logger.info(f"平均损失: {np.mean(losses):.4f}")

    if device.type == "cuda":
        logger.info(
            f"GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(
            f"GPU内存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    return {
        'device': device_name,
        'total_time': total_time,
        'avg_time_per_iteration': avg_time_per_iteration,
        'tokens_per_second': tokens_per_second,
        'avg_loss': np.mean(losses)
    }


def test_ensemble_operations(device_name):
    """测试集成操作的性能"""
    device = torch.device(device_name)
    logger.info(f"\n=== 集成操作测试: {device} ===")

    # 模拟多组logits
    num_groups = 6  # 6个不同的提示
    batch_size = 8
    vocab_size = 50000  # 更大的词汇表

    logits_set = torch.randn(num_groups, batch_size, vocab_size, device=device)
    weights = torch.rand(num_groups, batch_size, device=device)
    weights = weights / weights.sum(dim=0, keepdim=True)  # 归一化

    # 测试不同集成方法
    methods = {
        'avg': lambda x, w: x.mean(dim=0),
        'max': lambda x, w: x.softmax(dim=-1).max(dim=0).values,
        'weighted_avg': lambda x, w: (x * w.unsqueeze(-1)).sum(dim=0)
    }

    num_iterations = 1000

    for method_name, method_func in methods.items():
        logger.info(f"\n测试 {method_name} 方法...")

        start_time = time.time()
        for _ in range(num_iterations):
            result = method_func(logits_set, weights)
            next_tokens = torch.argmax(result, dim=-1)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations

        logger.info(f"{method_name}: {avg_time*1000:.3f} 毫秒/次")


def main():
    logger.info("=== RTX 4070 Ti 移动版性能测试 ===")

    # 检查CUDA可用性
    if not torch.cuda.is_available():
        logger.error("CUDA不可用！请检查PyTorch CUDA安装。")
        logger.info("当前PyTorch版本:", torch.__version__)
        return

    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA版本: {torch.version.cuda}")

    results = []

    # 测试CPU性能
    logger.info("\n" + "="*50)
    cpu_result = benchmark_device("cpu")
    results.append(cpu_result)

    # 测试GPU性能
    logger.info("\n" + "="*50)
    gpu_result = benchmark_device("cuda")
    results.append(gpu_result)

    # 性能对比
    logger.info("\n" + "="*50)
    logger.info("=== 性能对比 ===")
    speedup = cpu_result['avg_time_per_iteration'] / \
        gpu_result['avg_time_per_iteration']
    throughput_speedup = gpu_result['tokens_per_second'] / \
        cpu_result['tokens_per_second']

    logger.info(f"GPU加速比: {speedup:.2f}x (迭代时间)")
    logger.info(f"吞吐量提升: {throughput_speedup:.2f}x (tokens/秒)")

    if speedup > 2:
        logger.info("🚀 GPU显著提升性能，建议在实际项目中使用GPU！")
    elif speedup > 1.2:
        logger.info("⚡ GPU有一定性能提升，对大规模任务有帮助。")
    else:
        logger.info("💭 对于小规模调试任务，CPU已经足够。")

    # 测试集成操作
    logger.info("\n" + "="*50)
    test_ensemble_operations("cpu")
    test_ensemble_operations("cuda")

    logger.info("\n=== 建议 ===")
    logger.info("对于debug_generate.py调试脚本:")
    logger.info("- 学习代码逻辑: CPU足够")
    logger.info("- 性能对比测试: 可以尝试GPU")
    logger.info("- 大规模实验: 建议使用GPU")


if __name__ == "__main__":
    main()
