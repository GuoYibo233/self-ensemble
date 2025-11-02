#!/usr/bin/env python3
"""
测试可视化脚本 / Test Visualization Script

This script tests the key functions from the visualization notebook
without requiring jupyter to be installed.
"""

import numpy as np
import sys

def create_segment_positions(config):
    """
    根据配置创建分段位置
    Create segment positions based on configuration
    """
    segment_positions = []
    current_pos = 0
    
    for i in range(config['num_paraphrases']):
        if i > 0:
            current_pos += config['separator_tokens']
        
        start = current_pos
        end = current_pos + config['tokens_per_paraphrase']
        segment_positions.append((start, end))
        current_pos = end
    
    original_length = current_pos
    total_length = original_length + config['num_generated_tokens']
    
    return segment_positions, original_length, total_length


def create_attention_mask_function(segment_positions, original_length):
    """
    创建注意力掩码函数
    Create attention mask function
    """
    def mask_func(b, h, q_idx, kv_idx):
        # Causal constraint
        if q_idx < kv_idx:
            return False
        
        # Generated tokens can attend to all previous
        if q_idx >= original_length:
            return True
        
        # Original tokens only attend within same segment
        q_segment = None
        kv_segment = None
        
        for seg_id, (start, end) in enumerate(segment_positions):
            if start <= q_idx < end:
                q_segment = seg_id
            if start <= kv_idx < end:
                kv_segment = seg_id
        
        if q_segment is not None and kv_segment is not None:
            return q_segment == kv_segment
        
        return False
    
    return mask_func


def test_small_config():
    """测试小型配置 / Test small configuration"""
    print("\n" + "="*70)
    print("测试小型配置 / Testing Small Configuration")
    print("="*70)
    
    config = {
        'num_paraphrases': 3,
        'tokens_per_paraphrase': 15,
        'separator_tokens': 2,
        'num_generated_tokens': 5,
    }
    
    segment_positions, original_length, total_length = create_segment_positions(config)
    
    print(f"\n配置参数 / Configuration:")
    print(f"  改写数量 / Paraphrases: {config['num_paraphrases']}")
    print(f"  每段令牌数 / Tokens per paraphrase: {config['tokens_per_paraphrase']}")
    print(f"  分隔符令牌 / Separator tokens: {config['separator_tokens']}")
    print(f"  生成令牌数 / Generated tokens: {config['num_generated_tokens']}")
    
    print(f"\n计算结果 / Computed Results:")
    print(f"  分段位置 / Segment positions:")
    for i, (start, end) in enumerate(segment_positions):
        print(f"    Segment {i+1}: [{start:3d}, {end:3d}) - length {end-start}")
    print(f"  原始长度 / Original length: {original_length}")
    print(f"  总长度 / Total length: {total_length}")
    
    # Test mask function
    mask_func = create_attention_mask_function(segment_positions, original_length)
    
    # Create mask matrix
    mask_matrix = np.zeros((total_length, total_length))
    for q in range(total_length):
        for kv in range(total_length):
            mask_matrix[q, kv] = 1 if mask_func(0, 0, q, kv) else 0
    
    print(f"\n掩码统计 / Mask Statistics:")
    print(f"  矩阵大小 / Matrix size: {mask_matrix.shape}")
    print(f"  可关注位置 / Attention-allowed positions: {np.sum(mask_matrix):.0f}")
    print(f"  总位置数 / Total positions: {mask_matrix.size}")
    print(f"  注意力比例 / Attention ratio: {np.sum(mask_matrix)/mask_matrix.size*100:.1f}%")
    
    # Verify key properties
    print(f"\n验证关键属性 / Verifying Key Properties:")
    
    # 1. Causal constraint
    causal_violations = 0
    for q in range(total_length):
        for kv in range(q+1, total_length):
            if mask_matrix[q, kv] == 1:
                causal_violations += 1
    print(f"  ✓ 因果约束违规 / Causal violations: {causal_violations} (should be 0)")
    
    # 2. Segment isolation during encoding
    encoding_crossovers = 0
    for q in range(original_length):
        for kv in range(original_length):
            if mask_matrix[q, kv] == 1:
                # Check if q and kv are in same segment
                q_seg = None
                kv_seg = None
                for seg_id, (start, end) in enumerate(segment_positions):
                    if start <= q < end:
                        q_seg = seg_id
                    if start <= kv < end:
                        kv_seg = seg_id
                if q_seg != kv_seg:
                    encoding_crossovers += 1
    print(f"  ✓ 编码阶段跨段关注 / Encoding cross-segment attention: {encoding_crossovers} (should be 0)")
    
    # 3. Generation fusion
    if total_length > original_length:
        gen_pos = original_length
        gen_attention_count = np.sum(mask_matrix[gen_pos, :gen_pos+1])
        print(f"  ✓ 第一个生成令牌可关注位置 / First generated token attends to: {gen_attention_count:.0f}/{gen_pos+1}")
    
    return mask_matrix


def test_medium_config():
    """测试中型配置 / Test medium configuration"""
    print("\n" + "="*70)
    print("测试中型配置 / Testing Medium Configuration")
    print("="*70)
    
    config = {
        'num_paraphrases': 5,
        'tokens_per_paraphrase': 25,
        'separator_tokens': 3,
        'num_generated_tokens': 8,
    }
    
    segment_positions, original_length, total_length = create_segment_positions(config)
    mask_func = create_attention_mask_function(segment_positions, original_length)
    
    print(f"\n配置参数 / Configuration:")
    print(f"  改写数量 / Paraphrases: {config['num_paraphrases']}")
    print(f"  总长度 / Total length: {total_length}")
    
    # Sample some key positions for verification
    test_cases = [
        (10, 10, True, "Same position in segment 1"),
        (10, 5, True, "Within segment 1"),
        (10, 30, False, "Across segments (1 to 2)"),
        (original_length, 10, True, "Generated token to segment 1"),
        (original_length + 5, original_length + 2, True, "Generated token to earlier generated"),
    ]
    
    print(f"\n测试用例 / Test Cases:")
    all_passed = True
    for q, kv, expected, desc in test_cases:
        if q < total_length and kv < total_length:
            result = mask_func(0, 0, q, kv)
            status = "✓" if result == expected else "✗"
            if result != expected:
                all_passed = False
            print(f"  {status} Q={q:3d}, KV={kv:3d}: {result:5} (expected {expected:5}) - {desc}")
    
    if all_passed:
        print(f"\n✓ 所有测试通过！/ All tests passed!")
    else:
        print(f"\n✗ 部分测试失败 / Some tests failed")
        sys.exit(1)
    
    return True


def main():
    """主测试函数 / Main test function"""
    print("\n" + "="*70)
    print("FlexAttention 可视化笔记本测试 / FlexAttention Visualization Notebook Test")
    print("="*70)
    
    try:
        # Test small configuration
        mask_small = test_small_config()
        
        # Test medium configuration
        test_medium_config()
        
        print("\n" + "="*70)
        print("✓ 所有测试成功完成！/ All tests completed successfully!")
        print("="*70)
        print("\n笔记本功能验证通过 / Notebook functions verified")
        print("可以在 Jupyter 中运行完整的可视化 / Ready to run full visualizations in Jupyter")
        
    except Exception as e:
        print(f"\n✗ 测试失败 / Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
