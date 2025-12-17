#!/usr/bin/env python3
"""
演示脚本 - 生成可视化示例图片 / Demo Script - Generate Sample Visualizations

This script demonstrates what the notebook produces without requiring Jupyter.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')


def create_sample_flowchart():
    """创建简化的流程图示例 / Create simplified flowchart example"""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis('off')
    
    center_x = 5
    box_w = 3.5
    box_h = 0.6
    
    # Define simplified steps
    steps = [
        (13.5, 'Input Preparation\n输入准备', '#E3F2FD'),
        (12.0, 'Concatenate Paraphrases\n拼接改写', '#FFF3E0'),
        (10.5, 'Create FlexAttention Mask\n创建掩码', '#F3E5F5'),
        (9.0, 'Patch Model Layers\n模型打补丁', '#F3E5F5'),
        (7.5, 'Encoding Phase\n编码阶段', '#E8F5E9'),
        (6.0, 'Generation Loop\n生成循环', '#E8F5E9'),
        (4.5, 'Token Selection\n令牌选择', '#E8F5E9'),
        (3.0, 'Decode Output\n解码输出', '#FCE4EC'),
    ]
    
    # Draw boxes and arrows
    for i, (y, label, color) in enumerate(steps):
        # Draw box
        box = FancyBboxPatch(
            (center_x - box_w/2, y - box_h/2),
            box_w, box_h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='#424242',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(center_x, y, label, 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw arrow to next step
        if i < len(steps) - 1:
            y_from = y - box_h/2
            y_to = steps[i+1][0] + box_h/2
            arrow = FancyArrowPatch(
                (center_x, y_from),
                (center_x, y_to),
                arrowstyle='->,head_width=0.4,head_length=0.4',
                color='#424242',
                linewidth=2
            )
            ax.add_patch(arrow)
    
    # Add title
    ax.text(center_x, 14.5, 
            'FlexAttention Code Flowchart\nFlexAttention 代码流程图',
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#E3F2FD', label='Input / 输入'),
        mpatches.Patch(color='#FFF3E0', label='Processing / 处理'),
        mpatches.Patch(color='#F3E5F5', label='Attention / 注意力'),
        mpatches.Patch(color='#E8F5E9', label='Generation / 生成'),
        mpatches.Patch(color='#FCE4EC', label='Output / 输出')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_sample_attention_mask():
    """创建简化的注意力掩码示例 / Create simplified attention mask example"""
    # Configuration
    num_paraphrases = 3
    tokens_per_para = 10
    separator = 2
    num_generated = 3
    
    # Create segment positions
    segments = []
    pos = 0
    for i in range(num_paraphrases):
        if i > 0:
            pos += separator
        start = pos
        end = pos + tokens_per_para
        segments.append((start, end))
        pos = end
    
    original_length = pos
    total_length = original_length + num_generated
    
    # Create mask matrix
    mask = np.zeros((total_length, total_length))
    
    for q in range(total_length):
        for kv in range(total_length):
            # Causal constraint
            if q < kv:
                continue
            
            # Generated tokens can attend to all
            if q >= original_length:
                mask[q, kv] = 1
                continue
            
            # Original tokens only within segment
            q_seg = None
            kv_seg = None
            for seg_id, (start, end) in enumerate(segments):
                if start <= q < end:
                    q_seg = seg_id
                if start <= kv < end:
                    kv_seg = seg_id
            
            if q_seg is not None and kv_seg is not None and q_seg == kv_seg:
                mask[q, kv] = 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))
    
    cmap = plt.cm.colors.ListedColormap(['white', '#2E7D32'])
    im = ax.imshow(mask, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(range(0, total_length, 2))
    ax.set_yticks(range(0, total_length, 2))
    
    # Add grid
    ax.set_xticks(np.arange(total_length) - 0.5, minor=True)
    ax.set_yticks(np.arange(total_length) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Mark segment boundaries
    for start, end in segments:
        ax.axhline(y=start - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.6)
        ax.axvline(x=start - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.6)
    
    # Mark generation start
    ax.axhline(y=original_length - 0.5, color='blue', linewidth=2, linestyle='--', alpha=0.6)
    ax.axvline(x=original_length - 0.5, color='blue', linewidth=2, linestyle='--', alpha=0.6)
    
    # Labels
    ax.set_xlabel('Key/Value Position (KV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Position (Q)', fontsize=12, fontweight='bold')
    ax.set_title('Attention Mask Visualization\n注意力掩码可视化 (Simple Example)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Cannot Attend\n不可关注', 'Can Attend\n可关注'])
    
    # Add info text
    info_text = (
        f"Configuration / 配置:\n"
        f"  • Paraphrases / 改写数: {num_paraphrases}\n"
        f"  • Tokens per para / 每段令牌: {tokens_per_para}\n"
        f"  • Original length / 原始长度: {original_length}\n"
        f"  • Generated tokens / 生成数: {num_generated}\n"
        f"  • Total length / 总长度: {total_length}\n\n"
        f"Legend / 图例:\n"
        f"  ━━ Red / 红色: Segment boundary / 分段边界\n"
        f"  ━━ Blue / 蓝色: Generation start / 生成开始"
    )
    
    plt.gcf().text(0.98, 0.5, info_text, 
                   fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                   verticalalignment='center',
                   horizontalalignment='left')
    
    plt.tight_layout()
    return fig


def create_attention_pattern_comparison():
    """创建注意力模式对比图 / Create attention pattern comparison"""
    # Same configuration as above
    num_paraphrases = 3
    tokens_per_para = 10
    separator = 2
    num_generated = 3
    
    segments = []
    pos = 0
    for i in range(num_paraphrases):
        if i > 0:
            pos += separator
        start = pos
        end = pos + tokens_per_para
        segments.append((start, end))
        pos = end
    
    original_length = pos
    total_length = original_length + num_generated
    
    # Create mask function
    def can_attend(q, kv):
        if q < kv:
            return False
        if q >= original_length:
            return True
        q_seg = None
        kv_seg = None
        for seg_id, (start, end) in enumerate(segments):
            if start <= q < end:
                q_seg = seg_id
            if start <= kv < end:
                kv_seg = seg_id
        return q_seg is not None and kv_seg is not None and q_seg == kv_seg
    
    # Select representative positions
    query_positions = [
        5,                      # Middle of segment 1
        original_length,        # First generated token
    ]
    
    labels = [
        'Encoding Phase (Segment 1)\n编码阶段（分段1）',
        'Generation Phase (First Token)\n生成阶段（第一个令牌）',
    ]
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (q_pos, label) in enumerate(zip(query_positions, labels)):
        pattern = np.array([can_attend(q_pos, kv) for kv in range(total_length)])
        
        ax = axes[idx]
        colors = ['green' if val else 'lightgray' for val in pattern]
        ax.bar(range(total_length), pattern, color=colors, width=1.0, edgecolor='none')
        
        # Mark boundaries
        for start, end in segments:
            ax.axvline(x=start, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=original_length, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Key/Value Position', fontsize=10)
        ax.set_ylabel('Can Attend', fontsize=10)
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        attend_count = np.sum(pattern)
        ax.text(0.98, 0.95, f'Attend: {attend_count:.0f}/{total_length}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Attention Pattern Comparison / 注意力模式对比', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def main():
    """主函数 / Main function"""
    print("="*70)
    print("生成演示可视化 / Generating Demo Visualizations")
    print("="*70)
    
    output_dir = 'demo_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n输出目录 / Output directory: {output_dir}/")
    print("-"*70)
    
    # Generate flowchart
    print("\n1. 生成流程图 / Generating flowchart...")
    fig1 = create_sample_flowchart()
    flowchart_path = os.path.join(output_dir, 'demo_flowchart.png')
    fig1.savefig(flowchart_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 已保存 / Saved: {flowchart_path}")
    plt.close(fig1)
    
    # Generate attention mask
    print("\n2. 生成注意力掩码 / Generating attention mask...")
    fig2 = create_sample_attention_mask()
    mask_path = os.path.join(output_dir, 'demo_attention_mask.png')
    fig2.savefig(mask_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 已保存 / Saved: {mask_path}")
    plt.close(fig2)
    
    # Generate pattern comparison
    print("\n3. 生成注意力模式对比 / Generating pattern comparison...")
    fig3 = create_attention_pattern_comparison()
    pattern_path = os.path.join(output_dir, 'demo_attention_patterns.png')
    fig3.savefig(pattern_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 已保存 / Saved: {pattern_path}")
    plt.close(fig3)
    
    print("\n" + "="*70)
    print("✓ 所有演示可视化已生成！/ All demo visualizations generated!")
    print("="*70)
    print(f"\n查看生成的图片 / View generated images in: {output_dir}/")
    print("\n这些是笔记本生成的可视化示例。")
    print("These are examples of visualizations generated by the notebook.")
    print("\n要查看完整的交互式版本，请运行笔记本：")
    print("To see the full interactive version, run the notebook:")
    print("  jupyter notebook flowchart_and_attention_mask_visualization.ipynb")


if __name__ == "__main__":
    main()
