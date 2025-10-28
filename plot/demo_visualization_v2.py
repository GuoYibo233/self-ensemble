#!/usr/bin/env python3
"""
Demo Script - Generate English-only Visualizations with Prompt Structure

This script demonstrates FlexAttention visualizations with:
- English-only labels in all diagrams
- Clear question labeling (question1, question2, etc.)
- Prompt structure diagrams showing instruction + few-shot + question relationship
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')


def create_prompt_structure_diagram():
    """
    Create a diagram showing the structure of a single prompt.
    Shows: Instruction + Few-shot Examples + Question
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    y_start = 9
    box_height = 1.5
    box_width = 10
    x_center = 6
    
    # Title
    ax.text(x_center, 9.5, 'Single Prompt Structure', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Instruction box
    y_current = y_start - 1
    rect1 = FancyBboxPatch(
        (x_center - box_width/2, y_current - box_height),
        box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor='#E3F2FD',
        edgecolor='#1976D2',
        linewidth=2
    )
    ax.add_patch(rect1)
    ax.text(x_center, y_current - box_height/2, 
            'INSTRUCTION\n"Answer the question based on your knowledge."',
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.arrow(x_center, y_current - box_height, 0, -0.3,
             head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Few-shot examples box
    y_current -= (box_height + 0.4)
    box_height2 = 2.0
    rect2 = FancyBboxPatch(
        (x_center - box_width/2, y_current - box_height2),
        box_width, box_height2,
        boxstyle="round,pad=0.1",
        facecolor='#FFF3E0',
        edgecolor='#F57C00',
        linewidth=2
    )
    ax.add_patch(rect2)
    example_text = (
        'FEW-SHOT EXAMPLES (demonstrations)\n\n'
        'Q: What is 2+2?\nA: 4\n\n'
        'Q: What is the capital of France?\nA: Paris'
    )
    ax.text(x_center, y_current - box_height2/2, example_text,
            ha='center', va='center', fontsize=9, family='monospace')
    
    # Arrow
    ax.arrow(x_center, y_current - box_height2, 0, -0.3,
             head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Question box
    y_current -= (box_height2 + 0.4)
    box_height3 = 1.2
    rect3 = FancyBboxPatch(
        (x_center - box_width/2, y_current - box_height3),
        box_width, box_height3,
        boxstyle="round,pad=0.1",
        facecolor='#E8F5E9',
        edgecolor='#388E3C',
        linewidth=2
    )
    ax.add_patch(rect3)
    ax.text(x_center, y_current - box_height3/2,
            'QUESTION (to be answered)\nQ: What is the capital of Germany?\nA:',
            ha='center', va='center', fontsize=10, family='monospace', fontweight='bold')
    
    # Add legend
    legend_y = 0.5
    ax.text(1, legend_y, 'Components:', fontsize=10, fontweight='bold')
    ax.text(1, legend_y - 0.3, '• Instruction: Task description', fontsize=9)
    ax.text(1, legend_y - 0.5, '• Few-shot: Example Q&A pairs', fontsize=9)
    ax.text(1, legend_y - 0.7, '• Question: Query to answer', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_concatenated_prompts_diagram():
    """
    Create a diagram showing how multiple paraphrases are concatenated.
    Clearly labels question1, question2, etc.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(7, 11.5, 'FlexAttention: Concatenated Paraphrases Structure',
            ha='center', fontsize=16, fontweight='bold')
    
    # Configuration
    num_paraphrases = 5
    box_height = 1.5
    box_width = 12
    x_center = 7
    y_start = 10
    
    # Draw each paraphrase
    for i in range(num_paraphrases):
        y_pos = y_start - (i * (box_height + 0.3))
        
        # Main box for this paraphrase
        color = plt.cm.Set3(i)
        rect = FancyBboxPatch(
            (x_center - box_width/2, y_pos - box_height),
            box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='#424242',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Label: Question N
        ax.text(x_center - box_width/2 + 0.5, y_pos - box_height/2,
                f'Question{i+1}',
                ha='left', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))
        
        # Content structure within each question
        # Instruction + Few-shot + Question
        content_x = x_center - box_width/2 + 3
        ax.text(content_x, y_pos - box_height/2,
                f'[Instruction + Few-shot + Question{i+1}]',
                ha='left', va='center', fontsize=9, family='monospace')
        
        # Token count annotation
        token_count = 45 + i * 2  # Example varying lengths
        ax.text(x_center + box_width/2 - 0.5, y_pos - box_height/2,
                f'~{token_count} tokens',
                ha='right', va='center', fontsize=8, style='italic')
        
        # Separator (except for last one)
        if i < num_paraphrases - 1:
            sep_y = y_pos - box_height - 0.15
            ax.plot([x_center - box_width/2, x_center + box_width/2], 
                   [sep_y, sep_y], 
                   'r--', linewidth=2, label='[SEP]' if i == 0 else '')
            ax.text(x_center, sep_y - 0.05, '[SEP]',
                   ha='center', va='top', fontsize=9, color='red', fontweight='bold')
    
    # Add annotations
    # Original content
    ax.plot([x_center + box_width/2 + 0.5, x_center + box_width/2 + 0.5],
           [y_start, y_start - (num_paraphrases * (box_height + 0.3)) + 0.3],
           'b-', linewidth=3)
    ax.text(x_center + box_width/2 + 1, 
           y_start - (num_paraphrases * (box_height + 0.3))/2,
           'Original\nContent\n(Encoding\nPhase)',
           ha='left', va='center', fontsize=10, fontweight='bold', color='blue')
    
    # Generation area
    gen_y = y_start - (num_paraphrases * (box_height + 0.3))
    gen_box = Rectangle(
        (x_center - box_width/2, gen_y - 1),
        box_width, 0.8,
        facecolor='#C8E6C9',
        edgecolor='#2E7D32',
        linewidth=2
    )
    ax.add_patch(gen_box)
    ax.text(x_center, gen_y - 0.6,
           'Generated Answer (can attend to all questions)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Legend
    handles = [
        mpatches.Patch(color='lightblue', label='Paraphrase (isolated during encoding)'),
        mpatches.Patch(color='#C8E6C9', label='Generated tokens (fusion attention)'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_attention_mask_with_labels():
    """
    Create attention mask visualization with clear question labels.
    """
    # Configuration
    num_questions = 5
    tokens_per_question = 12
    separator = 2
    num_generated = 5
    
    # Create segment positions
    segments = []
    pos = 0
    for i in range(num_questions):
        if i > 0:
            pos += separator
        start = pos
        end = pos + tokens_per_question
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
    fig, ax = plt.subplots(figsize=(12, 11))
    
    cmap = plt.cm.colors.ListedColormap(['white', '#2E7D32'])
    im = ax.imshow(mask, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(range(0, total_length, 5))
    ax.set_yticks(range(0, total_length, 5))
    
    # Add grid
    ax.set_xticks(np.arange(total_length) - 0.5, minor=True)
    ax.set_yticks(np.arange(total_length) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Mark segment boundaries with labels
    colors = plt.cm.Set3(range(num_questions))
    for i, (start, end) in enumerate(segments):
        # Vertical line
        ax.axvline(x=start - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.6)
        # Horizontal line
        ax.axhline(y=start - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.6)
        
        # Label on top
        mid_point = (start + end - 1) / 2
        ax.text(mid_point, -2, f'Question{i+1}',
               ha='center', va='top', fontsize=10, fontweight='bold',
               rotation=0, color=colors[i],
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors[i], linewidth=2))
        
        # Label on left
        ax.text(-2, mid_point, f'Q{i+1}',
               ha='right', va='center', fontsize=10, fontweight='bold',
               color=colors[i],
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors[i], linewidth=2))
    
    # Mark generation start
    ax.axhline(y=original_length - 0.5, color='blue', linewidth=3, linestyle='--', alpha=0.8)
    ax.axvline(x=original_length - 0.5, color='blue', linewidth=3, linestyle='--', alpha=0.8)
    
    # Generation label
    gen_mid = (original_length + total_length - 1) / 2
    ax.text(gen_mid, -2, 'Generated',
           ha='center', va='top', fontsize=10, fontweight='bold',
           color='blue',
           bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax.text(-2, gen_mid, 'Gen',
           ha='right', va='center', fontsize=10, fontweight='bold',
           color='blue',
           bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='blue', linewidth=2))
    
    # Labels
    ax.set_xlabel('Key/Value Position (KV)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Query Position (Q)', fontsize=13, fontweight='bold')
    ax.set_title('Attention Mask: Segment Isolation + Fusion\n(Green = Can Attend, White = Cannot Attend)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Cannot Attend', 'Can Attend'])
    
    # Add info text
    info_text = (
        f"Configuration:\n"
        f"  • Questions: {num_questions}\n"
        f"  • Tokens per question: ~{tokens_per_question}\n"
        f"  • Separator tokens: {separator}\n"
        f"  • Original length: {original_length}\n"
        f"  • Generated tokens: {num_generated}\n"
        f"  • Total length: {total_length}\n\n"
        f"Attention Pattern:\n"
        f"  • Each question isolated during encoding\n"
        f"  • Generated tokens attend to all questions\n"
        f"  • Enables information fusion"
    )
    
    plt.gcf().text(0.98, 0.5, info_text, 
                   fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   verticalalignment='center',
                   horizontalalignment='left',
                   family='monospace')
    
    plt.tight_layout()
    return fig


def create_flowchart_english():
    """
    Create flowchart with English-only labels.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis('off')
    
    center_x = 5
    box_w = 3.5
    box_h = 0.6
    
    # Define steps (English only)
    steps = [
        (13.5, 'Input Preparation', '#E3F2FD'),
        (12.3, 'Generate 5 Paraphrases\n(Question1...Question5)', '#E3F2FD'),
        (11.1, 'Concatenate with [SEP]', '#FFF3E0'),
        (9.9, 'Track Segment Positions', '#FFF3E0'),
        (8.7, 'Create FlexAttention Mask', '#F3E5F5'),
        (7.5, 'Patch Model Layers', '#F3E5F5'),
        (6.3, 'Encoding Phase\n(Segment Isolation)', '#E8F5E9'),
        (5.1, 'Generation Loop\n(Fusion Attention)', '#E8F5E9'),
        (3.9, 'Token Selection (argmax)', '#E8F5E9'),
        (2.7, 'Decode Output', '#FCE4EC'),
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
            'FlexAttention Processing Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#E3F2FD', label='Input'),
        mpatches.Patch(color='#FFF3E0', label='Processing'),
        mpatches.Patch(color='#F3E5F5', label='Attention Setup'),
        mpatches.Patch(color='#E8F5E9', label='Generation'),
        mpatches.Patch(color='#FCE4EC', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig


def main():
    """Main function"""
    print("="*70)
    print("Generating English-Only Visualizations with Question Labels")
    print("="*70)
    
    output_dir = 'demo_outputs_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}/")
    print("-"*70)
    
    # 1. Prompt structure
    print("\n1. Generating prompt structure diagram...")
    fig1 = create_prompt_structure_diagram()
    fig1.savefig(os.path.join(output_dir, 'prompt_structure.png'), dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: prompt_structure.png")
    plt.close(fig1)
    
    # 2. Concatenated prompts
    print("\n2. Generating concatenated prompts diagram...")
    fig2 = create_concatenated_prompts_diagram()
    fig2.savefig(os.path.join(output_dir, 'concatenated_questions.png'), dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: concatenated_questions.png")
    plt.close(fig2)
    
    # 3. Attention mask with labels
    print("\n3. Generating attention mask with question labels...")
    fig3 = create_attention_mask_with_labels()
    fig3.savefig(os.path.join(output_dir, 'attention_mask_labeled.png'), dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: attention_mask_labeled.png")
    plt.close(fig3)
    
    # 4. Flowchart
    print("\n4. Generating English-only flowchart...")
    fig4 = create_flowchart_english()
    fig4.savefig(os.path.join(output_dir, 'flowchart_english.png'), dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: flowchart_english.png")
    plt.close(fig4)
    
    print("\n" + "="*70)
    print("✓ All visualizations generated!")
    print("="*70)
    print(f"\nView generated images in: {output_dir}/")
    print("\nKey improvements:")
    print("  • All diagram labels are in English only")
    print("  • Questions clearly labeled as Question1, Question2, etc.")
    print("  • New diagram showing prompt structure (Instruction + Few-shot + Question)")
    print("  • Clear visualization of how components relate")


if __name__ == "__main__":
    main()
