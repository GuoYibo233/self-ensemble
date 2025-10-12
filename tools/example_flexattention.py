#!/usr/bin/env python3
"""
Simple FlexAttention Example

This script demonstrates FlexAttention generation with a minimal example
that doesn't require downloading large datasets or models.

Usage:
    python3 example_flexattention.py
"""

import torch
import numpy as np

# Check if FlexAttention is available
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    print("✅ FlexAttention is available\n")
except ImportError:
    print("❌ FlexAttention not available. Install PyTorch 2.5+ or nightly.")
    print("   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121")
    exit(1)


def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def visualize_mask(mask_func, seq_len, max_display=15):
    """Visualize the attention mask as a matrix."""
    print("\nAttention Mask Visualization:")
    print("  (✓ = can attend, ✗ = cannot attend)")
    
    display_len = min(seq_len, max_display)
    
    # Print header
    print("\n  Q\\KV", end="")
    for kv in range(display_len):
        print(f" {kv:2d}", end="")
    print()
    
    # Print matrix
    for q in range(display_len):
        print(f"   {q:2d}  ", end="")
        for kv in range(display_len):
            can_attend = mask_func(0, 0, q, kv)
            symbol = "✓" if can_attend else "✗"
            print(f" {symbol} ", end="")
        print()
    
    if seq_len > max_display:
        print(f"\n  ... (showing first {max_display}x{max_display} of {seq_len}x{seq_len})")


def example_1_causal_mask():
    """Example 1: Simple causal (autoregressive) attention."""
    print_section("Example 1: Causal Attention")
    
    print("\nDescription:")
    print("  Standard autoregressive attention where each position can only")
    print("  attend to itself and previous positions.")
    
    # Define causal mask
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    seq_len = 10
    visualize_mask(causal_mask, seq_len)
    
    # Create tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, S, D = 1, 2, seq_len, 16
    
    Q = torch.randn(B, H, S, D, device=device)
    K = torch.randn(B, H, S, D, device=device)
    V = torch.randn(B, H, S, D, device=device)
    
    # Create block mask
    block_mask = create_block_mask(causal_mask, B=B, H=H, Q_LEN=S, KV_LEN=S, device=device)
    
    # Apply FlexAttention
    output = flex_attention(Q, K, V, block_mask=block_mask)
    
    print(f"\n✅ FlexAttention output shape: {output.shape}")
    print(f"   Expected: ({B}, {H}, {S}, {D})")


def example_2_segment_isolation():
    """Example 2: Segment isolation like in FlexAttention ensemble."""
    print_section("Example 2: Segment Isolation")
    
    print("\nDescription:")
    print("  Three segments where tokens can only attend within their own segment.")
    print("  This is similar to how paraphrases are isolated in FlexAttention ensemble.")
    
    # Define segments
    segments = [(0, 4), (4, 7), (7, 10)]
    seq_len = 10
    
    print(f"\nSegments:")
    for i, (start, end) in enumerate(segments):
        print(f"  Segment {i+1}: positions {start}-{end-1}")
    
    # Define segment isolation mask
    def segment_isolation_mask(b, h, q_idx, kv_idx):
        # Causal constraint
        if q_idx < kv_idx:
            return False
        
        # Find which segment q and kv belong to
        q_segment = None
        kv_segment = None
        
        for seg_id, (start, end) in enumerate(segments):
            if start <= q_idx < end:
                q_segment = seg_id
            if start <= kv_idx < end:
                kv_segment = seg_id
        
        # Only allow attention within same segment
        return q_segment is not None and q_segment == kv_segment
    
    visualize_mask(segment_isolation_mask, seq_len)
    
    # Verify isolation
    print("\n🔍 Verification:")
    for i, (start, end) in enumerate(segments):
        # Check that tokens in segment can attend to each other
        can_attend_self = segment_isolation_mask(0, 0, start, start)
        print(f"  Segment {i+1}: position {start} can attend to itself: {can_attend_self}")
        
        # Check that they cannot attend to other segments
        if i < len(segments) - 1:
            next_start = segments[i+1][0]
            cannot_attend_next = not segment_isolation_mask(0, 0, next_start, start)
            print(f"  Segment {i+2}: position {next_start} cannot attend to segment {i+1}: {cannot_attend_next}")


def example_3_fusion_after_encoding():
    """Example 3: Segment isolation during encoding, fusion during generation."""
    print_section("Example 3: Encoding + Generation (FlexAttention Ensemble)")
    
    print("\nDescription:")
    print("  Three segments (paraphrases) that are isolated during encoding,")
    print("  but new tokens (generation) can attend to all segments.")
    
    # Define segments and original length
    segments = [(0, 4), (4, 7), (7, 10)]
    original_length = 10
    current_length = 13  # 3 tokens have been generated
    
    print(f"\nConfiguration:")
    print(f"  Original segments: {segments}")
    print(f"  Original length: {original_length}")
    print(f"  Current length: {current_length}")
    print(f"  Generated tokens: {current_length - original_length} (positions {original_length}-{current_length-1})")
    
    # Define mask with generation fusion
    def encoding_generation_mask(b, h, q_idx, kv_idx):
        # Causal constraint
        if q_idx < kv_idx:
            return False
        
        # Generated tokens (beyond original) can attend to everything
        if q_idx >= original_length:
            return True
        
        # Original tokens only attend within segment
        q_segment = None
        kv_segment = None
        
        for seg_id, (start, end) in enumerate(segments):
            if start <= q_idx < end:
                q_segment = seg_id
            if start <= kv_idx < end:
                kv_segment = seg_id
        
        return q_segment is not None and q_segment == kv_segment
    
    visualize_mask(encoding_generation_mask, current_length)
    
    print("\n🔍 Key observations:")
    print(f"  ✅ Positions 0-3 (segment 1) only attend within segment 1")
    print(f"  ✅ Positions 4-6 (segment 2) only attend within segment 2")
    print(f"  ✅ Positions 7-9 (segment 3) only attend within segment 3")
    print(f"  ✅ Positions 10-12 (generated) attend to ALL previous positions")
    print(f"\n  This enables fusion: generated tokens see all paraphrases!")


def main():
    print("="*70)
    print("  FlexAttention Examples")
    print("="*70)
    print("\nThis script demonstrates how FlexAttention masks work.")
    print("Each example shows a different attention pattern.\n")
    
    # Run examples
    example_1_causal_mask()
    example_2_segment_isolation()
    example_3_fusion_after_encoding()
    
    print("\n" + "="*70)
    print("  Summary")
    print("="*70)
    print("\n✅ All examples completed successfully!")
    print("\nWhat you learned:")
    print("  1. How to create custom attention masks with FlexAttention")
    print("  2. How to implement causal (autoregressive) attention")
    print("  3. How to isolate segments (paraphrases) during encoding")
    print("  4. How to enable fusion during generation")
    print("\nNext steps:")
    print("  - Run: python3 tools/debug_flexattention.py --dataset webqa --max-samples 1")
    print("  - Read: docs/DELEGATE_PROMPT.md for full debugging guide")
    print("  - Explore: flex_attention_generate.py to see real implementation")


if __name__ == "__main__":
    main()
