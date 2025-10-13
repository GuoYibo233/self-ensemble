#!/usr/bin/env python3
"""
Test script to demonstrate the improved mask matrix visualization.
This can run without loading any models.
"""


def visualize_mask_old(mask_func, seq_len, max_display=20):
    """OLD version: Only shows first 20x20."""
    print("\n=== OLD VERSION: Limited to 20x20 ===")
    print("Attention Mask Visualization:")
    print("  (✓ = can attend, ✗ = cannot attend)")
    
    display_len = min(seq_len, max_display)
    
    # Print header
    print("\n  Q\\KV", end="")
    for kv in range(display_len):
        print(f" {kv:2d}", end="")
    print()
    
    # Print matrix
    for q in range(display_len):
        print(f"  {q:4d} ", end="")
        for kv in range(display_len):
            can_attend = mask_func(0, 0, q, kv)
            symbol = "✓" if can_attend else "✗"
            print(f" {symbol} ", end="")
        print()
    
    if seq_len > max_display:
        print(f"\n  ... (truncated, showing first {max_display}x{max_display} of {seq_len}x{seq_len})")
        print(f"  ❌ PROBLEM: Cannot see the overall structure of {seq_len} tokens!")


def visualize_mask_new(mask_func, seq_len, segment_positions, original_length, max_display=25):
    """NEW version: Smart sampling to show structure."""
    print("\n=== NEW VERSION: Smart Sampling for Large Sequences ===")
    print("Attention Mask Visualization:")
    
    # For large sequences, use smart sampling to show overall structure
    if seq_len > max_display:
        # Sample positions intelligently:
        # 1. Include segment boundaries to show structure
        # 2. Sample within each segment
        # 3. Include generated token positions
        positions = set()
        
        # Add segment boundaries and a few positions within each segment
        for start, end in segment_positions:
            positions.add(start)  # Start of segment
            positions.add(end - 1)  # End of segment
            # Add a few positions within segment
            segment_len = end - start
            if segment_len > 4:
                positions.add(start + segment_len // 3)
                positions.add(start + 2 * segment_len // 3)
        
        # Add original_length boundary
        if original_length < seq_len:
            positions.add(original_length)
            positions.add(original_length - 1)
        
        # Add some generated token positions
        if seq_len > original_length:
            gen_count = min(5, seq_len - original_length)
            for i in range(gen_count):
                positions.add(original_length + i)
        
        # Add last position
        positions.add(seq_len - 1)
        
        # Fill remaining slots with evenly spaced positions
        positions_list = sorted(list(positions))
        while len(positions_list) < max_display and len(positions_list) < seq_len:
            # Find largest gap
            max_gap = 0
            max_gap_idx = 0
            for i in range(len(positions_list) - 1):
                gap = positions_list[i+1] - positions_list[i]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_idx = i
            
            if max_gap <= 1:
                break
            
            # Insert midpoint of largest gap
            mid = (positions_list[max_gap_idx] + positions_list[max_gap_idx + 1]) // 2
            positions_list.insert(max_gap_idx + 1, mid)
        
        positions = sorted(positions_list[:max_display])
        print(f"\nMask Matrix ({seq_len}x{seq_len}):")
        print(f"  ✅ Showing {len(positions)} strategic positions (including segment boundaries)")
    else:
        positions = list(range(seq_len))
        print(f"\nMask Matrix ({seq_len}x{seq_len}):")
    
    # Create matrix for display positions
    matrix = [[0 for _ in range(len(positions))] for _ in range(len(positions))]
    for i, q in enumerate(positions):
        for j, kv in enumerate(positions):
            can_attend = mask_func(0, 0, q, kv)
            matrix[i][j] = 1 if can_attend else 0
    
    # Print header with position numbers
    print("  ", end="")
    print("Q\\KV ", end="")
    for kv in positions:
        print(f"{kv:3d}", end="")
    print()
    
    # Print matrix rows
    for i, q in enumerate(positions):
        # Mark segment boundaries with special indicators
        marker = " "
        for seg_idx, (start, end) in enumerate(segment_positions):
            if q == start:
                marker = f"S{seg_idx+1}"
                break
            elif q == end - 1:
                marker = f"E{seg_idx+1}"
                break
        if q == original_length:
            marker = "G0"  # Generation start
        
        print(f" {marker:>2} {q:3d} ", end="")
        for j, kv in enumerate(positions):
            symbol = "■" if matrix[i][j] else "·"  # Use filled square for attend, dot for no-attend
            print(f" {symbol} ", end="")
        print()
    
    # Print legend
    print("\n  Legend:")
    print("    ■ = can attend, · = cannot attend")
    print("    S# = segment start, E# = segment end, G0 = generation start")
    
    # Print segment boundaries
    print(f"\n  Segment Boundaries:")
    for i, (start, end) in enumerate(segment_positions):
        print(f"    Segment {i+1}: positions {start:3d}-{end-1:3d} (length {end-start})")
    print(f"  Original length: {original_length}")
    print(f"  Generated tokens: {seq_len - original_length}")
    
    if seq_len > max_display:
        print(f"\n  ✅ Matrix shows sampled positions emphasizing structure - you can see the overall pattern!")


def create_segment_mask_func(segment_positions, original_length):
    """Create a mask function for segment isolation."""
    def mask_func(b, h, q_idx, kv_idx):
        # Causal constraint
        if q_idx < kv_idx:
            return False
        
        # Generated tokens can attend to everything
        if q_idx >= original_length:
            return True
        
        # Original tokens only attend within segment
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


def test_separator_formatting():
    """Demonstrate the improved separator formatting."""
    print("\n" + "="*70)
    print("SEPARATOR FORMATTING COMPARISON")
    print("="*70)
    
    prompts = [
        "Q: What is the capital of France?\nA:",
        "Q: Which city is the capital of France?\nA:",
        "Q: France's capital city is called?\nA:",
    ]
    
    print("\n--- OLD VERSION: No spacing ---")
    old_concat = " [SEP] ".join(prompts)
    print(old_concat)
    print("\n❌ PROBLEM: Hard to see where one prompt ends and another begins!")
    
    print("\n" + "-"*70)
    print("--- NEW VERSION: Clear spacing with newlines ---")
    new_concat = "\n\n[SEP]\n\n".join(prompts)
    print(new_concat)
    print("\n✅ IMPROVEMENT: Much clearer separation between prompts!")


def main():
    print("="*70)
    print("MASK MATRIX VISUALIZATION IMPROVEMENTS TEST")
    print("="*70)
    print("\nThis test demonstrates the improvements for handling large sequences")
    print("with hundreds of tokens, where the old version only showed 20x20.")
    
    # Test case: Large sequence with multiple segments
    # Simulate: 5 prompts of ~50 tokens each = ~250 tokens, plus 10 generated tokens
    segment_positions = [
        (0, 48),      # Prompt 1
        (48, 95),     # Prompt 2
        (95, 143),    # Prompt 3
        (143, 192),   # Prompt 4
        (192, 238),   # Prompt 5
    ]
    original_length = 238
    seq_len = 248  # 238 original + 10 generated
    
    print(f"\n📊 Test Case:")
    print(f"   Total sequence length: {seq_len} tokens")
    print(f"   Number of prompts: {len(segment_positions)}")
    print(f"   Original length: {original_length}")
    print(f"   Generated tokens: {seq_len - original_length}")
    
    mask_func = create_segment_mask_func(segment_positions, original_length)
    
    # Show old version
    visualize_mask_old(mask_func, seq_len, max_display=20)
    
    # Show new version
    visualize_mask_new(mask_func, seq_len, segment_positions, original_length, max_display=25)
    
    # Test separator formatting
    test_separator_formatting()
    
    print("\n" + "="*70)
    print("SUMMARY OF IMPROVEMENTS")
    print("="*70)
    print("\n1. ✅ Mask Matrix Visualization:")
    print("   - OLD: Only showed first 20x20, couldn't see structure for large sequences")
    print("   - NEW: Smart sampling shows segment boundaries and overall structure")
    print("   - NEW: Uses better symbols (■/·) and adds segment markers (S#/E#/G0)")
    print("\n2. ✅ Prompt Separator Formatting:")
    print("   - OLD: Used ' [SEP] ' with no newlines, hard to read")
    print("   - NEW: Uses '\\n\\n[SEP]\\n\\n' for clear visual separation")
    print("\n3. ✅ Output Display:")
    print("   - NEW: Shows each prompt segment separately with position ranges")
    print("   - NEW: Clearly marks generated output section")
    print("\n✅ All improvements implemented successfully!")


if __name__ == "__main__":
    main()
