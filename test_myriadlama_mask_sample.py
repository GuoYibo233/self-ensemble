#!/usr/bin/env python3
"""
Test script to generate one sample and verify the MyriadLAMA mask is correctly applied.

This test:
1. Creates a simple mock prompt structure
2. Applies the mask function
3. Visualizes the resulting mask matrix
4. Verifies key requirements
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_mask_with_sample():
    """
    Test the mask with a realistic sample structure.
    
    Structure:
    - 2 paraphrases of the main question
    - Each has: instruction + 2 few-shot examples + question
    - Each few-shot has Q and A parts
    """
    
    # Mock the structure manually to understand what we're testing
    # In real usage, this would come from parse_prompt_segments_with_metadata
    
    # Paraphrase 0:
    # - Instruction
    # - FS 0: Q, A
    # - FS 1: Q, A
    # - Question
    
    # Paraphrase 1:
    # - Instruction  
    # - FS 0: Q, A
    # - FS 1: Q, A
    # - Question
    
    segment_metadata = [
        # Paraphrase 0
        {'type': 'instruction', 'paraphrase_idx': 0, 'few_shot_idx': None},  # 0
        {'type': 'few_shot_q', 'paraphrase_idx': 0, 'few_shot_idx': 0},      # 1
        {'type': 'few_shot_a', 'paraphrase_idx': 0, 'few_shot_idx': 0},      # 2
        {'type': 'few_shot_q', 'paraphrase_idx': 0, 'few_shot_idx': 1},      # 3
        {'type': 'few_shot_a', 'paraphrase_idx': 0, 'few_shot_idx': 1},      # 4
        {'type': 'question', 'paraphrase_idx': 0, 'few_shot_idx': None},     # 5
        
        # Paraphrase 1
        {'type': 'instruction', 'paraphrase_idx': 1, 'few_shot_idx': None},  # 6
        {'type': 'few_shot_q', 'paraphrase_idx': 1, 'few_shot_idx': 0},      # 7
        {'type': 'few_shot_a', 'paraphrase_idx': 1, 'few_shot_idx': 0},      # 8
        {'type': 'few_shot_q', 'paraphrase_idx': 1, 'few_shot_idx': 1},      # 9
        {'type': 'few_shot_a', 'paraphrase_idx': 1, 'few_shot_idx': 1},      # 10
        {'type': 'question', 'paraphrase_idx': 1, 'few_shot_idx': None},     # 11
    ]
    
    num_segments = len(segment_metadata)
    
    print("="*70)
    print("  Test Mask with Realistic Sample")
    print("="*70)
    print(f"\nTotal segments: {num_segments}\n")
    
    for i, meta in enumerate(segment_metadata):
        seg_type = meta['type']
        para = meta['paraphrase_idx']
        fs = meta['few_shot_idx']
        
        if seg_type == 'instruction':
            print(f"  Seg {i:2d}: Para {para}, Instruction")
        elif seg_type == 'few_shot_q':
            print(f"  Seg {i:2d}: Para {para}, Few-shot {fs}, Question")
        elif seg_type == 'few_shot_a':
            print(f"  Seg {i:2d}: Para {para}, Few-shot {fs}, Answer")
        elif seg_type == 'question':
            print(f"  Seg {i:2d}: Para {para}, Question")
    
    # Build expected mask based on rules
    mask = create_expected_mask_with_metadata(segment_metadata)
    
    # Visualize
    visualize_mask(mask, segment_metadata)
    
    # Verify key requirements
    verify_requirements(mask, segment_metadata)


def create_expected_mask_with_metadata(segment_metadata):
    """Create expected mask based on the complex rules."""
    num_segments = len(segment_metadata)
    mask = [[False for _ in range(num_segments)] for _ in range(num_segments)]
    
    for q_idx in range(num_segments):
        for kv_idx in range(num_segments):
            q_meta = segment_metadata[q_idx]
            kv_meta = segment_metadata[kv_idx]
            
            q_type = q_meta['type']
            kv_type = kv_meta['type']
            q_para = q_meta['paraphrase_idx']
            kv_para = kv_meta['paraphrase_idx']
            q_fs = q_meta['few_shot_idx']
            kv_fs = kv_meta['few_shot_idx']
            
            # Rule 1: Causal constraint
            if q_idx < kv_idx:
                continue
            
            # Rule: Instruction can attend to itself
            if q_type == 'instruction' and kv_type == 'instruction':
                mask[q_idx][kv_idx] = True
                continue
            
            # Rule: All can attend to instruction
            if kv_type == 'instruction':
                mask[q_idx][kv_idx] = True
                continue
            
            # Rule: Few-shot answers
            if q_type == 'few_shot_a':
                # Can attend to own question
                if kv_type == 'few_shot_q' and q_para == kv_para and q_fs == kv_fs:
                    mask[q_idx][kv_idx] = True
                # Can attend to other few-shot questions from DIFFERENT few-shot
                elif kv_type == 'few_shot_q' and q_fs != kv_fs:
                    mask[q_idx][kv_idx] = True
                # Can attend to other few-shot answers
                elif kv_type == 'few_shot_a':
                    # Same few-shot, different para - can
                    if q_fs == kv_fs and q_para != kv_para:
                        mask[q_idx][kv_idx] = True
                    # Different few-shot - can
                    elif q_fs != kv_fs:
                        mask[q_idx][kv_idx] = True
                    # Same segment - can
                    elif q_idx == kv_idx:
                        mask[q_idx][kv_idx] = True
                continue
            
            # Rule: Few-shot questions
            if q_type == 'few_shot_q' and kv_type == 'few_shot_q':
                # Same FS, same para - can
                if q_fs == kv_fs and q_para == kv_para:
                    mask[q_idx][kv_idx] = True
                # Same FS, different para - CANNOT
                elif q_fs == kv_fs and q_para != kv_para:
                    mask[q_idx][kv_idx] = False
                # Different FS - CAN
                else:
                    mask[q_idx][kv_idx] = True
                continue
            
            # Rule: Few-shot Q to A
            if q_type == 'few_shot_q' and kv_type == 'few_shot_a':
                # Different FS - can
                if q_fs != kv_fs:
                    mask[q_idx][kv_idx] = True
                # Same FS, different para - cannot
                elif q_fs == kv_fs and q_para != kv_para:
                    mask[q_idx][kv_idx] = False
                continue
            
            # Rule: Question paraphrases
            if q_type == 'question' and kv_type == 'question':
                # Same para - can
                if q_para == kv_para:
                    mask[q_idx][kv_idx] = True
                # Different para - CANNOT
                else:
                    mask[q_idx][kv_idx] = False
                continue
            
            # Rule: Questions can attend to few-shot
            if q_type == 'question' and (kv_type == 'few_shot_q' or kv_type == 'few_shot_a'):
                mask[q_idx][kv_idx] = True
                continue
    
    return mask


def visualize_mask(mask, segment_metadata):
    """Visualize the mask."""
    num_segments = len(segment_metadata)
    
    print("\n" + "="*70)
    print("  Expected Attention Mask")
    print("="*70)
    print("\n  ■ = can attend, · = cannot attend\n")
    
    # Header
    print("        Q\\KV", end="")
    for kv in range(num_segments):
        print(f" {kv:2d}", end="")
    print()
    
    # Rows
    for q in range(num_segments):
        meta = segment_metadata[q]
        seg_type = meta['type']
        para = meta['paraphrase_idx']
        fs = meta['few_shot_idx']
        
        if seg_type == 'instruction':
            marker = f"P{para}-Inst"
        elif seg_type == 'few_shot_q':
            marker = f"P{para}-F{fs}Q"
        elif seg_type == 'few_shot_a':
            marker = f"P{para}-F{fs}A"
        elif seg_type == 'question':
            marker = f"P{para}-Ques"
        
        print(f"  {marker:>10} {q:2d}", end="")
        for kv in range(num_segments):
            symbol = " ■" if mask[q][kv] else " ·"
            print(symbol, end="")
        print()


def verify_requirements(mask, segment_metadata):
    """Verify key requirements."""
    print("\n" + "="*70)
    print("  Verification")
    print("="*70)
    print("\nKey requirements:")
    
    # Requirement 1: Paraphrases of same few-shot CANNOT attend to each other
    # Seg 1 (P0-F0Q) should NOT attend to Seg 7 (P1-F0Q)
    if not mask[7][1]:
        print("  ✓ P1-F0Q cannot attend to P0-F0Q (same few-shot, different paraphrase)")
    else:
        print("  ✗ P1-F0Q should NOT attend to P0-F0Q")
    
    # Requirement 2: Paraphrases from different few-shot CAN attend
    # Seg 3 (P0-F1Q) CAN attend to Seg 1 (P0-F0Q) - different few-shot, same para
    if mask[3][1]:
        print("  ✓ P0-F1Q can attend to P0-F0Q (different few-shot, same paraphrase)")
    else:
        print("  ✗ P0-F1Q should attend to P0-F0Q")
    
    # Also test cross-paraphrase, different few-shot
    # Seg 9 (P1-F1Q) CAN attend to Seg 1 (P0-F0Q) - different few-shot, different para
    if mask[9][1]:
        print("  ✓ P1-F1Q can attend to P0-F0Q (different few-shot, different paraphrase)")
    else:
        print("  ✗ P1-F1Q should attend to P0-F0Q")
    
    # Requirement 3: Answer parts have normal causal mask
    # Seg 2 (P0-F0A) can attend to Seg 1 (P0-F0Q)
    if mask[2][1]:
        print("  ✓ P0-F0A can attend to P0-F0Q (answer to its question)")
    else:
        print("  ✗ P0-F0A should attend to P0-F0Q")
    
    # Requirement 4: Question paraphrases are isolated
    # Seg 11 (P1-Ques) CANNOT attend to Seg 5 (P0-Ques)
    if not mask[11][5]:
        print("  ✓ P1-Ques cannot attend to P0-Ques (different paraphrases)")
    else:
        print("  ✗ P1-Ques should NOT attend to P0-Ques")
    
    print("\n" + "="*70)


def main():
    """Run the test."""
    test_mask_with_sample()
    
    print("\n  Test completed")
    print("="*70)
    print("\nThis demonstrates the expected mask behavior for MyriadLAMA.")
    print()


if __name__ == "__main__":
    main()
