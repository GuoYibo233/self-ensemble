#!/usr/bin/env python3
"""
Test script to verify the MyriadLAMA FlexAttention mask behavior.

This script generates one sample and visualizes the attention mask to ensure
it follows the requirements:

1. Paraphrases of the same few-shot example cannot attend to each other
2. Paraphrases from different few-shot examples CAN attend to each other  
3. The answer part of few-shot examples has normal (causal) mask
4. Question paraphrases are isolated from each other
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



def create_test_mask_structure():
    """
    Create a test structure to understand the masking requirements.
    
    Example structure with 2 few-shot examples (each with 2 paraphrases) and 2 question paraphrases:
    
    Segments:
    0: Instruction
    1: Few-shot 1, para 1 (question part)
    2: Few-shot 1, para 1 (answer part)
    3: Few-shot 1, para 2 (question part)
    4: Few-shot 1, para 2 (answer part)
    5: Few-shot 2, para 1 (question part)
    6: Few-shot 2, para 1 (answer part)
    7: Few-shot 2, para 2 (question part)
    8: Few-shot 2, para 2 (answer part)
    9: Question para 1
    10: Question para 2
    
    Mask requirements:
    - Seg 1 (FS1-P1-Q) CANNOT attend to Seg 3 (FS1-P2-Q) - same few-shot, different paraphrases
    - Seg 1 (FS1-P1-Q) CAN attend to Seg 5 (FS2-P1-Q) - different few-shot
    - Seg 2 (FS1-P1-A) has normal causal mask (can attend to Seg 1)
    - Seg 9 (Q-P1) CANNOT attend to Seg 10 (Q-P2) - different question paraphrases
    """
    
    # Define segment structure
    # Each segment is (segment_type, few_shot_id, paraphrase_id, is_answer)
    # segment_type: "instruction", "few_shot", "question"
    # few_shot_id: which few-shot example (0, 1, 2, ...)
    # paraphrase_id: which paraphrase (0, 1, 2, ...)
    # is_answer: True if this is the answer part of a Q-A pair
    
    segments = [
        ("instruction", None, None, False),  # 0: Instruction
        
        # Few-shot 1
        ("few_shot", 0, 0, False),  # 1: FS1-P1 question
        ("few_shot", 0, 0, True),   # 2: FS1-P1 answer
        ("few_shot", 0, 1, False),  # 3: FS1-P2 question
        ("few_shot", 0, 1, True),   # 4: FS1-P2 answer
        
        # Few-shot 2
        ("few_shot", 1, 0, False),  # 5: FS2-P1 question
        ("few_shot", 1, 0, True),   # 6: FS2-P1 answer
        ("few_shot", 1, 1, False),  # 7: FS2-P2 question
        ("few_shot", 1, 1, True),   # 8: FS2-P2 answer
        
        # Question paraphrases
        ("question", None, 0, False),  # 9: Question para 1
        ("question", None, 1, False),  # 10: Question para 2
    ]
    
    num_segments = len(segments)
    
    print("="*70)
    print("  Test Mask Structure")
    print("="*70)
    print(f"\nTotal segments: {num_segments}\n")
    
    for i, seg in enumerate(segments):
        seg_type, fs_id, para_id, is_ans = seg
        if seg_type == "instruction":
            print(f"  Seg {i:2d}: Instruction")
        elif seg_type == "few_shot":
            part = "Answer" if is_ans else "Question"
            print(f"  Seg {i:2d}: Few-shot {fs_id}, Paraphrase {para_id}, {part}")
        elif seg_type == "question":
            print(f"  Seg {i:2d}: Question Paraphrase {para_id}")
    
    return segments


def create_expected_mask(segments):
    """
    Create the expected attention mask based on requirements.
    
    Rules:
    1. Causal constraint always applies (cannot attend to future)
    2. Paraphrases of same few-shot CANNOT attend to each other
    3. Different few-shot examples CAN attend to each other
    4. Answer parts have normal causal mask
    5. Question paraphrases are isolated from each other
    """
    num_segments = len(segments)
    mask = [[False for _ in range(num_segments)] for _ in range(num_segments)]
    
    for q_idx in range(num_segments):
        for kv_idx in range(num_segments):
            q_type, q_fs_id, q_para_id, q_is_ans = segments[q_idx]
            kv_type, kv_fs_id, kv_para_id, kv_is_ans = segments[kv_idx]
            
            # Rule 1: Causal constraint
            if q_idx < kv_idx:
                continue  # Cannot attend to future
            
            # Rule: Instruction can attend to itself
            if q_type == "instruction" and kv_type == "instruction":
                mask[q_idx][kv_idx] = True
                continue
            
            # Rule: Few-shot can attend to instruction
            if q_type == "few_shot" and kv_type == "instruction":
                mask[q_idx][kv_idx] = True
                continue
            
            # Rule: Question can attend to instruction
            if q_type == "question" and kv_type == "instruction":
                mask[q_idx][kv_idx] = True
                continue
            
            # Rule 4: Answer parts have normal causal mask
            if q_is_ans and kv_type == "few_shot":
                # Answer can attend to question part of same few-shot paraphrase
                if q_fs_id == kv_fs_id and q_para_id == kv_para_id:
                    mask[q_idx][kv_idx] = True
                # Answer can also attend to other few-shot segments (based on rules below)
                elif q_fs_id == kv_fs_id:
                    # Same few-shot, different paraphrase - CANNOT
                    continue
                else:
                    # Different few-shot - CAN
                    mask[q_idx][kv_idx] = True
            
            # Rule 2 & 3: Few-shot question parts
            elif q_type == "few_shot" and not q_is_ans and kv_type == "few_shot" and not kv_is_ans:
                if q_fs_id == kv_fs_id:
                    # Same few-shot example
                    if q_para_id == kv_para_id:
                        # Same paraphrase - CAN (causal)
                        mask[q_idx][kv_idx] = True
                    else:
                        # Different paraphrase - CANNOT
                        mask[q_idx][kv_idx] = False
                else:
                    # Different few-shot - CAN
                    mask[q_idx][kv_idx] = True
            
            # Rule 5: Question paraphrases isolated
            elif q_type == "question" and kv_type == "question":
                if q_para_id == kv_para_id:
                    # Same paraphrase - CAN (causal)
                    mask[q_idx][kv_idx] = True
                else:
                    # Different paraphrase - CANNOT
                    mask[q_idx][kv_idx] = False
            
            # Question can attend to few-shot
            elif q_type == "question" and kv_type == "few_shot":
                mask[q_idx][kv_idx] = True
    
    return mask


def visualize_mask(mask, segments):
    """Visualize the attention mask."""
    num_segments = len(segments)
    
    print("\n" + "="*70)
    print("  Expected Attention Mask")
    print("="*70)
    print("\n  ■ = can attend, · = cannot attend\n")
    
    # Header
    print("     Q\\KV", end="")
    for kv in range(num_segments):
        print(f" {kv:2d}", end="")
    print()
    
    # Rows
    for q in range(num_segments):
        seg_type, fs_id, para_id, is_ans = segments[q]
        
        if seg_type == "instruction":
            marker = "Inst"
        elif seg_type == "few_shot":
            part = "A" if is_ans else "Q"
            marker = f"F{fs_id}P{para_id}{part}"
        elif seg_type == "question":
            marker = f"Q-P{para_id}"
        
        print(f"  {marker:>6} {q:2d}", end="")
        for kv in range(num_segments):
            symbol = " ■" if mask[q][kv] else " ·"
            print(symbol, end="")
        print()
    
    print("\n" + "="*70)
    print("  Verification")
    print("="*70)
    
    # Verify key requirements
    print("\nKey requirements:")
    
    # Seg 1 (FS1-P1-Q) CANNOT attend to Seg 3 (FS1-P2-Q)
    if not mask[3][1]:
        print("  ✓ FS1-P2-Q cannot attend to FS1-P1-Q (same few-shot, different paraphrases)")
    else:
        print("  ✗ FS1-P2-Q should NOT attend to FS1-P1-Q")
    
    # Seg 1 (FS1-P1-Q) CAN attend to Seg 5 (FS2-P1-Q)
    if mask[5][1]:
        print("  ✓ FS2-P1-Q can attend to FS1-P1-Q (different few-shot)")
    else:
        print("  ✗ FS2-P1-Q should attend to FS1-P1-Q")
    
    # Seg 2 (FS1-P1-A) can attend to Seg 1 (FS1-P1-Q)
    if mask[2][1]:
        print("  ✓ FS1-P1-A can attend to FS1-P1-Q (answer to question, causal)")
    else:
        print("  ✗ FS1-P1-A should attend to FS1-P1-Q")
    
    # Seg 9 (Q-P1) CANNOT attend to Seg 10 (Q-P2)
    # Since 9 < 10, seg 9 cannot attend to seg 10 anyway (causal)
    # Check seg 10 cannot attend to seg 9
    if not mask[10][9]:
        print("  ✓ Q-P2 cannot attend to Q-P1 (different question paraphrases)")
    else:
        print("  ✗ Q-P2 should NOT attend to Q-P1")


def main():
    """Run the test."""
    segments = create_test_mask_structure()
    mask = create_expected_mask(segments)
    visualize_mask(mask, segments)
    
    print("\n" + "="*70)
    print("  Test completed")
    print("="*70)
    print("\nThis test file demonstrates the expected mask structure.")
    print("The actual implementation needs to:")
    print("  1. Track which segments belong to which few-shot example")
    print("  2. Track which segments are paraphrases of the same few-shot")
    print("  3. Track which segments are answers vs questions")
    print("  4. Apply the masking rules accordingly")
    print()


if __name__ == "__main__":
    main()
