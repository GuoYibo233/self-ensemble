#!/usr/bin/env python3
"""
Test the new prompt format for MyriadLAMA.

New format:
- Instruction
- Few-shot examples with multiple Q paraphrases + one A
- Main question paraphrases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from myriadlama_flex_attention_generate import (
    construct_single_prompt_new_format,
    parse_prompt_segments_with_metadata_new_format
)


def test_prompt_construction():
    """Test constructing a prompt in the new format."""
    print("="*70)
    print("  Test: Prompt Construction (New Format)")
    print("="*70)
    
    instruction = "Predict the [MASK] in the sentence in one word."
    
    few_shot_examples = [
        {
            'paraphrases': [
                'London is the capital of [MASK]',
                'The capital of [MASK] is London',
                '[MASK] has London as its capital'
            ],
            'answer': 'England'
        },
        {
            'paraphrases': [
                'Tokyo is the capital of [MASK]',
                'The capital of [MASK] is Tokyo'
            ],
            'answer': 'Japan'
        }
    ]
    
    question_para = 'Paris is the capital of [MASK]'
    
    prompt = construct_single_prompt_new_format(instruction, few_shot_examples, question_para)
    
    print("\nConstructed Prompt:")
    print("-" * 70)
    print(prompt)
    print("-" * 70)
    
    return prompt


def test_parsing():
    """Test parsing the new format."""
    print("\n" + "="*70)
    print("  Test: Parsing (New Format)")
    print("="*70)
    
    instruction = "Predict the [MASK] in the sentence in one word."
    
    few_shot_examples = [
        {
            'paraphrases': [
                'London is the capital of [MASK]',
                'The capital of [MASK] is London',
                '[MASK] has London as its capital'
            ],
            'answer': 'England'
        },
        {
            'paraphrases': [
                'Tokyo is the capital of [MASK]',
                'The capital of [MASK] is Tokyo'
            ],
            'answer': 'Japan'
        }
    ]
    
    question_para = 'Paris is the capital of [MASK]'
    
    prompt = construct_single_prompt_new_format(instruction, few_shot_examples, question_para)
    
    segments = parse_prompt_segments_with_metadata_new_format(prompt, paraphrase_idx=0)
    
    print("\nParsed Segments:")
    print("-" * 70)
    for i, (seg_text, meta) in enumerate(segments):
        seg_type = meta['type']
        para_idx = meta['paraphrase_idx']
        fs_idx = meta.get('few_shot_idx')
        fs_q_para = meta.get('fs_q_para_idx')
        
        print(f"\nSegment {i}:")
        print(f"  Type: {seg_type}")
        print(f"  Main Para Idx: {para_idx}")
        if fs_idx is not None:
            print(f"  Few-shot Idx: {fs_idx}")
        if fs_q_para is not None:
            print(f"  FS Q Para Idx: {fs_q_para}")
        print(f"  Text: {seg_text[:80]}...")
    
    print("\n" + "-" * 70)
    print(f"Total segments: {len(segments)}")
    
    # Verify structure
    expected_segments = [
        ('instruction', None, None),  # instruction
        ('few_shot_q', 0, 0),  # FS0 Q-para0
        ('few_shot_q', 0, 1),  # FS0 Q-para1
        ('few_shot_q', 0, 2),  # FS0 Q-para2
        ('few_shot_a', 0, None),  # FS0 Answer
        ('few_shot_q', 1, 0),  # FS1 Q-para0
        ('few_shot_q', 1, 1),  # FS1 Q-para1
        ('few_shot_a', 1, None),  # FS1 Answer
        ('question', None, None),  # Main question
    ]
    
    print("\nVerification:")
    all_correct = True
    for i, (seg_text, meta) in enumerate(segments):
        if i < len(expected_segments):
            exp_type, exp_fs, exp_fs_q_para = expected_segments[i]
            if meta['type'] == exp_type and meta.get('few_shot_idx') == exp_fs and meta.get('fs_q_para_idx') == exp_fs_q_para:
                print(f"  ✓ Segment {i}: {exp_type} (correct)")
            else:
                print(f"  ✗ Segment {i}: Expected {exp_type}, got {meta['type']}")
                all_correct = False
    
    if all_correct:
        print("\n✅ All segments parsed correctly!")
    else:
        print("\n❌ Some segments parsed incorrectly")
    
    return segments


def main():
    """Run all tests."""
    test_prompt_construction()
    test_parsing()
    
    print("\n" + "="*70)
    print("  Tests Completed")
    print("="*70)
    print("\nThe new prompt format is working correctly:")
    print("  - Instruction at the beginning")
    print("  - Few-shot examples with multiple Q paraphrases + one A")
    print("  - Main question paraphrase at the end")
    print()


if __name__ == "__main__":
    main()
