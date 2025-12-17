#!/usr/bin/env python3
"""
Test script for MyriadLAMA-specific FlexAttention implementation.

This test verifies (SIMPLIFIED VERSION):
1. Return types are correct (Tensor[bool])
2. Main question paraphrases are mutually invisible
3. All other parts use normal causal attention
4. Generation phase works correctly (full attention)
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from myriadlama_flex_attention_generate import create_myriadlama_mask


def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_return_types():
    """Test that mask_mod returns Tensor types, not Python bools."""
    print_section("Test 1: Return Type Verification")
    
    # Simple setup: instruction + 3 main question paraphrases
    segment_positions = [(0, 10), (10, 20), (20, 30), (30, 40)]
    segment_metadata = [
        {'type': 'instruction', 'paraphrase_idx': None, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 0, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 1, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 2, 'few_shot_idx': None},
    ]
    original_length = 40
    
    mask_mod = create_myriadlama_mask(segment_positions, segment_metadata, original_length)
    
    # Test with tensor indices
    q_idx = torch.tensor(5)
    kv_idx = torch.tensor(3)
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    result = mask_mod(b, h, q_idx, kv_idx)
    
    print(f"Query index: {q_idx.item()}")
    print(f"Key index: {kv_idx.item()}")
    print(f"Result type: {type(result)}")
    print(f"Result is Tensor: {isinstance(result, torch.Tensor)}")
    print(f"Result dtype: {result.dtype}")
    print(f"Result value: {result.item()}")
    
    assert isinstance(result, torch.Tensor), "mask_mod must return a Tensor"
    assert result.dtype == torch.bool, "mask_mod must return a bool Tensor"
    print("\n✅ PASS: mask_mod returns Tensor[bool]")


def test_question_paraphrase_isolation():
    """Test that main question paraphrases are mutually invisible (SIMPLIFIED)."""
    print_section("Test 2: Main Question Paraphrase Isolation (SIMPLIFIED)")
    
    # Instruction + 3 main question paraphrases
    segment_positions = [(0, 10), (10, 20), (20, 30), (30, 40)]
    segment_metadata = [
        {'type': 'instruction', 'paraphrase_idx': None, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 0, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 1, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 2, 'few_shot_idx': None},
    ]
    original_length = 40
    
    mask_mod = create_myriadlama_mask(segment_positions, segment_metadata, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nSIMPLIFIED logic: Only main question paraphrases are isolated")
    print("All other parts (instruction, few-shot) use normal causal attention")
    
    test_cases = [
        # Instruction segment - normal causal
        (5, 3, True, "Instruction: causal within segment"),
        
        # Question paraphrases - same paraphrase allowed (causal)
        (15, 12, True, "Q para 0: causal within same paraphrase"),
        (25, 22, True, "Q para 1: causal within same paraphrase"),
        
        # Question paraphrases - different paraphrases blocked
        (25, 15, False, "Q para 1 to Q para 0 - BLOCKED (isolated)"),
        (35, 15, False, "Q para 2 to Q para 0 - BLOCKED (isolated)"),
        (35, 25, False, "Q para 2 to Q para 1 - BLOCKED (isolated)"),
        
        # Questions can attend to instruction
        (15, 5, True, "Q para 0 to instruction - allowed"),
        (25, 5, True, "Q para 1 to instruction - allowed"),
        
        # Causal constraint still holds
        (3, 5, False, "Cannot attend to future"),
        (12, 15, False, "Cannot attend to future"),
    ]
    
    all_passed = True
    for q, kv, expected, desc in test_cases:
        q_idx = torch.tensor(q)
        kv_idx = torch.tensor(kv)
        result = mask_mod(b, h, q_idx, kv_idx)
        passed = result.item() == expected
        symbol = "✅" if passed else "❌"
        print(f"  {symbol} q={q:2d}, kv={kv:2d}: {result.item()!s:5s} (expected {expected!s:5s}) - {desc}")
        if not passed:
            all_passed = False
    
    assert all_passed, "Some question paraphrase isolation tests failed"
    print("\n✅ PASS: Main question paraphrase isolation working correctly")


def test_few_shot_normal_causal():
    """Test that few-shot examples use normal causal attention (SIMPLIFIED)."""
    print_section("Test 3: Few-Shot Normal Causal Attention (SIMPLIFIED)")
    
    # Instruction + few-shot + questions
    segment_positions = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    segment_metadata = [
        {'type': 'instruction', 'paraphrase_idx': None, 'few_shot_idx': None},
        {'type': 'few_shot_q', 'paraphrase_idx': None, 'few_shot_idx': 0, 'fs_q_para_idx': 0},
        {'type': 'few_shot_a', 'paraphrase_idx': None, 'few_shot_idx': 0, 'fs_q_para_idx': None},
        {'type': 'question', 'paraphrase_idx': 0, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 1, 'few_shot_idx': None},
    ]
    original_length = 50
    
    mask_mod = create_myriadlama_mask(segment_positions, segment_metadata, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nFew-shot examples use NORMAL causal attention (simplified)")
    
    test_cases = [
        # Few-shot Q and A can attend within themselves (causal)
        (15, 12, True, "FS Q: causal within segment"),
        (25, 22, True, "FS A: causal within segment"),
        
        # Few-shot A can attend to few-shot Q (causal, same few-shot)
        (25, 15, True, "FS A to FS Q - allowed (normal causal)"),
        
        # Few-shot can attend to instruction
        (15, 5, True, "FS Q to instruction - allowed"),
        
        # Questions can attend to few-shot (normal causal)
        (35, 15, True, "Q para 0 to FS Q - allowed (normal causal)"),
        (35, 25, True, "Q para 0 to FS A - allowed (normal causal)"),
        
        # Questions are still isolated from each other
        (45, 35, False, "Q para 1 to Q para 0 - BLOCKED (isolated)"),
        
        # Causal constraint
        (15, 25, False, "Cannot attend to future"),
    ]
    
    all_passed = True
    for q, kv, expected, desc in test_cases:
        q_idx = torch.tensor(q)
        kv_idx = torch.tensor(kv)
        result = mask_mod(b, h, q_idx, kv_idx)
        passed = result.item() == expected
        symbol = "✅" if passed else "❌"
        print(f"  {symbol} q={q:2d}, kv={kv:2d}: {result.item()!s:5s} (expected {expected!s:5s}) - {desc}")
        if not passed:
            all_passed = False
    
    assert all_passed, "Some few-shot normal causal tests failed"
    print("\n✅ PASS: Few-shot normal causal attention working correctly")


def test_generation_phase():
    """Test that generated tokens can attend to all segments."""
    print_section("Test 4: Generation Phase (Full Attention)")
    
    segment_positions = [(0, 10), (10, 20), (20, 30)]
    segment_metadata = [
        {'type': 'instruction', 'paraphrase_idx': None, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 0, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 1, 'few_shot_idx': None},
    ]
    original_length = 30
    
    mask_mod = create_myriadlama_mask(segment_positions, segment_metadata, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nGenerated tokens should attend to ALL previous segments:")
    
    test_cases = [
        # Generated tokens attending to all segments
        (30, 5, True, "Gen token 0 to instruction"),
        (30, 15, True, "Gen token 0 to Q para 0"),
        (30, 25, True, "Gen token 0 to Q para 1"),
        
        # Generated tokens attending to each other
        (31, 30, True, "Gen token 1 to Gen token 0"),
        
        # Still respects causal constraint
        (30, 31, False, "Cannot attend to future"),
    ]
    
    all_passed = True
    for q, kv, expected, desc in test_cases:
        q_idx = torch.tensor(q)
        kv_idx = torch.tensor(kv)
        result = mask_mod(b, h, q_idx, kv_idx)
        passed = result.item() == expected
        symbol = "✅" if passed else "❌"
        print(f"  {symbol} q={q:2d}, kv={kv:2d}: {result.item()!s:5s} (expected {expected!s:5s}) - {desc}")
        if not passed:
            all_passed = False
    
    assert all_passed, "Some generation phase tests failed"
    print("\n✅ PASS: Generation phase full attention working")


def visualize_simplified_mask():
    """Visualize the SIMPLIFIED MyriadLAMA mask pattern."""
    print_section("Test 5: Simplified Mask Visualization")
    
    # Setup: Inst [0-2], Q0 [3-5], Q1 [6-8], Generated [9-10]
    segment_positions = [(0, 3), (3, 6), (6, 9)]
    segment_metadata = [
        {'type': 'instruction', 'paraphrase_idx': None, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 0, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 1, 'few_shot_idx': None},
    ]
    original_length = 9
    seq_len = 11  # 9 original + 2 generated
    
    mask_mod = create_myriadlama_mask(segment_positions, segment_metadata, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    # Build full mask matrix
    matrix = []
    for q in range(seq_len):
        row = []
        for kv in range(seq_len):
            q_idx = torch.tensor(q)
            kv_idx = torch.tensor(kv)
            result = mask_mod(b, h, q_idx, kv_idx)
            row.append(result.item())
        matrix.append(row)
    
    # Print matrix
    print("\nSIMPLIFIED Mask Matrix Visualization:")
    print("  Inst: [0-2], Q0: [3-5], Q1: [6-8] | Gen: [9-10]")
    print("  ■ = can attend, · = cannot attend\n")
    
    # Header
    print("      Q\\KV", end="")
    for kv in range(seq_len):
        print(f" {kv:2d}", end="")
    print()
    
    # Rows
    for q in range(seq_len):
        # Mark segments
        if q < 3:
            marker = "IN"
        elif q < 6:
            marker = "Q0"
        elif q < 9:
            marker = "Q1"
        else:
            marker = f"G{q - 9}"
        
        print(f"  {marker:>4} {q:2d}", end="")
        for kv in range(seq_len):
            symbol = " ■" if matrix[q][kv] else " ·"
            print(symbol, end="")
        print()
    
    print("\nSIMPLIFIED Expected pattern:")
    print("  ✅ Main question paraphrases (Q0, Q1) are isolated from each other")
    print("  ✅ Instruction and all other parts use normal causal attention")
    print("  ✅ Generated (G) can attend to all previous tokens")
    
    # Verify question isolation
    assert not matrix[7][4], "Q1 should NOT attend to Q0 (isolated)"
    
    # Verify within-question causal
    assert matrix[4][3], "Q0 pos 4 should attend to Q0 pos 3 (causal)"
    
    # Verify instruction is normal causal
    assert matrix[1][0], "Instruction pos 1 should attend to pos 0 (causal)"
    
    # Verify questions can attend to instruction
    assert matrix[4][1], "Q0 should attend to instruction (normal)"
    
    # Verify generated attention
    assert matrix[9][1], "Generated should attend to instruction"
    assert matrix[9][4], "Generated should attend to Q0"
    assert matrix[9][7], "Generated should attend to Q1"
    
    print("\n✅ PASS: Visualization matches SIMPLIFIED MyriadLAMA pattern")


def main():
    """Run all tests."""
    print("="*70)
    print("  Testing SIMPLIFIED MyriadLAMA FlexAttention Implementation")
    print("="*70)
    print("\nThis test suite validates:")
    print("  1. Proper tensor return types")
    print("  2. Main question paraphrases are mutually invisible")
    print("  3. All other parts use normal causal attention")
    print("  4. Generation phase full attention")
    print("  5. Mask pattern visualization")
    
    try:
        test_return_types()
        test_question_paraphrase_isolation()
        test_few_shot_normal_causal()
        test_generation_phase()
        visualize_simplified_mask()
        
        print("\n" + "="*70)
        print("  ALL TESTS PASSED ✅")
        print("="*70)
        print("\nThe SIMPLIFIED MyriadLAMA mask implementation:")
        print("  ✅ Returns proper Tensor types (vmap compatible)")
        print("  ✅ Main question paraphrases are isolated (mutually invisible)")
        print("  ✅ Instruction and few-shot use normal causal attention")
        print("  ✅ Enables full attention during generation")
        print("\nReady for use with MyriadLAMA dataset!")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
