#!/usr/bin/env python3
"""
Test script for MyriadLAMA-specific FlexAttention implementation.

This test verifies:
1. All paraphrases are treated equally (all manually generated)
2. Each paraphrase is isolated during encoding
3. Within each paraphrase, only causal attention is allowed
4. Generation phase works correctly (full attention to all paraphrases)
5. Proper tensor return types
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from myriadlama_flex_attention_generate import create_myriadlama_mask


def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_return_types():
    """Test that mask_mod returns Tensor types, not Python bools."""
    print_section("Test 1: Return Type Verification")
    
    # 5 paraphrases (all manually generated)
    segment_positions = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    original_length = 50
    
    mask_mod = create_myriadlama_mask(segment_positions, original_length)
    
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


def test_paraphrase_isolation():
    """Test that each paraphrase is isolated during encoding."""
    print_section("Test 2: Paraphrase Isolation During Encoding")
    
    # 5 paraphrases (all manually generated)
    segment_positions = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    original_length = 50
    
    mask_mod = create_myriadlama_mask(segment_positions, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nAll paraphrases are manually generated and isolated:")
    print("Each paraphrase can only attend to itself (within-paraphrase causal)")
    
    test_cases = [
        # Within same paraphrase - should be True (causal)
        (5, 3, True, "Para 1 (pos 5) to Para 1 (pos 3) - same paraphrase, causal"),
        (15, 12, True, "Para 2 (pos 15) to Para 2 (pos 12) - same paraphrase, causal"),
        (35, 33, True, "Para 4 (pos 35) to Para 4 (pos 33) - same paraphrase, causal"),
        
        # Within same paraphrase but future - should be False (violates causal)
        (3, 5, False, "Para 1 (pos 3) to Para 1 (pos 5) - same paraphrase, future"),
        
        # Different paraphrases - should be False (isolated)
        (15, 5, False, "Para 2 (pos 15) to Para 1 (pos 5) - different paraphrases"),
        (25, 5, False, "Para 3 (pos 25) to Para 1 (pos 5) - different paraphrases"),
        (25, 15, False, "Para 3 (pos 25) to Para 2 (pos 15) - different paraphrases"),
        (35, 5, False, "Para 4 (pos 35) to Para 1 (pos 5) - different paraphrases"),
        (45, 15, False, "Para 5 (pos 45) to Para 2 (pos 15) - different paraphrases"),
        (45, 35, False, "Para 5 (pos 45) to Para 4 (pos 35) - different paraphrases"),
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
    
    assert all_passed, "Some paraphrase isolation tests failed"
    print("\n✅ PASS: Paraphrase isolation working correctly")


def test_segment_isolation():
    """Test standard segment isolation behavior."""
    print_section("Test 3: Segment Isolation (All Paraphrases Equal)")
    
    segment_positions = [(0, 10), (10, 20), (20, 30)]
    original_length = 30
    
    mask_mod = create_myriadlama_mask(segment_positions, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nAll paraphrases isolated (no cross-attention):")
    
    test_cases = [
        # Same segment - should be True
        (5, 3, True, "Same paraphrase, causal"),
        
        # Different segments - should be False
        (15, 5, False, "Different paraphrases (isolated)"),
        (25, 15, False, "Different paraphrases (isolated)"),
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
    
        symbol = "✅" if passed else "❌"
        print(f"  {symbol} q={q:2d}, kv={kv:2d}: {result.item()!s:5s} (expected {expected!s:5s}) - {desc}")
        if not passed:
            all_passed = False
    
    assert all_passed, "Some segment isolation tests failed"
    print("\n✅ PASS: Segment isolation working correctly")


def test_generation_phase():
    """Test that generated tokens can attend to all paraphrases."""
    print_section("Test 4: Generation Phase (Full Attention)")
    
    segment_positions = [(0, 10), (10, 20), (20, 30), (30, 40)]
    original_length = 40
    
    mask_mod = create_myriadlama_mask(segment_positions, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nGenerated tokens should attend to all paraphrases:")
    
    test_cases = [
        # Generated token attending to all paraphrases
        (40, 5, True, "Gen token 0 to Para 1"),
        (40, 15, True, "Gen token 0 to Para 2"),
        (40, 25, True, "Gen token 0 to Para 3"),
        (40, 35, True, "Gen token 0 to Para 4"),
        
        # Generated tokens attending to each other
        (41, 40, True, "Gen token 1 to Gen token 0"),
        (42, 41, True, "Gen token 2 to Gen token 1"),
        
        # Still respects causal constraint
        (40, 41, False, "Cannot attend to future"),
        (41, 42, False, "Cannot attend to future"),
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


def visualize_myriadlama_mask():
    """Visualize the MyriadLAMA-specific mask pattern."""
    print_section("Test 5: Mask Visualization")
    
    # Simple setup: 4 paraphrases + 2 generated
    # Para 1: [0-2], Para 2: [3-5], Para 3: [6-8], Para 4: [9-11]
    # Generated: [12-13]
    segment_positions = [(0, 3), (3, 6), (6, 9), (9, 12)]
    original_length = 12
    seq_len = 14  # 12 original + 2 generated
    
    mask_mod = create_myriadlama_mask(segment_positions, original_length)
    
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
    print("\nMask Matrix Visualization:")
    print("  Para 1: [0-2], Para 2: [3-5], Para 3: [6-8], Para 4: [9-11] | Gen: [12-13]")
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
            marker = "P1"
        elif q < 6:
            marker = "P2"
        elif q < 9:
            marker = "P3"
        elif q < 12:
            marker = "P4"
        else:
            marker = f"G{q - 12}"
        
        print(f"  {marker:>4} {q:2d}", end="")
        for kv in range(seq_len):
            symbol = " ■" if matrix[q][kv] else " ·"
            print(symbol, end="")
        print()
    
    print("\nExpected pattern:")
    print("  ✅ Each paraphrase (P1, P2, P3, P4) is isolated")
    print("  ✅ Within each paraphrase, only causal attention")
    print("  ✅ Generated (G) can attend to all previous tokens")
    
    # Verify paraphrase isolation
    assert not matrix[4][1], "Para 2 should NOT attend to Para 1"
    assert not matrix[10][7], "Para 4 should NOT attend to Para 3"
    
    # Verify within-paraphrase causal
    assert matrix[1][0], "Para 1 pos 1 should attend to Para 1 pos 0 (causal)"
    assert not matrix[0][1], "Para 1 pos 0 should NOT attend to Para 1 pos 1 (future)"
    
    # Verify generated attention
    assert matrix[12][1], "Generated should attend to Para 1"
    assert matrix[12][7], "Generated should attend to Para 3"
    
    print("\n✅ PASS: Visualization matches expected MyriadLAMA pattern")


def main():
    """Run all tests."""
    print("="*70)
    print("  Testing MyriadLAMA-Specific FlexAttention Implementation")
    print("="*70)
    print("\nThis test suite validates:")
    print("  1. Proper tensor return types")
    print("  2. Paraphrase isolation (all paraphrases treated equally)")
    print("  3. Within-paraphrase causal attention")
    print("  4. Generation phase full attention")
    print("  5. Mask pattern visualization")
    
    try:
        test_return_types()
        test_paraphrase_isolation()
        test_segment_isolation()
        test_generation_phase()
        visualize_myriadlama_mask()
        
        print("\n" + "="*70)
        print("  ALL TESTS PASSED ✅")
        print("="*70)
        print("\nThe MyriadLAMA-specific mask implementation:")
        print("  ✅ Returns proper Tensor types (vmap compatible)")
        print("  ✅ All paraphrases treated equally (all manually generated)")
        print("  ✅ Each paraphrase isolated during encoding")
        print("  ✅ Within-paraphrase causal attention only")
        print("  ✅ Enables full attention during generation")
        print("\nReady for use with MyriadLAMA dataset!")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
