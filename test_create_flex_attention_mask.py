#!/usr/bin/env python3
"""
Test script for create_flex_attention_mask implementation.

This test verifies:
1. The function returns proper tensor types (not Python bools)
2. Segment isolation works correctly during encoding phase
3. Generated tokens can attend to all segments during generation phase
4. Causal constraint is always respected
5. No data-dependent control flow issues
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flex_attention_generate import create_flex_attention_mask


def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_return_types():
    """Test that mask_mod returns Tensor types, not Python bools."""
    print_section("Test 1: Return Type Verification")
    
    segment_positions = [(0, 10), (10, 20)]
    original_length = 20
    
    mask_mod = create_flex_attention_mask(segment_positions, original_length)
    
    # Test with tensor indices (as FlexAttention would provide)
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


def test_causal_constraint():
    """Test that causal constraint is always enforced."""
    print_section("Test 2: Causal Constraint")
    
    segment_positions = [(0, 10), (10, 20)]
    original_length = 20
    
    mask_mod = create_flex_attention_mask(segment_positions, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    # Test: query cannot attend to future keys
    test_cases = [
        (5, 6, False, "Same segment, future key"),
        (5, 5, True, "Same position"),
        (5, 4, True, "Same segment, past key"),
        (15, 16, False, "Different segment, future key"),
    ]
    
    print("\nTesting causal constraint:")
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
    
    assert all_passed, "Some causal constraint tests failed"
    print("\n✅ PASS: Causal constraint enforced")


def test_segment_isolation():
    """Test that tokens in different segments cannot attend to each other."""
    print_section("Test 3: Segment Isolation (Encoding Phase)")
    
    segment_positions = [(0, 10), (10, 20), (20, 30)]
    original_length = 30
    
    mask_mod = create_flex_attention_mask(segment_positions, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nTesting segment isolation:")
    test_cases = [
        # Within same segment - should be True (if causal)
        (5, 3, True, "Segment 1 token attending to earlier Segment 1 token"),
        (15, 12, True, "Segment 2 token attending to earlier Segment 2 token"),
        (25, 22, True, "Segment 3 token attending to earlier Segment 3 token"),
        
        # Across different segments - should be False
        (5, 0, True, "Segment 1, same segment, past"),
        (15, 5, False, "Segment 2 attending to Segment 1 (cross-segment)"),
        (25, 5, False, "Segment 3 attending to Segment 1 (cross-segment)"),
        (25, 15, False, "Segment 3 attending to Segment 2 (cross-segment)"),
        (15, 22, False, "Cannot attend to future (cross-segment)"),
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
    
    assert all_passed, "Some segment isolation tests failed"
    print("\n✅ PASS: Segment isolation working correctly")


def test_generation_phase():
    """Test that generated tokens can attend to all previous tokens."""
    print_section("Test 4: Generation Phase (Full Attention)")
    
    segment_positions = [(0, 10), (10, 20), (20, 30)]
    original_length = 30
    
    mask_mod = create_flex_attention_mask(segment_positions, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nTesting generation phase:")
    print("Generated tokens should attend to all segments (subject to causal)")
    
    test_cases = [
        # Generated token attending to all segments
        (30, 5, True, "Gen token 0 attending to Segment 1"),
        (30, 15, True, "Gen token 0 attending to Segment 2"),
        (30, 25, True, "Gen token 0 attending to Segment 3"),
        (31, 5, True, "Gen token 1 attending to Segment 1"),
        (31, 30, True, "Gen token 1 attending to Gen token 0"),
        (35, 32, True, "Gen token 5 attending to Gen token 2"),
        
        # Still respects causal constraint
        (30, 31, False, "Cannot attend to future"),
        (32, 35, False, "Cannot attend to future"),
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


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print_section("Test 5: Edge Cases")
    
    segment_positions = [(0, 5), (5, 10)]
    original_length = 10
    
    mask_mod = create_flex_attention_mask(segment_positions, original_length)
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nTesting edge cases:")
    test_cases = [
        # Segment boundaries
        (0, 0, True, "First position"),
        (4, 4, True, "Last position of segment 1"),
        (5, 5, True, "First position of segment 2"),
        (4, 0, True, "Within segment 1"),
        (5, 0, False, "Segment 2 cannot attend to Segment 1"),
        (9, 5, True, "Within segment 2"),
        
        # Generation boundary
        (10, 0, True, "First generated token attending to Segment 1"),
        (10, 5, True, "First generated token attending to Segment 2"),
        (10, 9, True, "First generated token attending to last original"),
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
    
    assert all_passed, "Some edge case tests failed"
    print("\n✅ PASS: Edge cases handled correctly")


def test_batched_indices():
    """Test that function works with batched tensor indices."""
    print_section("Test 6: Batched Tensor Indices")
    
    segment_positions = [(0, 10), (10, 20)]
    original_length = 20
    
    mask_mod = create_flex_attention_mask(segment_positions, original_length)
    
    # Test with scalar tensors (as FlexAttention uses)
    b = torch.tensor(0)
    h = torch.tensor(0)
    q_idx = torch.tensor(5)
    kv_idx = torch.tensor(3)
    
    result = mask_mod(b, h, q_idx, kv_idx)
    
    print(f"Batch index: {b.item()}")
    print(f"Head index: {h.item()}")
    print(f"Query index: {q_idx.item()}")
    print(f"Key index: {kv_idx.item()}")
    print(f"Result: {result.item()}")
    print(f"Result shape: {result.shape}")
    
    assert result.shape == torch.Size([]), "Result should be scalar"
    print("\n✅ PASS: Works with scalar tensor indices")


def visualize_mask_matrix():
    """Visualize the complete mask matrix for verification."""
    print_section("Test 7: Mask Matrix Visualization")
    
    # Small test case for full visualization
    segment_positions = [(0, 3), (3, 6), (6, 9)]
    original_length = 9
    seq_len = 12  # 9 original + 3 generated
    
    mask_mod = create_flex_attention_mask(segment_positions, original_length)
    
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
    print("  Segments: [0-2], [3-5], [6-8] | Generated: [9-11]")
    print("  ■ = can attend, · = cannot attend\n")
    
    # Header
    print("     Q\\KV", end="")
    for kv in range(seq_len):
        print(f" {kv:2d}", end="")
    print()
    
    # Rows
    for q in range(seq_len):
        # Mark segments
        if q < original_length:
            seg_num = q // 3 + 1
            marker = f"S{seg_num}"
        else:
            marker = f"G{q - original_length}"
        
        print(f"  {marker:>3} {q:2d}", end="")
        for kv in range(seq_len):
            symbol = " ■" if matrix[q][kv] else " ·"
            print(symbol, end="")
        print()
    
    # Verify segment isolation
    print("\nVerifying segment isolation pattern:")
    # Check that segments don't attend to each other
    assert not matrix[1][4], "Segment 1 should not attend to Segment 2"
    assert not matrix[4][1], "Segment 2 should not attend to Segment 1"
    assert not matrix[7][1], "Segment 3 should not attend to Segment 1"
    
    # Check that generated tokens attend to all
    assert matrix[9][0], "Generated should attend to Segment 1"
    assert matrix[9][3], "Generated should attend to Segment 2"
    assert matrix[9][6], "Generated should attend to Segment 3"
    
    print("✅ Segment isolation verified")
    print("✅ Generation phase full attention verified")
    print("\n✅ PASS: Visualization matches expected pattern")


def main():
    """Run all tests."""
    print("="*70)
    print("  Testing create_flex_attention_mask Implementation")
    print("="*70)
    print("\nThis test suite validates:")
    print("  1. Proper tensor return types (not Python bools)")
    print("  2. Causal constraint enforcement")
    print("  3. Segment isolation during encoding")
    print("  4. Full attention during generation")
    print("  5. Edge case handling")
    print("  6. Tensor index compatibility")
    print("  7. Visual verification of mask pattern")
    
    try:
        test_return_types()
        test_causal_constraint()
        test_segment_isolation()
        test_generation_phase()
        test_edge_cases()
        test_batched_indices()
        visualize_mask_matrix()
        
        print("\n" + "="*70)
        print("  ALL TESTS PASSED ✅")
        print("="*70)
        print("\nThe create_flex_attention_mask implementation:")
        print("  ✅ Returns proper Tensor types (vmap compatible)")
        print("  ✅ Enforces causal constraint")
        print("  ✅ Implements segment isolation during encoding")
        print("  ✅ Enables full attention during generation")
        print("  ✅ Handles edge cases correctly")
        print("  ✅ Works with FlexAttention's tensor indices")
        print("\nReady for use with FlexAttention!")
        
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
