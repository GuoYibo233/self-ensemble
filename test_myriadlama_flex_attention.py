#!/usr/bin/env python3
"""
Test script for MyriadLAMA-specific FlexAttention implementation.

This test verifies:
1. The modified mask with manual cross-attention works correctly
2. Manual templates can attend to each other when enabled
3. Auto templates remain isolated
4. Generation phase works correctly with manual cross-attention
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
    
    # 3 manual templates + 2 auto templates
    segment_positions = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    original_length = 50
    manual_count = 3
    
    mask_mod = create_myriadlama_mask(segment_positions, original_length, manual_count)
    
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


def test_manual_cross_attention():
    """Test that manual templates can attend to each other."""
    print_section("Test 2: Manual Template Cross-Attention")
    
    # 3 manual templates: [0-10), [10-20), [20-30)
    # 2 auto templates: [30-40), [40-50)
    segment_positions = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    original_length = 50
    manual_count = 3
    
    mask_mod = create_myriadlama_mask(
        segment_positions, original_length, manual_count=manual_count
    )
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print(f"\nManual templates: segments 0-{manual_count-1}")
    print(f"Auto templates: segments {manual_count}-{len(segment_positions)-1}")
    print("\nTesting manual cross-attention:")
    
    test_cases = [
        # Manual to manual - should be True (cross-attention allowed)
        (5, 3, True, "Manual 1 (pos 5) to Manual 1 (pos 3) - same segment"),
        (15, 5, True, "Manual 2 (pos 15) to Manual 1 (pos 5) - cross manual"),
        (25, 5, True, "Manual 3 (pos 25) to Manual 1 (pos 5) - cross manual"),
        (25, 15, True, "Manual 3 (pos 25) to Manual 2 (pos 15) - cross manual"),
        
        # Manual to auto - should be False (different groups)
        (15, 35, False, "Manual 2 (pos 15) to Auto 1 (pos 35) - future & different group"),
        (25, 35, False, "Manual 3 (pos 25) to Auto 1 (pos 35) - future & different group"),
        
        # Auto to manual - should be False (different groups)
        (35, 5, False, "Auto 1 (pos 35) to Manual 1 (pos 5) - different group"),
        (45, 15, False, "Auto 2 (pos 45) to Manual 2 (pos 15) - different group"),
        
        # Auto to auto (same segment) - should be True
        (35, 33, True, "Auto 1 (pos 35) to Auto 1 (pos 33) - same segment"),
        (45, 43, True, "Auto 2 (pos 45) to Auto 2 (pos 43) - same segment"),
        
        # Auto to auto (different segments) - should be False
        (45, 35, False, "Auto 2 (pos 45) to Auto 1 (pos 35) - different auto segments"),
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
    
    assert all_passed, "Some manual cross-attention tests failed"
    print("\n✅ PASS: Manual cross-attention working correctly")


def test_no_manual_cross_attention():
    """Test behavior when manual_count=0 (no cross-attention)."""
    print_section("Test 3: No Manual Cross-Attention (manual_count=0)")
    
    segment_positions = [(0, 10), (10, 20), (20, 30)]
    original_length = 30
    manual_count = 0  # Disable manual cross-attention
    
    mask_mod = create_myriadlama_mask(
        segment_positions, original_length, manual_count=manual_count
    )
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nWith manual_count=0, should behave like standard segment isolation:")
    
    test_cases = [
        # Same segment - should be True
        (5, 3, True, "Same segment"),
        
        # Different segments - should be False
        (15, 5, False, "Different segments (no cross-attention)"),
        (25, 15, False, "Different segments (no cross-attention)"),
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
    
    assert all_passed, "Some no cross-attention tests failed"
    print("\n✅ PASS: Correctly disables manual cross-attention when manual_count=0")


def test_generation_phase():
    """Test that generated tokens can attend to all templates."""
    print_section("Test 4: Generation Phase with Manual Cross-Attention")
    
    segment_positions = [(0, 10), (10, 20), (20, 30), (30, 40)]
    original_length = 40
    manual_count = 2
    
    mask_mod = create_myriadlama_mask(
        segment_positions, original_length, manual_count=manual_count
    )
    
    b = torch.tensor(0)
    h = torch.tensor(0)
    
    print("\nGenerated tokens should attend to all templates (manual + auto):")
    
    test_cases = [
        # Generated token attending to manual templates
        (40, 5, True, "Gen token 0 to Manual 1"),
        (40, 15, True, "Gen token 0 to Manual 2"),
        
        # Generated token attending to auto templates
        (40, 25, True, "Gen token 0 to Auto 1"),
        (40, 35, True, "Gen token 0 to Auto 2"),
        
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
    
    # Simple setup: 2 manual + 2 auto + 2 generated
    # Manual: [0-2], [3-5]
    # Auto: [6-8], [9-11]
    # Generated: [12-13]
    segment_positions = [(0, 3), (3, 6), (6, 9), (9, 12)]
    original_length = 12
    manual_count = 2
    seq_len = 14  # 12 original + 2 generated
    
    mask_mod = create_myriadlama_mask(
        segment_positions, original_length, manual_count=manual_count
    )
    
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
    print("  Manual: [0-2], [3-5] | Auto: [6-8], [9-11] | Gen: [12-13]")
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
            marker = "M1"
        elif q < 6:
            marker = "M2"
        elif q < 9:
            marker = "A1"
        elif q < 12:
            marker = "A2"
        else:
            marker = f"G{q - 12}"
        
        print(f"  {marker:>4} {q:2d}", end="")
        for kv in range(seq_len):
            symbol = " ■" if matrix[q][kv] else " ·"
            print(symbol, end="")
        print()
    
    print("\nExpected pattern:")
    print("  ✅ Manual templates (M1, M2) can attend to each other")
    print("  ✅ Auto templates (A1, A2) isolated from each other")
    print("  ✅ Manual and Auto cannot attend to each other")
    print("  ✅ Generated (G) can attend to all previous tokens")
    
    # Verify manual cross-attention
    assert matrix[4][1], "Manual 2 should attend to Manual 1"
    assert matrix[1][4], "Manual 1 should attend to Manual 2 (if causal)"
    
    # Verify auto isolation
    assert not matrix[10][7], "Auto 2 should not attend to Auto 1"
    
    # Verify manual-auto isolation
    assert not matrix[4][7], "Manual 2 should not attend to Auto 1"
    assert not matrix[7][4], "Auto 1 should not attend to Manual 2"
    
    # Verify generated attention
    assert matrix[12][1], "Generated should attend to Manual 1"
    assert matrix[12][7], "Generated should attend to Auto 1"
    
    print("\n✅ PASS: Visualization matches expected MyriadLAMA pattern")


def main():
    """Run all tests."""
    print("="*70)
    print("  Testing MyriadLAMA-Specific FlexAttention Implementation")
    print("="*70)
    print("\nThis test suite validates:")
    print("  1. Proper tensor return types")
    print("  2. Manual template cross-attention")
    print("  3. Auto template isolation")
    print("  4. Generation phase full attention")
    print("  5. Mask pattern visualization")
    
    try:
        test_return_types()
        test_manual_cross_attention()
        test_no_manual_cross_attention()
        test_generation_phase()
        visualize_myriadlama_mask()
        
        print("\n" + "="*70)
        print("  ALL TESTS PASSED ✅")
        print("="*70)
        print("\nThe MyriadLAMA-specific mask implementation:")
        print("  ✅ Returns proper Tensor types (vmap compatible)")
        print("  ✅ Enables manual template cross-attention")
        print("  ✅ Maintains auto template isolation")
        print("  ✅ Enables full attention during generation")
        print("  ✅ Correctly handles manual_count parameter")
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
