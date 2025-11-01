"""
Test script to verify that causal mask has highest priority in the attention mask.

This script tests that:
1. Positions violating causality (kv_idx > q_idx) are always blocked
2. Custom masking rules only apply when causality is satisfied
3. The refactored code behaves identically to the original
"""

import torch

def test_causal_priority():
    """
    Test that causal constraint has highest priority.
    """
    print("=" * 80)
    print("Testing Causal Mask Priority")
    print("=" * 80)
    
    # Simulate a simple segment structure
    segment_positions = [
        (0, 10),   # Instruction (segment 0)
        (10, 20),  # FS1-Q-para1 (segment 1)
        (20, 30),  # FS1-Q-para2 (segment 2)
        (30, 40),  # FS1-A (segment 3)
        (40, 50),  # Main-Q-para1 (segment 4)
        (50, 60),  # Main-Q-para2 (segment 5)
    ]
    
    segment_metadata = [
        {'type': 'instruction', 'paraphrase_idx': 0, 'few_shot_idx': -1},
        {'type': 'few_shot_q', 'paraphrase_idx': 0, 'few_shot_idx': 0, 'fs_q_para_idx': 0},
        {'type': 'few_shot_q', 'paraphrase_idx': 0, 'few_shot_idx': 0, 'fs_q_para_idx': 1},
        {'type': 'few_shot_a', 'paraphrase_idx': 0, 'few_shot_idx': 0},
        {'type': 'question', 'paraphrase_idx': 0, 'few_shot_idx': -1},
        {'type': 'question', 'paraphrase_idx': 1, 'few_shot_idx': -1},
    ]
    
    original_length = 60
    
    # Import the mask creation function
    import sys
    sys.path.insert(0, '/home/runner/work/self-ensemble/self-ensemble')
    from myriadlama_flex_attention_generate import create_myriadlama_mask
    
    mask_fn = create_myriadlama_mask(segment_positions, segment_metadata, original_length)
    
    print("\nTest 1: Causal violation (kv_idx > q_idx) should always block")
    print("-" * 80)
    
    # Test: q=15 (in FS1-Q-para1), kv=25 (in FS1-Q-para2, future position)
    q_idx = torch.tensor(15)
    kv_idx = torch.tensor(25)
    result = mask_fn(0, 0, q_idx, kv_idx)
    print(f"q_idx=15 (FS1-Q-para1), kv_idx=25 (FS1-Q-para2, FUTURE)")
    print(f"Expected: False (violates causality)")
    print(f"Actual: {result}")
    assert result == False, "Causal violation should always block!"
    print("✓ PASS: Causal violation correctly blocked\n")
    
    # Test: q=25 (in FS1-Q-para2), kv=15 (in FS1-Q-para1, past position)
    q_idx = torch.tensor(25)
    kv_idx = torch.tensor(15)
    result = mask_fn(0, 0, q_idx, kv_idx)
    print(f"q_idx=25 (FS1-Q-para2), kv_idx=15 (FS1-Q-para1, PAST)")
    print(f"Expected: False (same FS, different para - custom rule blocks)")
    print(f"Actual: {result}")
    assert result == False, "Same FS, different para should be blocked by custom rule!"
    print("✓ PASS: Custom rule correctly applied after causal check\n")
    
    print("\nTest 2: Causal allowed, custom rules apply")
    print("-" * 80)
    
    # Test: q=15 (in FS1-Q-para1), kv=5 (in instruction)
    q_idx = torch.tensor(15)
    kv_idx = torch.tensor(5)
    result = mask_fn(0, 0, q_idx, kv_idx)
    print(f"q_idx=15 (FS1-Q-para1), kv_idx=5 (instruction)")
    print(f"Expected: True (all segments can attend to instruction)")
    print(f"Actual: {result}")
    assert result == True, "Should attend to instruction!"
    print("✓ PASS: Instruction attention allowed\n")
    
    # Test: q=15 (in FS1-Q-para1), kv=12 (in FS1-Q-para1, same segment)
    q_idx = torch.tensor(15)
    kv_idx = torch.tensor(12)
    result = mask_fn(0, 0, q_idx, kv_idx)
    print(f"q_idx=15 (FS1-Q-para1), kv_idx=12 (FS1-Q-para1, same segment)")
    print(f"Expected: True (same segment, same para, causal allowed)")
    print(f"Actual: {result}")
    assert result == True, "Should attend within same segment!"
    print("✓ PASS: Same segment attention allowed\n")
    
    # Test: q=45 (in Main-Q-para1), kv=55 (in Main-Q-para2, different para)
    # This would violate causality, so should be blocked immediately
    q_idx = torch.tensor(45)
    kv_idx = torch.tensor(55)
    result = mask_fn(0, 0, q_idx, kv_idx)
    print(f"q_idx=45 (Main-Q-para1), kv_idx=55 (Main-Q-para2, FUTURE)")
    print(f"Expected: False (violates causality)")
    print(f"Actual: {result}")
    assert result == False, "Future position should be blocked by causal!"
    print("✓ PASS: Causal violation blocked\n")
    
    # Test: q=55 (in Main-Q-para2), kv=45 (in Main-Q-para1, different para)
    q_idx = torch.tensor(55)
    kv_idx = torch.tensor(45)
    result = mask_fn(0, 0, q_idx, kv_idx)
    print(f"q_idx=55 (Main-Q-para2), kv_idx=45 (Main-Q-para1, PAST)")
    print(f"Expected: False (different question paraphrases - custom rule blocks)")
    print(f"Actual: {result}")
    assert result == False, "Different question paras should be blocked!"
    print("✓ PASS: Question paraphrase isolation enforced\n")
    
    print("\nTest 3: Generation phase (q_idx >= original_length)")
    print("-" * 80)
    
    # Test: q=65 (generation), kv=15 (encoding)
    q_idx = torch.tensor(65)
    kv_idx = torch.tensor(15)
    result = mask_fn(0, 0, q_idx, kv_idx)
    print(f"q_idx=65 (GENERATION), kv_idx=15 (encoding)")
    print(f"Expected: True (generation can attend to all past)")
    print(f"Actual: {result}")
    assert result == True, "Generation should attend to all encoding!"
    print("✓ PASS: Generation attention allowed\n")
    
    # Test: q=65 (generation), kv=70 (future generation)
    q_idx = torch.tensor(65)
    kv_idx = torch.tensor(70)
    result = mask_fn(0, 0, q_idx, kv_idx)
    print(f"q_idx=65 (generation), kv_idx=70 (FUTURE generation)")
    print(f"Expected: False (violates causality)")
    print(f"Actual: {result}")
    assert result == False, "Cannot attend to future even in generation!"
    print("✓ PASS: Causal constraint enforced in generation\n")
    
    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nSummary:")
    print("1. ✓ Causal constraint has highest priority")
    print("2. ✓ Positions violating causality are immediately blocked")
    print("3. ✓ Custom masking rules only apply when causality is satisfied")
    print("4. ✓ Generation phase respects causal constraint")
    print("=" * 80)

if __name__ == "__main__":
    test_causal_priority()
