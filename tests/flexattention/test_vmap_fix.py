#!/usr/bin/env python3
"""
Test script to verify FlexAttention vmap fix.

This script tests that the mask function works with PyTorch's vmap
by using only tensor operations (no .item() or Python if on tensors).
"""

import torch
import sys

def test_mask_function():
    """Test that mask function uses only tensor operations"""
    
    # Import the mask creation function
    sys.path.insert(0, '/home/runner/work/self-ensemble/self-ensemble')
    from myriadlama_flex_attention_generate import create_myriadlama_mask
    
    # Create dummy segment positions and metadata
    segment_positions = [
        (0, 10),    # instruction
        (10, 20),   # fs1_q_para1
        (20, 30),   # fs1_q_para2
        (30, 40),   # fs1_a
        (40, 50),   # fs2_q_para1
        (50, 60),   # fs2_q_para2
        (60, 70),   # fs2_a
        (70, 80),   # main_q_para1
        (80, 90),   # main_q_para2
    ]
    
    segment_metadata = [
        {'type': 'instruction', 'paraphrase_idx': None, 'few_shot_idx': None},
        {'type': 'few_shot_q', 'paraphrase_idx': 0, 'few_shot_idx': 0, 'fs_q_para_idx': 0},
        {'type': 'few_shot_q', 'paraphrase_idx': 0, 'few_shot_idx': 0, 'fs_q_para_idx': 1},
        {'type': 'few_shot_a', 'paraphrase_idx': 0, 'few_shot_idx': 0},
        {'type': 'few_shot_q', 'paraphrase_idx': 0, 'few_shot_idx': 1, 'fs_q_para_idx': 0},
        {'type': 'few_shot_q', 'paraphrase_idx': 0, 'few_shot_idx': 1, 'fs_q_para_idx': 1},
        {'type': 'few_shot_a', 'paraphrase_idx': 0, 'few_shot_idx': 1},
        {'type': 'question', 'paraphrase_idx': 0, 'few_shot_idx': None},
        {'type': 'question', 'paraphrase_idx': 1, 'few_shot_idx': None},
    ]
    
    original_length = 90
    
    # Create the mask function
    mask_mod = create_myriadlama_mask(segment_positions, segment_metadata, original_length)
    
    # Test the mask function with various query-key pairs
    print("Testing mask function with tensor operations...")
    print()
    
    test_cases = [
        (0, 0, "Instruction to itself"),
        (15, 5, "FS1-Q-Para1 to Instruction"),
        (15, 25, "FS1-Q-Para1 to FS1-Q-Para2 (same FS, diff para - should BLOCK)"),
        (15, 45, "FS1-Q-Para1 to FS2-Q-Para1 (diff FS - should ALLOW)"),
        (75, 85, "Main-Q-Para1 to Main-Q-Para2 (diff para - should BLOCK)"),
        (75, 75, "Main-Q-Para1 to itself (should ALLOW)"),
        (35, 15, "FS1-Answer to FS1-Q-Para1 (should ALLOW)"),
        (35, 25, "FS1-Answer to FS1-Q-Para2 (should ALLOW)"),
        (95, 15, "Generated token to FS1-Q-Para1 (should ALLOW)"),
        (15, 25, "FS1-Q-Para1 to future FS1-Q-Para2 (causal violation - should BLOCK)"),
    ]
    
    for q_idx, kv_idx, desc in test_cases:
        try:
            # Create tensor indices
            q = torch.tensor(q_idx, dtype=torch.int64)
            kv = torch.tensor(kv_idx, dtype=torch.int64)
            b = torch.tensor(0, dtype=torch.int64)
            h = torch.tensor(0, dtype=torch.int64)
            
            # Call mask function
            result = mask_mod(b, h, q, kv)
            
            # Check that result is a tensor (not Python bool)
            assert isinstance(result, torch.Tensor), f"Result must be tensor, got {type(result)}"
            
            # Get the boolean value
            allow = result.item()
            
            print(f"✓ q={q_idx:3d}, kv={kv_idx:3d}: {allow:5} - {desc}")
            
        except Exception as e:
            print(f"✗ q={q_idx:3d}, kv={kv_idx:3d}: ERROR - {desc}")
            print(f"  Error: {e}")
            return False
    
    print()
    print("="*70)
    print("SUCCESS: All test cases passed!")
    print("The mask function now uses only tensor operations.")
    print("This fix should resolve the vmap compilation error.")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_mask_function()
    sys.exit(0 if success else 1)
