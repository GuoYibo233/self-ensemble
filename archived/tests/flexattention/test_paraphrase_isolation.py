"""
Test to verify that paraphrases within the same group are isolated.

This test verifies the user's requirement:
- Q: {fs1_para3} can't attend to Q: {fs1_para2} and Q: {fs1_para1}
- Q: {main_q_para3} can't attend to Q: {main_q_para2} and Q: {main_q_para1}
"""

import sys
sys.path.insert(0, '/home/runner/work/self-ensemble/self-ensemble')

import torch
from myriadlama_flex_attention_generate import (
    parse_prompt_segments_with_metadata_new_format,
    concatenate_paraphrases_with_positions,
    create_myriadlama_mask
)

def test_paraphrase_isolation():
    """Test that paraphrases within the same group cannot attend to each other."""
    
    # Create a test prompt with the new format
    prompt = """Predict the [MASK] in the sentence in one word.

Q: London is the capital of [MASK]
Q: The capital of [MASK] is London
Q: [MASK] has London as its capital
A: England

Q: Tokyo is the capital of [MASK]
Q: The capital of [MASK] is Tokyo
Q: Japanese capital is [MASK]
A: Japan

Q: Paris is the capital of [MASK]
Q: The capital of [MASK] is Paris
Q: France's capital is [MASK]
A:"""
    
    print("=" * 80)
    print("Test: Paraphrase Isolation")
    print("=" * 80)
    print("\nPrompt:")
    print(prompt)
    print("\n" + "=" * 80)
    
    # Parse the prompt
    segments = parse_prompt_segments_with_metadata_new_format(prompt)
    
    print("\nParsed segments:")
    for i, (text, meta) in enumerate(segments):
        print(f"{i:2d}. {text[:50]:50s} | {meta}")
    
    # Concatenate to get positions (simulating tokenization with dummy tokenizer)
    # For simplicity, assume each segment is 10 tokens
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Return dummy tokens (length proportional to text length)
            return [0] * max(1, len(text) // 10)
    
    tokenizer = DummyTokenizer()
    segment_positions, segment_metadata, total_length = concatenate_paraphrases_with_positions(
        prompt, tokenizer
    )
    
    print(f"\nTotal length: {total_length}")
    print(f"Number of segments: {len(segment_positions)}")
    
    # Create the mask function
    mask_fn = create_myriadlama_mask(segment_positions, segment_metadata, total_length)
    
    # Test cases
    print("\n" + "=" * 80)
    print("Testing Mask Behavior:")
    print("=" * 80)
    
    # Find segment indices for testing
    fs1_para1_idx = None
    fs1_para2_idx = None
    fs1_para3_idx = None
    fs2_para1_idx = None
    fs2_para2_idx = None
    main_para1_idx = None
    main_para2_idx = None
    main_para3_idx = None
    
    for i, meta in enumerate(segment_metadata):
        if meta['type'] == 'few_shot_q' and meta['few_shot_idx'] == 0:
            if meta['fs_q_para_idx'] == 0:
                fs1_para1_idx = i
            elif meta['fs_q_para_idx'] == 1:
                fs1_para2_idx = i
            elif meta['fs_q_para_idx'] == 2:
                fs1_para3_idx = i
        elif meta['type'] == 'few_shot_q' and meta['few_shot_idx'] == 1:
            if meta['fs_q_para_idx'] == 0:
                fs2_para1_idx = i
            elif meta['fs_q_para_idx'] == 1:
                fs2_para2_idx = i
        elif meta['type'] == 'question':
            if meta['paraphrase_idx'] == 0:
                main_para1_idx = i
            elif meta['paraphrase_idx'] == 1:
                main_para2_idx = i
            elif meta['paraphrase_idx'] == 2:
                main_para3_idx = i
    
    # Test: fs1_para3 should NOT attend to fs1_para1 and fs1_para2
    if fs1_para3_idx and fs1_para1_idx and fs1_para2_idx:
        # Get a position within each segment for testing
        q_pos = segment_positions[fs1_para3_idx][0]  # First token of fs1_para3
        kv_pos1 = segment_positions[fs1_para1_idx][0]  # First token of fs1_para1
        kv_pos2 = segment_positions[fs1_para2_idx][0]  # First token of fs1_para2
        
        result1 = mask_fn(
            torch.tensor(0), torch.tensor(0),  # batch, head (dummy)
            torch.tensor(q_pos), torch.tensor(kv_pos1)
        )
        result2 = mask_fn(
            torch.tensor(0), torch.tensor(0),
            torch.tensor(q_pos), torch.tensor(kv_pos2)
        )
        
        print(f"\n1. FS1-Para3 attend to FS1-Para1: {result1.item()}")
        print(f"   Expected: False (CANNOT attend)")
        print(f"   Result: {'✅ PASS' if not result1.item() else '❌ FAIL'}")
        
        print(f"\n2. FS1-Para3 attend to FS1-Para2: {result2.item()}")
        print(f"   Expected: False (CANNOT attend)")
        print(f"   Result: {'✅ PASS' if not result2.item() else '❌ FAIL'}")
    
    # Test: fs1_para1 should NOT attend to fs2_para1 (different FS, but causal may block)
    # Actually they CAN attend if causal allows (fs1 comes before fs2)
    if fs2_para1_idx and fs1_para1_idx:
        q_pos = segment_positions[fs2_para1_idx][0]
        kv_pos = segment_positions[fs1_para1_idx][0]
        
        result = mask_fn(
            torch.tensor(0), torch.tensor(0),
            torch.tensor(q_pos), torch.tensor(kv_pos)
        )
        
        print(f"\n3. FS2-Para1 attend to FS1-Para1 (different FS): {result.item()}")
        print(f"   Expected: True (CAN attend - different FS, and causal allows)")
        print(f"   Result: {'✅ PASS' if result.item() else '❌ FAIL'}")
    
    # Test: main_para3 should NOT attend to main_para1 and main_para2
    if main_para3_idx and main_para1_idx and main_para2_idx:
        q_pos = segment_positions[main_para3_idx][0]
        kv_pos1 = segment_positions[main_para1_idx][0]
        kv_pos2 = segment_positions[main_para2_idx][0]
        
        result1 = mask_fn(
            torch.tensor(0), torch.tensor(0),
            torch.tensor(q_pos), torch.tensor(kv_pos1)
        )
        result2 = mask_fn(
            torch.tensor(0), torch.tensor(0),
            torch.tensor(q_pos), torch.tensor(kv_pos2)
        )
        
        print(f"\n4. Main-Para3 attend to Main-Para1: {result1.item()}")
        print(f"   Expected: False (CANNOT attend)")
        print(f"   Result: {'✅ PASS' if not result1.item() else '❌ FAIL'}")
        
        print(f"\n5. Main-Para3 attend to Main-Para2: {result2.item()}")
        print(f"   Expected: False (CANNOT attend)")
        print(f"   Result: {'✅ PASS' if not result2.item() else '❌ FAIL'}")
    
    # Test: fs1_para1 within itself should allow causal attention
    if fs1_para1_idx:
        seg_start, seg_end = segment_positions[fs1_para1_idx]
        if seg_end - seg_start > 1:  # If segment has more than 1 token
            q_pos = seg_start + 1  # Second token
            kv_pos = seg_start  # First token
            
            result = mask_fn(
                torch.tensor(0), torch.tensor(0),
                torch.tensor(q_pos), torch.tensor(kv_pos)
            )
            
            print(f"\n6. FS1-Para1 token 2 attend to FS1-Para1 token 1 (same segment): {result.item()}")
            print(f"   Expected: True (CAN attend - within same segment, causal)")
            print(f"   Result: {'✅ PASS' if result.item() else '❌ FAIL'}")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_paraphrase_isolation()
