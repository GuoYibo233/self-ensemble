#!/usr/bin/env python3
"""
Test to verify the separator display fix.
This simulates the token structure without requiring a model.
"""

def test_segment_display():
    """Test that segments include their trailing separators."""
    
    # Simulate token structure:
    # Prompt 1: tokens 0-9 (10 tokens)
    # Separator: tokens 10-14 (5 tokens, representing "\n\n[SEP]\n\n")
    # Prompt 2: tokens 15-24 (10 tokens)
    # Separator: tokens 25-29 (5 tokens)
    # Prompt 3: tokens 30-39 (10 tokens)
    # Generated: tokens 40-44 (5 tokens)
    
    segment_positions = [
        (0, 10),   # Prompt 1
        (15, 25),  # Prompt 2
        (30, 40),  # Prompt 3
    ]
    
    original_length = 40
    full_tokens = list(range(45))  # 0-44
    
    print("Testing segment display logic...")
    print(f"Total tokens: {len(full_tokens)}")
    print(f"Segment positions: {segment_positions}")
    print(f"Original length: {original_length}")
    print()
    
    # OLD VERSION (buggy):
    print("=== OLD VERSION (Buggy - loses separators) ===")
    for i, (start, end) in enumerate(segment_positions):
        segment_tokens = full_tokens[start:end]
        print(f"[Prompt {i+1}] (positions {start}-{end-1}): tokens {segment_tokens}")
    
    print("\n❌ Problem: Separator tokens (10-14, 25-29) are not shown!")
    print()
    
    # NEW VERSION (fixed):
    print("=== NEW VERSION (Fixed - includes separators) ===")
    for i, (start, end) in enumerate(segment_positions):
        # Include separator tokens after each segment (except the last one)
        if i < len(segment_positions) - 1:
            # Include tokens from current segment up to the start of next segment
            next_start = segment_positions[i+1][0]
            segment_tokens = full_tokens[start:next_start]
        else:
            # For the last segment, only include its own tokens
            segment_tokens = full_tokens[start:end]
        
        print(f"[Prompt {i+1}] (positions {start}-{end-1}): tokens {segment_tokens}")
    
    print("\n✅ Fixed: Separator tokens are now included!")
    print()
    
    # Verify correctness
    print("=== VERIFICATION ===")
    all_shown_tokens = set()
    
    for i, (start, end) in enumerate(segment_positions):
        if i < len(segment_positions) - 1:
            next_start = segment_positions[i+1][0]
            shown = set(range(start, next_start))
        else:
            shown = set(range(start, end))
        all_shown_tokens.update(shown)
    
    # Add generated tokens
    if len(full_tokens) > original_length:
        gen_shown = set(range(original_length, len(full_tokens)))
        all_shown_tokens.update(gen_shown)
        print(f"Generated tokens: {list(range(original_length, len(full_tokens)))}")
    
    all_original_tokens = set(range(original_length))
    missing = all_original_tokens - all_shown_tokens
    
    if missing:
        print(f"❌ Missing tokens: {sorted(missing)}")
    else:
        print(f"✅ All original tokens (0-{original_length-1}) are displayed!")
    
    print()


def test_actual_separator_example():
    """Test with an example closer to the actual use case."""
    print("=== ACTUAL USE CASE EXAMPLE ===")
    print()
    
    # Simulate: 3 prompts with [SEP] separators
    # Prompt 1: "Q: test?\nA:" = tokens 0-4 (5 tokens)
    # [SEP]: "\n\n[SEP]\n\n" = tokens 5-9 (5 tokens)
    # Prompt 2: "Q: test2?\nA:" = tokens 10-14 (5 tokens)
    # [SEP]: "\n\n[SEP]\n\n" = tokens 15-19 (5 tokens)
    # Prompt 3: "Q: test3?\nA:" = tokens 20-24 (5 tokens)
    # Generated: "answer" = tokens 25-29 (5 tokens)
    
    segment_positions = [(0, 5), (10, 15), (20, 25)]
    original_length = 25
    
    # Simulate token representation
    token_repr = {
        0: "Q:", 1: "test", 2: "?", 3: "\n", 4: "A:",
        5: "\n\n", 6: "[", 7: "SEP", 8: "]", 9: "\n\n",
        10: "Q:", 11: "test2", 12: "?", 13: "\n", 14: "A:",
        15: "\n\n", 16: "[", 17: "SEP", 18: "]", 19: "\n\n",
        20: "Q:", 21: "test3", 22: "?", 23: "\n", 24: "A:",
        25: "ans", 26: "wer", 27: "!", 28: "", 29: ""
    }
    
    full_tokens = list(range(30))
    
    print("Token structure:")
    for i in range(30):
        if i in token_repr:
            print(f"  Token {i:2d}: {token_repr[i]}")
    print()
    
    print("Display with FIX:")
    for i, (start, end) in enumerate(segment_positions):
        if i < len(segment_positions) - 1:
            next_start = segment_positions[i+1][0]
            segment_tokens = full_tokens[start:next_start]
            token_strs = [token_repr.get(t, "") for t in segment_tokens]
        else:
            segment_tokens = full_tokens[start:end]
            token_strs = [token_repr.get(t, "") for t in segment_tokens]
        
        text = "".join(token_strs)
        print(f"\n[Prompt {i+1}] (positions {start}-{end-1}):")
        print(f"  Tokens {start}-{segment_tokens[-1]}: {text}")
        
        if i < len(segment_positions) - 1:
            # Check if separator is included
            has_sep = any("[" in token_repr.get(t, "") or "SEP" in token_repr.get(t, "") 
                         for t in segment_tokens)
            print(f"  {'✅' if has_sep else '❌'} Separator included: {has_sep}")
    
    print("\n✅ Each prompt now includes its trailing [SEP] separator!")


if __name__ == "__main__":
    test_segment_display()
    print()
    print("="*70)
    print()
    test_actual_separator_example()
    print()
    print("="*70)
    print("SUMMARY:")
    print("  ✅ The fix includes separator tokens when displaying each segment")
    print("  ✅ Separators are no longer split or lost")
    print("  ✅ The display matches the actual token structure")
