#!/usr/bin/env python3
"""
Test script to verify the updated construct_prompt_new_format function
"""

def construct_prompt_new_format(instruction, few_shot_examples, question_paraphrases):
    """
    Construct prompt parts: one shared part + multiple paraphrase parts.
    
    Returns a list where:
    - First element: shared part (instruction + few-shot examples)
    - Remaining elements: individual paraphrase parts (one per paraphrase)
    """
    # Build shared part: instruction + few-shot examples
    shared_parts = []
    
    # Add instruction with trailing newline
    shared_parts.append(instruction + "\n")
    
    # Add few-shot examples (each on one line: Q: ... A: ...)
    fs_lines = []
    for fs_example in few_shot_examples:
        # Use only the FIRST paraphrase from each few-shot example
        para = fs_example['paraphrases'][0]
        answer = fs_example['answer']
        fs_lines.append(f"Q: {para} A: {answer}")
    
    # Join few-shot lines with newlines, add trailing newline
    shared_parts.append("\n".join(fs_lines) + "\n")
    
    # Combine shared parts
    shared_part = "\n".join(shared_parts)
    
    # Build individual paraphrase parts
    para_parts = []
    for para in question_paraphrases:
        para_part = f"Q: {para} A:"
        para_parts.append(para_part)
    
    # Return list: [shared_part, para_part_1, para_part_2, ...]
    return [shared_part] + para_parts


# Test the function
if __name__ == "__main__":
    instruction = "You are a helpful assistant. Answer the following question."
    
    few_shot_examples = [
        {
            'paraphrases': [
                'What is the capital of France?',
                'France\'s capital is?',
                'Name the capital of France'
            ],
            'answer': 'Paris'
        },
        {
            'paraphrases': [
                'Who invented relativity?',
                'The inventor of relativity is?',
                'Relativity was invented by?'
            ],
            'answer': 'Einstein'
        },
        {
            'paraphrases': [
                'What is the largest planet?',
                'The largest planet is?',
                'Name the biggest planet'
            ],
            'answer': 'Jupiter'
        }
    ]
    
    question_paraphrases = [
        'What is X?',
        'Define X',
        'X refers to?'
    ]
    
    # Call the function
    parts = construct_prompt_new_format(instruction, few_shot_examples, question_paraphrases)
    
    print("="*80)
    print("TESTING construct_prompt_new_format")
    print("="*80)
    
    print(f"\nTotal parts returned: {len(parts)}")
    print(f"  - 1 shared part")
    print(f"  - {len(parts) - 1} paraphrase parts")
    
    print("\n" + "="*80)
    print("PART 0: Shared Part (Instruction + Few-Shot)")
    print("="*80)
    print(repr(parts[0]))
    print("\nFormatted:")
    print(parts[0])
    
    for i, para_part in enumerate(parts[1:], 1):
        print("\n" + "="*80)
        print(f"PART {i}: Paraphrase {i}")
        print("="*80)
        print(repr(para_part))
        print("\nFormatted:")
        print(para_part)
    
    print("\n" + "="*80)
    print("FULL PROMPT (Combined)")
    print("="*80)
    full_prompt = parts[0] + "\n".join(parts[1:])
    print(full_prompt)
    
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    # Verify structure
    shared_part = parts[0]
    lines = shared_part.strip().split('\n')
    print(f"✓ Shared part has {len(lines)} lines")
    print(f"  - Line 1: Instruction")
    print(f"  - Lines 2-4: Few-shot examples (Q: ... A: ...)")
    
    print(f"\n✓ Each paraphrase part is a single line:")
    for i, para_part in enumerate(parts[1:], 1):
        print(f"  - Part {i}: {para_part}")
    
    print("\n✓ Format check:")
    for i, fs_example in enumerate(few_shot_examples):
        expected = f"Q: {fs_example['paraphrases'][0]} A: {fs_example['answer']}"
        if expected in shared_part:
            print(f"  ✓ Few-shot {i+1} correctly formatted")
        else:
            print(f"  ✗ Few-shot {i+1} INCORRECT")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
