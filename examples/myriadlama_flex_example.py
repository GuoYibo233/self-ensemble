#!/usr/bin/env python3
"""
Example script demonstrating MyriadLAMA FlexAttention usage.

Run with: python examples/myriadlama_flex_example.py
"""

def example_prompt_construction():
    """Demonstrate prompt construction for MyriadLAMA."""
    print("="*70)
    print("  Example: MyriadLAMA Prompt Construction")
    print("="*70)
    
    print("\nExample MyriadLAMA data:")
    print("  Relation: capital_of")
    print("  Answers: ['France', 'French Republic']")
    print("\n  Manual templates (3):")
    print("    1. Paris is the capital of [MASK]")
    print("    2. The capital of [MASK] is Paris")
    print("    3. [MASK]'s capital city is Paris")
    print("\n  Auto templates (5):")
    print("    1. Paris, capital of [MASK]")
    print("    2. The city Paris is [MASK]'s capital")
    print("    ... (3 more)")
    
    print("\n" + "-"*70)
    print("  Configuration: All manual + 5 auto with cross-attention")
    print("-"*70)
    print("\nManual templates (0-2): Can attend to each other")
    print("Auto templates (3-7): Isolated from each other")
    print("Generation: New tokens attend to all templates")

print("\n" + "="*70)
print("  MyriadLAMA FlexAttention Examples")
print("="*70)

example_prompt_construction()

print("\n" + "="*70)
print("  Usage Examples")
print("="*70)
print("\nGenerate with manual cross-attention:")
command = """  python myriadlama_flex_attention_generate.py \\
    --model llama3.2_3b_it \\
    --allow_manual_cross_attention"""
print(command)
print("\nFor more details, see:")
print("  - MYRIADLAMA_FLEX_USAGE.md")
print("  - docs/MYRIADLAMA_FLEX_ATTENTION.md")
print()
