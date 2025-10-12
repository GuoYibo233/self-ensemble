#!/usr/bin/env python3
"""
FlexAttention Debug Script

This script provides step-by-step debugging capabilities for FlexAttention generation.
It shows detailed information about each step of the generation process including:
- Tensor shapes and values
- Attention mask visualization
- Token-by-token generation
- Segment isolation verification

Usage:
    python3 tools/debug_flexattention.py --dataset webqa --model llama3.2_3b_it --max-samples 2
    python3 tools/debug_flexattention.py --dataset webqa --model llama3.2_3b_it --max-samples 1 --verbose
    python3 tools/debug_flexattention.py --dataset webqa --model llama3.2_3b_it --indexs 0,1,2 --max-samples 1
"""

import os
import sys
import argparse

# Add parent directory to path to import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from constants import MODEL_PATHs

# Try to import FlexAttention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print("⚠️  FlexAttention not available. Run validate_flexattention_env.py first.")
    sys.exit(1)


def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_subsection(title):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def print_tensor_info(tensor, name="Tensor"):
    """Print detailed information about a tensor."""
    print(f"\n📊 {name}:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Device: {tensor.device}")
    if tensor.numel() > 0:
        print(f"   Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        if tensor.dtype.is_floating_point:
            print(f"   Mean: {tensor.mean().item():.4f}")
        # Show first few values if small enough
        if tensor.numel() <= 20:
            print(f"   Values: {tensor.flatten().tolist()}")
        else:
            print(f"   First 5: {tensor.flatten()[:5].tolist()}")


def visualize_attention_mask(mask_func, seq_len, segment_positions, original_length):
    """Visualize the attention mask as a matrix."""
    print_subsection("Attention Mask Visualization")
    
    # Create a 2D matrix representation
    matrix = np.zeros((seq_len, seq_len), dtype=int)
    
    for q in range(seq_len):
        for kv in range(seq_len):
            # mask_func expects (batch, head, q_idx, kv_idx)
            can_attend = mask_func(0, 0, q, kv)
            matrix[q, kv] = 1 if can_attend else 0
    
    # Print the matrix
    print(f"\nMask Matrix ({seq_len}x{seq_len}):")
    print("  Q\\KV", end="")
    for kv in range(min(seq_len, 20)):  # Limit width
        print(f" {kv:2d}", end="")
    print()
    
    for q in range(min(seq_len, 20)):  # Limit height
        print(f"  {q:4d} ", end="")
        for kv in range(min(seq_len, 20)):
            symbol = "✓" if matrix[q, kv] else "✗"
            print(f" {symbol} ", end="")
        print()
    
    if seq_len > 20:
        print("  ... (truncated, showing first 20x20)")
    
    # Print segment boundaries
    print(f"\nSegment Boundaries:")
    for i, (start, end) in enumerate(segment_positions):
        print(f"  Segment {i+1}: positions {start}-{end-1}")
    print(f"  Original length: {original_length}")
    print(f"  Generated tokens: {seq_len - original_length}")


def verify_segment_isolation(mask_func, segment_positions, original_length):
    """Verify that segments are properly isolated."""
    print_subsection("Segment Isolation Verification")
    
    issues_found = False
    
    # Check each segment
    for i, (start_i, end_i) in enumerate(segment_positions):
        for q in range(start_i, end_i):
            # Check that this position can attend to its own segment
            can_attend_self = all(
                mask_func(0, 0, q, kv) 
                for kv in range(start_i, min(q+1, end_i))
            )
            
            if not can_attend_self:
                print(f"  ❌ Position {q} in segment {i+1} cannot attend to its own segment")
                issues_found = True
            
            # Check that it CANNOT attend to other segments
            for j, (start_j, end_j) in enumerate(segment_positions):
                if i != j:
                    for kv in range(start_j, end_j):
                        if kv <= q:  # Only check causal positions
                            can_attend_other = mask_func(0, 0, q, kv)
                            if can_attend_other:
                                print(f"  ❌ Position {q} in segment {i+1} can attend to segment {j+1} position {kv}")
                                issues_found = True
    
    if not issues_found:
        print("  ✅ All segments properly isolated")
    
    return not issues_found


def debug_concatenation(prompts, tokenizer, separator=" [SEP] "):
    """Debug the concatenation step."""
    print_section("Step 1: Paraphrase Concatenation")
    
    print(f"\nInput: {len(prompts)} paraphrases")
    for i, prompt in enumerate(prompts):
        print(f"\n  Paraphrase {i+1}:")
        print(f"    Text: {prompt[:100]}..." if len(prompt) > 100 else f"    Text: {prompt}")
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        print(f"    Tokens: {len(tokens)}")
    
    # Tokenize each prompt
    tokenized_prompts = []
    sep_tokens = tokenizer.encode(separator, add_special_tokens=False)
    print(f"\nSeparator: '{separator}'")
    print(f"  Separator tokens: {sep_tokens}")
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tokenized_prompts.append(tokens)
    
    # Build full sequence
    full_tokens = []
    segment_positions = []
    current_pos = 0
    
    for i, tokens in enumerate(tokenized_prompts):
        if i > 0:
            full_tokens.extend(sep_tokens)
            current_pos += len(sep_tokens)
        
        start_pos = current_pos
        full_tokens.extend(tokens)
        current_pos += len(tokens)
        end_pos = current_pos
        
        segment_positions.append((start_pos, end_pos))
        print(f"\n  Segment {i+1}: positions [{start_pos}, {end_pos})")
    
    concatenated_text = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    print(f"\nResult:")
    print(f"  Total tokens: {len(full_tokens)}")
    print(f"  Segment positions: {segment_positions}")
    print(f"  Concatenated text (first 200 chars): {concatenated_text[:200]}...")
    
    return concatenated_text, segment_positions, len(full_tokens)


def debug_mask_creation(segment_positions, original_length, current_length=None):
    """Debug the mask creation step."""
    if current_length is None:
        current_length = original_length
    
    print_section("Step 2: Attention Mask Creation")
    
    print(f"\nParameters:")
    print(f"  Original length: {original_length}")
    print(f"  Current length: {current_length}")
    print(f"  Generated tokens: {current_length - original_length}")
    print(f"  Number of segments: {len(segment_positions)}")
    
    # Create mask function
    def mask_mod(b, h, q_idx, kv_idx):
        # Causal constraint
        if q_idx < kv_idx:
            return False
        
        # Generated tokens can attend to everything
        if q_idx >= original_length:
            return True
        
        # Original tokens only attend within segment
        q_segment = None
        kv_segment = None
        
        for seg_id, (start, end) in enumerate(segment_positions):
            if start <= q_idx < end:
                q_segment = seg_id
            if start <= kv_idx < end:
                kv_segment = seg_id
        
        if q_segment is not None and kv_segment is not None:
            return q_segment == kv_segment
        
        return False
    
    # Visualize and verify
    visualize_attention_mask(mask_mod, current_length, segment_positions, original_length)
    verify_segment_isolation(mask_mod, segment_positions, original_length)
    
    return mask_mod


def debug_generation_step(step, input_ids, logits, next_token, tokenizer):
    """Debug a single generation step."""
    print_subsection(f"Generation Step {step}")
    
    print_tensor_info(input_ids, "Input IDs")
    print_tensor_info(logits, "Logits")
    
    # Show top-k predictions
    top_k = 5
    top_probs, top_indices = torch.topk(torch.softmax(logits[0], dim=-1), top_k)
    
    print(f"\n📈 Top {top_k} predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token_str = tokenizer.decode([idx.item()])
        print(f"   {i+1}. '{token_str}' (token {idx.item()}) - {prob.item():.4f}")
    
    selected_token_str = tokenizer.decode([next_token.item()])
    print(f"\n🎯 Selected token: '{selected_token_str}' (token {next_token.item()})")
    
    # Show generated sequence so far
    current_seq = tokenizer.decode(input_ids[0])
    print(f"\n📝 Current sequence: {current_seq}")


@torch.no_grad()
def debug_flex_attention_generation(prompts, tokenizer, model, max_new_tokens=5):
    """Debug version of flex_attention_generation with detailed output."""
    print_section("FlexAttention Generation Debug")
    
    # Configure model
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    
    # Step 1: Concatenate
    concatenated_text, segment_positions, original_length = debug_concatenation(
        prompts, tokenizer
    )
    
    # Step 2: Create mask
    mask_mod = debug_mask_creation(segment_positions, original_length)
    
    # Tokenize
    print_section("Step 3: Tokenization")
    inputs = tokenizer(
        concatenated_text,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True
    ).to(model.device)
    
    print_tensor_info(inputs["input_ids"], "Tokenized Input")
    
    # Generation loop
    print_section("Step 4: Auto-regressive Generation")
    
    generated = None
    
    for step in range(max_new_tokens):
        current_length = inputs["input_ids"].shape[1]
        
        print(f"\n{'─'*70}")
        print(f"Generation Step {step + 1}/{max_new_tokens}")
        print(f"{'─'*70}")
        print(f"Current sequence length: {current_length}")
        
        # Show current mask for this length
        if step == 0 or step == max_new_tokens - 1:
            # Only visualize mask on first and last step to avoid clutter
            mask_mod = debug_mask_creation(segment_positions, original_length, current_length)
        
        # Forward pass
        try:
            logits = model(inputs["input_ids"]).logits[:, -1, :]
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            break
        
        # Select next token
        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
        
        # Debug info
        debug_generation_step(step + 1, inputs["input_ids"], logits, next_token, tokenizer)
        
        # Update inputs
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        
        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)
        
        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            print(f"\n🛑 EOS token generated, stopping")
            break
    
    # Final output
    print_section("Final Output")
    
    generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    full_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    
    print(f"\nGenerated tokens: {generated.shape[1]}")
    print(f"Generated text: '{generated_text}'")
    print(f"\nFull sequence: {full_text}")
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Debug FlexAttention generation with detailed output"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["webqa", "myriadlama"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--model", type=str, default="llama3.2_3b_it",
        help="Model name from constants.MODEL_PATHs"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for model (default: cuda)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=1,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=5,
        help="Maximum tokens to generate per sample"
    )
    parser.add_argument(
        "--indexs", type=str, default=None,
        help="Specific paraphrase indices (comma-separated)"
    )
    parser.add_argument(
        "--num-paraphrases", type=int, default=5,
        help="Number of paraphrases to use"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()
    
    print("="*70)
    print("🐛 FlexAttention Debug Mode")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples}")
    print(f"Max tokens: {args.max_tokens}")
    
    # Load dataset
    print_section("Loading Dataset")
    if args.dataset == "webqa":
        from dataset import WebQADataset
        dataset = WebQADataset(model_name=args.model)
    elif args.dataset == "myriadlama":
        from dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name=args.model)
    else:
        raise ValueError("Unsupported dataset")
    
    print(f"✅ Dataset loaded: {args.dataset}")
    
    # Load model
    print_section("Loading Model")
    if args.model not in MODEL_PATHs:
        raise ValueError(f"Model {args.model} not in MODEL_PATHs")
    
    model_path = MODEL_PATHs[args.model]
    print(f"Model path: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=args.device, torch_dtype="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded: {args.model}")
    print(f"   Device: {next(model.parameters()).device}")
    
    # Get data
    print_section("Preparing Data")
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
    few_shot_context = dataset.get_few_shot_examples()
    
    # Process samples
    sample_count = 0
    for uuids, answers, all_paraphrases in dataloader:
        if sample_count >= args.max_samples:
            break
        
        sample_count += 1
        print(f"\n{'='*70}")
        print(f"Sample {sample_count}/{args.max_samples}")
        print(f"{'='*70}")
        print(f"UUID: {uuids[0]}")
        print(f"Answer: {answers[0]}")
        
        # Select paraphrases
        if args.indexs is not None:
            indices = [int(idx) for idx in args.indexs.split(",")]
            selected = [all_paraphrases[idx][0] for idx in indices]
        else:
            selected = [p[0] for p in all_paraphrases[:args.num_paraphrases]]
        
        print(f"\nUsing {len(selected)} paraphrases")
        
        # Construct prompts
        prompts = []
        for paraphrase in selected:
            prompt = dataset.construct_prompts(few_shot_context, [paraphrase])
            prompts.append(prompt[0])
        
        # Debug generation
        result = debug_flex_attention_generation(
            prompts, tokenizer, model, max_new_tokens=args.max_tokens
        )
        
        print(f"\n{'='*70}")
        print(f"Sample {sample_count} Complete")
        print(f"{'='*70}")
    
    print(f"\n✅ Debug session complete!")
    print(f"   Processed {sample_count} sample(s)")


if __name__ == "__main__":
    main()
