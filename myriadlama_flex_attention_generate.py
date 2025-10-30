"""
MyriadLama-specific FlexAttention generation.

This script implements a modified FlexAttention-based ensemble generation specifically
designed for the MyriadLAMA dataset, with custom prompt construction and mask logic.

Key differences from flex_attention_generate.py:
- Custom prompt formatting for [MASK] token prediction
- Modified mask logic to handle template-based prompts
- Optimized for MyriadLAMA's fill-in-the-blank task structure
- Separates manual and auto-generated templates during concatenation

Features:
- Concatenates manual and auto paraphrases separately with tracking
- Uses FlexAttention to isolate attention within template groups during encoding
- Allows generated tokens to attend to all templates for fusion
- Specifically designed for one-word prediction tasks
"""

from pdb import set_trace
import os
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from constants import MODEL_PATHs

# Try to import FlexAttention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    FLEX_ATTENTION_AVAILABLE = True
    print("✅ FlexAttention is available")
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print("⚠️  FlexAttention not available. This script requires PyTorch 2.5+ or nightly.")
    print("    Install with: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121")

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

# ==============================================================================
# REUSED FROM generate.py - Lemmatization functions
# ==============================================================================

nlp = None
num_parts = 8

def init_spacy():
    """Reused from generate.py"""
    global nlp
    nlp = spacy.load("en_core_web_lg")

def lemmaize_predicts(predict):
    """Reused from generate.py"""
    global nlp
    doc = nlp(predict)
    return [token.lemma_.lower() for token in doc]

def lemmaize_chunk(chunk):
    """Reused from generate.py"""
    predict_lemmas = []
    answer_lemmas = []
    for prediction, answers in tqdm(zip(chunk["prediction"], chunk["answers"]), total=len(chunk)):
        predict_lemmas.append(lemmaize_predicts(prediction))
        answer_lemmas.append([lemmaize_predicts(ans) for ans in answers])
    return predict_lemmas, answer_lemmas

def append_lemmas(df, results):
    """Reused from generate.py"""
    all_predict_lemmas = []
    all_answer_lemmas = []
    for predict_lemmas, answer_lemmas in results:
        all_predict_lemmas.extend(predict_lemmas)
        all_answer_lemmas.extend(answer_lemmas)
    df["predict_lemma"] = pd.Series(all_predict_lemmas, dtype=object)
    df["answer_lemmas"] = pd.Series(all_answer_lemmas, dtype=object)
    return df

# ==============================================================================
# MODIFIED - MyriadLama-specific paraphrase concatenation
# ==============================================================================

def concatenate_paraphrases_with_positions(prompts, tokenizer, separator="\n\n"):
    """
    Concatenate multiple prompts with optimized separator for MyriadLAMA.
    
    Modified for MyriadLAMA:
    - Uses simpler separator (just double newline) since templates are shorter
    - Tracks manual vs auto template boundaries
    - Optimized for fill-in-the-blank task structure
    
    Args:
        prompts: List of prompt strings (template-based paraphrases)
        tokenizer: HuggingFace tokenizer
        separator: Separator token between prompts (default: double newline)
        
    Returns:
        concatenated_text: Single concatenated string
        segment_positions: List of (start, end) tuples for each template
        total_length: Total number of tokens
    """
    # Tokenize each prompt individually to get accurate lengths
    tokenized_prompts = []
    sep_tokens = tokenizer.encode(separator, add_special_tokens=False)
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tokenized_prompts.append(tokens)
    
    # Build full token sequence and track positions
    full_tokens = []
    segment_positions = []
    current_pos = 0
    
    for i, tokens in enumerate(tokenized_prompts):
        if i > 0:
            # Add separator
            full_tokens.extend(sep_tokens)
            current_pos += len(sep_tokens)
        
        start_pos = current_pos
        full_tokens.extend(tokens)
        current_pos += len(tokens)
        end_pos = current_pos
        
        segment_positions.append((start_pos, end_pos))
    
    # Decode back to text
    concatenated_text = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    return concatenated_text, segment_positions, len(full_tokens)

# ==============================================================================
# MODIFIED - MyriadLama-specific mask creation
# ==============================================================================

def create_myriadlama_mask(segment_positions, original_length, manual_count=None):
    """
    Create attention mask for MyriadLAMA template-based ensemble generation.
    
    Modified mask logic for MyriadLAMA:
    - Templates are organized as: [manual_templates, auto_templates]
    - During encoding: Each template group attends within itself
    - Optional: Allow cross-attention between manual templates (they're human-written)
    - During generation: New tokens attend to all templates for fusion
    - Optimized for short, fill-in-the-blank style templates
    
    Args:
        segment_positions: List of (start, end) tuples defining template boundaries
        original_length: Length of the original concatenated sequence
        manual_count: Number of manual templates (optional, for special handling)
        
    Returns:
        mask_mod: Function (b, h, q_idx, kv_idx) -> Tensor[bool]
    """
    import torch
    
    # Convert segment positions to tensors
    segment_starts = torch.tensor([start for start, _ in segment_positions], dtype=torch.int64)
    segment_ends = torch.tensor([end for _, end in segment_positions], dtype=torch.int64)
    num_segments = len(segment_positions)
    
    # If manual_count is provided, we can create template group boundaries
    # Manual templates: [0, manual_count)
    # Auto templates: [manual_count, num_segments)
    if manual_count is None:
        manual_count = 0  # No special handling for manual templates
    
    def mask_mod(b, h, q_idx, kv_idx):
        """
        Mask function for MyriadLAMA FlexAttention.
        
        Modified logic:
        1. Always enforce causal constraint
        2. Generated tokens (>= original_length) attend to all
        3. Within encoding phase:
           - Each template attends to itself
           - Optionally: manual templates can attend to each other
        """
        # Move segment tensors to same device as indices
        device = q_idx.device
        seg_starts = segment_starts.to(device)
        seg_ends = segment_ends.to(device)
        
        # Causal constraint - cannot attend to future
        causal_mask = q_idx >= kv_idx
        
        # If query is in generation phase, allow attention to all previous tokens
        is_generated = q_idx >= original_length
        
        # For original tokens, determine segment membership
        q_in_segment = (q_idx >= seg_starts) & (q_idx < seg_ends)
        kv_in_segment = (kv_idx >= seg_starts) & (kv_idx < seg_ends)
        
        # Same segment check
        same_segment = (q_in_segment & kv_in_segment).any()
        
        # MODIFIED: Allow manual templates to attend to each other
        # This is beneficial for MyriadLAMA as manual templates are high-quality
        # and can share information during encoding
        if manual_count > 0:
            # Check if both q and kv are in manual template region
            q_in_manual = q_idx < seg_ends[min(manual_count - 1, num_segments - 1)]
            kv_in_manual = kv_idx < seg_ends[min(manual_count - 1, num_segments - 1)]
            within_manual_group = q_in_manual & kv_in_manual
        else:
            within_manual_group = torch.tensor(False, device=device)
        
        # Combine all constraints:
        # 1. Must satisfy causal constraint
        # 2. Either: generated token OR same segment OR both in manual group
        result = causal_mask & (is_generated | same_segment | within_manual_group)
        
        return result
    
    return mask_mod

# ==============================================================================
# REUSED FROM flex_attention_generate.py - Attention wrapper
# ==============================================================================

class FlexAttentionWrapper:
    """
    Wrapper that patches model attention layers to use FlexAttention.
    Reused from flex_attention_generate.py with no modifications.
    """
    def __init__(self, model):
        self.model = model
        self.original_forwards = {}
        self.is_patched = False
        self.current_mask_mod = None
    
    def create_patched_forward(self, layer_idx, original_attn):
        """Create a patched forward function for an attention layer."""
        def patched_forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **kwargs
        ):
            # If no custom mask or sequence is too short, use original
            bsz, q_len, _ = hidden_states.size()
            if self.current_mask_mod is None or q_len == 1:
                return self.original_forwards[layer_idx](
                    hidden_states, position_embeddings, attention_mask,
                    past_key_value, cache_position, **kwargs
                )
            
            # Extract position embeddings
            cos, sin = position_embeddings
            
            # Compute Q, K, V projections
            query_states = original_attn.q_proj(hidden_states)
            key_states = original_attn.k_proj(hidden_states)
            value_states = original_attn.v_proj(hidden_states)
            
            # Reshape to multi-head format
            num_heads = original_attn.config.num_attention_heads
            num_key_value_heads = original_attn.config.num_key_value_heads
            head_dim = original_attn.head_dim
            
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            
            # Apply rotary position embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            
            # Expand key and value states for GQA
            if num_key_value_heads != num_heads:
                key_states = key_states.repeat_interleave(num_heads // num_key_value_heads, dim=1)
                value_states = value_states.repeat_interleave(num_heads // num_key_value_heads, dim=1)
            
            # Create block mask and use FlexAttention
            try:
                block_mask = create_block_mask(
                    self.current_mask_mod,
                    B=bsz,
                    H=num_heads,
                    Q_LEN=q_len,
                    KV_LEN=q_len,
                    device=query_states.device
                )
                
                attn_output = flex_attention(
                    query_states,
                    key_states,
                    value_states,
                    block_mask=block_mask
                )
            except Exception as e:
                # Fallback to standard SDPA
                import traceback
                print(f"⚠️  FlexAttention failed in layer {layer_idx}: {type(e).__name__}: {e}")
                print(f"    Falling back to standard attention")
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    is_causal=True
                )
            
            # Reshape output
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
            
            # Output projection
            attn_output = original_attn.o_proj(attn_output)
            
            return attn_output, attn_output
        
        return patched_forward
    
    def patch_model(self, mask_mod):
        """Patch all attention layers with FlexAttention."""
        if self.is_patched:
            self.unpatch_model()
        
        self.current_mask_mod = mask_mod
        
        for i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            self.original_forwards[i] = attn.forward
            attn.forward = self.create_patched_forward(i, attn)
        
        self.is_patched = True
    
    def unpatch_model(self):
        """Restore original attention implementation."""
        if not self.is_patched:
            return
        
        for i, layer in enumerate(self.model.model.layers):
            if i in self.original_forwards:
                layer.self_attn.forward = self.original_forwards[i]
        
        self.original_forwards = {}
        self.current_mask_mod = None
        self.is_patched = False

# ==============================================================================
# MODIFIED - MyriadLama-specific generation function
# ==============================================================================

@torch.no_grad()
def myriadlama_flex_generation(prompts, manual_count=0, max_new_tokens=10):
    """
    Generate text using FlexAttention for MyriadLAMA.
    
    Modified for MyriadLAMA:
    - Shorter max_new_tokens (10 instead of 20) for one-word answers
    - Uses modified mask that allows manual template cross-attention
    - Optimized for fill-in-the-blank task
    
    Args:
        prompts: List of template-based prompts
        manual_count: Number of manual templates (for special mask handling)
        max_new_tokens: Maximum tokens to generate (default: 10 for one-word answers)
        
    Returns:
        Generated text string
    """
    # Set model config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    # Concatenate templates with position tracking
    concatenated_text, segment_positions, original_length = \
        concatenate_paraphrases_with_positions(prompts, tokenizer)
    
    # Tokenize input
    inputs = tokenizer(
        concatenated_text,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True
    ).to(model.device)
    
    # Create FlexAttention wrapper
    flex_wrapper = FlexAttentionWrapper(model)
    
    generated = None
    
    # Generation loop
    for step in range(max_new_tokens):
        current_length = inputs["input_ids"].shape[1]
        
        # Create mask with MyriadLAMA-specific logic
        mask_mod = create_myriadlama_mask(
            segment_positions, 
            original_length,
            manual_count=manual_count
        )
        
        # Patch model with FlexAttention
        flex_wrapper.patch_model(mask_mod)
        
        try:
            # Forward pass
            logits = model(inputs["input_ids"]).logits[:, -1, :]
        except Exception as e:
            import traceback
            print(f"⚠️  Generation step {step} failed: {type(e).__name__}: {e}")
            print(f"    Falling back to unpatched model...")
            flex_wrapper.unpatch_model()
            logits = model(inputs["input_ids"]).logits[:, -1, :]
        finally:
            # Always unpatch after each step
            flex_wrapper.unpatch_model()
        
        # Token selection
        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        
        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)
        
        # Check for EOS or newline (likely end of one-word answer)
        if next_token.item() == tokenizer.eos_token_id:
            break
        # Also check if we generated a newline or space (end of word)
        decoded = tokenizer.decode(next_token[0], skip_special_tokens=False)
        if '\n' in decoded and step > 0:  # Allow at least one token
            break
    
    # Decode output
    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return generated_texts[0].strip()

# ==============================================================================
# Main script
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MyriadLAMA-specific FlexAttention generation"
    )
    parser.add_argument(
        "--model", type=str, default="llama3.2_3b_it",
        help="Model name from constants.MODEL_PATHs"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for model (default: auto)"
    )
    parser.add_argument(
        "--lemmaize", action="store_true",
        help="Normalize predictions and answers to lemmas"
    )
    parser.add_argument(
        "--num_manual", type=int, default=None,
        help="Number of manual templates to use (default: all available)"
    )
    parser.add_argument(
        "--num_auto", type=int, default=5,
        help="Number of auto templates to use (default: 5)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to generate (default: None, process all)"
    )
    parser.add_argument(
        "--allow_manual_cross_attention", action="store_true",
        help="Allow manual templates to attend to each other during encoding"
    )
    args = parser.parse_args()
    
    # Check FlexAttention availability
    if not FLEX_ATTENTION_AVAILABLE:
        print("❌ FlexAttention is required for this script.")
        print("   Please install PyTorch 2.5+ or nightly:")
        print("   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121")
        exit(1)
    
    # Load MyriadLAMA dataset
    from dataset import MyriadLamaDataset
    dataset = MyriadLamaDataset(model_name=args.model)
    
    # Use batch_size=1 for sequential processing
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
    
    if args.model not in MODEL_PATHs:
        raise ValueError(
            f"Model {args.model} not supported. "
            f"Choose from {list(MODEL_PATHs.keys())}"
        )
    
    model_path = MODEL_PATHs.get(args.model, args.model)
    
    # Output file setup
    local_output_dir = f"/net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/{args.model}"
    os.makedirs(local_output_dir, exist_ok=True)
    
    # Determine file name based on template configuration
    if args.num_manual is not None:
        template_config = f"m{args.num_manual}_a{args.num_auto}"
    else:
        template_config = f"all_a{args.num_auto}"
    
    if args.allow_manual_cross_attention:
        template_config += "_xattn"
    
    dump_file = f"{local_output_dir}/myriadlama_flex_{template_config}.feather"
    
    print(f"Output file: {dump_file}")
    
    # Lemmatization mode
    if args.lemmaize:
        assert os.path.exists(dump_file), \
            f"File {dump_file} does not exist. Run without --lemmaize first."
        
        df = pd.read_feather(dump_file)
        if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
            print(f"Lemmatized data already exists in {dump_file}")
            exit(0)
        
        chunks = np.array_split(df, num_parts)
        with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
            results = pool.map(lemmaize_chunk, chunks)
        
        df = append_lemmas(df, results)
        df.to_feather(dump_file)
        exit(0)
    
    if os.path.exists(dump_file):
        print(f"File {dump_file} already exists, skipping generation.")
        exit(0)
    
    # Model loading
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=args.device, torch_dtype="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Print model info
    print(f"🔍 Model: {args.model}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Using MyriadLAMA-specific FlexAttention")
    print(f"   Template configuration: {template_config}")
    if hasattr(model.config, 'num_attention_heads'):
        print(f"   Attention heads: {model.config.num_attention_heads}")
    
    # DataFrame initialization
    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation", "templates"])
    print(f"\nMyriadLAMA FlexAttention generation")
    if args.max_samples:
        print(f"Processing maximum {args.max_samples} samples")
    
    # Few-shot context
    few_shot_context = dataset.get_few_shot_examples()
    
    # Main generation loop
    sample_count = 0
    for uuids, answers, all_paraphrases in tqdm(dataloader):
        batch_predictions = []
        batch_generations = []
        batch_templates = []
        
        # Process each question in batch
        for i, paraphrases in enumerate(zip(*all_paraphrases)):
            # Separate manual and auto templates
            # According to MyriadLamaDataset.collate_fn:
            # - manual_paraphrases come first (variable count)
            # - then 5 auto_paraphrases
            
            # Count manual templates (all templates before the last 5 auto ones)
            manual_templates = []
            auto_templates = []
            
            # The paraphrases tuple contains all templates for this question
            # We need to separate them based on the original structure
            # Since we don't have direct access here, we'll use a simple heuristic:
            # - If num_manual is specified, use that
            # - Otherwise, assume first templates are manual
            
            all_templates = list(paraphrases)
            
            if args.num_manual is not None:
                manual_templates = all_templates[:args.num_manual]
                remaining_templates = all_templates[args.num_manual:]
            else:
                # Use all manual templates (typically the first few)
                # For MyriadLAMA, manual templates come first
                # We'll take all but the last 5 (which are auto-generated)
                if len(all_templates) > 5:
                    manual_templates = all_templates[:-5]
                    remaining_templates = all_templates[-5:]
                else:
                    manual_templates = []
                    remaining_templates = all_templates
            
            # Select auto templates
            auto_templates = remaining_templates[:args.num_auto]
            
            # Combine templates
            selected_templates = manual_templates + auto_templates
            manual_count = len(manual_templates) if args.allow_manual_cross_attention else 0
            
            # Construct prompts for each template
            prompts = []
            for template in selected_templates:
                prompt = dataset.construct_prompts(few_shot_context, [template])
                prompts.append(prompt[0])
            
            # Generate using MyriadLAMA-specific FlexAttention
            generation = myriadlama_flex_generation(
                prompts,
                manual_count=manual_count,
                max_new_tokens=10
            )
            
            # Extract prediction (first word only for MyriadLAMA)
            prediction = generation.strip().split()[0] if generation.strip() else ""
            
            batch_predictions.append(prediction)
            batch_generations.append(generation)
            batch_templates.append(selected_templates)
        
        # Store results
        items = {
            "uuid": uuids,
            "templates": batch_templates,
            "answers": answers,
            "prediction": batch_predictions,
            "generation": batch_generations,
        }
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)
        
        # Check if we've reached max_samples
        sample_count += len(uuids)
        if args.max_samples and sample_count >= args.max_samples:
            print(f"Reached max_samples limit ({args.max_samples}), stopping generation")
            break
    
    # Save results
    df.to_feather(dump_file)
    print(f"✅ Results saved to {dump_file}")
