"""
FlexAttention-based ensemble generation.

This script implements a new ensemble method that concatenates multiple paraphrases
and uses FlexAttention to isolate attention within each paraphrase during encoding,
then allows fusion during generation.

Key features:
- Concatenates 5 paraphrases into a single prompt with position tracking
- Uses FlexAttention to prevent cross-paraphrase attention during encoding
- Allows generated tokens to attend to all previous content for fusion
- Reuses functions and patterns from generate.py where possible
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
# NEW - Paraphrase concatenation with position tracking
# ==============================================================================

def concatenate_paraphrases_with_positions(prompts, tokenizer, separator="\n\n[SEP]\n\n"):
    """
    Concatenate multiple prompts and track token positions for each segment.
    
    Args:
        prompts: List of prompt strings (paraphrases)
        tokenizer: HuggingFace tokenizer
        separator: Separator token between prompts (default: newlines around [SEP])
        
    Returns:
        concatenated_text: Single concatenated string
        segment_positions: List of (start, end) tuples for each paraphrase
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
    
    # Decode back to text for verification
    concatenated_text = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    return concatenated_text, segment_positions, len(full_tokens)

# ==============================================================================
# NEW - FlexAttention mask creation
# ==============================================================================

def create_segment_isolation_mask(segment_positions, original_length):
    """
    Create a mask function for FlexAttention that isolates paraphrase segments.
    
    During encoding (original sequence):
    - Each paraphrase token can only attend to tokens in its own segment
    
    During generation (beyond original_length):
    - New tokens can attend to all previous tokens (enables fusion)
    
    Args:
        segment_positions: List of (start, end) tuples
        original_length: Length of the original concatenated sequence
        
    Returns:
        mask_mod: Function (b, h, q_idx, kv_idx) -> bool
    """
    def mask_mod(b, h, q_idx, kv_idx):
        # Rule 1: Causal constraint - cannot attend to future tokens
        if q_idx < kv_idx:
            return False
        
        # Rule 2: Generated tokens (beyond original) can attend to everything before them
        if q_idx >= original_length:
            return True
        
        # Rule 3: Original tokens can only attend within their segment
        q_segment = None
        kv_segment = None
        
        # Find which segment q_idx belongs to
        for seg_id, (start, end) in enumerate(segment_positions):
            if start <= q_idx < end:
                q_segment = seg_id
                break
        
        # Find which segment kv_idx belongs to
        for seg_id, (start, end) in enumerate(segment_positions):
            if start <= kv_idx < end:
                kv_segment = seg_id
                break
        
        # Both must be in valid segments and the same segment
        if q_segment is not None and kv_segment is not None:
            return q_segment == kv_segment
        
        # Default: don't allow attention
        return False
    
    return mask_mod

# ==============================================================================
# NEW - FlexAttention-based generation with monkey patching
# ==============================================================================

class FlexAttentionWrapper:
    """
    Wrapper that patches model attention layers to use FlexAttention.
    Based on patterns from attention-gym repository.
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
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs
        ):
            # If no custom mask or sequence is too short, use original
            bsz, q_len, _ = hidden_states.size()
            if self.current_mask_mod is None or q_len == 1:
                return self.original_forwards[layer_idx](
                    hidden_states, attention_mask, position_ids,
                    past_key_value, output_attentions, use_cache,
                    cache_position, **kwargs
                )
            
            # Compute Q, K, V projections
            query_states = original_attn.q_proj(hidden_states)
            key_states = original_attn.k_proj(hidden_states)
            value_states = original_attn.v_proj(hidden_states)
            
            # Reshape to multi-head format
            num_heads = original_attn.num_heads
            head_dim = original_attn.head_dim
            
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            
            # Apply rotary position embeddings if available
            if hasattr(original_attn, 'rotary_emb'):
                cos, sin = original_attn.rotary_emb(value_states, position_ids)
                query_states, key_states = original_attn.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
            
            # Create block mask for FlexAttention
            try:
                block_mask = create_block_mask(
                    self.current_mask_mod,
                    B=bsz,
                    H=num_heads, 
                    Q_LEN=q_len,
                    KV_LEN=q_len,
                    device=query_states.device
                )
                
                # Use FlexAttention
                attn_output = flex_attention(
                    query_states,
                    key_states,
                    value_states,
                    block_mask=block_mask
                )
            except Exception as e:
                # Fallback to standard SDPA
                print(f"⚠️  FlexAttention failed in layer {layer_idx}: {e}")
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    is_causal=True
                )
            
            # Reshape output
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
            
            # Output projection
            attn_output = original_attn.o_proj(attn_output)
            
            return attn_output, None, past_key_value
        
        return patched_forward
    
    def patch_model(self, mask_mod):
        """Patch all attention layers with FlexAttention."""
        if self.is_patched:
            self.unpatch_model()
        
        self.current_mask_mod = mask_mod
        
        for i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            self.original_forwards[i] = attn.forward
            attn.forward = self.create_patched_forward(i, attn).__get__(attn)
        
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
# NEW - Main generation function using FlexAttention
# ==============================================================================

@torch.no_grad()
def flex_attention_generation(prompts, max_new_tokens=20):
    """
    Generate text using FlexAttention-based ensemble.
    
    This is the NEW implementation that:
    1. Concatenates prompts with position tracking
    2. Uses FlexAttention for segment isolation
    3. Generates tokens that can attend to all segments
    
    Args:
        prompts: List of paraphrase prompts (typically 5)
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated text string
    """
    # Reused pattern: Set model config (from generate.py single_generation)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    # NEW: Concatenate paraphrases with position tracking
    concatenated_text, segment_positions, original_length = \
        concatenate_paraphrases_with_positions(prompts, tokenizer)
    
    print(f"  Concatenated {len(prompts)} prompts into {original_length} tokens")
    print(f"  Segment positions: {segment_positions}")
    
    # Reused pattern: Tokenize input (from generate.py)
    inputs = tokenizer(
        concatenated_text,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True
    ).to(model.device)
    
    # NEW: Create FlexAttention wrapper
    flex_wrapper = FlexAttentionWrapper(model)
    
    generated = None
    
    # Reused pattern: Generation loop (from generate.py single_generation)
    for step in range(max_new_tokens):
        current_length = inputs["input_ids"].shape[1]
        
        # NEW: Create mask for current sequence length
        mask_mod = create_segment_isolation_mask(segment_positions, original_length)
        
        # NEW: Patch model with FlexAttention
        flex_wrapper.patch_model(mask_mod)
        
        try:
            # Reused pattern: Forward pass (from generate.py)
            logits = model(inputs["input_ids"]).logits[:, -1, :]
        except Exception as e:
            print(f"⚠️  Generation step {step} failed: {e}")
            # Fallback to unpatchedmodel
            flex_wrapper.unpatch_model()
            logits = model(inputs["input_ids"]).logits[:, -1, :]
        finally:
            # NEW: Always unpatch after each step to avoid state issues
            flex_wrapper.unpatch_model()
        
        # Reused pattern: Token selection and update (from generate.py)
        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        
        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)
        
        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Reused pattern: Decode output (from generate.py)
    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return generated_texts[0].strip()

# ==============================================================================
# Main script
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    # Reused pattern: Argument parser (from generate.py)
    parser = argparse.ArgumentParser(
        description="FlexAttention-based ensemble generation"
    )
    parser.add_argument(
        "--method", type=str, default="flex_attention",
        help="Generation method (always flex_attention for this script)"
    )
    parser.add_argument(
        "--model", type=str, default="llama3.2_3b_it",
        help="Model name from constants.MODEL_PATHs"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["webqa", "myriadlama"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for model (default: cuda)"
    )
    parser.add_argument(
        "--lemmaize", action="store_true",
        help="Normalize predictions and answers to lemmas"
    )
    parser.add_argument(
        "--indexs", type=str, default=None,
        help="Specific paraphrase indices to use (comma-separated)"
    )
    parser.add_argument(
        "--num_paraphrases", type=int, default=5,
        help="Number of paraphrases to concatenate (default: 5)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to generate (default: None, process all)"
    )
    args = parser.parse_args()
    
    # Check FlexAttention availability
    if not FLEX_ATTENTION_AVAILABLE:
        print("❌ FlexAttention is required for this script.")
        print("   Please install PyTorch 2.5+ or nightly:")
        print("   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121")
        exit(1)
    
    # Reused pattern: Dataset loading (from generate.py)
    if args.dataset == "webqa":
        from dataset import WebQADataset
        dataset = WebQADataset(model_name=args.model)
    elif args.dataset == "myriadlama":
        from dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name=args.model)
    else:
        raise ValueError("Unsupported dataset")
    
    # Reused pattern: Dataloader setup (from generate.py)
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    
    if args.model not in MODEL_PATHs:
        raise ValueError(
            f"Model {args.model} not supported. "
            f"Choose from {list(MODEL_PATHs.keys())}"
        )
    
    model_path = MODEL_PATHs.get(args.model, args.model)
    
    # Reused pattern: Output file setup (from generate.py)
    if args.indexs is not None:
        _root = os.path.join(dataset.dataset_root, "diversity")
        os.makedirs(_root, exist_ok=True)
        dump_file = f"{_root}/flex_attention-{args.indexs}.feather"
    else:
        dump_file = f"{dataset.dataset_root}/flex_attention-{args.num_paraphrases}.feather"
    
    print(f"Output file: {dump_file}")
    
    # Reused pattern: Lemmatization mode (from generate.py)
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
    
    # Reused pattern: Model loading (from generate.py)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=args.device, torch_dtype="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Print model info
    print(f"🔍 Model: {args.model}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Using FlexAttention for segment isolation")
    if hasattr(model.config, 'num_attention_heads'):
        print(f"   Attention heads: {model.config.num_attention_heads}")
    
    # Reused pattern: DataFrame initialization (from generate.py)
    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation"])
    print(f"FlexAttention generation with {args.num_paraphrases} paraphrases")
    if args.max_samples:
        print(f"Processing maximum {args.max_samples} samples")
    
    # Reused pattern: Few-shot context (from generate.py)
    few_shot_context = dataset.get_few_shot_examples()
    
    # Main generation loop
    sample_count = 0
    for uuids, answers, all_paraphrases in tqdm(dataloader):
        # Reused pattern: Paraphrase selection (from generate.py)
        if args.indexs is not None:
            selected_paraphrases = [
                all_paraphrases[int(idx)] for idx in args.indexs.split(",")
            ]
        else:
            selected_paraphrases = all_paraphrases[:args.num_paraphrases]
        
        batch_predictions = []
        batch_generations = []
        
        # Process each question in batch
        for i, paraphrases in enumerate(zip(*selected_paraphrases)):
            # Reused pattern: Construct prompts (from generate.py)
            prompts = []
            for paraphrase in paraphrases:
                prompt = dataset.construct_prompts(few_shot_context, [paraphrase])
                prompts.append(prompt[0])
            
            # NEW: Use FlexAttention generation
            generation = flex_attention_generation(prompts, max_new_tokens=20)
            
            # Reused pattern: Extract prediction (from generate.py)
            prediction = generation.strip().split('\n')[0]
            
            batch_predictions.append(prediction)
            batch_generations.append(generation)
        
        # Reused pattern: Store results (from generate.py)
        items = {
            "uuid": uuids,
            "paraphrases": list(zip(*selected_paraphrases)),
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
    
    # Reused pattern: Save results (from generate.py)
    df.to_feather(dump_file)
    print(f"✅ Results saved to {dump_file}")
