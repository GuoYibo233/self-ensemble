# FlexAttention Implementation Summary

## Overview
This document explains the implementation of `flex_attention_generate.py` and details which components are reused from `generate.py` and which are new.

## Architecture

### 1. Paraphrase Processing
- **Input**: 5 paraphrases per question
- **Process**: Concatenate with `[SEP]` separator while tracking token positions
- **Output**: Single prompt with segment position metadata

### 2. Attention Mechanism
- **During Encoding**: Each paraphrase can only attend to itself (segment isolation)
- **During Generation**: New tokens can attend to all previous tokens (enables fusion)

### 3. Generation Process
- Use FlexAttention with custom mask during forward pass
- Generate tokens auto-regressively
- Unpatch model after each step to avoid state accumulation

## Reused Components from `generate.py`

### ✅ Fully Reused Functions (No Modification)

1. **Lemmatization Functions**
   ```python
   - init_spacy()
   - lemmaize_predicts()
   - lemmaize_chunk()
   - append_lemmas()
   ```
   - Location in original: Lines 19-44
   - Purpose: Normalize predictions and answers for evaluation

2. **Model Configuration Pattern**
   ```python
   tokenizer.pad_token_id = tokenizer.eos_token_id
   model.generation_config.temperature = None
   model.generation_config.top_p = None
   ```
   - Location in original: Lines 47-50, 78-81
   - Purpose: Set deterministic generation parameters

3. **Tokenization Pattern**
   ```python
   inputs = tokenizer(
       text, return_tensors="pt",
       truncation=True, add_special_tokens=True
   ).to(model.device)
   ```
   - Location in original: Lines 52-55
   - Purpose: Convert text to model inputs

4. **Generation Loop Structure**
   ```python
   for step in range(max_new_tokens):
       logits = model(inputs["input_ids"]).logits[:, -1, :]
       next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
       inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
   ```
   - Location in original: Lines 59-70
   - Purpose: Auto-regressive generation

5. **Dataset Loading**
   ```python
   if args.dataset == "webqa":
       dataset = WebQADataset(model_name=args.model)
   elif args.dataset == "myriadlama":
       dataset = MyriadLamaDataset(model_name=args.model)
   ```
   - Location in original: Lines 157-164
   - Purpose: Load appropriate dataset

6. **Dataloader Setup**
   ```python
   dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
   ```
   - Location in original: Line 167
   - Purpose: Batch processing

7. **Model Loading**
   ```python
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   model = AutoModelForCausalLM.from_pretrained(
       model_path, device_map=args.device, torch_dtype="auto"
   )
   tokenizer.pad_token = tokenizer.eos_token
   ```
   - Location in original: Lines 178-180, 240-242
   - Purpose: Initialize model and tokenizer

8. **Output File Management**
   ```python
   if args.indexs is not None:
       dump_file = f"{_root}/method-{args.indexs}.feather"
   else:
       dump_file = f"{dataset.dataset_root}/method-{args.num}.feather"
   ```
   - Location in original: Lines 212-217
   - Purpose: Determine output path

9. **Paraphrase Selection**
   ```python
   if args.indexs is not None:
       selected = [all_paraphrases[int(idx)] for idx in args.indexs.split(",")]
   else:
       selected = all_paraphrases[:args.num_paraphrases]
   ```
   - Location in original: Lines 251-255
   - Purpose: Select which paraphrases to use

10. **Prompt Construction**
    ```python
    few_shot_context = dataset.get_few_shot_examples()
    prompts = dataset.construct_prompts(few_shot_context, paraphrases)
    ```
    - Location in original: Lines 190, 193, 249, 267
    - Purpose: Build prompts with few-shot examples

11. **Result Storage**
    ```python
    items = {
        "uuid": uuids,
        "paraphrases": list(zip(*selected_paraphrases)),
        "answers": answers,
        "prediction": predictions,
        "generation": generations,
    }
    df = pd.concat([df, pd.DataFrame(items)], ignore_ignore=True)
    ```
    - Location in original: Lines 272-279
    - Purpose: Collect results

12. **Prediction Extraction**
    ```python
    prediction = generation.strip().split('\n')[0]
    ```
    - Location in original: Lines 195, 270
    - Purpose: Extract answer from generation

13. **DataFrame Saving**
    ```python
    df.to_feather(dump_file)
    ```
    - Location in original: Line 288
    - Purpose: Save results

### ✅ Reused Patterns (Adapted)

1. **Argument Parser Structure**
   - Similar arguments: `--method`, `--model`, `--dataset`, `--device`, `--lemmaize`, `--indexs`
   - New argument: `--num_paraphrases` (replaces `--num_ensemble`)
   - Purpose: Command-line interface

2. **Main Loop Structure**
   ```python
   for uuids, answers, all_paraphrases in tqdm(dataloader):
       # Process each batch
   ```
   - Location in original: Lines 183, 250
   - Purpose: Batch processing iteration

## New Components

### 🆕 New Functions

1. **`concatenate_paraphrases_with_positions()`**
   - Purpose: Concatenate multiple prompts while tracking token positions
   - Why needed: FlexAttention requires knowing which tokens belong to which segment
   - Key features:
     - Tokenizes each prompt separately
     - Adds separator tokens between prompts
     - Returns (text, positions, total_length)

2. **`create_segment_isolation_mask()`**
   - Purpose: Create mask function for FlexAttention
   - Why needed: Implements the segment isolation logic
   - Mask rules:
     - Causal: Cannot attend to future tokens
     - Isolation: Original tokens only attend within their segment
     - Fusion: Generated tokens attend to all previous tokens

3. **`FlexAttentionWrapper` class**
   - Purpose: Monkey-patch model attention layers
   - Why needed: Integrate FlexAttention into existing model
   - Methods:
     - `patch_model()`: Replace attention with FlexAttention
     - `unpatch_model()`: Restore original attention
     - `create_patched_forward()`: Create custom forward function
   - Based on: PyTorch attention-gym repository patterns

4. **`flex_attention_generation()`**
   - Purpose: Main generation function using FlexAttention
   - Why needed: Orchestrate the new generation process
   - Flow:
     1. Concatenate prompts with position tracking
     2. Create FlexAttention wrapper
     3. For each generation step:
        - Create mask for current length
        - Patch model with mask
        - Generate next token
        - Unpatch model
     4. Return generated text

### 🆕 New Implementation Details

1. **Segment Position Tracking**
   ```python
   segment_positions = [(0, 120), (125, 245), ...]  # (start, end) for each paraphrase
   original_length = 245  # Total tokens in concatenated input
   ```

2. **Dynamic Mask Updates**
   - Mask is recreated at each generation step
   - Accounts for growing sequence length
   - Maintains segment boundaries

3. **Attention Layer Patching**
   - Intercepts attention computation
   - Applies FlexAttention with block_mask
   - Falls back to SDPA if FlexAttention fails
   - Patches/unpatches to avoid state issues

4. **Error Handling**
   - FlexAttention availability check at import
   - Try-except for mask creation
   - Fallback to standard attention if needed
   - Always unpatch in finally block

## Key Differences from `generate.py`

| Aspect | `generate.py` | `flex_attention_generate.py` |
|--------|---------------|------------------------------|
| **Input Processing** | Process paraphrases separately | Concatenate into single sequence |
| **Attention** | Standard self-attention | FlexAttention with segment isolation |
| **Fusion Method** | Logit averaging/max | Attention-based during generation |
| **Model Modification** | None | Temporary monkey-patching |
| **Complexity** | Lower | Higher (custom attention) |

## Usage Example

```bash
# Run FlexAttention generation on WebQA with 5 paraphrases
python flex_attention_generate.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5 \
    --device auto

# Run with specific paraphrase indices
python flex_attention_generate.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --indexs 0,1,2,3,4

# Lemmatize existing results
python flex_attention_generate.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5 \
    --lemmaize
```

## Requirements

- PyTorch 2.5+ or nightly build
- FlexAttention API: `torch.nn.attention.flex_attention`
- All other dependencies same as `generate.py`

## Technical Notes

1. **Why Unpatch After Each Step?**
   - Avoids accumulating state in patched functions
   - Ensures clean state for next iteration
   - Prevents memory leaks

2. **Why Segment Isolation During Encoding?**
   - Forces model to process each paraphrase independently
   - Prevents information leakage between paraphrases
   - Maintains diversity in representations

3. **Why Full Attention During Generation?**
   - Allows fusion of information from all paraphrases
   - Enables ensemble-like behavior
   - Generated tokens benefit from all input variations

4. **Why FlexAttention Instead of 4D Mask?**
   - More efficient for sparse patterns
   - Better GPU kernel optimization
   - Supports more complex mask patterns
   - Official PyTorch recommendation for custom attention

## Validation

To verify the implementation:

1. Check that segment positions are correctly computed
2. Verify mask allows intra-segment attention
3. Verify mask blocks cross-segment attention
4. Confirm generated tokens attend to all segments
5. Compare outputs with standard ensemble methods

## Future Improvements

1. Support for variable number of paraphrases per question
2. Different fusion strategies (weighted attention)
3. Cache optimizations for faster generation
4. Support for batched generation (multiple questions)
5. Integration with beam search
