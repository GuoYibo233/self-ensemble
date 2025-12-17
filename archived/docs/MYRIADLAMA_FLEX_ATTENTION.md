# MyriadLAMA FlexAttention Implementation

## Overview

This document describes the MyriadLAMA-specific FlexAttention implementation in `myriadlama_flex_attention_generate.py`. This is a specialized version of the FlexAttention approach, tailored for the unique characteristics of the MyriadLAMA dataset.

## Key Differences from `flex_attention_generate.py`

### 1. Task Structure
- **MyriadLAMA**: Fill-in-the-blank task with `[MASK]` token
- **WebQA**: Question-answering task
- **Impact**: Shorter generation (10 tokens vs 20), different prompt format

### 2. Template Organization
MyriadLAMA provides two types of templates:
- **Manual templates**: Human-written, high-quality templates (variable count)
- **Auto templates**: Automatically generated templates (typically 5)

### 3. Modified Prompt Construction

#### Standard Approach (flex_attention_generate.py)
```python
prompts = dataset.construct_prompts(few_shot_context, paraphrases)
```

#### MyriadLAMA Approach (myriadlama_flex_attention_generate.py)
```python
# Separate manual and auto templates
manual_templates = all_templates[:num_manual]
auto_templates = all_templates[num_manual:num_manual + num_auto]
selected_templates = manual_templates + auto_templates

# Construct prompts
prompts = [dataset.construct_prompts(few_shot_context, [t])[0] 
           for t in selected_templates]
```

### 4. Modified Mask Logic

#### Standard Mask
```python
def create_flex_attention_mask(segment_positions, original_length):
    """Each segment isolated during encoding, full attention during generation"""
    def mask_mod(b, h, q_idx, kv_idx):
        same_segment = (q_in_segment & kv_in_segment).any()
        result = causal_mask & (is_generated | same_segment)
        return result
    return mask_mod
```

#### MyriadLAMA Mask
```python
def create_myriadlama_mask(segment_positions, original_length, manual_count=None):
    """
    Enhanced mask with optional cross-attention for manual templates
    """
    def mask_mod(b, h, q_idx, kv_idx):
        same_segment = (q_in_segment & kv_in_segment).any()
        
        # NEW: Allow manual templates to attend to each other
        if manual_count > 0:
            q_in_manual = q_idx < seg_ends[manual_count - 1]
            kv_in_manual = kv_idx < seg_ends[manual_count - 1]
            within_manual_group = q_in_manual & kv_in_manual
        else:
            within_manual_group = False
        
        # Enhanced result with manual cross-attention
        result = causal_mask & (is_generated | same_segment | within_manual_group)
        return result
    return mask_mod
```

## Rationale for Manual Cross-Attention

### Why Allow Manual Templates to Attend to Each Other?

1. **Quality**: Manual templates are human-written and high-quality
2. **Semantic Similarity**: They express the same relation in different ways
3. **Information Sharing**: Cross-attention allows templates to learn from each other's context
4. **Better Encoding**: Richer representations lead to better fusion during generation

### Example

For the relation "capital_of", manual templates might include:
- "Paris is the capital of [MASK]"
- "The capital of [MASK] is Paris"
- "[MASK]'s capital city is Paris"

These templates benefit from attending to each other because:
- They share semantic structure
- They contain the same entity (Paris)
- They express the same fact differently

## Architecture

```
Input Templates:
┌──────────────────────────────────────┐
│ Manual Templates (human-written)     │
│ - Template 1                         │
│ - Template 2                         │
│ - Template 3                         │
├──────────────────────────────────────┤
│ Auto Templates (generated)           │
│ - Template 4                         │
│ - Template 5                         │
└──────────────────────────────────────┘
           ↓
    Concatenation
           ↓
┌──────────────────────────────────────┐
│ Concatenated Sequence                │
│ [T1] [SEP] [T2] [SEP] [T3] [SEP] ... │
└──────────────────────────────────────┘
           ↓
  FlexAttention Mask
           ↓
┌──────────────────────────────────────┐
│ Encoding Phase:                      │
│ - Manual templates: cross-attention  │
│ - Auto templates: self-attention     │
│ - No attention between groups        │
└──────────────────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│ Generation Phase:                    │
│ - New tokens attend to all templates │
│ - Fusion of information              │
└──────────────────────────────────────┘
```

## Mask Visualization

### Standard Mask (flex_attention_generate.py)
```
Encoding Phase (5 templates, no cross-attention):
     T1  T2  T3  T4  T5
T1 [ ✓  ✗  ✗  ✗  ✗ ]  Each template
T2 [ ✗  ✓  ✗  ✗  ✗ ]  attends only
T3 [ ✗  ✗  ✓  ✗  ✗ ]  to itself
T4 [ ✗  ✗  ✗  ✓  ✗ ]
T5 [ ✗  ✗  ✗  ✗  ✓ ]

Generation Phase:
     T1  T2  T3  T4  T5  G1  G2
G1 [ ✓  ✓  ✓  ✓  ✓  ✗  ✗ ]  Generated tokens
G2 [ ✓  ✓  ✓  ✓  ✓  ✓  ✗ ]  attend to all
```

### MyriadLAMA Mask (with manual_count=3)
```
Encoding Phase (3 manual + 2 auto templates):
     M1  M2  M3  A1  A2
M1 [ ✓  ✓  ✓  ✗  ✗ ]  Manual templates
M2 [ ✓  ✓  ✓  ✗  ✗ ]  attend to each other
M3 [ ✓  ✓  ✓  ✗  ✗ ]
A1 [ ✗  ✗  ✗  ✓  ✗ ]  Auto templates
A2 [ ✗  ✗  ✗  ✗  ✓ ]  isolated

Generation Phase (same as standard):
     M1  M2  M3  A1  A2  G1  G2
G1 [ ✓  ✓  ✓  ✓  ✓  ✗  ✗ ]  Generated tokens
G2 [ ✓  ✓  ✓  ✓  ✓  ✓  ✗ ]  attend to all
```

## Usage Examples

### Basic Usage
```bash
# Generate with all manual templates and 5 auto templates
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --device auto
```

### With Manual Cross-Attention
```bash
# Enable cross-attention between manual templates
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --allow_manual_cross_attention \
    --device auto
```

### Custom Template Configuration
```bash
# Use specific number of manual and auto templates
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --num_manual 3 \
    --num_auto 2 \
    --allow_manual_cross_attention \
    --device auto
```

### Limited Sample Generation (for testing)
```bash
# Generate only 100 samples for quick testing
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --max_samples 100 \
    --device auto
```

### Lemmatization
```bash
# First generate, then lemmatize
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --device auto

python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --lemmaize
```

## Output Files

Files are saved in `/net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/{model}/`:

- `myriadlama_flex_all_a5.feather`: All manual + 5 auto templates
- `myriadlama_flex_m3_a5.feather`: 3 manual + 5 auto templates
- `myriadlama_flex_all_a5_xattn.feather`: With manual cross-attention enabled

## Output Format

The generated DataFrame contains:
- `uuid`: Unique identifier for the question
- `templates`: List of templates used for this question
- `answers`: Ground truth answers
- `prediction`: Predicted answer (first word)
- `generation`: Full generated text
- `predict_lemma`: Lemmatized prediction (after --lemmaize)
- `answer_lemmas`: Lemmatized answers (after --lemmaize)

## Implementation Details

### 1. Shorter Generation
```python
max_new_tokens = 10  # vs 20 for WebQA
```
MyriadLAMA requires one-word answers, so we generate fewer tokens.

### 2. Early Stopping
```python
# Stop on newline (end of word)
if '\n' in decoded and step > 0:
    break
```

### 3. Template Separation
```python
# Separate manual and auto templates
if len(all_templates) > 5:
    manual_templates = all_templates[:-5]
    auto_templates = all_templates[-5:]
```

### 4. First Word Extraction
```python
# Extract first word as prediction
prediction = generation.strip().split()[0]
```

## Performance Considerations

### Sequential Processing
Like `flex_attention_generate.py`, this implementation processes samples sequentially (batch_size=1) due to variable-length concatenation.

### Expected GPU Utilization
- **Low utilization (5-15%)**: Due to sequential processing
- **Normal**: This is expected for FlexAttention with dynamic masks

### Memory Usage
- **Lower than WebQA**: Shorter templates and generation length
- **Can handle larger batches**: If batching is implemented in the future

## Comparison with Other Methods

| Aspect | Origin | Per-Prompt | Ensemble (avg/max) | Flex (WebQA) | Flex (MyriadLAMA) |
|--------|--------|------------|-------------------|--------------|-------------------|
| **Templates Used** | 1 (original) | 1 per inference | Multiple | Multiple | Manual + Auto |
| **Fusion Method** | None | None | Logit combination | Attention-based | Attention-based |
| **Template Types** | N/A | All equal | All equal | All equal | Manual vs Auto |
| **Cross-Attention** | N/A | N/A | N/A | No | Optional (manual) |
| **Task Specificity** | General | General | General | General | MyriadLAMA-specific |

## Validation

To verify the implementation works correctly:

```bash
# 1. Test with small sample
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --max_samples 10 \
    --device auto

# 2. Check output file
python -c "
import pandas as pd
df = pd.read_feather('/net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/llama3.2_3b_it/myriadlama_flex_all_a5.feather')
print(df.head())
print(f'Columns: {df.columns.tolist()}')
print(f'Total samples: {len(df)}')
"

# 3. Verify predictions are one-word
python -c "
import pandas as pd
df = pd.read_feather('/net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/llama3.2_3b_it/myriadlama_flex_all_a5.feather')
multi_word = df[df['prediction'].str.contains(' ', na=False)]
print(f'Multi-word predictions: {len(multi_word)} / {len(df)}')
"
```

## Future Improvements

1. **Batched Generation**: Implement padding to allow batch processing
2. **Weighted Manual Templates**: Use confidence scores for manual templates
3. **Adaptive Template Selection**: Dynamically choose best templates
4. **Relation-Aware Masking**: Different masks for different relation types
5. **Hybrid Fusion**: Combine attention-based and logit-based fusion

## References

- Base implementation: `flex_attention_generate.py`
- Dataset: `dataset.py` (MyriadLamaDataset class)
- FlexAttention docs: `docs/FLEX_ATTENTION_IMPLEMENTATION.md`
