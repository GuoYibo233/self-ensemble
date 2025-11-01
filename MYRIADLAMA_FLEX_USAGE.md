# MyriadLAMA FlexAttention Usage Guide

## Quick Start

### Basic Usage

Generate predictions for MyriadLAMA using FlexAttention with all manual templates and 5 auto templates:

```bash
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --device auto
```

### Enable Manual Cross-Attention

Allow manual templates to share information during encoding:

```bash
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --allow_manual_cross_attention \
    --device auto
```

### Custom Template Configuration

Specify exactly which templates to use:

```bash
# Use 3 manual templates and 2 auto templates
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --num_manual 3 \
    --num_auto 2 \
    --allow_manual_cross_attention \
    --device auto
```

### Testing with Limited Samples

Process only 100 samples for quick testing:

```bash
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --max_samples 100 \
    --device auto
```

## Complete Workflow

### 1. Generate Predictions

```bash
# Full generation with manual cross-attention
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --allow_manual_cross_attention \
    --device auto
```

Output: `/net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/llama3.2_3b_it/myriadlama_flex_all_a5_xattn.feather`

### 2. Lemmatize Results

```bash
# Add lemmatized columns
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --allow_manual_cross_attention \
    --lemmaize
```

This updates the feather file with `predict_lemma` and `answer_lemmas` columns.

### 3. Analyze Results

```python
import pandas as pd

# Load results
df = pd.read_feather(
    '/net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/'
    'llama3.2_3b_it/myriadlama_flex_all_a5_xattn.feather'
)

# Basic statistics
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Check accuracy (if lemmatized)
if 'predict_lemma' in df.columns:
    correct = 0
    for pred_lemma, ans_lemmas in zip(df['predict_lemma'], df['answer_lemmas']):
        if any(p in ans_lemmas for p in pred_lemma):
            correct += 1
    accuracy = correct / len(df) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Sample predictions
print("\nSample predictions:")
print(df[['uuid', 'prediction', 'answers']].head())
```

## Configuration Options

### Template Selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_manual` | None (all) | Number of manual templates to use |
| `--num_auto` | 5 | Number of auto templates to use |

Examples:
- `--num_manual 3 --num_auto 2`: Use 3 manual + 2 auto templates
- `--num_auto 3`: Use all manual + 3 auto templates

### Attention Behavior

| Flag | Default | Effect |
|------|---------|--------|
| `--allow_manual_cross_attention` | False | Enable manual templates to attend to each other |

**Recommendation**: Use `--allow_manual_cross_attention` for better performance, as manual templates are high-quality and benefit from information sharing.

### Model and Device

| Argument | Default | Options |
|----------|---------|---------|
| `--model` | llama3.2_3b_it | See `constants.MODEL_PATHs` |
| `--device` | auto | `cuda`, `cpu`, `auto`, `cuda:0`, etc. |

### Processing Control

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_samples` | None (all) | Maximum samples to process |
| `--lemmaize` | False | Lemmatize existing results |

## Output Files

Files are saved in: `/net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/{model}/`

### File Naming Convention

- `myriadlama_flex_all_a5.feather`: All manual + 5 auto, no cross-attention
- `myriadlama_flex_all_a5_xattn.feather`: All manual + 5 auto, with cross-attention
- `myriadlama_flex_m3_a2.feather`: 3 manual + 2 auto, no cross-attention
- `myriadlama_flex_m3_a2_xattn.feather`: 3 manual + 2 auto, with cross-attention

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | str | Unique identifier for the question |
| `templates` | list | Templates used for this question |
| `answers` | list | Ground truth answers (aliases) |
| `prediction` | str | Predicted answer (first word) |
| `generation` | str | Full generated text |
| `predict_lemma` | list | Lemmatized prediction (after --lemmaize) |
| `answer_lemmas` | list | Lemmatized answers (after --lemmaize) |

## Comparison with Other Methods

### Example: Generate with Different Configurations

```bash
# 1. Standard FlexAttention (no cross-attention)
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --device auto

# 2. With manual cross-attention
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --allow_manual_cross_attention \
    --device auto

# 3. Limited templates
python myriadlama_flex_attention_generate.py \
    --model llama3.2_3b_it \
    --num_manual 2 \
    --num_auto 3 \
    --device auto
```

### Compare Results

```python
import pandas as pd

# Load different configurations
df_no_xattn = pd.read_feather('.../myriadlama_flex_all_a5.feather')
df_xattn = pd.read_feather('.../myriadlama_flex_all_a5_xattn.feather')

# Compare predictions
diff = df_no_xattn['prediction'] != df_xattn['prediction']
print(f"Predictions differ: {diff.sum()} / {len(df_no_xattn)}")

# Show different predictions
different = df_no_xattn[diff][['uuid', 'prediction', 'answers']]
different['prediction_xattn'] = df_xattn[diff]['prediction'].values
print(different.head())
```

## Performance Expectations

### Processing Speed

- **Sequential processing**: One sample at a time (due to variable-length concatenation)
- **GPU utilization**: Low (5-15%) - this is expected
- **Time estimate**: ~1-2 hours for full MyriadLAMA test set (2000 samples)

### Memory Usage

- **Model**: ~6-8 GB for llama3.2_3b_it
- **Intermediate**: ~2-4 GB for FlexAttention masks
- **Total**: ~10-12 GB GPU memory

### Accuracy

Expected improvements over baseline:
- **vs Origin (single question)**: +5-10%
- **vs Per-Prompt (average)**: +2-5%
- **vs Standard ensemble (logit avg/max)**: +1-3%

Manual cross-attention typically adds +1-2% over standard FlexAttention.

## Troubleshooting

### PyTorch Version Error

```
ImportError: cannot import name 'flex_attention'
```

**Solution**: Install PyTorch 2.5+ or nightly:
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce number of templates: `--num_manual 2 --num_auto 2`
2. Use CPU: `--device cpu` (slower but works)
3. Use smaller model (if available)

### File Already Exists

```
File {dump_file} already exists, skipping generation.
```

**Solution**: Delete the existing file to regenerate:
```bash
rm /net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/llama3.2_3b_it/myriadlama_flex_*.feather
```

### Dataset Not Found

```
FileNotFoundError: Dataset not found at {dataset_path}
```

**Solution**: The dataset needs to be prepared first. Check if the dataset path exists:
```bash
ls /net/tokyo100-10g/data/str01_01/y-guo/datasets/myriadlama/paraphrases_dataset/
```

If not, the dataset needs to be created from the HuggingFace dataset first.

## Testing

Run the test suite to verify the implementation:

```bash
# Run MyriadLAMA-specific tests
python test_myriadlama_flex_attention.py
```

Expected output:
```
======================================================================
  Testing MyriadLAMA-Specific FlexAttention Implementation
======================================================================
...
======================================================================
  ALL TESTS PASSED ✅
======================================================================
```

## Advanced Usage

### Custom Analysis Script

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_feather('.../myriadlama_flex_all_a5_xattn.feather')

# Analyze by relation type (if available in dataset)
# Group by relation and calculate accuracy

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(df['prediction'].str.len(), bins=20)
plt.xlabel('Prediction Length (characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Lengths')
plt.savefig('prediction_lengths.png')
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple models

for model in llama3.2_3b_it llama3.1_8b_it; do
    echo "Processing $model..."
    python myriadlama_flex_attention_generate.py \
        --model $model \
        --allow_manual_cross_attention \
        --device auto
    
    python myriadlama_flex_attention_generate.py \
        --model $model \
        --allow_manual_cross_attention \
        --lemmaize
done
```

## References

- Implementation: `myriadlama_flex_attention_generate.py`
- Documentation: `docs/MYRIADLAMA_FLEX_ATTENTION.md`
- Base implementation: `flex_attention_generate.py`
- Dataset class: `dataset.py` (MyriadLamaDataset)
- Test suite: `test_myriadlama_flex_attention.py`
