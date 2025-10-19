# Baseline Generation Usage Guide

This guide explains how to generate and analyze baseline results for self-ensemble experiments.

## Overview

Baseline results are essential for evaluating the effectiveness of ensemble methods. This repository provides two types of baselines:

### Baseline Types

| Baseline | Description | Use Case | Output File |
|----------|-------------|----------|-------------|
| **Baseline 1 (origin)** | Uses only the original question | Attention mode baseline | `baseline_origin.feather` |
| **Baseline 2 (per_prompt)** | Each paraphrase separately | Second baseline for attention mode | `baseline_per_prompt.feather` |

### Why Two Baselines?

1. **Baseline 1 (origin)**: Establishes the performance without any paraphrases
   - Pure baseline for comparing the impact of paraphrases
   - Single forward pass per sample (fastest)

2. **Baseline 2 (per_prompt)**: Shows performance with paraphrases but without fusion
   - When using auto-generated prompts, this is the second baseline for attention mode
   - Multiple forward passes but no ensemble (one per paraphrase)

## Quick Start

### Generate Baseline 1 (Origin)

```bash
python baseline_generate.py \
    --method origin \
    --dataset webqa \
    --model llama3.2_3b_it
```

**Output**: `datasets/webqa/llama3.2_3b_it/baseline_origin.feather`

### Generate Baseline 2 (Per-Prompt)

```bash
python baseline_generate.py \
    --method per_prompt \
    --dataset webqa \
    --model llama3.2_3b_it
```

**Output**: `datasets/webqa/llama3.2_3b_it/baseline_per_prompt.feather`

### Generate Both Baselines

```bash
python baseline_generate.py \
    --method all \
    --dataset webqa \
    --model llama3.2_3b_it
```

## Analysis

### Analyze Baseline Results

```bash
# Analyze both baselines
python analysis/analyze_baseline.py \
    --dataset webqa \
    --model llama3.2_3b_it

# Compare with ensemble methods
python analysis/analyze_baseline.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --compare
```

### Example Analysis Output

```
======================================================================
Baseline 1 Analysis: Origin (Attention Mode Baseline)
======================================================================
✅ Loading baseline origin results: .../baseline_origin.feather

📊 Baseline 1 (Origin) Accuracy: 0.653

📈 Dataset Statistics:
   Total samples: 1000
   Unique questions: 1000

======================================================================
Baseline 2 Analysis: Per-Prompt (Attention Mode Second Baseline)
======================================================================
✅ Loading baseline per_prompt results: .../baseline_per_prompt.feather

📊 Baseline 2 (Per-Prompt) Overall Accuracy: 0.678

📈 Per-Paraphrase Statistics:
   Number of paraphrases per question: 6
   Paraphrase 0: 0.653
   Paraphrase 1: 0.681
   Paraphrase 2: 0.675
   ...
   Average accuracy across paraphrases: 0.678

======================================================================
Comparison: Baselines vs Ensemble Methods
======================================================================

📊 Accuracy Comparison:
Method                          Accuracy
------------------------------------------
FlexAttention-5                    0.712
Ensemble max-6                     0.705
Ensemble avg-6                     0.698
Baseline 2 (Per-Prompt)            0.678
Baseline 1 (Origin)                0.653

📈 Improvements over Baseline 1 (Origin):
   FlexAttention-5                +0.059 (+9.0%)
   Ensemble max-6                 +0.052 (+8.0%)
   Ensemble avg-6                 +0.045 (+6.9%)
   Baseline 2 (Per-Prompt)        +0.025 (+3.8%)
```

## Detailed Usage

### Command-Line Arguments

#### baseline_generate.py

```bash
python baseline_generate.py [OPTIONS]

Required arguments:
  --method {origin,per_prompt,all}
                        Baseline method to generate
  --dataset {webqa,myriadlama}
                        Dataset to use

Optional arguments:
  --model MODEL         Model name (default: llama3.2_3b_it)
  --device DEVICE       Device (default: cuda)
  --rewrite             Regenerate even if file exists
```

#### analysis/analyze_baseline.py

```bash
python analysis/analyze_baseline.py [OPTIONS]

Required arguments:
  --dataset {webqa,myriadlama}
                        Dataset to use
  --model MODEL         Model name

Optional arguments:
  --compare             Compare with ensemble methods
```

### Output Format

Both baseline methods generate `.feather` files with the following columns:

**Baseline 1 (origin):**
- `uuid`: Unique question identifier
- `answers`: Ground truth answers
- `question`: Original question
- `prompt`: Full prompt sent to model
- `prediction`: Extracted prediction
- `generation`: Full generated text
- `predict_lemma`: Lemmatized prediction tokens
- `answer_lemmas`: Lemmatized answer tokens

**Baseline 2 (per_prompt):**
- `uuid`: Unique question identifier
- `answers`: Ground truth answers
- `paraphrase`: Paraphrase text
- `prompt`: Full prompt sent to model
- `prediction`: Extracted prediction
- `generation`: Full generated text
- `predict_lemma`: Lemmatized prediction tokens
- `answer_lemmas`: Lemmatized answer tokens

## Comparison with Other Methods

### Method Comparison Table

| Method | Baselines | Efficiency | Forward Passes | Fusion Level |
|--------|-----------|------------|----------------|--------------|
| **Baseline 1 (origin)** | ✅ Yes | Fastest | 1× | None |
| **Baseline 2 (per_prompt)** | ✅ Yes | Standard | N× | None |
| per_prompt (original) | No | Standard | N× | None |
| avg/max ensemble | No | Standard | N× | Logit-level |
| weighted_avg/max | No | Standard | N× | Logit + confidence |
| **flex_attention** | No | **Most efficient** | **1×** | **Attention-level** |

### When to Use Each Method

**Use Baseline 1 (origin)** when:
- Establishing baseline performance without paraphrases
- Comparing the impact of paraphrasing
- Quick single-pass evaluation

**Use Baseline 2 (per_prompt)** when:
- Evaluating paraphrases individually
- Comparing with ensemble methods
- Analyzing per-paraphrase performance

**Use ensemble methods** when:
- Combining multiple paraphrases for better accuracy
- Comparing different fusion strategies

**Use flex_attention** when:
- Need efficient attention-level fusion
- Want to minimize inference time
- Exploring attention-based ensemble

## Integration with Existing Workflow

### Complete Experiment Workflow

```bash
# Step 1: Generate baselines
python baseline_generate.py --method all --dataset webqa --model llama3.2_3b_it

# Step 2: Generate ensemble results
python generate.py --method avg --dataset webqa --model llama3.2_3b_it --num_ensemble 6
python generate.py --method max --dataset webqa --model llama3.2_3b_it --num_ensemble 6

# Step 3: Generate FlexAttention results
python flex_attention_generate.py --dataset webqa --model llama3.2_3b_it --num_paraphrases 5

# Step 4: Analyze and compare all methods
python analysis/analyze_baseline.py --dataset webqa --model llama3.2_3b_it --compare
python analysis/analyze_flexattention.py --dataset webqa --model llama3.2_3b_it
```

## Technical Details

### Lemmatization

Both baselines automatically perform lemmatization using spaCy:
- Normalizes words to their base form (e.g., "running" → "run")
- Improves matching with answer variations
- Uses `en_core_web_lg` model

### Generation Settings

- **Decoding**: Greedy decoding (argmax)
- **Max tokens**: 20 new tokens
- **Temperature**: None (deterministic)
- **Batch size**: 8 samples per batch

### Performance Notes

**Baseline 1 (origin)**:
- Processing time: ~1-2 hours for 1000 samples (on GPU)
- Memory usage: ~8GB GPU memory
- Disk space: ~100MB output file

**Baseline 2 (per_prompt)**:
- Processing time: ~6-12 hours for 1000 samples with 6 paraphrases
- Memory usage: ~8GB GPU memory
- Disk space: ~500MB output file

## Troubleshooting

### File Already Exists

```bash
# Use --rewrite to regenerate
python baseline_generate.py --method origin --dataset webqa --rewrite
```

### Out of Memory

```bash
# Use CPU instead of GPU
python baseline_generate.py --method origin --dataset webqa --device cpu

# Or reduce batch size in the code (edit baseline_generate.py)
dataloader = dataset.get_dataloader(batch_size=4, shuffle=False)
```

### Missing spaCy Model

```bash
# Download the required model
python -m spacy download en_core_web_lg
```

## Related Documentation

- [README.md](README.md) - Main project overview
- [FLEXATTENTION_USAGE.md](FLEXATTENTION_USAGE.md) - FlexAttention usage
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - Quick start guide
- [CHANGELOG.md](CHANGELOG.md) - All changes and updates

## Examples

### Example 1: Quick Testing with Limited Samples

Currently baseline_generate.py processes all samples. For testing, you can modify the dataset loading code or filter the results after generation.

### Example 2: Analyzing Specific Paraphrases

Use pandas to analyze specific aspects:

```python
import pandas as pd

# Load baseline 2 results
df = pd.read_feather("datasets/webqa/llama3.2_3b_it/baseline_per_prompt.feather")

# Group by UUID to see all paraphrases for each question
for uuid in df['uuid'].unique()[:3]:
    subset = df[df['uuid'] == uuid]
    print(f"\nQuestion UUID: {uuid}")
    for i, row in subset.iterrows():
        print(f"  Paraphrase: {row['paraphrase'][:60]}...")
        print(f"  Prediction: {row['prediction']}")
```

## Summary

The baseline generation system provides:

✅ **Two complementary baselines** for comprehensive evaluation
✅ **Automated lemmatization** for robust matching
✅ **Easy comparison** with ensemble methods
✅ **Consistent output format** with other methods
✅ **Integrated analysis tools** for quick insights

Use baselines to establish performance bounds and measure the effectiveness of ensemble and attention-based fusion methods.
