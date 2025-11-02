# FlexAttention Generation and Analysis Guide

This guide covers the features for FlexAttention-based ensemble generation and traditional ensemble methods, including limiting sample generation and analyzing results.

## Generation Methods Overview

### Available Methods

| Method | Description | Output File | Usage |
|--------|-------------|-------------|-------|
| **origin** | Baseline using only original questions (no paraphrases) | `origin.feather` | `generate.py --method origin` |
| **per_prompt** | Generate with each paraphrase separately | `per_prompt.feather` | `generate.py --method per_prompt` |
| **avg** | Logit-level fusion with averaging | `ensemble_avg-N.feather` | `generate.py --method avg` |
| **max** | Logit-level fusion with max pooling | `ensemble_max-N.feather` | `generate.py --method max` |
| **weighted_avg** | Weighted averaging based on confidence | `ensemble_weighted_avg-N.feather` | `generate.py --method weighted_avg` |
| **weighted_max** | Weighted max based on confidence | `ensemble_weighted_max-N.feather` | `generate.py --method weighted_max` |
| **flex_attention** | Attention-level fusion (most efficient) | `flex_attention-N.feather` | `flex_attention_generate.py` |

### Method Comparison

| Method | Fusion Level | Efficiency | Forward Passes | Paraphrases Used |
|--------|--------------|------------|----------------|------------------|
| origin | None | Fastest | 1× per step | 0 (original only) |
| per_prompt | None | Baseline | N× per step | All separately |
| avg/max | Logit | N× cost | N× per step | All (logit fusion) |
| weighted_* | Logit + confidence | N× cost | N× per step | All (weighted fusion) |
| **flex_attention** | **Attention** | **Most efficient** | **1× per step** | **All (attention fusion)** |

## Usage Examples

### 1. Baseline Generation (Original Questions Only)

Generate results using only the original questions without any paraphrases:

```bash
# Generate baseline results
python generate.py \
    --method origin \
    --dataset webqa \
    --model llama3.2_3b_it

# Output: /path/to/datasets/webqa/llama3.2_3b_it/origin.feather
```

**When to use:**
- Establish baseline performance
- Compare impact of paraphrases
- Quick single-pass generation

### 2. Per-Prompt Generation

Generate with each paraphrase separately (no ensemble):

```bash
python generate.py \
    --method per_prompt \
    --dataset webqa \
    --model llama3.2_3b_it

# Output: /path/to/datasets/webqa/llama3.2_3b_it/per_prompt.feather
```

### 3. Traditional Ensemble Methods

```bash
# Average ensemble
python generate.py \
    --method avg \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6

# Max ensemble
python generate.py \
    --method max \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6

# Weighted methods (requires confidence.feather)
python generate.py \
    --method weighted_avg \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6
```

### 4. FlexAttention Generation

Most efficient method with attention-level fusion:

```bash
# FlexAttention with 5 paraphrases
python flex_attention_generate.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5

# Quick testing with limited samples
python flex_attention_generate.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5 \
    --max_samples 100
```

## New Features

### 1. Limit Number of Samples to Generate

You can now specify `--max_samples` to limit how many samples are generated. This is useful for:
- Quick testing with a small subset
- Running experiments on limited compute resources
- Creating smaller datasets for prototyping

**Usage:**
```bash
# Generate only 100 samples
python flex_attention_generate.py \
    --dataset myriadlama \
    --model qwen2.5_7b_it \
    --num_paraphrases 5 \
    --max_samples 100

# Generate only 50 samples for quick testing
python flex_attention_generate.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 3 \
    --max_samples 50
```

**Parameters:**
- `--max_samples N`: Generate at most N samples (default: None, process all)

### 2. Analysis Tools for FlexAttention Results

Two new analysis tools are provided:

#### A. Command-line Analysis Script

`analysis/analyze_flexattention.py` - Analyze FlexAttention results from the command line.

**Basic Usage:**
```bash
# Analyze FlexAttention results
python analysis/analyze_flexattention.py \
    --dataset myriadlama \
    --model qwen2.5_7b_it \
    --num_paraphrases 5

# Compare with different numbers of paraphrases
python analysis/analyze_flexattention.py \
    --dataset myriadlama \
    --model qwen2.5_7b_it \
    --num_paraphrases 5 \
    --compare_all
```

**Features:**
- ✅ Compute FlexAttention accuracy
- ✅ Compare with traditional ensemble methods (avg, max, weighted_avg, weighted_max)
- ✅ Show sample generations
- ✅ Analyze effect of different numbers of paraphrases
- ✅ Calculate improvement over traditional methods

**Parameters:**
- `--dataset`: Dataset name (webqa or myriadlama)
- `--model`: Model name (e.g., qwen2.5_7b_it, llama3.2_3b_it)
- `--num_paraphrases`: Number of paraphrases to analyze (default: 5)
- `--compare_all`: Compare results with different numbers of paraphrases

**Output Example:**
```
======================================================================
FlexAttention Generation Analysis
======================================================================
Dataset root: datasets/myriadlama/qwen2.5_7b_it
Number of paraphrases: 5

✅ Loading FlexAttention results: datasets/myriadlama/qwen2.5_7b_it/flex_attention-5.feather

📊 FlexAttention Accuracy (with lemmatization): 0.756

📈 Dataset Statistics:
   Total samples: 1000
   Unique UUIDs: 1000

======================================================================
Comparison with Traditional Ensemble Methods
======================================================================
  avg            : 0.742
  max            : 0.738
  weighted_avg   : 0.745
  weighted_max   : 0.741

  FlexAttention  : 0.756

📊 Summary:
   Average traditional ensemble: 0.742
   FlexAttention: 0.756
   Improvement: +0.014 (+1.9%)
```

#### B. Jupyter Notebook for Interactive Analysis

`analysis/flexattention_analysis.ipynb` - Interactive analysis with visualizations.

**Features:**
- 📊 Load and explore FlexAttention results
- 📈 Visualize method comparisons (bar charts)
- 📉 Plot accuracy vs. number of paraphrases
- 🔍 Error analysis with examples
- ✅ Compare with traditional ensemble methods

**Usage:**
1. Open the notebook: `jupyter notebook analysis/flexattention_analysis.ipynb`
2. Set the configuration in the second cell:
   ```python
   DATASET = "myriadlama"
   MODEL = "qwen2.5_7b_it"
   NUM_PARAPHRASES = 5
   ```
3. Run all cells to see the analysis

**Key Sections:**
1. **Setup** - Import libraries and utilities
2. **Configuration** - Set dataset and model
3. **Load Results** - Load FlexAttention results
4. **Compute Accuracy** - Calculate accuracy metrics
5. **Sample Generations** - View example outputs
6. **Method Comparison** - Compare with traditional ensembles (with bar chart)
7. **Effect of Paraphrases** - Analyze impact of number of paraphrases (with line plot)
8. **Error Analysis** - Examine incorrect predictions
9. **Summary** - Overall statistics and improvements

## Complete Workflow Example

Here's a complete workflow from generation to analysis:

### Step 1: Generate FlexAttention Results

```bash
# Generate with 5 paraphrases (100 samples for testing)
python flex_attention_generate.py \
    --dataset myriadlama \
    --model qwen2.5_7b_it \
    --num_paraphrases 5 \
    --max_samples 100

# Add lemmatization for accuracy computation
python flex_attention_generate.py \
    --dataset myriadlama \
    --model qwen2.5_7b_it \
    --num_paraphrases 5 \
    --lemmaize
```

### Step 2: Analyze Results

**Option A: Command-line Analysis**
```bash
python analysis/analyze_flexattention.py \
    --dataset myriadlama \
    --model qwen2.5_7b_it \
    --num_paraphrases 5
```

**Option B: Interactive Notebook**
```bash
jupyter notebook analysis/flexattention_analysis.ipynb
```

### Step 3: Compare Different Configurations

Generate and compare results with different numbers of paraphrases:

```bash
# Generate with 2-10 paraphrases
for n in {2..10}; do
    python flex_attention_generate.py \
        --dataset myriadlama \
        --model qwen2.5_7b_it \
        --num_paraphrases $n \
        --max_samples 100
done

# Lemmatize all results
for n in {2..10}; do
    python flex_attention_generate.py \
        --dataset myriadlama \
        --model qwen2.5_7b_it \
        --num_paraphrases $n \
        --lemmaize
done

# Analyze and compare
python analysis/analyze_flexattention.py \
    --dataset myriadlama \
    --model qwen2.5_7b_it \
    --compare_all
```

## Tips and Best Practices

1. **Start Small**: Use `--max_samples` to test with a small dataset first
   ```bash
   python flex_attention_generate.py --dataset webqa --max_samples 10
   ```

2. **Always Lemmatize**: Run with `--lemmaize` flag to enable accuracy computation
   ```bash
   python flex_attention_generate.py --dataset webqa --lemmaize
   ```

3. **Compare Methods**: Generate results for traditional ensemble methods too for comparison
   ```bash
   python generate.py --dataset myriadlama --method avg --num_ensemble 5
   python generate.py --dataset myriadlama --method max --num_ensemble 5
   ```

4. **Use Notebooks for Exploration**: The Jupyter notebook provides interactive visualizations

5. **Check Output Files**: Results are saved in `datasets/{dataset}/{model}/flex_attention-{n}.feather`

## File Locations

- **Generation script**: `flex_attention_generate.py`
- **Analysis script**: `analysis/analyze_flexattention.py`
- **Analysis notebook**: `analysis/flexattention_analysis.ipynb`
- **Output directory**: `datasets/{dataset}/{model}/`
- **Output files**: `flex_attention-{num_paraphrases}.feather`

## Troubleshooting

**Q: How do I know if lemmatization is complete?**
A: Check if the output file contains `predict_lemma` and `answer_lemmas` columns.

**Q: Analysis script shows "Results not found"**
A: Make sure you've run `flex_attention_generate.py` first and the output file exists.

**Q: Can I use custom models?**
A: Yes, add your model path to `constants.MODEL_PATHs` dictionary.

**Q: How do I stop generation early?**
A: Use Ctrl+C. Results will be saved for samples processed so far (unless you want to discard them).

## Related Files

- `flex_attention_generate.py` - Main generation script
- `tools/debug_flexattention.py` - Debugging tool with detailed output
- `tools/example_flexattention.py` - Simple examples
- `test_mask_visualization.py` - Test mask visualization improvements
- `IMPROVEMENTS_SUMMARY.md` - Details on mask matrix and formatting improvements
