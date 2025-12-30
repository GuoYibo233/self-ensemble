# Diversity Comparison Notebook

## Overview

The `diversity_comparison.ipynb` notebook computes semantic diversity of paraphrases for each prompt combination using BERT embeddings. It provides both traditional (Jaccard-based) and modern (BERT-based) diversity metrics.

## Quick Start

### 1. Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Key dependencies added for this notebook:
- `scikit-learn>=1.3.0` - For cosine similarity calculations
- `transformers>=4.30.0` - For BERT models (already in requirements)
- `torch` - For PyTorch operations (already in requirements)

### 2. Data Preparation

The notebook expects diversity data to be generated and stored in the following structure:

```
datasets/myriadlama/<model_name>/
├── confidence.feather
└── diversity/
    ├── prompt1,prompt2.feather
    ├── prompt1,prompt3.feather
    └── ...
```

Generate this data using the appropriate generation scripts in the repository before running the notebook.

### 3. Running the Notebook

1. Open `diversity_comparison.ipynb` in Jupyter Lab/Notebook
2. Update the data path in the "Configure data paths" cell to point to your dataset
3. Run all cells in order

The notebook will:
- Automatically try to load a modern BERT model (with fallbacks)
- Process each diversity file in your dataset  
- Calculate BERT-based semantic diversity
- Compare with traditional metrics
- Display comprehensive results

## BERT Model Selection

The notebook tries models in this order:
1. `sentence-transformers/all-MiniLM-L6-v2` (recommended - efficient and accurate)
2. `sentence-transformers/all-mpnet-base-v2` (higher quality, slower)
3. `bert-base-uncased` (standard BERT, widely available)

**First-time run**: Models are downloaded from HuggingFace (requires internet)  
**Subsequent runs**: Uses cached models from `~/.cache/huggingface/`

## Understanding the Output

### Metrics Computed

1. **BERT Diversity Score**
   - Range: 0.0 to 1.0
   - Formula: `1 - average_pairwise_cosine_similarity`
   - Interpretation:
     - High (0.7-1.0): Paraphrases are semantically diverse
     - Medium (0.3-0.7): Moderate semantic variation
     - Low (0.0-0.3): Paraphrases are very similar

2. **Jaccard Diversity** (Traditional)
   - Formula: `OR_matches / AND_matches`
   - Based on prediction agreement

3. **Consistency Score**
   - Proportion of matching predictions across paraphrases

4. **Ensemble Scores**
   - Accuracy of ensemble predictions

### Results Tables

The notebook outputs:
- Per-file diversity scores for each prompt combination
- Correlation analysis between metrics
- Top 5 most diverse prompt combinations
- Top 5 least diverse prompt combinations
- Statistical summaries (mean, std, min, max)

## Customization

### Change Data Source

Edit the `root_options` list in the "Configure data paths" cell:

```python
root_options = [
    "path/to/your/dataset",
    # Add more paths as fallbacks
]
```

### Use Different BERT Model

Edit the `model_options` list in the "Load BERT model" cell:

```python
model_options = [
    'your-preferred-model',
    'fallback-model',
    'bert-base-uncased',  # Keep as last fallback
]
```

### Adjust Batch Size

For memory-constrained environments, reduce the batch size in `get_bert_embeddings()`:

```python
def get_bert_embeddings(texts, batch_size=16):  # Default is 32
```

## Performance Tips

- **GPU Usage**: The notebook automatically uses GPU if available (significantly faster)
- **Memory**: Large datasets may require reducing batch size or processing in chunks
- **Caching**: Models are cached after first download (in `~/.cache/huggingface/`)
- **Parallel Processing**: The notebook processes files sequentially with progress bars

## Troubleshooting

### No Data Found
```
Diversity directory not found: ...
```
**Solution**: Generate diversity data first using the generation scripts

### Model Download Fails
```
Failed to load any BERT model
```
**Solution**: 
- Ensure internet access
- Check HuggingFace is not blocked
- Try downloading model manually first

### Out of Memory
```
CUDA out of memory
```
**Solution**:
- Reduce batch size in `get_bert_embeddings()`
- Use CPU instead: `device = torch.device('cpu')`
- Process fewer files at once

### Import Errors
```
No module named 'sklearn'
```
**Solution**:
```bash
pip install scikit-learn
# or
pip install -r requirements.txt
```

## Technical Details

### Diversity Calculation

For a set of paraphrases P = {p₁, p₂, ..., pₙ}:

1. **Get BERT embeddings**: E = {e₁, e₂, ..., eₙ}
2. **Compute pairwise similarities**: sim(eᵢ, eⱼ) for all i < j
3. **Average similarity**: avg_sim = mean(all pairwise similarities)
4. **Diversity**: diversity = 1 - avg_sim

### Embedding Method

Uses **mean pooling** over BERT token embeddings:
- Tokenize text with padding/truncation
- Get BERT last hidden states
- Apply attention mask
- Average token embeddings

This produces a single fixed-size vector per text, suitable for semantic comparison.

## Examples

### Example Output

```
DIVERSITY ANALYSIS RESULTS
================================================================================
                    file  bert_diversity  num_paraphrases  ensemble_score
0  paraphrase0,paraphrase1       0.3245               10          0.8234
1  paraphrase0,paraphrase2       0.4123               10          0.8156
...

CORRELATION ANALYSIS
================================================================================
Pearson correlation between ensemble and BERT diversity scores: -0.6432
Pearson correlation between ensemble and consistency scores: 0.7181
Mean BERT diversity score: 0.3567
================================================================================
```

### Interpreting Results

- **Negative correlation** between diversity and ensemble score is expected: 
  - More diverse paraphrases → harder to get consistent answers
  - Less diverse paraphrases → easier to achieve consensus

- **Positive correlation** between consistency and ensemble score:
  - Consistent predictions across paraphrases help ensemble

## Integration with Existing Analysis

This notebook complements the existing `archived/analysis/diversity.ipynb` by:
- Adding semantic (BERT-based) diversity metrics
- Providing modern transformer-based analysis
- Offering detailed per-combination insights
- Supporting newer BERT models

Both notebooks can be used together for comprehensive analysis.

## License

Part of the self-ensemble project. See main repository LICENSE.

## Citation

If you use this diversity analysis in your research, please cite the main self-ensemble paper and this notebook.
