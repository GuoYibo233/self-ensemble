# MyriadLAMA Analysis Notebook

This notebook provides comprehensive analysis tools for MyriadLAMA generation results.

## Features

### 1. One-Click Search for All Results 🔍
The notebook automatically discovers and loads all result files in your dataset directory:
- Baseline results (`baseline_origin`, `baseline_per_prompt`)
- Ensemble methods (`ensemble_avg`, `ensemble_max`, etc.)
- FlexAttention results (`flex_attention-*`)
- Any other `.feather` files in the dataset directory

### 2. Built-in Lemmatization 📝
- Automatic lemmatization using Spacy
- On-the-fly lemmatization for files that don't have lemmatized results
- Reusable lemmatization functions for custom analysis

### 3. Comparison Tables 📊
Generate comprehensive comparison tables showing:
- Method names and categories
- Accuracy scores
- Sample counts and unique questions
- Improvements over baseline methods
- Sorted by performance

### 4. Detailed Examples 🔬
View detailed examples with:
- Original questions and paraphrases
- Model inputs (prompts)
- Generated outputs
- Predictions and correct answers
- Lemmatized versions
- Correctness indicators (✅/❌)

Additional features:
- **Cross-method comparison**: Compare the same examples across different methods
- **Error analysis**: Analyze patterns in incorrect predictions
- **CSV export**: Export detailed analysis for external review
- **Customizable filters**: Show only correct/incorrect predictions

## Installation

### Prerequisites
```bash
# Install required packages
pip install pandas numpy spacy openpyxl

# Download Spacy language model
python -m spacy download en_core_web_lg
```

### Required Data Structure
Your dataset should be organized as follows:
```
datasets/
└── myriadlama/
    └── {model_name}/
        ├── baseline_origin.feather
        ├── baseline_per_prompt.feather
        ├── ensemble_avg-5.feather
        ├── flex_attention-5.feather
        └── ... (other result files)
```

## Usage

### Quick Start

1. **Open the notebook**:
   ```bash
   jupyter notebook analysis/myriadlama_analysis.ipynb
   ```

2. **Configure your dataset** (Cell 2):
   ```python
   DATASET_NAME = "myriadlama"
   MODEL_NAME = "qwen2.5_7b_it"  # Change to your model
   ```

3. **Run all cells** to:
   - Discover all result files
   - Load and process results
   - Calculate accuracies
   - Generate comparison table
   - Show sample examples

### Advanced Usage

#### Compare Specific Methods
```python
methods_to_compare = ['baseline_origin', 'flex_attention-5']
compare_examples_across_methods(methods_to_compare, loaded_results, num_examples=5)
```

#### Show Only Incorrect Predictions
```python
generate_detailed_examples(
    'flex_attention-5',
    loaded_results['flex_attention-5'],
    num_examples=10,
    show_correct=False,
    show_incorrect=True
)
```

#### Export Analysis to CSV
```python
export_detailed_analysis('flex_attention-5', loaded_results['flex_attention-5'])
```

#### Apply Lemmatization On-the-Fly
```python
loaded_results = load_all_results(all_results, apply_lemmatization=True)
```

#### Analyze Error Patterns
```python
analyze_error_patterns('flex_attention-5', loaded_results['flex_attention-5'])
```

## Notebook Structure

1. **Setup and Imports** - Import required libraries
2. **Configuration** - Set dataset paths
3. **Lemmatization Setup** - Initialize Spacy for lemmatization
4. **File Discovery** - Automatically find all result files
5. **Load Results** - Load all discovered files
6. **Calculate Accuracies** - Compute accuracy metrics
7. **Comparison Table** - Generate and display comparison table
8. **Detailed Examples** - View examples with full details
9. **Export Analysis** - Export to CSV for external review
10. **Summary Statistics** - Overall statistics and insights
11. **Custom Analysis** - Add your own analysis functions

## Output Examples

### Comparison Table
```
Method                          Category    Accuracy    Total Samples    Unique Questions
flex_attention-5                FlexAttention   0.853       1000            1000
ensemble_weighted_max-5         Ensemble        0.847       1000            1000
baseline_per_prompt             Baseline        0.821       5000            1000
baseline_origin                 Baseline        0.815       1000            1000
```

### Detailed Example
```
Example 1/5
────────────────────────────────────────────────────────────────
UUID: abc123

Original Question:
The capital of France is [MASK].

Paraphrases:
  1. France's capital city is [MASK].
  2. [MASK] is the capital of France.

Prediction: Paris
Correct Answers: ['Paris', 'paris']

Prediction (lemmatized): ['paris']
Answer Lemmas: [['paris'], ['paris']]

Status: ✅ CORRECT
```

## Troubleshooting

### Spacy Model Not Found
```bash
python -m spacy download en_core_web_lg
```

### No Results Found
Make sure:
- `DATASET_ROOT` path is correct
- Result files are in `.feather` format
- Files are in the correct directory structure

### Lemmatization Not Available
Some older result files may not have lemmatized predictions. You can:
1. Re-generate with `--lemmaize` flag
2. Use `apply_lemmatization=True` when loading (slower)

## Customization

You can add custom analysis functions in Section 11 of the notebook. Example:

```python
def analyze_by_question_length(df):
    """Analyze accuracy by question length."""
    df['question_length'] = df['question'].apply(len)
    # Add your analysis logic here
    pass
```

## Tips

1. **Performance**: If you have many large files, loading might take time. Consider loading specific files only.

2. **Memory**: Large datasets may require significant RAM. Monitor memory usage when working with many files.

3. **Filtering**: Use DataFrame filtering to focus on specific subsets:
   ```python
   subset = df[df['uuid'].str.startswith('cat')]
   ```

4. **Visualization**: While this notebook focuses on tables and text, you can add visualization code in the custom analysis section.

## Support

For questions or issues:
1. Check that all prerequisites are installed
2. Verify dataset paths are correct
3. Ensure result files are properly formatted
4. Review error messages for specific issues

## Future Enhancements

Potential additions:
- Visualization support (accuracy curves, distributions)
- Statistical significance tests
- More sophisticated error analysis
- Support for additional file formats
- Interactive widgets for method selection
