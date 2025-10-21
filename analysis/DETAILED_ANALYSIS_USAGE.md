# Detailed Analysis Export

This document describes the enhanced analysis script that exports detailed features from generation results.

## Overview

The `analyze_detailed.py` script provides comprehensive analysis of generation results, including:

- **Accuracy calculation**: Overall accuracy using lemmatized matching
- **Detailed feature export**: All information in an easy-to-view table format
- **Multiple export formats**: CSV and Excel support

## Features Exported

The detailed table includes the following information:

1. **Index**: Row number in the dataset
2. **UUID**: Unique identifier for each question
3. **Original_Question**: The original question text
4. **Paraphrase/Paraphrases**: Paraphrased versions of the question (if available)
5. **Model_Input_Prompt**: The complete prompt sent to the model (including few-shot examples)
6. **Model_Output_Generation**: Raw output from the model
7. **Processed_Output_Prediction**: Extracted prediction (first line, cleaned)
8. **Correct_Answers**: List of acceptable correct answers
9. **Prediction_Lemma**: Lemmatized version of the prediction
10. **Answer_Lemmas**: Lemmatized versions of correct answers
11. **Is_Correct**: ✓ or ✗ indicating if the prediction matches any correct answer

## Usage

### Basic Usage

```bash
# Analyze baseline origin results (CSV export)
python analysis/analyze_detailed.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --method baseline_origin

# Analyze baseline per_prompt results
python analysis/analyze_detailed.py \
    --dataset myriadlama \
    --model qwen2.5_7b_it \
    --method baseline_per_prompt
```

### FlexAttention Results

```bash
# Analyze FlexAttention results with 5 paraphrases
python analysis/analyze_detailed.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --method flex_attention \
    --num_paraphrases 5
```

### Ensemble Methods

```bash
# Analyze ensemble_avg results with 5 paraphrases
python analysis/analyze_detailed.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --method ensemble_avg \
    --num_paraphrases 5
```

### Excel Export

```bash
# Export to Excel format
python analysis/analyze_detailed.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --method baseline_origin \
    --export-format excel
```

### Custom Output Directory

```bash
# Specify custom output directory
python analysis/analyze_detailed.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --method baseline_origin \
    --output-dir ./results
```

### Suppress Display Output

```bash
# Only export, don't display sample data
python analysis/analyze_detailed.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --method baseline_origin \
    --no-display
```

## Arguments

- `--dataset`: Dataset name (choices: webqa, myriadlama)
- `--model`: Model name (e.g., llama3.2_3b_it, qwen2.5_7b_it)
- `--method`: Method name (baseline_origin, baseline_per_prompt, flex_attention, ensemble_avg, ensemble_max, ensemble_weighted_avg, ensemble_weighted_max)
- `--num_paraphrases`: Number of paraphrases (required for flex_attention and ensemble methods)
- `--export-format`: Export format (choices: csv, excel; default: csv)
- `--output-dir`: Output directory for exported files (default: same as dataset root)
- `--no-display`: Don't display sample data (only export)

## Output Files

The script generates files with the following naming pattern:

- CSV: `{method}_detailed.csv` or `{method}-{num_paraphrases}_detailed.csv`
- Excel: `{method}_detailed.xlsx` or `{method}-{num_paraphrases}_detailed.xlsx`

Output files are saved in the dataset root directory by default (e.g., `datasets/webqa/llama3.2_3b_it/`) unless a custom output directory is specified.

## Example Output

The exported table will look like this:

| Index | UUID | Original_Question | Model_Input_Prompt | Model_Output_Generation | Processed_Output_Prediction | Correct_Answers | Is_Correct |
|-------|------|-------------------|-------------------|------------------------|----------------------------|-----------------|------------|
| 0 | q1 | What is the capital of France? | Q: What is...\nA: | Paris is the capital of France | Paris | ['Paris', 'paris'] | ✓ |
| 1 | q2 | What is 2+2? | Q: What is...\nA: | 2+2 equals 4 | 4 | ['4', 'four'] | ✓ |

## Testing

Run the test suite to verify the script works correctly:

```bash
python test/test_analyze_detailed.py
```

This will test:
- Table preparation with different data formats
- Accuracy calculation
- CSV and Excel export functionality
- Different method types (baseline_origin, baseline_per_prompt, flex_attention)

## Dependencies

Required Python packages:
- pandas
- numpy
- pyarrow (for feather file support)
- openpyxl (for Excel export)
- torch (for utils.py)
- tqdm (for utils.py)

Install with:
```bash
pip install pandas numpy pyarrow openpyxl
```

## Notes

- The script currently does **not** include paraphrase number comparison or plotting functionality (as per requirements)
- Accuracy is calculated using lemmatized partial matching
- The script handles missing data gracefully and marks unavailable fields as "N/A"
- Different methods may have different available columns (e.g., baseline_per_prompt has individual paraphrases, flex_attention has paraphrase tuples)
