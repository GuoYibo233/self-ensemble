# Analysis Scripts Overview

This directory contains scripts for analyzing generation results from the self-ensemble experiments.

## Scripts

### 1. analyze_detailed.py (NEW - Enhanced Analysis)

**Purpose**: Export detailed features for comprehensive analysis

**Features**:
- Calculates accuracy using lemmatized matching
- Exports ALL information to CSV/Excel tables for easy viewing
- Shows: question, paraphrases, model input (prompt), model output (generation), processed output (prediction), correct answers, and accuracy markers
- Supports all generation methods (baseline_origin, baseline_per_prompt, flex_attention, ensemble_*)
- User-friendly table format for manual review and analysis

**Use when**:
- You need to review individual predictions and their details
- You want to export data for external analysis (Excel, data science tools)
- You need a comprehensive view of all features for debugging or analysis

**Example**:
```bash
# Export baseline results to CSV
python analysis/analyze_detailed.py --dataset webqa --model llama3.2_3b_it --method baseline_origin

# Export FlexAttention results to Excel
python analysis/analyze_detailed.py --dataset webqa --model llama3.2_3b_it --method flex_attention --num_paraphrases 5 --export-format excel
```

See [DETAILED_ANALYSIS_USAGE.md](DETAILED_ANALYSIS_USAGE.md) for complete documentation.

---

### 2. analyze_baseline.py (Baseline-specific Analysis)

**Purpose**: Analyze baseline generation results with comparison

**Features**:
- Focused analysis for baseline_origin and baseline_per_prompt methods
- Shows accuracy and sample predictions
- Compares baselines with ensemble methods
- Per-paraphrase statistics for baseline_per_prompt

**Use when**:
- You want to quickly check baseline accuracy
- You need to compare baseline performance with ensemble methods
- You want to see per-paraphrase breakdown for baseline_per_prompt

**Example**:
```bash
# Analyze baselines
python analysis/analyze_baseline.py --dataset webqa --model llama3.2_3b_it

# Compare with ensemble methods
python analysis/analyze_baseline.py --dataset webqa --model llama3.2_3b_it --compare
```

---

### 3. analyze_flexattention.py (FlexAttention-specific Analysis)

**Purpose**: Analyze FlexAttention generation results with comparison

**Features**:
- Focused analysis for FlexAttention method
- Shows accuracy and sample generations
- Compares with traditional ensemble methods
- Can compare different numbers of paraphrases

**Use when**:
- You want to quickly check FlexAttention accuracy
- You need to compare FlexAttention with ensemble methods
- You want to find the optimal number of paraphrases

**Example**:
```bash
# Analyze FlexAttention results
python analysis/analyze_flexattention.py --dataset webqa --model llama3.2_3b_it --num_paraphrases 5

# Compare different numbers of paraphrases
python analysis/analyze_flexattention.py --dataset webqa --model llama3.2_3b_it --compare_all
```

---

### 4. demo_detailed_analysis.py (Demo Script)

**Purpose**: Demonstrate the analyze_detailed.py functionality

**Features**:
- Creates sample data
- Runs analyze_detailed.py on the demo data
- Shows example output

**Use when**:
- You want to see how analyze_detailed.py works
- You want to test the script without real data

**Example**:
```bash
python analysis/demo_detailed_analysis.py
```

---

## Quick Comparison

| Script | Output Format | Comparison | Export to File | Detailed Features |
|--------|--------------|------------|----------------|-------------------|
| analyze_detailed.py | CSV/Excel | ❌ | ✅ | ✅ (All fields) |
| analyze_baseline.py | Console | ✅ | ❌ | ❌ |
| analyze_flexattention.py | Console | ✅ | ❌ | ❌ |

## Typical Workflow

1. **Generate results** using baseline_generate.py, flex_attention_generate.py, or generate.py
2. **Quick check**: Use analyze_baseline.py or analyze_flexattention.py for console output
3. **Detailed review**: Use analyze_detailed.py to export comprehensive data
4. **Analysis**: Open exported CSV/Excel in your preferred tool for detailed analysis

## Notes

- All scripts require that generation has been completed first
- All scripts use lemmatized matching for accuracy calculation
- analyze_detailed.py does NOT include paraphrase number comparison or plotting (as per current requirements)
- The detailed export is especially useful for:
  - Manual error analysis
  - Identifying patterns in failures
  - Exporting data for further statistical analysis
  - Creating reports or visualizations in external tools
