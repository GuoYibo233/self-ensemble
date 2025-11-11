# Analysis Scripts Overview

This directory contains scripts and notebooks for analyzing generation results from the self-ensemble experiments.

## 📓 Notebooks

### myriadlama_analysis.ipynb (NEW - Comprehensive Analysis Notebook)

**Purpose**: Interactive Jupyter notebook for comprehensive MyriadLAMA results analysis

**Features**:
- ✅ **One-click search**: Automatically discover all generation files and baselines
- ✅ **Built-in lemmatization**: Integrated Spacy lemmatization with on-the-fly support
- ✅ **Comparison tables**: Sortable accuracy comparisons with improvement calculations
- ✅ **Detailed examples**: View questions, predictions, and answers with correctness indicators
- ✅ **Cross-method comparison**: Compare same examples across different methods
- ✅ **Error analysis**: Analyze patterns in incorrect predictions
- ✅ **CSV export**: Export detailed analysis for external review
- ✅ **No plotting**: Text and table-based analysis only

**Use when**:
- You want an interactive analysis environment
- You need to explore results from multiple methods at once
- You want to generate comparison tables and detailed examples
- You need flexible, customizable analysis workflows

**Quick Start**:
```bash
# Launch Jupyter
jupyter notebook analysis/myriadlama_analysis.ipynb

# Or use JupyterLab
jupyter lab analysis/myriadlama_analysis.ipynb
```

See [MYRIADLAMA_ANALYSIS_README.md](MYRIADLAMA_ANALYSIS_README.md) for complete documentation.

---

## 🐍 Python Scripts

### 1. analyze_detailed.py (Enhanced Analysis)

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

| Tool | Output Format | Comparison | Export to File | Detailed Features | Interactive |
|------|--------------|------------|----------------|-------------------|-------------|
| **myriadlama_analysis.ipynb** | Notebook/CSV | ✅ | ✅ | ✅ (All fields) | ✅ |
| analyze_detailed.py | CSV/Excel | ❌ | ✅ | ✅ (All fields) | ❌ |
| analyze_baseline.py | Console | ✅ | ❌ | ❌ | ❌ |
| analyze_flexattention.py | Console | ✅ | ❌ | ❌ | ❌ |

## Typical Workflow

1. **Generate results** using baseline_generate.py, flex_attention_generate.py, or generate.py
2. **Interactive analysis**: Use **myriadlama_analysis.ipynb** for comprehensive exploration (RECOMMENDED)
3. **Quick check**: Use analyze_baseline.py or analyze_flexattention.py for console output
4. **Batch export**: Use analyze_detailed.py to export comprehensive data for specific methods
5. **Further analysis**: Open exported CSV/Excel in your preferred tool for additional analysis

## Notes

- All scripts and notebooks require that generation has been completed first
- All analysis tools use lemmatized matching for accuracy calculation
- **myriadlama_analysis.ipynb** is recommended for most analysis tasks as it combines all features in an interactive environment
- analyze_detailed.py does NOT include paraphrase number comparison or plotting (as per current requirements)
- The detailed export is especially useful for:
  - Manual error analysis
  - Identifying patterns in failures
  - Exporting data for further statistical analysis
  - Creating reports or visualizations in external tools
