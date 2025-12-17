# Notebooks Directory

This directory contains Jupyter notebooks for interactive analysis and visualization of self-ensemble results.

## Available Notebooks

### Analysis Notebooks

- **flexattention_analysis.ipynb**: Interactive analysis of FlexAttention results
  - Compare with baseline methods
  - Visualize attention patterns
  - Analyze performance metrics

- **diversity.ipynb**: Analyze diversity of generated outputs
  - Measure paraphrase diversity
  - Compare diversity across methods
  - Visualize diversity metrics

- **new_fact_occur.ipynb**: Analyze occurrence of new facts in generated outputs
  - Track novel information
  - Compare information gain across methods

- **num_prompts.ipynb**: Analyze impact of number of paraphrases
  - Performance vs. number of paraphrases
  - Efficiency analysis
  - Optimal paraphrase count

- **report_accs.ipynb**: Report and compare accuracy metrics
  - Aggregate results across datasets
  - Generate comparison tables
  - Create visualization plots

## Usage

### Starting Jupyter

From the repository root:
```bash
jupyter notebook notebooks/
```

Or start a specific notebook:
```bash
jupyter notebook notebooks/flexattention_analysis.ipynb
```

### Running in JupyterLab

For a more modern interface:
```bash
jupyter lab notebooks/
```

### Google Colab

Notebooks can also be run in Google Colab:
1. Upload the notebook to Google Drive
2. Open with Google Colab
3. Install required dependencies in the first cell

## Workflow

1. **Generate Results**: Run generation scripts from `src/`
2. **Analyze Results**: Use notebooks to analyze outputs
3. **Visualize**: Create plots and visualizations
4. **Export**: Save figures and tables for reports

## Adding New Notebooks

When creating new analysis notebooks:

1. Place in this directory with a descriptive name
2. Include markdown cells explaining the analysis
3. Document required input files and formats
4. Add visualization outputs
5. Update this README

## Dependencies

Notebooks may require additional dependencies:
```bash
pip install jupyter matplotlib seaborn pandas numpy
```

See individual notebooks for specific requirements.

## Tips

- Use `%matplotlib inline` for inline plots
- Save plots to `plot/` directory for documentation
- Use meaningful variable names
- Add markdown explanations between code cells
- Clear outputs before committing (to reduce file size)
