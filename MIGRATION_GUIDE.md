# Migration Guide - New Repository Structure

This guide helps you transition to the reorganized repository structure.

## What Changed

### Directory Structure

**Old Structure:**
```
.
├── generate.py
├── flex_attention_generate.py
├── myriadlama_flex_attention_generate.py
├── baseline_generate.py
├── dataset.py
├── constants.py
├── utils.py
├── paraphrase.py
├── confidence.py
├── test_*.py
└── test/
    └── *.ipynb
```

**New Structure:**
```
.
├── src/
│   ├── core/                      # Shared modules
│   │   ├── dataset.py
│   │   ├── constants.py
│   │   ├── utils.py
│   │   ├── paraphrase.py
│   │   ├── confidence.py
│   │   └── interactive.py         # NEW
│   ├── generate_original.py       # Renamed from generate.py
│   ├── generate_flex_attention.py # Renamed
│   ├── generate_myriadlama.py     # Renamed
│   ├── generate_baseline.py       # Renamed
│   └── run_interactive.py         # NEW
├── tests/                         # Moved from root and test/
│   ├── test_*.py
│   └── *.ipynb
└── notebooks/                     # Moved from analysis/
    └── *.ipynb
```

## Command Changes

### Generation Scripts

**Old:**
```bash
python generate.py --method avg --dataset webqa --model llama3.2_3b_it
python flex_attention_generate.py --dataset webqa --model llama3.2_3b_it
python myriadlama_flex_attention_generate.py --dataset myriadlama --model llama3.2_3b_it
python baseline_generate.py --method origin --dataset webqa --model llama3.2_3b_it
```

**New:**
```bash
python src/generate_original.py --method avg --dataset webqa --model llama3.2_3b_it
python src/generate_flex_attention.py --dataset webqa --model llama3.2_3b_it
python src/generate_myriadlama.py --dataset myriadlama --model llama3.2_3b_it
python src/generate_baseline.py --method origin --dataset webqa --model llama3.2_3b_it
```

**Or use the new interactive mode:**
```bash
python src/run_interactive.py
```

### Test Scripts

**Old:**
```bash
python test_causal_priority.py
jupyter notebook test/test_generate.ipynb
```

**New:**
```bash
python tests/test_causal_priority.py
jupyter notebook tests/test_generate.ipynb
```

### Analysis Notebooks

**Old:**
```bash
jupyter notebook analysis/flexattention_analysis.ipynb
```

**New:**
```bash
jupyter notebook notebooks/flexattention_analysis.ipynb
```

## Import Changes

If you have custom scripts that import from the repository:

**Old:**
```python
from constants import MODEL_PATHs
from dataset import WebQADataset
from utils import save_jsonl
```

**New:**
```python
from src.core.constants import MODEL_PATHs
from src.core.dataset import WebQADataset
from src.core.utils import save_jsonl
```

## New Features

### 1. Interactive Mode

The new `src/run_interactive.py` script provides guided prompts for:
- Selecting generation type
- Choosing dataset and model
- Configuring parameters
- Setting optional flags

**Usage:**
```bash
python src/run_interactive.py
```

### 2. Organized Documentation

- `src/README.md`: Source code documentation
- `tests/README.md`: Testing guide
- `notebooks/README.md`: Notebook documentation
- `docs/IMPLEMENTATION_SUMMARY.md`: English translation of implementation details

### 3. Better Module Organization

Core utilities are now in `src/core/`:
- `constants.py`: Model paths and configurations
- `dataset.py`: Dataset loaders
- `utils.py`: General utilities
- `paraphrase.py`: Paraphrase generation
- `confidence.py`: Confidence computation
- `interactive.py`: Interactive parameter prompts (NEW)

## Backward Compatibility

### Old Scripts Still Work

The original files (`generate.py`, `flex_attention_generate.py`, etc.) are still present in the root directory for backward compatibility. However, they are no longer maintained and will be removed in a future release.

**Recommendation:** Update your scripts to use the new structure.

### Gradual Migration

You can migrate gradually:

1. **Week 1-2**: Use new paths but keep old scripts as backup
2. **Week 3-4**: Update any custom scripts and automation
3. **Week 5+**: Remove old scripts from your workflows

## Common Issues

### Issue 1: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'dataset'
```

**Solution:**
Update imports to use `src.core.dataset`:
```python
from src.core.dataset import WebQADataset
```

### Issue 2: Command Not Found

**Error:**
```
python: can't open file 'generate.py': [Errno 2] No such file or directory
```

**Solution:**
Use the new path:
```bash
python src/generate_original.py [args]
```

### Issue 3: Test Files Not Found

**Error:**
```
FileNotFoundError: test_generate.ipynb
```

**Solution:**
Tests are now in the `tests/` directory:
```bash
jupyter notebook tests/test_generate.ipynb
```

## Getting Help

If you encounter issues:

1. Check this migration guide
2. Review the documentation in `src/README.md`
3. Look at examples in `notebooks/`
4. Open an issue on GitHub

## Rollback

If you need to rollback to the old structure:

```bash
git checkout <previous-commit-hash>
```

Find the commit hash before the reorganization:
```bash
git log --oneline
```

## Summary

The reorganization improves:
- ✅ Code organization and clarity
- ✅ Module separation and reusability  
- ✅ Documentation structure
- ✅ User experience with interactive mode
- ✅ Maintainability

Update your scripts to use the new paths and enjoy the improved structure!
