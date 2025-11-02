# Source Code Directory

This directory contains all generation scripts and core utilities for the self-ensemble project.

## Directory Structure

```
src/
├── core/                         # Shared utilities and modules
│   ├── __init__.py              # Package initialization
│   ├── constants.py             # Model paths and configurations
│   ├── dataset.py               # Dataset loaders (WebQA, MyriadLAMA)
│   ├── utils.py                 # General utility functions
│   ├── paraphrase.py            # Paraphrase generation utilities
│   ├── confidence.py            # Confidence computation
│   └── interactive.py           # Interactive parameter prompts
│
├── generate_original.py         # Original ensemble methods (avg, max, weighted)
├── generate_flex_attention.py   # FlexAttention-based generation
├── generate_myriadlama.py       # MyriadLAMA-specific FlexAttention
├── generate_baseline.py         # Baseline generation (origin, per_prompt)
└── run_interactive.py           # Interactive mode with parameter prompts
```

## Generation Scripts

### 1. Original Ensemble Methods (`generate_original.py`)

Implements traditional ensemble methods with multiple forward passes:
- `avg`: Average logits across paraphrases
- `max`: Maximum logits across paraphrases
- `weighted_avg`: Confidence-weighted average
- `weighted_max`: Confidence-weighted maximum
- `per_prompt`: Generate with each paraphrase separately

**Usage:**
```bash
python src/generate_original.py \
    --method avg \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6
```

### 2. FlexAttention Generation (`generate_flex_attention.py`)

Efficient attention-based ensemble using FlexAttention:
- Concatenates multiple paraphrases in a single prompt
- Isolates paraphrases during encoding
- Fuses information during generation
- Single forward pass (5× faster)

**Usage:**
```bash
python src/generate_flex_attention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5
```

### 3. MyriadLAMA-Specific (`generate_myriadlama.py`)

Specialized FlexAttention for MyriadLAMA dataset:
- Custom prompt formatting for [MASK] token prediction
- Modified mask logic for few-shot examples
- Optimized for fill-in-the-blank tasks

**Usage:**
```bash
python src/generate_myriadlama.py \
    --dataset myriadlama \
    --model llama3.2_3b_it \
    --num_paraphrases 5
```

### 4. Baseline Generation (`generate_baseline.py`)

Baseline methods for comparison:
- `origin`: Original question only
- `per_prompt`: Each paraphrase separately
- `all`: Generate both baselines

**Usage:**
```bash
python src/generate_baseline.py \
    --method all \
    --dataset webqa \
    --model llama3.2_3b_it
```

## Interactive Mode

Run any generation script interactively with guided prompts:

```bash
python src/run_interactive.py
```

The interactive mode will:
1. Prompt you to select generation type
2. Select dataset from available options
3. Select model from available options
4. Configure method-specific parameters
5. Set optional parameters (max_samples, device)
6. Show and confirm the command before execution

## Core Modules

### constants.py
Contains model paths and global configurations.

### dataset.py
Dataset loaders for:
- WebQA: Question answering dataset
- MyriadLAMA: Knowledge probing dataset

### utils.py
General utility functions for JSON handling, file I/O, etc.

### paraphrase.py
Utilities for generating and managing paraphrases.

### confidence.py
Functions for computing generation confidence scores.

### interactive.py
Helper functions for interactive parameter input:
- `prompt_for_parameter()`: Prompt for single parameter
- `select_from_options()`: Select from list of options
- `confirm_action()`: Yes/no confirmation

## Import Guidelines

All generation scripts import from `core` modules:

```python
from core.constants import MODEL_PATHs
from core.dataset import WebQADataset, MyriadLamaDataset
from core.utils import save_jsonl, load_jsonl
```

## Adding New Generation Methods

To add a new generation method:

1. Create a new script in `src/` (e.g., `generate_new_method.py`)
2. Import required utilities from `core/`
3. Follow the pattern of existing scripts for argument parsing
4. Update `run_interactive.py` to include the new method
5. Add documentation to this README

## Common Parameters

All generation scripts support these common parameters:

- `--dataset`: Dataset name (webqa, myriadlama)
- `--model`: Model name from constants.MODEL_PATHs
- `--device`: Device to use (cuda, cpu)
- `--max_samples`: Limit number of samples (for testing)
- `--output_dir`: Output directory for results

See individual script help for method-specific parameters:
```bash
python src/generate_original.py --help
```
