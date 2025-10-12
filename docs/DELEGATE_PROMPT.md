# FlexAttention Debugging and Validation Guide

## Overview

This guide provides step-by-step instructions for understanding, validating, and debugging the FlexAttention-based ensemble generation code in this repository.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Validation](#validation)
4. [Understanding the Code](#understanding-the-code)
5. [Debugging Guide](#debugging-guide)
6. [Resource Management](#resource-management)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### One-Command Setup and Validation

```bash
# Validate your environment
python3 tools/validate_flexattention_env.py

# Download necessary resources (datasets and models)
bash tools/download_resources.sh --dataset webqa --model llama3.2_3b_it

# Run a debug session to understand the code flow
python3 tools/debug_flexattention.py --dataset webqa --model llama3.2_3b_it --max-samples 2
```

---

## Environment Setup

### Prerequisites

- **Python**: 3.9 or higher (tested on 3.9 and 3.10)
- **PyTorch**: 2.5+ or nightly build (for FlexAttention support)
- **CUDA**: 11.8+ or 12.1+ (required for GPU)
- **RAM**: 16GB minimum, 32GB recommended
- **Disk Space**: 20GB minimum for models and datasets
- **Conda**: Miniconda or Anaconda installed

### Step 1: Install Base Dependencies

```bash
# Create and activate conda environment (recommended)

# Option 1: Linux system with CUDA 12.1 (Ubuntu 22.04+)
# Uses stable PyTorch 2.5.1 with pre-configured dependencies
conda env create -f environment_linux.yml
conda activate self-ensemble-debug

# Option 2: General conda environment with PyTorch nightly
conda env create -f environment.yml
conda activate flexattention

# Option 3: Install from requirements.txt
conda create -n flexattention python=3.9 -y
conda activate flexattention
pip install -r requirements.txt

# Option 4: Install manually
conda create -n flexattention python=3.9 -y
conda activate flexattention
# Install PyTorch with FlexAttention support (CUDA version)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# Install other dependencies
pip install transformers pandas numpy tqdm datasets spacy

# Download spaCy model for lemmatization
python3 -m spacy download en_core_web_lg
```

### Step 2: Verify Installation

```bash
python3 tools/validate_flexattention_env.py
```

This script checks:
- ✅ Python version compatibility
- ✅ PyTorch version and FlexAttention availability
- ✅ CUDA availability and version
- ✅ Required packages (transformers, pandas, numpy, etc.)
- ✅ Disk space availability
- ✅ spaCy model availability

---

## Validation

### Environment Validation

The `validate_flexattention_env.py` script performs comprehensive checks:

```bash
# Basic validation
python3 tools/validate_flexattention_env.py

# Detailed validation with diagnostic output
python3 tools/validate_flexattention_env.py --verbose

# Test FlexAttention functionality
python3 tools/validate_flexattention_env.py --test-flex-attention
```

**What it checks:**
1. **System Requirements**: Python version, available memory, disk space
2. **PyTorch Setup**: Version, CUDA availability, FlexAttention API
3. **Dependencies**: All required packages and their versions
4. **Dataset Access**: Ability to load datasets from Hugging Face
5. **Model Access**: Ability to download/access models
6. **FlexAttention**: Creates a small test to verify FlexAttention works

### Code Validation

Run the test notebooks to validate core functionality:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Run dataset tests
jupyter notebook test/test_dataset.ipynb

# Run generation tests
jupyter notebook test/test_generate.ipynb
```

---

## Understanding the Code

### Architecture Overview

The FlexAttention implementation has these key components:

```
┌─────────────────────────────────────────────────────────┐
│                   Input: 5 Paraphrases                   │
│   "What is X?"  "Tell me X"  "X is what?"  etc.         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  concatenate_paraphrases_with_positions()                │
│  → "Para1 [SEP] Para2 [SEP] ... Para5"                  │
│  → Positions: [(0,45), (50,92), ...]                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  create_segment_isolation_mask()                         │
│  → Mask ensures paraphrases don't attend to each other  │
│  → Generated tokens can attend to all paraphrases       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  FlexAttentionWrapper                                    │
│  → Patches model attention layers                       │
│  → Applies custom mask during forward pass              │
│  → Unpatches after each step                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  flex_attention_generation()                             │
│  → Auto-regressive generation with custom attention     │
│  → Each step: patch → forward → unpatch                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Output: Generated Text                 │
│                   "Paris"                                │
└─────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose | Lines of Code |
|------|---------|---------------|
| `flex_attention_generate.py` | Main implementation | 554 |
| `dataset.py` | Dataset loading and processing | ~400 |
| `constants.py` | Model paths and configurations | ~50 |
| `generate.py` | Original generation methods | ~400 |

### Understanding the Flow

**Read the documentation in this order:**

1. **[README_FLEXATTENTION.md](README_FLEXATTENTION.md)** - Start here for overview
2. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick usage examples
3. **[FLEX_ATTENTION_IMPLEMENTATION.md](FLEX_ATTENTION_IMPLEMENTATION.md)** - Technical details
4. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Visual diagrams
5. **[REUSE_VS_NEW_DETAILED.md](REUSE_VS_NEW_DETAILED.md)** - Code breakdown

**Then examine the code:**

```bash
# View the main implementation
less flex_attention_generate.py

# View supporting modules
less dataset.py
less constants.py
less utils.py
```

---

## Debugging Guide

### Interactive Debugging with debug_flexattention.py

The `debug_flexattention.py` script provides detailed step-by-step insights:

```bash
# Basic debug run (2 samples, verbose output)
python3 tools/debug_flexattention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --max-samples 2 \
    --verbose

# Debug specific paraphrases
python3 tools/debug_flexattention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --indexs 0,1,2 \
    --max-samples 1

# Interactive mode with breakpoints
python3 tools/debug_flexattention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --max-samples 1 \
    --interactive
```

**Debug output includes:**
- 📊 Tensor shapes at each step
- 🎯 Attention mask visualization
- 🔍 Token-by-token generation details
- 📈 Logit distributions
- ✅ Mask validation (segment isolation check)

### Using Python Debugger (pdb)

Set breakpoints in the code:

```python
# In flex_attention_generate.py, add:
import pdb; pdb.set_trace()
```

Then run:

```bash
python3 -m pdb flex_attention_generate.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 3
```

**Useful pdb commands:**
- `n` - Next line
- `s` - Step into function
- `c` - Continue
- `p variable_name` - Print variable
- `pp variable_name` - Pretty print
- `l` - List source code
- `bt` - Backtrace

### Visual Debugging with VSCode

1. Install VSCode and Python extension
2. Open the repository folder
3. Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug FlexAttention",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/flex_attention_generate.py",
            "console": "integratedTerminal",
            "args": [
                "--dataset", "webqa",
                "--model", "llama3.2_3b_it",
                "--num_paraphrases", "3"
            ]
        },
        {
            "name": "Debug FlexAttention (Debug Script)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tools/debug_flexattention.py",
            "console": "integratedTerminal",
            "args": [
                "--dataset", "webqa",
                "--model", "llama3.2_3b_it",
                "--max-samples", "2",
                "--verbose"
            ]
        }
    ]
}
```

4. Set breakpoints by clicking left of line numbers
5. Press F5 to start debugging

### Logging and Monitoring

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='flexattention_debug.log'
)
```

Monitor GPU usage:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

---

## Resource Management

### Downloading Datasets and Models

Use the provided download script:

```bash
# Download WebQA dataset and Llama model
bash tools/download_resources.sh --dataset webqa --model llama3.2_3b_it

# Download MyriadLAMA dataset
bash tools/download_resources.sh --dataset myriadlama --model llama3.2_3b_it

# Download only datasets (no models)
bash tools/download_resources.sh --dataset-only webqa

# List available resources
bash tools/download_resources.sh --list
```

### Manual Dataset Download

If you prefer manual download:

```python
from datasets import load_dataset

# WebQA
webqa = load_dataset("WebQA", trust_remote_code=True)

# MyriadLAMA
# Follow instructions in dataset.py
```

### Managing Model Cache

Models are cached in `~/.cache/huggingface/`:

```bash
# Check cache size
du -sh ~/.cache/huggingface/

# Clear cache (careful!)
rm -rf ~/.cache/huggingface/hub/*

# Move cache to different location
export HF_HOME=/path/to/large/disk/huggingface
```

### Disk Space Management

```bash
# Check available space
df -h

# Find large files in repository
find . -type f -size +100M -exec ls -lh {} \;

# Clean up generated results (careful!)
find . -name "*.feather" -type f -delete
```

---

## Troubleshooting

### Common Issues

#### 1. FlexAttention Not Available

**Error:** `FlexAttention not available`

**Solution:**
```bash
# Install PyTorch nightly
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# Verify
python3 -c "from torch.nn.attention.flex_attention import flex_attention; print('✅ Available')"
```

#### 2. CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**
```bash
# Use smaller model
python3 flex_attention_generate.py --model smaller_model_name --device cuda

# Reduce batch size in dataset.py
# Edit: dataloader = dataset.get_dataloader(batch_size=4, shuffle=False)

# If still having issues, use CPU (slower)
python3 flex_attention_generate.py --device cpu
```

#### 3. Dataset Download Fails

**Error:** `ConnectionError` or `TimeoutError`

**Solutions:**
```bash
# Use proxy
export HF_ENDPOINT=https://hf-mirror.com
export HTTP_PROXY=http://your.proxy:port

# Download manually and place in correct directory
# See dataset.py for expected paths
```

#### 4. spaCy Model Not Found

**Error:** `Can't find model 'en_core_web_lg'`

**Solution:**
```bash
python3 -m spacy download en_core_web_lg

# Or use smaller model (edit code)
python3 -m spacy download en_core_web_sm
```

#### 5. Import Errors

**Error:** `ModuleNotFoundError: No module named 'XXX'`

**Solution:**
```bash
# Install missing packages
pip install transformers pandas numpy tqdm datasets spacy

# Verify environment
python3 tools/validate_flexattention_env.py
```

### Debug Checklist

When things don't work:

- [ ] Run `python3 tools/validate_flexattention_env.py` to check environment
- [ ] Check Python version: `python3 --version` (should be 3.10+)
- [ ] Check PyTorch version: `python3 -c "import torch; print(torch.__version__)"`
- [ ] Check CUDA availability: `python3 -c "import torch; print(torch.cuda.is_available())"`
- [ ] Check disk space: `df -h`
- [ ] Check available RAM: `free -h`
- [ ] Review logs in `flexattention_debug.log`
- [ ] Try with minimal example: `python3 tools/debug_flexattention.py --max-samples 1`

### Getting Help

1. **Check Documentation**: Read all .md files in the repository
2. **Review Issues**: Check GitHub issues for similar problems
3. **Run Tests**: `jupyter notebook test/test_generate.ipynb`
4. **Enable Debug Mode**: Use `--verbose` flag
5. **Minimal Reproduction**: Try smallest possible example

---

## Advanced Usage

### Custom Datasets

To use your own dataset:

1. Create a dataset class inheriting from `ParaPharaseDataset`
2. Implement required methods: `load_dataset()`, `get_dataloader()`, `collate_fn()`
3. Register in the argument parser

Example:
```python
from dataset import ParaPharaseDataset

class MyCustomDataset(ParaPharaseDataset):
    def load_dataset(self):
        # Your implementation
        pass
    
    def get_dataloader(self, batch_size=8, shuffle=False):
        # Your implementation
        pass
```

### Custom Models

To use different models:

1. Add model path to `constants.py`:
```python
MODEL_PATHs = {
    "my_model": "path/to/model",
    # ... existing models
}
```

2. Run with your model:
```bash
python3 flex_attention_generate.py --model my_model --dataset webqa
```

### Performance Tuning

**GPU Optimization:**
```bash
# Use mixed precision
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use TF32 for better performance
python3 -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True"
```

**CPU Optimization:**
```bash
# Use all CPU cores
export OMP_NUM_THREADS=$(nproc)

# Disable GPU
export CUDA_VISIBLE_DEVICES=""
```

---

## Summary

This guide provided:

✅ **Environment Setup**: Complete installation and configuration  
✅ **Validation**: Tools to verify your setup  
✅ **Code Understanding**: Architecture overview and documentation  
✅ **Debugging**: Multiple debugging approaches (CLI, pdb, VSCode)  
✅ **Resource Management**: Dataset and model downloading  
✅ **Troubleshooting**: Common issues and solutions  

**Next Steps:**

1. ✅ Run `python3 tools/validate_flexattention_env.py`
2. ✅ Download resources: `bash tools/download_resources.sh --dataset webqa --model llama3.2_3b_it`
3. ✅ Try debugging: `python3 tools/debug_flexattention.py --max-samples 1 --verbose`
4. ✅ Read the code: Start with `flex_attention_generate.py`
5. ✅ Run full generation: `python3 flex_attention_generate.py --dataset webqa --model llama3.2_3b_it`

**Happy Debugging! 🐛🔍**
