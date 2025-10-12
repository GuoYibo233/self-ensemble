# Linux System Configuration Guide

This guide provides configuration instructions for running FlexAttention on Linux systems, specifically optimized for Ubuntu 22.04 with NVIDIA RTX A6000 GPUs.

## System Requirements

**Verified Configuration:**
- OS: Ubuntu 22.04.4 LTS
- Python: 3.9.23
- PyTorch: 2.5.1
- CUDA: 12.1
- GPU: 10× NVIDIA RTX A6000 (48GB each)
- RAM: 251GB

## Quick Start for Linux Systems

### 1. Use Existing Environment (Recommended)

If you already have the `self-ensemble-debug` environment:

```bash
# Activate the environment
conda activate self-ensemble-debug

# Verify FlexAttention is available
python -c "from torch.nn.attention.flex_attention import flex_attention; print('✅ FlexAttention available')"

# Download spaCy model if not already installed
python -m spacy download en_core_web_lg
```

### 2. Create New Environment from Linux Config

```bash
# Create environment from Linux-specific config
conda env create -f environment_linux.yml

# Activate environment
conda activate self-ensemble-debug

# Download spaCy model
python -m spacy download en_core_web_lg

# Verify installation
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
from torch.nn.attention.flex_attention import flex_attention
print('FlexAttention: ✅')
import transformers
print('Transformers:', transformers.__version__)
"
```

## Storage and Cache Configuration

### Set Cache Directories

For systems with network storage, configure cache paths:

```python
import os

# Set cache directories for Hugging Face
os.environ['HF_HOME'] = '/net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache'
```

Or add to your `~/.bashrc`:

```bash
# Add to ~/.bashrc
export HF_HOME=/net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache
export TRANSFORMERS_CACHE=/net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache
export HF_DATASETS_CACHE=/net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache
```

### Directory Structure

```
Project Root: /home/y-guo/self-ensemble/self-ensemble/
Network Storage: /net/tokyo100-10g/data/str01_01/y-guo/
├── huggingface_cache/    # Models and datasets cache
├── models/               # Local model storage
└── datasets/             # Local dataset storage
```

## GPU Configuration

### Single GPU Usage

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python3 flex_attention_generate.py --dataset webqa --model llama3.2_3b_it
```

### Multi-GPU Usage

```bash
# Use GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 flex_attention_generate.py --dataset webqa --model llama3.2_3b_it

# Use all 10 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python3 flex_attention_generate.py --dataset webqa --model llama3.2_3b_it
```

### GPU Monitoring

```bash
# Real-time monitoring
nvidia-smi -l 1

# Or use gpustat for better formatting
gpustat -i 1
```

## Performance Optimization

### PyTorch Settings for RTX A6000

Add to your scripts for better performance:

```python
import torch

# Enable TF32 for better performance on Ampere GPUs (RTX A6000)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

# Set memory allocator settings (optional)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

### Multi-GPU Data Parallel

```python
# Use DataParallel for multi-GPU
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    print(f"Using {torch.cuda.device_count()} GPUs")
```

## Running the Tools

All tools work the same way on Linux:

```bash
# Validation
python tools/validate_flexattention_env.py --test-flex-attention

# Debugging (uses GPU by default)
python tools/debug_flexattention.py --dataset webqa --max-samples 1 --verbose

# Download resources
bash tools/download_resources.sh --dataset webqa --model llama3.2_3b_it

# Examples
python tools/example_flexattention.py
```

## Common Issues and Solutions

### CUDA Version Mismatch

If you see CUDA version warnings:

```bash
# Check CUDA versions
nvidia-smi  # System CUDA (12.2)
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA (12.1)
```

The system CUDA (12.2) is newer than PyTorch CUDA (12.1), which is fine and fully compatible.

### Network Storage Latency

For better I/O performance with network storage:

1. Copy frequently accessed files to local storage
2. Use local storage for temporary files
3. Write final results to both local and network storage

```bash
# Example: Use local tmp for processing
export TMPDIR=/tmp/y-guo
mkdir -p $TMPDIR
```

### Memory Management

With 251GB RAM and 10× 48GB GPUs:

- **System RAM**: More than sufficient for large models
- **GPU Memory**: Each A6000 has 48GB, can handle large models
- **Recommendation**: Use 1-4 GPUs for most experiments

### Python Version Compatibility

The code works with Python 3.9 and 3.10:
- **3.9.23**: Currently installed, fully tested
- **3.10+**: Also supported, newer features available

## Verification Checklist

Run these commands to verify your setup:

```bash
# 1. Activate environment
conda activate self-ensemble-debug

# 2. Check Python version
python --version  # Should show 3.9.23

# 3. Check PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 4. Check GPU count
python -c "import torch; print('GPUs:', torch.cuda.device_count())"

# 5. Check FlexAttention
python -c "from torch.nn.attention.flex_attention import flex_attention; print('FlexAttention: OK')"

# 6. Check transformers
python -c "import transformers; print('Transformers:', transformers.__version__)"

# 7. Check spaCy
python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('spaCy: OK')"

# 8. Run validation tool
python tools/validate_flexattention_env.py --test-flex-attention
```

## Environment Variables

Add these to your `~/.bashrc` for convenience:

```bash
# Hugging Face cache
export HF_HOME=/net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# CUDA settings
export CUDA_VISIBLE_DEVICES=0  # Default to GPU 0, change as needed

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Example Workflow

```bash
# 1. Activate environment
conda activate self-ensemble-debug

# 2. Set cache directories (if not in ~/.bashrc)
export HF_HOME=/net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache

# 3. Validate environment
python tools/validate_flexattention_env.py

# 4. Download resources (stores in HF_HOME)
bash tools/download_resources.sh --dataset webqa --model llama3.2_3b_it

# 5. Run debugging on GPU 0
CUDA_VISIBLE_DEVICES=0 python tools/debug_flexattention.py \
    --dataset webqa \
    --max-samples 2 \
    --verbose

# 6. Run full generation on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 python flex_attention_generate.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5
```

## System Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Check disk usage
df -h
du -sh /net/tokyo100-10g/data/str01_01/y-guo/*

# Monitor network storage
ls -lh /net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache/
```

## Notes

1. **Storage**: Network storage is 93% full - monitor space usage
2. **GPUs**: 10 GPUs available - use CUDA_VISIBLE_DEVICES to select
3. **Memory**: 251GB RAM is more than sufficient for all operations
4. **CUDA**: Version mismatch (12.2 system vs 12.1 PyTorch) is not a problem

---

**For More Information:**
- See [QUICKSTART.md](QUICKSTART.md) for general setup
- See [DELEGATE_PROMPT.md](DELEGATE_PROMPT.md) for detailed debugging guide
- Check `environment_linux.yml` for exact package versions
