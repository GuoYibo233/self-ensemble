# Self-Ensemble with FlexAttention

This repository implements multiple self-ensemble text generation methods, including an efficient attention-level fusion approach based on FlexAttention.

## 📖 Documentation Navigation

**First-time users, start here:**

1. **[GENERATE_README.md](GENERATE_README.md)** - Detailed explanation and comparison of all generation scripts
2. **[archived/docs/](archived/docs/)** - Complete technical documentation (archived)

## 🎯 Core Features

This repository provides **four generation methods** for different use cases:

1. **Baseline Generation** - Baseline comparison methods (origin/per_prompt)
2. **Original Ensemble** - Traditional logit-level fusion (max/avg/weighted)
3. **FlexAttention Ensemble** - Efficient attention-level fusion (WebQA)
4. **MyriadLAMA Ensemble** - FlexAttention optimized for fill-in-the-blank tasks

### Method Comparison

| Method | Fusion Type | Efficiency | Use Case |
|--------|------------|------------|----------|
| Baseline | No fusion | Fastest | Comparison baseline |
| Original | Logit-level | Standard (N× forward) | Research different fusion strategies |
| **FlexAttention** | **Attention-level** | **Most efficient (1× forward)** | **WebQA Q&A (Recommended)** |
| MyriadLAMA | Attention-level | Most efficient (1× forward) | Fill-in-the-blank tasks |

For detailed comparison, see: [GENERATE_README.md](GENERATE_README.md)

## 🔧 Environment Setup

### System Requirements

- Python 3.10+
- PyTorch 2.5+ or nightly (required for FlexAttention)
- NVIDIA GPU with CUDA
- Conda/Miniconda

### Quick Installation

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate flexattention

# Or use Linux-specific environment (Ubuntu 22.04+)
conda env create -f environment_linux.yml
conda activate self-ensemble-debug

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 3. Install PyTorch nightly (supports FlexAttention)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

For detailed setup instructions: [archived/docs/QUICKSTART.md](archived/docs/QUICKSTART.md)

## 📖 Usage

### Option 1: Interactive Mode (Recommended)

```bash
python src/run_interactive.py
```

The interactive interface will guide you to select:
- Generation type (baseline/original/flex_attention/myriadlama)
- Dataset (webqa/myriadlama)
- Model
- Method-specific parameters

### Option 2: Direct Execution

```bash
# Baseline generation
python src/generate_baseline.py --method origin --dataset webqa --model llama3.2_3b_it

# Original ensemble methods
python src/generate_original.py --method max --dataset webqa --model llama3.2_3b_it --num_ensemble 6

# FlexAttention ensemble (recommended)
python src/generate_flex_attention.py --dataset webqa --model llama3.2_3b_it --num_paraphrases 5

# MyriadLAMA fill-in-the-blank tasks
python src/generate_myriadlama.py --dataset myriadlama --model llama3.2_3b_it --num_paraphrases 5
```

### Quick Testing

```bash
# Limit sample count for quick testing
python src/generate_flex_attention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5 \
    --max_samples 100
```

**Complete usage guide**: [GENERATE_README.md](GENERATE_README.md)

## 📁 Repository Structure

```
.
├── src/                           # Core generation scripts
│   ├── core/                      # Shared modules
│   │   ├── constants.py           # Model configuration
│   │   ├── dataset.py             # Dataset loading
│   │   ├── paraphrase.py          # Paraphrase generation
│   │   ├── confidence.py          # Confidence calculation
│   │   ├── utils.py               # Utility functions
│   │   └── interactive.py         # Interactive input
│   │
│   ├── generate_baseline.py      # Baseline generation
│   ├── generate_original.py      # Original ensemble
│   ├── generate_flex_attention.py # FlexAttention ensemble
│   ├── generate_myriadlama.py    # MyriadLAMA ensemble
│   └── run_interactive.py        # Interactive runner
│
├── mask_visualization.py         # Mask visualization
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── environment_linux.yml         # Linux environment
│
├── GENERATE_README.md            # Detailed generate scripts documentation
├── README.md                     # This file
│
└── archived/                     # Archived files
    ├── docs/                     # Detailed documentation
    ├── tests/                    # Test files
    ├── tools/                    # Tool scripts
    ├── analysis/                 # Analysis tools
    ├── notebooks/                # Jupyter notebooks
    └── ...                       # Other archived content
```

## 💡 Common Issues

### FlexAttention not available

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

### CUDA out of memory

```bash
python src/generate_flex_attention.py --device cpu
```

### Dataset download failed

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

For more troubleshooting: [archived/docs/DELEGATE_PROMPT.md](archived/docs/DELEGATE_PROMPT.md)

## 📚 Detailed Documentation

- **[GENERATE_README.md](GENERATE_README.md)** - Detailed generate scripts documentation (Must read)
- **[archived/docs/](archived/docs/)** - Complete technical documentation
  - [QUICKSTART.md](archived/docs/QUICKSTART.md) - Quick start guide
  - [README_FLEXATTENTION.md](archived/docs/README_FLEXATTENTION.md) - FlexAttention overview
  - [ARCHITECTURE.md](archived/docs/ARCHITECTURE.md) - Architecture diagrams
  - [实现总结.md](archived/docs/实现总结.md) - Implementation summary (Chinese)

## 🔗 Related Tools (Archived)

Testing, analysis, and debugging tools have been moved to the `archived/` directory:

- **Debugging tools**: `archived/tools/debug_flexattention.py`
- **Test scripts**: `archived/tests/`
- **Analysis tools**: `archived/analysis/`
- **Visualization**: `archived/plot/`
- **Examples**: `archived/examples/`

These tools are still available but not required for running generate scripts.

---

**Last Updated**: 2025-12-17

**Status**: ✅ Production Ready | 📖 Documented | 🧹 Organized
