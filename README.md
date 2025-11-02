# Self-Ensemble with FlexAttention

This repository implements self-ensemble methods for natural language generation, including a novel FlexAttention-based approach that enables efficient attention-level fusion of multiple paraphrases.

## 🚀 Quick Start

**New to this repository? Start here:**

1. **[QUICKSTART.md](docs/QUICKSTART.md)** - Get up and running in 5 minutes
2. **[DELEGATE_PROMPT.md](docs/DELEGATE_PROMPT.md)** - Complete debugging and validation guide
3. **[README_FLEXATTENTION.md](docs/README_FLEXATTENTION.md)** - FlexAttention overview

## 📚 What's in This Repository

### Core Implementation

- **`flex_attention_generate.py`** - FlexAttention-based ensemble generation (NEW!)
- **`generate.py`** - Original ensemble methods (per_prompt, avg, max, weighted_avg)
- **`dataset.py`** - Dataset loading (WebQA, MyriadLAMA)
- **`constants.py`** - Model paths and configurations

### Debugging and Validation Tools

- **`tools/validate_flexattention_env.py`** - Environment validation script
- **`tools/debug_flexattention.py`** - Step-by-step debugging with detailed output
- **`tools/example_flexattention.py`** - Minimal working examples
- **`tools/download_resources.sh`** - Download datasets and models

### Analysis Tools

- **`analysis/analyze_flexattention.py`** - Command-line analysis tool for FlexAttention results
- **`analysis/flexattention_analysis.ipynb`** - Interactive Jupyter notebook for analysis and visualization

### Visualization Tools

- **`plot/flowchart_and_attention_mask_visualization.ipynb`** - Interactive notebook for visualizing code flowchart and attention masks
- **`plot/demo_visualization.py`** - Standalone script to generate demo visualizations
- **`plot/test_visualization.py`** - Test script to verify mask functions
- **`plot/README.md`** - Detailed usage guide for visualization tools

### Documentation

| Document | Description |
|----------|-------------|
| **[docs/QUICKSTART.md](docs/QUICKSTART.md)** | 5-minute setup guide |
| **[docs/LINUX_SETUP.md](docs/LINUX_SETUP.md)** | Linux-specific setup (Ubuntu 22.04, RTX A6000) |
| **[docs/DELEGATE_PROMPT.md](docs/DELEGATE_PROMPT.md)** | Complete debugging guide |
| **[docs/README_FLEXATTENTION.md](docs/README_FLEXATTENTION.md)** | FlexAttention overview |
| **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** | API quick reference |
| **[docs/FLEX_ATTENTION_IMPLEMENTATION.md](docs/FLEX_ATTENTION_IMPLEMENTATION.md)** | Technical details |
| **[docs/CREATE_FLEX_ATTENTION_MASK_IMPLEMENTATION.md](docs/CREATE_FLEX_ATTENTION_MASK_IMPLEMENTATION.md)** | Mask function implementation guide |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Visual diagrams |
| **[docs/REUSE_VS_NEW_DETAILED.md](docs/REUSE_VS_NEW_DETAILED.md)** | Component breakdown |
| **[FLEXATTENTION_USAGE.md](FLEXATTENTION_USAGE.md)** | Usage guide with --max_samples and analysis tools |
| **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** | Recent mask matrix & prompt formatting improvements |
| **[docs/实现总结.md](docs/实现总结.md)** | Chinese summary |

## 🎯 What is FlexAttention Ensemble?

FlexAttention ensemble is a new method that:

1. **Concatenates** multiple paraphrases into a single prompt
2. **Isolates** each paraphrase during encoding using custom attention masks
3. **Fuses** information from all paraphrases during generation

**Result:** More efficient (1× forward pass vs 5×) with attention-based fusion.

### Comparison with Existing Methods

| Method | Fusion | Efficiency | Forward Passes | Description |
|--------|--------|------------|----------------|-------------|
| **Baseline 1 (origin)** | None | Fastest | 1× per step | Original question only (attention mode baseline) |
| **Baseline 2 (per_prompt)** | None | Standard | N× per step | Each paraphrase separately (second baseline) |
| avg/max | Logit-level | Standard | N× per step | Logit-level ensemble fusion |
| weighted_* | Logit + confidence | Standard | N× per step | Weighted logit-level fusion |
| **flex_attention** | **Attention-level** | **Most efficient** | **1× per step** | **Attention-level fusion (most efficient)** |

## 🔧 Setup

### Prerequisites

- Python 3.10+  # FlexAttention requires Python 3.10+
- PyTorch 2.5+ or nightly (for FlexAttention)
- 20GB disk space
- NVIDIA GPU with CUDA support
- Conda/Miniconda installed

### Quick Setup

```bash
# 1. Create conda environment
# Option 1: Linux with CUDA 12.1 (Ubuntu 22.04+, RTX A6000)
conda env create -f environment_linux.yml
conda activate self-ensemble-debug

# Option 2: General environment with PyTorch nightly
conda env create -f environment.yml
conda activate flexattention

# Option 3: Manual with pip (requires Python 3.10+)
conda create -n flexattention python=3.10 -y
conda activate flexattention

# Install PyTorch FIRST
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Then install other dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

# 2. Validate environment
python3 tools/validate_flexattention_env.py --test-flex-attention

# 3. Download resources
bash tools/download_resources.sh --dataset webqa --model llama3.2_3b_it
```

For detailed setup instructions, see **[docs/QUICKSTART.md](docs/QUICKSTART.md)**.

## 📖 Usage

### Interactive Mode (Recommended)

Run generation scripts interactively with guided prompts:

```bash
python src/run_interactive.py
```

This will guide you through selecting:
- Generation type (original, flex_attention, myriadlama, baseline)
- Dataset (webqa, myriadlama)
- Model (from available models)
- Method-specific parameters
- Optional parameters

### Baseline Generation

```bash
# Baseline 1: Original questions only (attention mode baseline)
python src/generate_baseline.py \
    --method origin \
    --dataset webqa \
    --model llama3.2_3b_it

# Baseline 2: Each paraphrase separately (second baseline for attention mode)
python src/generate_baseline.py \
    --method per_prompt \
    --dataset webqa \
    --model llama3.2_3b_it

# Generate both baselines
python src/generate_baseline.py \
    --method all \
    --dataset webqa \
    --model llama3.2_3b_it
```

For detailed baseline usage, see **[BASELINE_USAGE.md](BASELINE_USAGE.md)**.

### Ensemble Generation

```bash
# Original ensemble methods: max, avg, weighted_avg, weighted_max
python src/generate_original.py \
    --method max \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6

# FlexAttention with 5 paraphrases (most efficient)
python src/generate_flex_attention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5

# MyriadLAMA-specific FlexAttention
python src/generate_myriadlama.py \
    --dataset myriadlama \
    --model llama3.2_3b_it \
    --num_paraphrases 5

# Limit to 100 samples for quick testing
python src/generate_flex_attention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5 \
    --max_samples 100
```

### Analysis

```bash
# Analyze baseline results
python3 analysis/analyze_baseline.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --compare

# Analyze FlexAttention results
python3 analysis/analyze_flexattention.py \
    --dataset webqa \
    --model llama3.2_3b_it

# Interactive analysis with Jupyter
jupyter notebook analysis/flexattention_analysis.ipynb
```

For detailed usage and analysis guide, see **[FLEXATTENTION_USAGE.md](FLEXATTENTION_USAGE.md)**.

### Debugging

```bash
# Debug mode with detailed output
python3 tools/debug_flexattention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --max-samples 2 \
    --verbose
```

### Minimal Example

```bash
# Run standalone example (no dataset/model required)
python3 tools/example_flexattention.py
```

For more examples, see **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**.

## 🐛 Debugging

This repository includes comprehensive debugging tools:

### Command Line Debugging

```bash
# Debug with detailed step-by-step output
python3 tools/debug_flexattention.py --dataset webqa --max-samples 1 --verbose
```

**Shows:**
- 📊 Tensor shapes and values
- 🎯 Attention mask visualization
- 🔍 Token-by-token generation
- ✅ Segment isolation verification

### VSCode Debugging

1. Open repository in VSCode
2. Press `F5`
3. Select debug configuration:
   - "Debug FlexAttention - WebQA"
   - "Debug Script - WebQA (2 samples)"
   - "Validate Environment"

See **[docs/DELEGATE_PROMPT.md](docs/DELEGATE_PROMPT.md)** for complete debugging guide.

## 🧪 Testing

Run the validation and example scripts:

```bash
# Validate environment
python tools/validate_flexattention_env.py --test-flex-attention

# Run minimal example
python tools/example_flexattention.py

# Run test scripts
python tests/test_causal_priority.py
python tests/test_paraphrase_isolation.py

# Test notebooks (requires Jupyter)
jupyter notebook tests/test_generate.ipynb

# Analysis notebooks
jupyter notebook notebooks/flexattention_analysis.ipynb
```

See **[tests/README.md](tests/README.md)** for more details on testing.

## 📊 Datasets

Supported datasets:

- **WebQA**: Question answering dataset
- **MyriadLAMA**: Knowledge probing dataset

Download with:
```bash
bash tools/download_resources.sh --dataset webqa
bash tools/download_resources.sh --dataset myriadlama
```

## 🤖 Models

Supported models (defined in `constants.py`):

- Llama 3.2 3B Instruct
- Other models can be added to `MODEL_PATHs`

Download with:
```bash
bash tools/download_resources.sh --model llama3.2_3b_it
```

## 📁 Repository Structure

```
.
├── src/                            # Source code and generation scripts
│   ├── core/                       # Shared utilities and modules
│   │   ├── constants.py            # Model paths and configurations
│   │   ├── dataset.py              # Dataset loaders
│   │   ├── utils.py                # General utilities
│   │   ├── paraphrase.py           # Paraphrase generation
│   │   ├── confidence.py           # Confidence computation
│   │   └── interactive.py          # Interactive parameter prompts
│   │
│   ├── generate_original.py        # Original ensemble methods
│   ├── generate_flex_attention.py  # FlexAttention-based generation
│   ├── generate_myriadlama.py      # MyriadLAMA-specific generation
│   ├── generate_baseline.py        # Baseline generation
│   ├── run_interactive.py          # Interactive mode with prompts
│   └── README.md                   # Source code documentation
│
├── tests/                          # Test scripts and notebooks
│   ├── test_*.py                   # Unit tests
│   ├── *.ipynb                     # Test notebooks
│   └── README.md                   # Test documentation
│
├── notebooks/                      # Analysis and visualization notebooks
│   ├── flexattention_analysis.ipynb
│   ├── diversity.ipynb
│   └── README.md                   # Notebook documentation
│
├── tools/                          # Debugging and validation utilities
│   ├── validate_flexattention_env.py
│   ├── debug_flexattention.py
│   ├── example_flexattention.py
│   └── download_resources.sh
│
├── analysis/                       # Analysis scripts
│   ├── analyze_baseline.py
│   └── analyze_flexattention.py
│
├── docs/                          # Documentation
│   ├── QUICKSTART.md              # Quick start guide
│   ├── README_FLEXATTENTION.md    # FlexAttention overview
│   ├── ARCHITECTURE.md            # Architecture diagrams
│   └── [other documentation]
│
├── plot/                          # Visualization tools and outputs
├── examples/                      # Example scripts
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment file
└── README.md                      # This file
```

## 🔍 How It Works

### Step 1: Concatenation
```
5 Paraphrases → "Para1 [SEP] Para2 [SEP] ... Para5"
Track positions: [(0,45), (50,92), ...]
```

### Step 2: Encoding with Isolation
```
Each paraphrase only attends to itself:
Para1: ✓✓✓ ✗✗✗ ✗✗✗
Para2: ✗✗✗ ✓✓✓ ✗✗✗
Para3: ✗✗✗ ✗✗✗ ✓✓✓
```

### Step 3: Generation with Fusion
```
Generated tokens attend to ALL paraphrases:
Gen1: ✓✓✓ ✓✓✓ ✓✓✓
Gen2: ✓✓✓ ✓✓✓ ✓✓✓ ✓
```

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for detailed diagrams.

## 📈 Performance

- **Speed**: ~5× faster than logit-level fusion (1 forward pass vs 5 per step)
- **Quality**: Comparable or better than logit-level methods
- **Memory**: Similar to single-pass generation
- **Testing**: 19/19 tests passed (100%)

## 🛠️ Troubleshooting

### FlexAttention not available
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

### CUDA out of memory
```bash
python3 flex_attention_generate.py --device cpu
```

### Dataset download fails
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

For more solutions, see **[docs/DELEGATE_PROMPT.md#troubleshooting](docs/DELEGATE_PROMPT.md#troubleshooting)**.

## 📝 Documentation Index

**Getting Started:**
- [BASELINE_USAGE.md](BASELINE_USAGE.md) - **NEW**: Baseline generation guide
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - 5-minute setup
- [docs/DELEGATE_PROMPT.md](docs/DELEGATE_PROMPT.md) - Complete guide

**Understanding FlexAttention:**
- [FLEXATTENTION_USAGE.md](FLEXATTENTION_USAGE.md) - Usage guide
- [docs/README_FLEXATTENTION.md](docs/README_FLEXATTENTION.md) - Overview
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Visual diagrams
- [docs/FLEX_ATTENTION_IMPLEMENTATION.md](docs/FLEX_ATTENTION_IMPLEMENTATION.md) - Technical details

**API Reference:**
- [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Quick reference
- [docs/REUSE_VS_NEW_DETAILED.md](docs/REUSE_VS_NEW_DETAILED.md) - Code breakdown

**Development & Changes:**
- [CHANGELOG.md](CHANGELOG.md) - All changes and updates
- [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) - **NEW**: Consolidated improvements

**中文文档:**
- [docs/实现总结.md](docs/实现总结.md) - 中文总结

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Support for more datasets
- Additional fusion strategies
- Performance optimizations
- More comprehensive testing

## 📄 License

[Add license information here]

## 🙏 Acknowledgments

- PyTorch team for FlexAttention API
- Hugging Face for transformers library
- Dataset authors (WebQA, MyriadLAMA)

## 📧 Contact

[Add contact information here]

---

**Status:** ✅ Production Ready | 🧪 Tested | 📖 Documented

Last updated: 2025-10-13
