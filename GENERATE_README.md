# Generate Scripts - Usage Guide

This project contains four different generation scripts, each implementing a different generation method. This document explains the differences between all generate scripts and how to use them.

## 📋 Scripts Overview

| Script | Purpose | Dataset | Method Type |
|--------|---------|---------|-------------|
| `generate_baseline.py` | Baseline testing | WebQA | Individual generation (origin/per_prompt) |
| `generate_original.py` | Original ensemble methods | WebQA | Logit-level fusion (max/avg/weighted) |
| `generate_flex_attention.py` | FlexAttention ensemble | WebQA | Attention-level fusion |
| `generate_myriadlama.py` | MyriadLAMA-specific method | MyriadLAMA | FlexAttention (for fill-in-the-blank) |

---

## 1. generate_baseline.py - Baseline Generation

**Purpose**: Provide baseline comparison results for ensemble methods

**Dataset**: WebQA

**Supported Methods**:
- `origin`: Use only original question (Baseline 1)
- `per_prompt`: Generate each paraphrase separately (Baseline 2)

### Usage

```bash
# Baseline 1: Original question only
python src/generate_baseline.py \
    --method origin \
    --dataset webqa \
    --model llama3.2_3b_it

# Baseline 2: Each paraphrase separately
python src/generate_baseline.py \
    --method per_prompt \
    --dataset webqa \
    --model llama3.2_3b_it

# Generate all baselines
python src/generate_baseline.py \
    --method all \
    --dataset webqa \
    --model llama3.2_3b_it
```

### Features
- ✅ Simplest generation method
- ✅ No ensemble techniques used
- ✅ Suitable as comparison baseline

---

## 2. generate_original.py - Original Ensemble Methods

**Purpose**: Auto-generate paraphrases and ensemble them using WebQA dataset

**Dataset**: WebQA

**Supported Methods**:
- `max`: Select maximum logit at each step
- `avg`: Average all logits
- `weighted_avg`: Confidence-weighted average
- `weighted_max`: Confidence-weighted maximum

### Usage

```bash
# Maximum ensemble
python src/generate_original.py \
    --method max \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6

# Average ensemble
python src/generate_original.py \
    --method avg \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6

# Weighted average ensemble
python src/generate_original.py \
    --method weighted_avg \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6
```

### Features
- ✅ Traditional ensemble methods
- ✅ Logit-level fusion
- ❌ N forward passes required per step (less efficient)
- ✅ Supports multiple fusion strategies

---

## 3. generate_flex_attention.py - FlexAttention Ensemble

**Purpose**: Efficient attention-level ensemble using FlexAttention

**Dataset**: WebQA

**Method**: FlexAttention - Fuses multiple paraphrases in a single forward pass

### Usage

```bash
# FlexAttention ensemble (5 paraphrases)
python src/generate_flex_attention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5

# Limit sample count (quick testing)
python src/generate_flex_attention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5 \
    --max_samples 100
```

### How It Works

1. **Concatenation**: Concatenate paraphrases together
   ```
   [ins fewshot paraphrase1] [ins fewshot paraphrase2] [ins fewshot paraphrase3] [ins fewshot paraphrase4] [ins fewshot paraphrase5]
   ```

2. **Isolated Encoding**: Each paraphrase isolated during encoding
   ```
   Para1: ✓✓✓ ✗✗✗ ✗✗✗ ✗✗✗ ✗✗✗
   Para2: ✗✗✗ ✓✓✓ ✗✗✗ ✗✗✗ ✗✗✗
   Para3: ✗✗✗ ✗✗✗ ✓✓✓ ✗✗✗ ✗✗✗
   ```

3. **Fused Generation**: Generated tokens attend to all paraphrases
   ```
   Gen1: ✓✓✓ ✓✓✓ ✓✓✓ ✓✓✓ ✓✓✓
   Gen2: ✓✓✓ ✓✓✓ ✓✓✓ ✓✓✓ ✓✓✓ ✓
   ```

### Features
- ✅ **Most efficient**: Only 1 forward pass per step (vs N)
- ✅ Attention-level fusion
- ✅ Quality comparable to or better than logit-level methods
- ⚠️ Requires PyTorch 2.5+ or nightly

### Environment Requirements

```bash
# Install PyTorch nightly (supports FlexAttention)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

---

## 4. generate_myriadlama.py - MyriadLAMA-Specific Generation

**Purpose**: Only use MyriadLAMA dataset for fill-in-the-blank tasks

**Dataset**: MyriadLAMA

**Method**: FlexAttention (optimized for fill-in-the-blank tasks)

### Usage

```bash
# MyriadLAMA FlexAttention generation
python src/generate_myriadlama.py \
    --dataset myriadlama \
    --model llama3.2_3b_it \
    --num_paraphrases 5
```

### Differences from generate_flex_attention.py

| Feature | generate_flex_attention.py | generate_myriadlama.py |
|---------|---------------------------|------------------------|
| **Dataset** | WebQA (Q&A) | MyriadLAMA (fill-in-the-blank) |
| **Task Type** | Long text generation | Word prediction |
| **Prompt Format** | Standard Q&A format | [MASK] fill-in-the-blank format |
| **Mask Logic** | Paraphrase isolation | Paraphrase + Few-shot example isolation |
| **Few-shot** | Standard few-shot | Each example independently isolated |
| **Output Length** | Variable length | Usually single token |

### Features
- ✅ Optimized for MyriadLAMA
- ✅ Supports [MASK] fill-in-the-blank tasks
- ✅ Few-shot examples mutually isolated
- ✅ Optimized for word prediction

---

## 5. myriadlama_custom_attention_generate.py - MyriadLAMA Custom Attention

**Purpose**: Custom attention-based ensemble generation specifically designed for MyriadLAMA dataset

**Dataset**: MyriadLAMA

**Method**: Uses HuggingFace's native attention mask mechanism with custom structure masks

### Usage

```bash
# MyriadLAMA custom attention generation
python myriadlama_custom_attention_generate.py \
    --dataset myriadlama \
    --model llama3.2_3b_it
```

### Key Features
- Uses HuggingFace's native attention mask mechanism
- Patches LLaMA's `_update_causal_mask` to inject custom structure masks
- Implements segment-based masking:
  * Causal mask is always applied
  * Shared part (instruction + few-shot) uses normal causal attention
  * Each para part can attend to itself and the shared part
  * Para parts are isolated from each other

### Attention Pattern

```
    shared_part | para_1 | para_2 | para_3 | ...
    ------------+--------+--------+--------+----
    normal      | see    | see    | see    | ...   <- shared tokens
    causal      | shared | shared | shared | ...
                | only   | only   | only   |
    ------------+--------+--------+--------+----
    can see     | normal | X      | X      | ...   <- para_1 tokens
    shared      | causal |        |        |
    ------------+--------+--------+--------+----
    can see     | X      | normal | X      | ...   <- para_2 tokens
    shared      |        | causal |        |
    ------------+--------+--------+--------+----
    ...
```

### Features
- ✅ Native HuggingFace integration
- ✅ Flexible attention masking
- ✅ Optimized for MyriadLAMA structure
- ✅ Provides placeholder interface for Qwen model support

---

## 🔄 Method Comparison Summary

### Efficiency Comparison

| Method | Forward Passes per Step | Relative Speed | Fusion Level |
|--------|------------------------|----------------|--------------|
| baseline (origin) | 1× | Fastest | No fusion |
| baseline (per_prompt) | N× | Standard | No fusion |
| original (max/avg) | N× | Standard | Logit-level |
| flex_attention | 1× | **Fastest** | **Attention-level** |
| myriadlama | 1× | **Fastest** | **Attention-level** |
| myriadlama_custom_attention | 1× | **Fastest** | **Attention-level** |

### Quality Comparison

| Method | Accuracy | Diversity | Use Case |
|--------|----------|-----------|----------|
| baseline (origin) | Baseline | Low | Comparison baseline |
| baseline (per_prompt) | Medium | High | Comparison baseline |
| original (max/avg) | High | Medium | General Q&A |
| flex_attention | **High** | High | General Q&A (Recommended) |
| myriadlama | **High** | High | Fill-in-the-blank |
| myriadlama_custom_attention | **High** | High | Fill-in-the-blank (custom masks) |

---

## 📦 Dependency Files

All generate scripts depend on the following core modules:

### Core Modules (src/core/)
- `constants.py` - Model path configuration
- `dataset.py` - Dataset loaders
- `paraphrase.py` - Paraphrase generation
- `confidence.py` - Confidence calculation
- `utils.py` - General utility functions
- `interactive.py` - Interactive parameter input

### Root Directory Modules
- `mask_visualization.py` - Attention mask visualization (used only by flex_attention and myriadlama)

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate flexattention

# Or use Linux-specific environment
conda env create -f environment_linux.yml
conda activate self-ensemble-debug
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 3. Run Generation

```bash
# Recommended: Use interactive mode
python src/run_interactive.py

# Or run specific script directly
python src/generate_flex_attention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5
```

---

## 📊 Choosing the Right Script

### By Task Type

- **WebQA Q&A tasks** → `generate_flex_attention.py` (Recommended)
- **MyriadLAMA fill-in-the-blank** → `generate_myriadlama.py` or `myriadlama_custom_attention_generate.py`
- **Need comparison baseline** → `generate_baseline.py`
- **Research different fusion methods** → `generate_original.py`

### By Efficiency Requirements

- **Fastest speed** → `generate_flex_attention.py`, `generate_myriadlama.py`, or `myriadlama_custom_attention_generate.py`
- **Standard speed, multiple methods** → `generate_original.py`
- **Simple baseline** → `generate_baseline.py`

---

## 🗂️ File Structure

```
.
├── src/
│   ├── core/                      # Core modules
│   │   ├── constants.py           # Configuration
│   │   ├── dataset.py             # Datasets
│   │   ├── paraphrase.py          # Paraphrases
│   │   ├── confidence.py          # Confidence
│   │   ├── utils.py               # Utilities
│   │   └── interactive.py         # Interactive
│   │
│   ├── generate_baseline.py       # Baseline generation
│   ├── generate_original.py       # Original ensemble
│   ├── generate_flex_attention.py # FlexAttention
│   ├── generate_myriadlama.py     # MyriadLAMA
│   └── run_interactive.py         # Interactive runner
│
├── myriadlama_custom_attention_generate.py  # MyriadLAMA custom attention
├── mask_visualization.py          # Mask visualization
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
└── archived/                      # Archived files
    ├── docs/                      # Detailed documentation
    ├── tests/                     # Test files
    ├── tools/                     # Tool scripts
    └── ...                        # Other archived files
```

---

## 📝 Notes

1. **FlexAttention scripts** (`generate_flex_attention.py` and `generate_myriadlama.py`) require PyTorch 2.5+ or nightly
2. All scripts support `--max_samples` parameter for quick testing
3. Use `--help` to view complete parameter list for each script
4. Archived documentation (`archived/docs/`) contains more detailed technical explanations

---

## 🔗 Related Links

- Main README: [README.md](README.md)
- Archived Documentation: [archived/docs/](archived/docs/)
- Interactive Runner: `python src/run_interactive.py`

---

**Last Updated**: 2025-12-17
