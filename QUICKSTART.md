# Quick Start Guide - FlexAttention Debugging

This guide gets you up and running with FlexAttention debugging in under 5 minutes.

## Prerequisites

- Python 3.10+
- 20GB free disk space
- (Optional) NVIDIA GPU with CUDA support

## Step 1: Install Dependencies (2 minutes)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# or: venv\Scripts\activate  # On Windows

# Install PyTorch with FlexAttention support
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# Install other dependencies
pip install transformers pandas numpy tqdm datasets spacy

# Download spaCy model
python3 -m spacy download en_core_web_lg
```

## Step 2: Validate Environment (30 seconds)

```bash
python3 validate_flexattention_env.py --test-flex-attention
```

You should see:
```
✅ Python 3.12.3 - Compatible
✅ PyTorch 2.x - Installed
✅ FlexAttention API available
...
🎉 Environment validation PASSED!
```

## Step 3: Download Resources (3-10 minutes depending on network)

```bash
# Download WebQA dataset and model
bash download_resources.sh --dataset webqa --model llama3.2_3b_it
```

This will:
- Download the WebQA dataset (~100MB)
- Download the Llama 3.2 3B model (~6GB)
- Download spaCy language model (~500MB)

## Step 4: Run Debug Session (1 minute)

```bash
# Run on 1 sample to see how the code works
python3 debug_flexattention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --max-samples 1 \
    --max-tokens 5 \
    --verbose
```

This will show you:
- 📊 How paraphrases are concatenated
- 🎯 How the attention mask isolates segments
- 🔍 Token-by-token generation with logits
- ✅ Verification that segment isolation works

## Step 5: Run Full Generation (5-30 minutes depending on dataset size)

```bash
# Run FlexAttention generation on WebQA
python3 flex_attention_generate.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5
```

Results will be saved to `datasets/webqa/flex_attention-5.feather`

## Quick Commands Reference

### Debugging
```bash
# Debug 1 sample with verbose output
python3 debug_flexattention.py --dataset webqa --model llama3.2_3b_it --max-samples 1 --verbose

# Debug with specific paraphrases
python3 debug_flexattention.py --dataset webqa --model llama3.2_3b_it --indexs 0,1,2 --max-samples 1
```

### Validation
```bash
# Quick validation
python3 validate_flexattention_env.py

# Full validation with FlexAttention test
python3 validate_flexattention_env.py --test-flex-attention --verbose
```

### Resource Management
```bash
# List available resources
bash download_resources.sh --list

# Download only dataset (no model)
bash download_resources.sh --dataset-only webqa

# Download only spaCy model
bash download_resources.sh --spacy
```

### Generation
```bash
# FlexAttention with 5 paraphrases
python3 flex_attention_generate.py --dataset webqa --model llama3.2_3b_it --num_paraphrases 5

# With specific paraphrase indices
python3 flex_attention_generate.py --dataset webqa --model llama3.2_3b_it --indexs 0,1,2,3,4

# Lemmatize results (after generation)
python3 flex_attention_generate.py --dataset webqa --model llama3.2_3b_it --num_paraphrases 5 --lemmaize
```

## VSCode Debugging

1. Open the repository in VSCode
2. Press `F5` or click Run → Start Debugging
3. Select one of the debug configurations:
   - **Debug FlexAttention - WebQA**: Run full generation with breakpoints
   - **Debug Script - WebQA (2 samples)**: Run debug script with breakpoints
   - **Validate Environment**: Run environment validation

## Troubleshooting

### Issue: FlexAttention not available
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
python3 -c "from torch.nn.attention.flex_attention import flex_attention; print('OK')"
```

### Issue: CUDA out of memory
```bash
# Use CPU instead
python3 flex_attention_generate.py --dataset webqa --model llama3.2_3b_it --device cpu

# Or use smaller batch size (edit dataset.py)
```

### Issue: Dataset download fails
```bash
# Check network connection
ping huggingface.co

# Set proxy if needed
export HTTP_PROXY=http://your.proxy:port
export HTTPS_PROXY=http://your.proxy:port
```

## Next Steps

1. **Read the documentation**:
   - [DELEGATE_PROMPT.md](DELEGATE_PROMPT.md) - Full debugging guide
   - [README_FLEXATTENTION.md](README_FLEXATTENTION.md) - FlexAttention overview
   - [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - API reference

2. **Understand the code**:
   - Read `flex_attention_generate.py` with comments
   - Check out the architecture diagrams in [ARCHITECTURE.md](ARCHITECTURE.md)

3. **Experiment**:
   - Try different numbers of paraphrases
   - Compare with original methods (per_prompt, avg, max)
   - Analyze the results

## Summary

✅ You've learned to:
- Install and validate the FlexAttention environment
- Download necessary datasets and models
- Run debug sessions to understand the code
- Generate results using FlexAttention

**Total setup time: ~5-15 minutes** (plus download time)

For detailed information, see [DELEGATE_PROMPT.md](DELEGATE_PROMPT.md).
