# Delegate Task Prompt for FlexAttention Testing Setup

## Task Overview
Create comprehensive testing files for the FlexAttention-based ensemble generation project. The goal is to help me understand how the code runs,validate the environment, download necessary resources, and enable step-by-step debugging with detailed inspection capabilities.

## ⚠️ CRITICAL REQUIREMENTS

### File Management Rules
- **DO NOT MODIFY** existing core files: `generate.py`, `flex_attention_generate.py`, `dataset.py`, `constants.py`, `utils.py`, `paraphrase.py`, `confidence.py`
- **DO NOT MODIFY** existing scripts in `scripts/` directory
- **DO NOT MODIFY** existing analysis files in `analysis/` directory
- **CREATE NEW FILES ONLY** in designated testing directories
- **ORGANIZE** all new test files in proper directory structure

### Project Directory Refactoring
**IMPORTANT**: Propose a minimal, professional, and well-organized directory structure for this small research project. Follow Python project conventions and machine learning best practices. Consider:
- Separating core code, tests, documentation, and data
- Clear naming conventions for different file types
- Logical grouping of related functionality
- Easy navigation for developers and researchers
- Standard project layout that scales well

Current structure has scattered files - please suggest a clean reorganization while preserving all existing functionality.

## Requirements

### 1. Cache Warming (Resource Download)
- **Download and cache datasets**: WebQA and MyriadLAMA from Hugging Face
- **Download and cache models**: Small models like `Qwen/Qwen2.5-3B-Instruct` for quick testing
- **Verify cache locations**: Ensure all downloads go to `/net/tokyo100-10g/data/str01_01/y-guo/` directories
- **Create cache test**: Script to verify cached resources are accessible

### 2. Environment Validation
- **Test PyTorch 2.5.1**: Verify FlexAttention APIs (`flex_attention`, `create_mask`) work correctly
- **Test transformers 4.55.2**: Ensure model loading and tokenization work
- **Test dependencies**: Verify pandas, numpy, spacy, tqdm are functional
- **Test GPU access**: Check CUDA availability and memory
- **Integration test**: Minimal end-to-end test with 1-2 samples

### 3. Step-by-Step Debugging Framework
- **Minimal dataset**: Create test with only 2-3 questions for fast iteration
- **Breakpoint suggestions**: Identify key inspection points in the generation pipeline
- **Inspection functions**: Create helper functions to visualize intermediate states
- **Progress logging**: Add detailed logging at each major step

## Specific Files to Create

### Directory: `testing/cache_warmup/`
**File 1: `test_cache_warmup.py`**
- Download 1 small model (Qwen/Qwen2.5-3B-Instruct)
- Download WebQA dataset (first 10 samples only)
- Download MyriadLAMA dataset (first 10 samples only)
- Verify cache integrity
- Print cache locations and sizes

**File 2: `cache_verification.py`**
- Verify cached resources are accessible and intact
- Test model loading speed from cache
- Report cache usage statistics

### Directory: `testing/environment/`
**File 3: `test_environment.py`**
- Test all import statements
- Test FlexAttention API with simple examples
- Test model loading and tokenization
- Test basic tensor operations
- Report environment status

**File 4: `environment_report.py`**
- Generate comprehensive environment report
- Test GPU memory and CUDA capabilities
- Benchmark basic operations

### Directory: `testing/debugging/`
**File 5: `debug_helpers.py`**
- `inspect_tokenization()`: Show token IDs, positions, attention masks
- `inspect_concatenation()`: Show how paraphrases are joined and segment positions
- `inspect_attention_mask()`: Visualize FlexAttention mask patterns
- `inspect_generation_step()`: Show logits, token selection, attention patterns
- `save_debug_info()`: Save all intermediate states to files

**File 6: `test_minimal_pipeline.py`**
- Process exactly 2 questions with 5 paraphrases each
- Add detailed logging at each step
- Include timing measurements
- Save intermediate outputs for inspection
- Use small model for fast execution

**File 7: `run_debug_session.py`**
- Main script that orchestrates debugging
- Suggested breakpoints with explanations
- Interactive mode for step-by-step execution
- Automated mode for complete pipeline testing

### Directory: `testing/`
**File 8: `README.md`**
- Complete testing guide
- How to run each test
- Troubleshooting common issues
- File structure explanation

## Debugging Breakpoints Suggestions

1. **After tokenization**: Inspect input_ids, attention_mask, segment_positions
2. **Before FlexAttention**: Check mask creation and model patching
3. **During generation loop**: Examine logits, token selection, attention patterns
4. **After each generation step**: Verify sequence extension and mask updates
5. **Final output**: Compare with expected results and measure performance

## Inspection Function Examples

```python
def inspect_step(step_name, **kwargs):
    print(f"\n=== DEBUG: {step_name} ===")
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            print(f"  Sample values: {value.flatten()[:5]}")
        else:
            print(f"{key}: {type(value)} = {value}")
    print("=" * (len(step_name) + 16))
```

## Performance Requirements
- **Fast execution**: Each test should complete in under 2 minutes
- **Small memory footprint**: Use models under 4GB
- **Clear output**: All logs should be easily readable and informative
- **Error handling**: Graceful failure with helpful error messages

## Expected Deliverables
1. **8 Python files** organized in the specified directory structure
2. **Complete testing directory** with proper organization
3. **Each file standalone and executable** with clear documentation
4. **README.md** with comprehensive testing guide
5. **Example usage commands** for each test scenario
6. **Error handling** with helpful error messages
7. **NO MODIFICATIONS** to existing core project files

## File Safety Guidelines
- **Read-only access** to existing files for reference
- **Import existing modules** rather than copying code
- **Create wrapper functions** if needed to extend functionality
- **Document dependencies** on existing files clearly
- **Test in isolation** to avoid breaking existing functionality

## Quality Standards
- **Fast execution**: Each test completes in under 2 minutes
- **Small memory footprint**: Use models under 4GB
- **Clear output**: All logs easily readable and informative
- **Graceful error handling**: Helpful error messages
- **Cross-platform compatibility**: Works on different systems

## Technical Context
- **Project**: FlexAttention-based text generation ensemble
- **Framework**: PyTorch 2.5.1, transformers 4.55.2
- **Environment**: conda environment `self-ensemble-debug`
- **Main script**: `flex_attention_generate.py`
- **Data format**: Feather files for input/output
- **Model type**: Causal language models (LLaMA, Qwen series)

## Success Criteria
- ✅ All tests pass without errors
- ✅ Cache is properly populated in user's directory (`/net/tokyo100-10g/data/str01_01/y-guo/`)
- ✅ Debugging framework provides clear insights into pipeline behavior
- ✅ User can easily identify and fix issues if they arise
- ✅ Documentation is clear for non-native English speakers
- ✅ **NO EXISTING FILES ARE MODIFIED OR BROKEN**
- ✅ **Clean, organized directory structure is maintained**
- ✅ All new files follow consistent naming and documentation standards

## Example Usage Commands
```bash
# Navigate to project directory
cd /home/y-guo/self-ensemble/self-ensemble

# Run cache warmup
python testing/cache_warmup/test_cache_warmup.py

# Test environment
python testing/environment/test_environment.py

# Run minimal pipeline test
python testing/debugging/test_minimal_pipeline.py

# Start interactive debugging session
python testing/debugging/run_debug_session.py --interactive

# Generate environment report
python testing/environment/environment_report.py
```

Please create these testing files with comprehensive error handling, clear documentation, helpful debugging capabilities, and **strict adherence to the file safety guidelines**.