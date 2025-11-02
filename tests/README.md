# Tests Directory

This directory contains test scripts and notebooks for validating the self-ensemble implementation.

## Contents

### Jupyter Notebooks

Interactive notebooks for testing and exploration:

- **test_generate.ipynb**: Test generation functionality
- **test_dataset.ipynb**: Test dataset loading and processing
- **test_confidence.ipynb**: Test confidence computation
- **baselines.ipynb**: Test baseline generation methods

### Test Scripts

Python scripts for validating specific functionality:

- **test_causal_priority.py**: Verify causal mask has highest priority
- **test_create_flex_attention_mask.py**: Test FlexAttention mask creation
- **test_mask_visualization.py**: Visualize attention masks
- **test_myriadlama_flex_attention.py**: Test MyriadLAMA FlexAttention
- **test_myriadlama_mask_sample.py**: Test MyriadLAMA mask samples
- **test_myriadlama_mask_structure.py**: Test MyriadLAMA mask structure
- **test_new_prompt_format.py**: Test new prompt formatting
- **test_paraphrase_isolation.py**: Test paraphrase isolation in masks
- **test_separator_fix.py**: Test separator token handling
- **test_vmap_fix.py**: Test vmap functionality

## Running Tests

### Jupyter Notebooks

Open notebooks with Jupyter:
```bash
jupyter notebook tests/test_generate.ipynb
```

### Python Test Scripts

Run individual test scripts:
```bash
python tests/test_causal_priority.py
```

Run all tests:
```bash
for test in tests/test_*.py; do
    echo "Running $test..."
    python "$test"
done
```

## Test Organization

Tests are organized by functionality:

1. **Core Functionality Tests**: Dataset, generation, confidence
2. **FlexAttention Tests**: Mask creation, isolation, causality
3. **MyriadLAMA Tests**: Dataset-specific functionality
4. **Visualization Tests**: Mask and result visualization

## Adding New Tests

When adding new functionality:

1. Create a test script or notebook in this directory
2. Name it descriptively (test_<feature>.py or test_<feature>.ipynb)
3. Include clear documentation and expected outcomes
4. Update this README with the new test

## Test Guidelines

- Use descriptive test names
- Include clear success/failure messages
- Document expected behavior
- Test edge cases
- Keep tests independent (don't rely on other test outputs)
