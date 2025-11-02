# MyriadLAMA Tests

Tests specific to MyriadLAMA dataset and its FlexAttention implementation.

## Test Files

- **test_myriadlama_flex_attention.py**: Test MyriadLAMA-specific FlexAttention implementation
- **test_myriadlama_mask_sample.py**: Test mask creation with sample MyriadLAMA prompts
- **test_myriadlama_mask_structure.py**: Verify mask structure for MyriadLAMA prompts

## Running Tests

Run individual tests:
```bash
python tests/myriadlama/test_myriadlama_flex_attention.py
```

Run all MyriadLAMA tests:
```bash
for test in tests/myriadlama/test_*.py; do
    echo "Running $test..."
    python "$test"
done
```

## Test Coverage

These tests verify:
- MyriadLAMA-specific prompt formatting
- Mask logic for few-shot examples
- Segment isolation for fill-in-the-blank tasks
- Proper handling of [MASK] token prediction
