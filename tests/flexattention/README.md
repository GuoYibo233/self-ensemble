# FlexAttention Tests

Tests for FlexAttention functionality and mask creation.

## Test Files

- **test_causal_priority.py**: Verify that causal mask has highest priority in attention masks
- **test_create_flex_attention_mask.py**: Test FlexAttention mask creation and segment isolation
- **test_mask_visualization.py**: Visualize attention masks for debugging
- **test_paraphrase_isolation.py**: Test paraphrase isolation in FlexAttention masks
- **test_separator_fix.py**: Test separator token handling in concatenated prompts
- **test_vmap_fix.py**: Test vmap functionality for batched operations

## Verification Scripts

- **verify_causal_priority.py**: Standalone verification of causal mask priority
- **verify_mask_logic.py**: Standalone verification of mask logic

## Running Tests

Run individual tests:
```bash
python tests/flexattention/test_causal_priority.py
```

Run all FlexAttention tests:
```bash
for test in tests/flexattention/test_*.py; do
    echo "Running $test..."
    python "$test"
done
```

Run verification scripts:
```bash
python tests/flexattention/verify_causal_priority.py
python tests/flexattention/verify_mask_logic.py
```

## Test Coverage

These tests verify:
- Causal mask priority in attention mechanisms
- Segment isolation during encoding phase
- Proper attention fusion during generation
- Separator token handling
- Mask visualization for debugging
