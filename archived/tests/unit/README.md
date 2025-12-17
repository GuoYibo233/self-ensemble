# Unit Tests

General unit tests for various components.

## Test Files

- **test_analyze_detailed.py**: Test detailed analysis functionality
- **test_new_prompt_format.py**: Test new prompt formatting
- **test_plot_visualization.py**: Test plot and visualization utilities

## Running Tests

Run individual tests:
```bash
python tests/unit/test_analyze_detailed.py
```

Run all unit tests:
```bash
for test in tests/unit/test_*.py; do
    echo "Running $test..."
    python "$test"
done
```

## Test Coverage

These tests verify:
- Analysis and reporting functions
- Prompt formatting and construction
- Visualization utilities
