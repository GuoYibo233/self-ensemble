# Tests Directory

This directory contains test scripts and notebooks for validating the self-ensemble implementation, organized into subdirectories by functionality.

## Directory Structure

```
tests/
├── flexattention/      # FlexAttention-related tests
├── myriadlama/        # MyriadLAMA-specific tests
├── unit/              # General unit tests
└── notebooks/         # Interactive Jupyter notebooks
```

## Subdirectories

### flexattention/
Tests for FlexAttention functionality and mask creation:
- Causal mask priority verification
- Mask creation and segment isolation
- Paraphrase isolation testing
- Separator token handling
- Visualization utilities

See [flexattention/README.md](flexattention/README.md) for details.

### myriadlama/
Tests specific to MyriadLAMA dataset:
- MyriadLAMA-specific FlexAttention implementation
- Mask creation with sample prompts
- Mask structure verification
- Few-shot example handling

See [myriadlama/README.md](myriadlama/README.md) for details.

### unit/
General unit tests for various components:
- Analysis and reporting functions
- Prompt formatting
- Visualization utilities

See [unit/README.md](unit/README.md) for details.

### notebooks/
Interactive Jupyter notebooks for testing:
- Generation functionality testing
- Dataset exploration
- Confidence computation testing
- Baseline method testing

See [notebooks/README.md](notebooks/README.md) for details.

## Running Tests

### Run tests from a specific category

```bash
# FlexAttention tests
for test in tests/flexattention/test_*.py; do python "$test"; done

# MyriadLAMA tests
for test in tests/myriadlama/test_*.py; do python "$test"; done

# Unit tests
for test in tests/unit/test_*.py; do python "$test"; done
```

### Run all Python tests

```bash
find tests/ -name "test_*.py" -type f -exec python {} \;
```

### Run Jupyter notebooks

```bash
jupyter notebook tests/notebooks/
```

## Test Guidelines

- Use descriptive test names
- Include clear success/failure messages
- Document expected behavior
- Test edge cases
- Keep tests independent (don't rely on other test outputs)

## Adding New Tests

When adding new functionality:

1. Determine the appropriate subdirectory (flexattention, myriadlama, unit, or notebooks)
2. Create a test file with descriptive name
3. Include clear documentation
4. Update the subdirectory's README.md
5. Update this main README if adding a new category
