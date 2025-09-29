# Verbose Test Logging Guide

## Overview

The test framework now supports detailed verbose logging that can be toggled with a command-line flag.

## Usage

### Run tests with verbose logging:
```bash
pytest --verbose-tests
```

### Run specific test with verbose logging:
```bash
pytest --verbose-tests test_framework.py::test_tensor_compression
```

### Run with pytest's verbose mode AND logging:
```bash
pytest -v --verbose-tests -s
```

## Logging Levels

When `--verbose-tests` is enabled:
- **INFO** level: High-level test progress (test start/end)
- **DEBUG** level: Detailed operations (tensor shapes, graph stats, entity details)

When verbose mode is OFF:
- **WARNING** level only (minimal noise)

## Example Output

### Without `--verbose-tests`:
```
============================= test session starts ==============================
test_framework.py .....                                                  [100%]
========================= 5 passed in 2.81s =========================
```

### With `--verbose-tests`:
```
============================= test session starts ==============================
test_framework.py 16:23:42 - conftest - INFO - ================================================================================
16:23:42 - conftest - INFO - VERBOSE TEST MODE ENABLED
16:23:42 - conftest - INFO - ================================================================================
16:23:42 - test_framework - INFO - Starting test_tensor_compression
16:23:42 - test_framework - DEBUG - Creating random tensor: shape (10, 32)
16:23:42 - test_framework - DEBUG - Tensor created with shape: (10, 32), mean: -0.0454, std: 0.9271
16:23:42 - test_framework - DEBUG - Compressing with PCA (n_components=8)
16:23:42 - test_framework - DEBUG - PCA compression result length: 80
16:23:42 - test_framework - DEBUG - Compressing with SVD (n_components=8)
16:23:42 - test_framework - DEBUG - SVD compression result length: 80
16:23:42 - test_framework - INFO - ✓ test_tensor_compression passed
.
```

## What Gets Logged

### Fixtures
- GraphStore creation
- LLMClient initialization
- Test graph generation with node/edge counts
- Timeline graph creation

### Unit Tests

**test_tensor_compression:**
- Tensor shape and statistics
- Compression method and results
- Output lengths

**test_entity_storage:**
- Entity creation details
- Save/load operations
- Entity IDs and types

**test_validation_registry:**
- Entity metadata
- Validation context
- Individual validator results
- Violation details (if any)

**test_graph_creation_property:**
- Graph size (n_entities)
- Nodes and edges created
- Multiple hypothesis test cases

**test_full_workflow:**
- Workflow creation
- Initial state setup
- Workflow execution
- Final state summary
- Results breakdown

## Configuration

The verbose mode is configured in `conftest.py`:
```python
def pytest_addoption(parser):
    parser.addoption(
        "--verbose-tests",
        action="store_true",
        default=False,
        help="Enable verbose logging during tests"
    )
```

## Adding Verbose Logging to New Tests

```python
def test_my_feature(verbose_mode):
    logger.info("Starting test_my_feature")
    
    logger.debug("Setting up test data")
    data = create_test_data()
    logger.debug(f"Data created: {len(data)} items")
    
    logger.debug("Running operation")
    result = my_function(data)
    logger.debug(f"Result: {result}")
    
    assert result is not None
    logger.info("✓ test_my_feature passed")
```

**Note:** For hypothesis tests, don't add `verbose_mode` parameter:
```python
@given(st.integers(min_value=1, max_value=100))
def test_with_hypothesis(value):
    # logger is already configured globally
    logger.info(f"Testing with value={value}")
    # ... rest of test
```

## Tips

1. **Use `-s` flag** to see output in real-time: `pytest --verbose-tests -s`
2. **Combine with pytest -v** for maximum detail: `pytest -v --verbose-tests -s`
3. **Filter by test name** to focus on specific tests: `pytest --verbose-tests -k "tensor"`
4. **Check log format** in `conftest.py` if you need to customize timestamps or formats
