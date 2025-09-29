# timepoint-daedalus

Temporal entity simulation with LLM-driven training and tensor compression.

## Overview

Timepoint-Daedalus is a framework for simulating temporal entities with:
- **LLM-driven entity population** using Instructor for structured outputs
- **Graph-based entity relationships** using NetworkX
- **Tensor compression** (PCA, SVD, NMF) for efficient entity representation
- **Temporal validation** ensuring biological plausibility and information conservation
- **LangGraph workflows** for parallel entity training
- **SQLModel persistence** for entities, timelines, and graphs

## Installation

### Quick Install (Recommended)

Use the provided installation script that handles all compatibility issues:

```bash
./install.sh
```

### Using Poetry (Recommended)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clean any existing locks and cache (if reinstalling)
poetry cache clear pypi --all
rm -f poetry.lock

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

**macOS Apple Silicon (M1/M2/M3) Users:**

If you encounter grpcio build errors, try:

```bash
# Set environment variables for better compatibility
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Install with pre-built wheels
poetry install

# Or install grpcio separately first
pip install --upgrade grpcio
poetry install
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `conf/config.yaml` to configure:
- Database connection
- LLM settings (API key, base URL)
- Autopilot parameters
- Training settings

## Usage

### Run Autopilot Mode
```bash
python cli.py mode=autopilot
```

### Run Evaluation
```bash
python cli.py mode=evaluate
```

### Run Training
```bash
python cli.py mode=train
```

### Override Configuration
```bash
# Change graph sizes for autopilot
python cli.py mode=autopilot autopilot.graph_sizes=[5,10,20]

# Enable dry-run mode
python cli.py mode=train llm.dry_run=true

# Change target resolution
python cli.py mode=train training.target_resolution=scene
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run with verbose logging
pytest --verbose-tests -s

# Run specific test
pytest test_framework.py::test_tensor_compression

# Run specific test with verbose logging
pytest --verbose-tests -s test_framework.py::test_tensor_compression

# Run property-based tests
pytest test_framework.py::test_graph_creation_property
```

See [VERBOSE_TESTING.md](VERBOSE_TESTING.md) for detailed logging documentation.

## Architecture

### Core Components

- **schemas.py**: SQLModel schemas (Entity, Timeline, SystemPrompt, ValidationRule)
- **storage.py**: Database and graph persistence layer
- **llm.py**: LLM client with Instructor integration
- **workflows.py**: LangGraph workflow definitions
- **validation.py**: Pluggable validation framework
- **tensors.py**: Tensor compression with plugin registry
- **evaluation.py**: Evaluation metrics (coherence, consistency, plausibility)
- **graph.py**: NetworkX graph creation and centrality metrics
- **test_framework.py**: Pytest fixtures and tests

### Resolution Levels

1. **TENSOR_ONLY**: Compressed tensor representation only
2. **SCENE**: Scene-level context
3. **GRAPH**: Full graph context
4. **DIALOG**: Dialog-level detail
5. **TRAINED**: Fully trained entity

### Validation Rules

- **Information Conservation**: Knowledge ⊆ exposure history
- **Energy Budget**: Interaction costs ≤ capacity
- **Behavioral Inertia**: Gradual personality drift
- **Biological Constraints**: Age-dependent capabilities

## Development

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy .
```

### Adding New Validators

```python
@Validator.register("custom_validator", "WARNING")
def validate_custom(entity: Entity, context: Dict) -> Dict:
    # Your validation logic
    return {"valid": True, "message": "Custom validation passed"}
```

### Adding New Tensor Compressors

```python
@TensorCompressor.register("custom_method")
def custom_compress(tensor: np.ndarray, n_components: int = 8) -> np.ndarray:
    # Your compression logic
    return compressed_tensor
```

## License

[Your License Here]