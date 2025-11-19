# PyPI Readiness Plan: Timepoint Package Restructuring

**Version:** 1.0
**Date:** 2025-11-19
**Target:** Package `timepoint` for PyPI with proper src-layout structure

---

## Executive Summary

This plan outlines the restructuring of `timepoint-daedalus` into a PyPI-ready package named `timepoint` using Python's standard src-layout pattern. The migration involves organizing scattered root-level modules into a proper package hierarchy, updating imports, and preparing all necessary packaging metadata.

**Estimated Effort:** 4-8 hours
**Risk Level:** Medium (requires careful import updates and testing)

---

## Current State Assessment

### Problems with Current Structure

1. **Root-level module scatter**: 15+ core modules at repository root
2. **Import inconsistency**: Mix of root imports and package imports
3. **Test organization**: 80+ test files at root instead of `tests/` directory
4. **Package configuration**: `pyproject.toml` has incorrect package specification
5. **Name mismatch**: Project named "timepoint-daedalus" but should be "timepoint"

### Existing Assets (Good)

✅ Poetry-based dependency management
✅ Proper sub-packages with `__init__.py` files
✅ CLI entry point configured
✅ Comprehensive test suite with 80%+ coverage target
✅ Code quality tools configured (ruff, black, mypy)

---

## Target Structure

```
timepoint-daedalus/                    # Repository root
├── src/
│   └── timepoint/                     # Main package namespace
│       ├── __init__.py                # Package initialization, version export
│       │
│       ├── cli.py                     # CLI entry point (moved from root)
│       ├── orchestrator.py            # Core orchestration (moved from root)
│       ├── schemas.py                 # Core schemas (moved from root)
│       ├── storage.py                 # Storage layer (moved from root)
│       ├── llm_v2.py                  # LLM interface (moved from root)
│       ├── query_interface.py         # Query interface (moved from root)
│       ├── config_manager.py          # Config management (moved from root)
│       ├── data_models.py             # Data models (moved from root)
│       ├── entity_models.py           # Entity models (moved from root)
│       ├── simulation.py              # Simulation engine (moved from root)
│       ├── tensor_compression.py      # Compression (moved from root)
│       ├── training.py                # Training logic (moved from root)
│       ├── utils.py                   # Utilities (moved from root)
│       │
│       ├── llm_service/               # LLM service package (moved from root)
│       │   ├── __init__.py
│       │   ├── client.py
│       │   ├── prompts.py
│       │   └── ... (9 modules)
│       │
│       ├── nl_interface/              # Natural language interface
│       │   ├── __init__.py
│       │   └── ... (5 modules)
│       │
│       ├── workflows/                 # Workflow orchestration
│       │   ├── __init__.py
│       │   └── ... (modules)
│       │
│       ├── reporting/                 # Reporting & export
│       │   ├── __init__.py
│       │   └── ... (6 modules)
│       │
│       ├── generation/                # Generation engine
│       │   ├── __init__.py
│       │   └── ... (11 modules)
│       │
│       ├── monitoring/                # Real-time monitoring
│       │   ├── __init__.py
│       │   └── ... (7 modules)
│       │
│       ├── metadata/                  # Run tracking & coverage
│       │   ├── __init__.py
│       │   └── ... (6 modules)
│       │
│       └── andos/                     # ANDOS layer system
│           ├── __init__.py
│           └── ... (modules)
│
├── tests/                             # All tests moved here
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   ├── test_cli.py
│   ├── test_orchestrator.py
│   ├── test_schemas.py
│   └── ... (80+ test files)
│
├── conf/                              # Hydra configuration (stays)
├── datasets/                          # Data files (stays)
├── docs/                              # Documentation (create if needed)
│
├── pyproject.toml                     # Updated package config
├── README.md                          # User-facing documentation
├── LICENSE                            # License file (ADD)
├── CHANGELOG.md                       # Version history (ADD)
├── .gitignore                         # Git ignores
└── CLAUDE.md                          # Project instructions
```

---

## Phase 1: Preparation (1 hour)

### 1.1 Create Branch Protection
```bash
git checkout -b feature/pypi-restructure
```

### 1.2 Inventory All Files to Move

**Root Python modules to move:**
- cli.py
- orchestrator.py
- schemas.py
- storage.py
- llm_v2.py
- query_interface.py
- config_manager.py
- data_models.py
- entity_models.py
- simulation.py
- tensor_compression.py
- training.py
- utils.py
- workflows.py (if exists at root, may conflict with workflows/)
- Any other *.py files at root

**Sub-packages to move:**
- llm_service/
- nl_interface/
- workflows/
- reporting/
- generation/
- monitoring/
- metadata/
- andos/
- oxen_integration/
- e2e_workflows/

**Test files to move:**
- All test_*.py files
- conftest.py

### 1.3 Analyze Import Dependencies

Run import analysis to understand cross-module dependencies:
```bash
# Find all imports from root modules
grep -r "^import " *.py | sort | uniq
grep -r "^from " *.py | sort | uniq
```

Create import map: `IMPORT_MIGRATION_MAP.md`

---

## Phase 2: Create New Structure (30 min)

### 2.1 Create Directory Structure
```bash
mkdir -p src/timepoint
mkdir -p tests
mkdir -p docs
```

### 2.2 Create Package __init__.py

**File:** `src/timepoint/__init__.py`
```python
"""Timepoint: Temporal entity simulation with LLM-driven training.

A framework for simulating temporal entities with LLM-driven training,
tensor compression, and comprehensive query interfaces.
"""

__version__ = "0.1.0"
__author__ = "Sean McDonald"

# Core exports
from timepoint.orchestrator import Orchestrator
from timepoint.schemas import (
    Entity,
    TimePoint,
    SimulationConfig,
    # ... other key schemas
)
from timepoint.simulation import SimulationEngine
from timepoint.storage import StorageManager

__all__ = [
    "__version__",
    "Orchestrator",
    "Entity",
    "TimePoint",
    "SimulationConfig",
    "SimulationEngine",
    "StorageManager",
]
```

### 2.3 Create Tests __init__.py
```bash
touch tests/__init__.py
```

---

## Phase 3: File Migration (1-2 hours)

### 3.1 Move Core Modules

Use git mv to preserve history:
```bash
# Move core modules
git mv cli.py src/timepoint/cli.py
git mv orchestrator.py src/timepoint/orchestrator.py
git mv schemas.py src/timepoint/schemas.py
git mv storage.py src/timepoint/storage.py
git mv llm_v2.py src/timepoint/llm_v2.py
git mv query_interface.py src/timepoint/query_interface.py
git mv config_manager.py src/timepoint/config_manager.py
git mv data_models.py src/timepoint/data_models.py
git mv entity_models.py src/timepoint/entity_models.py
git mv simulation.py src/timepoint/simulation.py
git mv tensor_compression.py src/timepoint/tensor_compression.py
git mv training.py src/timepoint/training.py
git mv utils.py src/timepoint/utils.py

# Move any other root *.py files
```

### 3.2 Move Sub-packages
```bash
git mv llm_service/ src/timepoint/llm_service/
git mv nl_interface/ src/timepoint/nl_interface/
git mv workflows/ src/timepoint/workflows/
git mv reporting/ src/timepoint/reporting/
git mv generation/ src/timepoint/generation/
git mv monitoring/ src/timepoint/monitoring/
git mv metadata/ src/timepoint/metadata/
git mv andos/ src/timepoint/andos/
# Move others as needed
```

### 3.3 Move Test Files
```bash
# Move all test files
git mv test_*.py tests/
git mv conftest.py tests/
```

### 3.4 Commit Migration
```bash
git commit -m "refactor(structure): migrate to src-layout for PyPI packaging"
```

---

## Phase 4: Import Updates (2-3 hours)

### 4.1 Update Imports in Moved Modules

**Old imports (root-level):**
```python
from schemas import Entity, TimePoint
from storage import StorageManager
import llm_v2
from llm_service.client import LLMClient
```

**New imports (package-level):**
```python
from timepoint.schemas import Entity, TimePoint
from timepoint.storage import StorageManager
from timepoint import llm_v2
from timepoint.llm_service.client import LLMClient
```

### 4.2 Update Strategy

**Automated approach:**
```bash
# Find and replace imports across all Python files
find src/timepoint -name "*.py" -type f -exec sed -i 's/^import \([a-z_]*\)$/import timepoint.\1/g' {} \;
find src/timepoint -name "*.py" -type f -exec sed -i 's/^from \([a-z_]*\) import/from timepoint.\1 import/g' {} \;
```

**Manual verification required** - automated replacement may have false positives.

### 4.3 Update Test Imports

Tests should import from package:
```python
# Old
from orchestrator import Orchestrator

# New
from timepoint.orchestrator import Orchestrator
```

### 4.4 Handle Circular Import Issues

Watch for circular dependencies that may surface during restructuring:
- Use TYPE_CHECKING imports
- Lazy imports where appropriate
- Protocol classes for type hints

### 4.5 Commit Import Updates
```bash
git commit -m "refactor(imports): update all imports for src-layout structure"
```

---

## Phase 5: Configuration Updates (30 min)

### 5.1 Update pyproject.toml

**Full updated configuration:**

```toml
[tool.poetry]
name = "timepoint"
version = "0.1.0"
description = "Temporal entity simulation with LLM-driven training and tensor compression"
authors = ["Sean McDonald <sean@example.com>"]
readme = "README.md"
license = "MIT"  # Choose appropriate license
homepage = "https://github.com/realityinspector/timepoint-daedalus"
repository = "https://github.com/realityinspector/timepoint-daedalus"
documentation = "https://timepoint.readthedocs.io"  # If applicable
keywords = ["simulation", "llm", "temporal", "ai", "machine-learning"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# CRITICAL: Update package location
packages = [{include = "timepoint", from = "src"}]

[tool.poetry.dependencies]
python = "^3.11"

# Core dependencies
hydra-core = "^1.3.2"
omegaconf = "^2.3.0"
pydantic = "^2.10.0"

# LLM and AI
instructor = "^1.7.0"
httpx = "^0.27.0"
langgraph = "^0.2.62"

# Graph and network analysis
networkx = "^3.4.2"

# Database and ORM
sqlmodel = "^0.0.22"

# Scientific computing and ML
numpy = "^2.2.1"
scipy = "^1.15.0"
scikit-learn = "^1.6.1"

# Serialization
msgspec = "^0.19.0"

# Web framework
fastapi = "^0.118.0"
uvicorn = "^0.37.0"

# Security
bleach = "^6.1.0"

# Build fixes
grpcio = "^1.68.1"

# Optional dependencies
reportlab = {version = "^4.0.0", optional = true}
logfire = {version = "*", optional = true}
oxenai = {version = "*", optional = true}

[tool.poetry.group.dev.dependencies]
# Testing
pytest = "^7.3.1"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"
pytest-mock = "^3.11.1"
hypothesis = "^6.122.3"

# Security and validation
python-dotenv = "^1.0.0"

# Code quality
ruff = "^0.8.5"
black = "^24.10.0"
mypy = "^1.14.1"

[tool.poetry.extras]
reporting = ["reportlab"]
monitoring = ["logfire"]
oxen = ["oxenai"]
all = ["reportlab", "logfire", "oxenai"]

[tool.poetry.scripts]
timepoint = "timepoint.cli:main"

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src"]

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501", # line too long (handled by black)
    "B008", # do not perform function calls in argument defaults
    "C901", # too complex
]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--cov=src/timepoint",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80",
]

[tool.coverage.run]
source = ["src/timepoint"]
omit = [
    "*/tests/*",
    "*/test_*.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### 5.2 Update Hydra Configuration Paths

If Hydra configs reference Python modules, update paths:
- Check `conf/` YAML files
- Update any module references from root to `timepoint.`

### 5.3 Commit Configuration
```bash
git commit -m "chore(config): update pyproject.toml for src-layout package"
```

---

## Phase 6: Testing & Validation (1-2 hours)

### 6.1 Install in Development Mode
```bash
poetry install
```

**Expected result:** No import errors, all dependencies resolved

### 6.2 Run Type Checking
```bash
poetry run mypy src/timepoint
```

**Fix any type errors that surface**

### 6.3 Run Linting
```bash
poetry run ruff check src/timepoint tests
poetry run black --check src/timepoint tests
```

**Fix formatting and linting issues**

### 6.4 Run Test Suite
```bash
poetry run pytest -v --cov=src/timepoint --cov-fail-under=80
```

**Success criteria:**
- All tests pass
- Coverage ≥80%
- No import errors

### 6.5 Test CLI Entry Point
```bash
poetry run timepoint --help
poetry run timepoint --version
```

**Verify CLI works correctly**

### 6.6 Test Package Build
```bash
poetry build
```

**Expected output:**
```
Building timepoint (0.1.0)
  - Building sdist
  - Built timepoint-0.1.0.tar.gz
  - Building wheel
  - Built timepoint-0.1.0-py3-none-any.whl
```

### 6.7 Test Local Installation
```bash
# Create test virtualenv
python -m venv /tmp/test-timepoint
source /tmp/test-timepoint/bin/activate

# Install from built wheel
pip install dist/timepoint-0.1.0-py3-none-any.whl

# Test import
python -c "import timepoint; print(timepoint.__version__)"

# Test CLI
timepoint --help

# Clean up
deactivate
rm -rf /tmp/test-timepoint
```

### 6.8 Commit Fixes
```bash
git commit -m "fix(tests): address test failures after restructuring"
```

---

## Phase 7: Documentation (1 hour)

### 7.1 Create/Update README.md

**Minimum sections:**
```markdown
# Timepoint

Temporal entity simulation with LLM-driven training and tensor compression.

## Installation

### From PyPI (once published)
```bash
pip install timepoint
```

### From source
```bash
git clone https://github.com/realityinspector/timepoint-daedalus.git
cd timepoint-daedalus
poetry install
```

## Quick Start

```python
from timepoint import Orchestrator, SimulationConfig

# Initialize orchestrator
orchestrator = Orchestrator()

# Run simulation
config = SimulationConfig(...)
results = orchestrator.run(config)
```

## CLI Usage

```bash
timepoint --help
```

## Features

- 🧠 LLM-driven entity simulation
- 📊 Tensor compression for efficient storage
- 🔍 Natural language query interface
- 📈 Real-time monitoring and reporting
- 🔄 Workflow orchestration

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

## License

MIT License - see [LICENSE](LICENSE)
```

### 7.2 Create LICENSE File

Choose license (MIT recommended for open source):
```bash
# Add MIT License file
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Sean McDonald

Permission is hereby granted, free of charge, to any person obtaining a copy...
EOF
```

### 7.3 Create CHANGELOG.md

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-19

### Added
- Initial PyPI package release
- Core simulation engine
- LLM integration layer
- Natural language query interface
- Workflow orchestration
- Real-time monitoring
- Comprehensive test suite (80%+ coverage)

### Changed
- Restructured to src-layout for PyPI distribution
- Updated imports to use `timepoint` namespace

[0.1.0]: https://github.com/realityinspector/timepoint-daedalus/releases/tag/v0.1.0
```

### 7.4 Commit Documentation
```bash
git commit -m "docs: add README, LICENSE, and CHANGELOG for PyPI release"
```

---

## Phase 8: Pre-publish Checklist

### 8.1 Version Management
- [ ] Update version in `pyproject.toml`
- [ ] Update version in `src/timepoint/__init__.py`
- [ ] Update CHANGELOG.md with release notes
- [ ] Create git tag: `git tag -a v0.1.0 -m "Initial PyPI release"`

### 8.2 Package Metadata
- [ ] All classifiers appropriate
- [ ] Keywords relevant
- [ ] README renders correctly on PyPI
- [ ] License file present
- [ ] Author info correct

### 8.3 Quality Checks
- [ ] All tests pass: `poetry run pytest`
- [ ] Coverage ≥80%
- [ ] Type checking passes: `poetry run mypy`
- [ ] Linting passes: `poetry run ruff check`
- [ ] Formatting correct: `poetry run black --check`

### 8.4 Security Checks
- [ ] No secrets in code
- [ ] Dependencies up to date
- [ ] Known vulnerabilities checked: `poetry show --outdated`
- [ ] `.gitignore` excludes sensitive files

### 8.5 Build Verification
- [ ] `poetry build` succeeds
- [ ] Wheel installs cleanly
- [ ] CLI works after installation
- [ ] Imports work correctly

---

## Phase 9: Publishing

### 9.1 Test on TestPyPI First

```bash
# Configure TestPyPI repository
poetry config repositories.testpypi https://test.pypi.org/legacy/

# Get TestPyPI token from https://test.pypi.org/manage/account/token/
poetry config pypi-token.testpypi pypi-XXXXXXXXXXXX

# Build and publish to TestPyPI
poetry build
poetry publish -r testpypi

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ timepoint

# Verify it works
python -c "import timepoint; print(timepoint.__version__)"
```

### 9.2 Publish to PyPI

```bash
# Get PyPI token from https://pypi.org/manage/account/token/
poetry config pypi-token.pypi pypi-XXXXXXXXXXXX

# Publish to PyPI
poetry publish

# Verify on PyPI
# Visit: https://pypi.org/project/timepoint/
```

### 9.3 Post-publish Verification

```bash
# Install from PyPI
pip install timepoint

# Verify version
python -c "import timepoint; print(timepoint.__version__)"

# Test CLI
timepoint --help
```

### 9.4 Create GitHub Release

```bash
# Push tags
git push origin v0.1.0

# Create release on GitHub
# - Go to https://github.com/realityinspector/timepoint-daedalus/releases
# - Create new release from tag v0.1.0
# - Copy CHANGELOG content
# - Attach wheel and sdist files
```

---

## Phase 10: Post-publish Tasks

### 10.1 Update Repository
- [ ] Add PyPI badge to README
- [ ] Add documentation links
- [ ] Update main branch with restructuring
- [ ] Archive old structure (if needed)

### 10.2 Documentation
- [ ] Set up ReadTheDocs (optional)
- [ ] Create API documentation
- [ ] Add usage examples
- [ ] Create tutorials

### 10.3 Continuous Integration
- [ ] Set up GitHub Actions for testing
- [ ] Automatic PyPI publishing on tags
- [ ] Code quality checks in CI
- [ ] Coverage reporting

---

## Rollback Plan

If issues arise during migration:

### Emergency Rollback
```bash
# Discard all changes
git reset --hard origin/main

# Or revert specific commits
git revert <commit-hash>
```

### Partial Rollback
- Keep feature branch
- Fix specific issues
- Test thoroughly before re-attempting

---

## Risk Mitigation

### High-Risk Areas

1. **Import circular dependencies**
   - Mitigation: Use TYPE_CHECKING, lazy imports
   - Test: Run all tests after each import update batch

2. **Hydra configuration paths**
   - Mitigation: Update configs incrementally
   - Test: Verify CLI loads configs correctly

3. **Database file paths**
   - Mitigation: Use relative paths from package root
   - Test: Run integration tests with databases

4. **Test discovery**
   - Mitigation: Update pytest configuration
   - Test: Run pytest with `-v` to verify all tests found

---

## Success Criteria

✅ **Package published to PyPI**
✅ **`pip install timepoint` works**
✅ **All tests passing with 80%+ coverage**
✅ **CLI functional after installation**
✅ **No import errors**
✅ **Type checking passes**
✅ **Documentation complete and accurate**
✅ **Version tagged in git**

---

## Timeline Estimate

| Phase | Description | Time | Cumulative |
|-------|-------------|------|------------|
| 1 | Preparation | 1h | 1h |
| 2 | Create structure | 0.5h | 1.5h |
| 3 | File migration | 1-2h | 3.5h |
| 4 | Import updates | 2-3h | 6.5h |
| 5 | Configuration | 0.5h | 7h |
| 6 | Testing | 1-2h | 9h |
| 7 | Documentation | 1h | 10h |
| 8 | Pre-publish checks | 0.5h | 10.5h |
| 9 | Publishing | 0.5h | 11h |
| 10 | Post-publish | 1h | 12h |

**Total: 10-12 hours** (spread over 2-3 days recommended)

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Create feature branch**: `feature/pypi-restructure`
3. **Execute Phase 1**: Preparation and inventory
4. **Proceed incrementally**, testing after each phase
5. **Document issues** encountered for future reference

---

## Questions to Resolve Before Starting

- [ ] Confirm package name: "timepoint" vs "timepoint-daedalus"?
- [ ] License choice: MIT, Apache 2.0, or other?
- [ ] Version numbering: Start at 0.1.0 or 1.0.0?
- [ ] Optional dependencies strategy?
- [ ] Documentation hosting plan?
- [ ] CI/CD pipeline requirements?

---

## References

- [Python Packaging User Guide](https://packaging.python.org/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [PyPI Publishing Guide](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)

---

**Plan Prepared By:** Claude
**Last Updated:** 2025-11-19
