# Timepoint-Daedalus: Codebase Structure for PyPI Packaging

## Executive Summary
The **Timepoint-Daedalus** project is a sophisticated temporal knowledge graph system with LLM-driven entity simulation. It's currently structured as a flat module in the root directory with several sub-packages. For PyPI packaging as "timepoint", the codebase will need to be reorganized into a proper Python package structure.

---

## Current Directory Structure

### Root Level Organization
```
timepoint-daedalus/
├── pyproject.toml              # Poetry configuration (PRIMARY)
├── requirements.txt            # Requirements (mirrors pyproject.toml)
├── requirements-test.txt       # Testing requirements
├── poetry.lock                 # Locked dependencies
├── pytest.ini                  # Pytest configuration
├── README.md                   # Project documentation
├── CLAUDE.md                   # Project instructions
├── MECHANICS.md                # Architecture documentation
├── HANDOFF.md                  # Handoff documentation
│
├── [CORE MODULES - ROOT LEVEL]
├── cli.py                      # CLI entry point (Hydra-based)
├── orchestrator.py             # Scene-to-Specification compiler
├── schemas.py                  # SQLModel data definitions
├── storage.py                  # Database & graph persistence
├── llm.py / llm_v2.py          # LLM client implementations
├── query_interface.py          # Query execution engine
├── validation.py               # Validation system
├── workflows.py                # Workflow orchestration
├── temporal_chain.py           # Temporal chain building
├── resolution_engine.py        # Resolution logic
├── graph.py                    # Graph utilities
├── tensors.py                  # Tensor compression
├── tensor_initialization.py    # Tensor initialization
├── reporting.py                # Simple reporting functions
├──
│
├── [SUB-PACKAGES - WITH __init__.py]
├── llm_service/                # Centralized LLM integration
│   ├── __init__.py             # Exports LLMProvider, LLMService, LLMServiceConfig
│   ├── service.py              # Main LLM service
│   ├── provider.py             # Provider interface
│   ├── config.py               # Service configuration
│   ├── error_handler.py        # Error handling
│   ├── call_logger.py          # Call logging
│   ├── prompt_manager.py       # Prompt management
│   ├── response_parser.py      # Response parsing
│   ├── security_filter.py      # Input/output filtering
│   └── providers/              # Provider implementations
│
├── nl_interface/               # Natural Language Interface
│   ├── __init__.py             # Exports NLConfigGenerator, ConfigValidator, etc.
│   ├── nl_to_config.py         # NL → Config conversion
│   ├── config_validator.py     # Config validation
│   ├── interactive_refiner.py  # Interactive refinement
│   ├── clarification_engine.py # Clarification questions
│   └── prompts.py              # Prompt templates
│
├── workflows/                  # Workflow Orchestration (large module)
│   ├── __init__.py             # Main workflow exports
│   └── [Large module - 33K+ tokens]
│
├── reporting/                  # Reporting & Export
│   ├── __init__.py             # Exports report generators, formatters
│   ├── query_engine.py         # Enhanced query engine
│   ├── report_generator.py     # Report generation
│   ├── formatters.py           # Output formatters (MD, JSON, CSV)
│   ├── export_formats.py       # Export formats
│   ├── export_pipeline.py      # Export pipeline
│   └── script_generator.py     # Script generation
│
├── generation/                 # Generation Engine
│   ├── __init__.py
│   ├── checkpoint_manager.py   # Checkpoint management
│   ├── fault_handler.py        # Fault handling
│   ├── horizontal_generator.py # Horizontal generation
│   ├── vertical_generator.py   # Vertical generation
│   ├── world_manager.py        # World management
│   ├── progress_tracker.py     # Progress tracking
│   ├── temporal_expansion.py   # Temporal expansion
│   ├── variation_strategies.py # Variation strategies
│   ├── resilience_orchestrator.py # Resilience
│   └── config_schema.py        # Configuration schema (320K!)
│
├── andos/                      # ANDOS Layer System
│   ├── __init__.py             # Exports: compute_andos_layers, etc.
│   └── layer_computer.py       # Core ANDOS algorithm
│
├── monitoring/                 # Monitoring & Explanation
│   ├── __init__.py
│   ├── db_inspector.py         # Database inspection
│   ├── llm_explainer.py        # LLM explanation
│   ├── monitor_runner.py       # Monitoring runner
│   ├── stream_parser.py        # Stream parsing
│   ├── config.py               # Configuration
│   └── prompts/                # Prompt templates
│
├── metadata/                   # Metadata & Tracking
│   ├── __init__.py
│   ├── tracking.py             # Mechanism tracking
│   ├── run_tracker.py          # Run tracking
│   ├── run_summarizer.py       # Run summarization
│   ├── narrative_exporter.py   # Narrative export
│   ├── coverage_matrix.py      # Coverage tracking
│   └── logfire_setup.py        # Logfire setup
│
├── oxen_integration/           # Oxen Integration
├── e2e_workflows/              # End-to-End Workflows
│
├── [TEST FILES - 80+ test files]
├── test_*.py                   # Unit/integration tests
├── conftest.py                 # Pytest fixtures & configuration
├── pytest.ini                  # Pytest configuration
│
├── [DEMO/EXAMPLE FILES]
├── demo_*.py                   # Demo scripts
├── examples/                   # Example configurations
├── run_*.py                    # Execution scripts
├── validate_*.py               # Validation scripts
│
└── [DATA & CONFIG]
    ├── datasets/               # Dataset directories
    ├── conf/                   # Hydra configuration files
    ├── logs/                   # Log output
    ├── *.db                    # SQLite databases
    └── *.json                  # Configuration & metadata

---

## 1. Main Source Code Structure

### Core Modules (Root Level)
These modules form the foundation and are currently at the root level:

| Module | Purpose | Lines | Key Classes |
|--------|---------|-------|------------|
| `cli.py` | Hydra-based CLI entry point | ~1000+ | CLI command handlers |
| `orchestrator.py` | Scene → Specification compiler (OrchestratorAgent) | ~2000+ | SceneParser, KnowledgeSeeder |
| `schemas.py` | SQLModel data definitions | ~1000+ | Entity, Timepoint, Dialog, etc. |
| `storage.py` | Database & graph persistence | ~500 | GraphStore |
| `llm_v2.py` | Centralized LLM client (NEW) | ~1500+ | LLMClient |
| `llm.py` | Legacy LLM implementation | ~800+ | LLMClient, EntityPopulation |
| `query_interface.py` | Query execution engine | ~2000+ | QueryInterface, QueryResult |
| `workflows.py` | Workflow orchestration | ~33K tokens! | TemporalAgent, create_entity_training_workflow |
| `validation.py` | Validation system | ~2000+ | Validator |
| `temporal_chain.py` | Temporal chain building | ~400 | build_temporal_chain |
| `resolution_engine.py` | Resolution logic | ~600 | ResolutionEngine |
| `graph.py` | Graph utilities | ~200 | create_test_graph, export_graph_data |
| `tensors.py` | Tensor compression | ~400 | TensorCompressor |
| `tensor_initialization.py` | Tensor initialization | ~1000+ | TensorInitializer |

### Sub-Packages (with __init__.py)

#### 1. **llm_service/** (Centralized LLM Integration)
```
Structure:
- __init__.py: Exports LLMProvider, LLMResponse, LLMService, LLMServiceConfig
- service.py: Main LLMService class (12K tokens)
- provider.py: Abstract provider interface
- config.py: Service configuration
- error_handler.py: Error handling & retry logic
- call_logger.py: Call logging for tracking
- prompt_manager.py: Prompt management
- response_parser.py: Response parsing
- security_filter.py: Input bleaching & output sanitization
- providers/: Provider implementations (Mirascope, custom OpenRouter, test)
```

#### 2. **nl_interface/** (Natural Language Interface - Sprint 3)
```
Structure:
- __init__.py: Exports main classes
- nl_to_config.py: NL → SimulationConfig conversion (uses LLM)
- config_validator.py: Semantic validation (10K tokens)
- interactive_refiner.py: Interactive CLI refinement
- clarification_engine.py: Ambiguity detection
- prompts.py: LLM prompt templates

Key Feature: Convert natural language descriptions to valid simulation configs
```

#### 3. **workflows/** (Workflow Orchestration)
```
Large module (33K+ tokens) containing:
- Workflow state machines
- Entity training workflows
- Temporal agents
- Dialog synthesis
- Key exports via __init__.py
```

#### 4. **reporting/** (Reporting & Export)
```
Structure:
- query_engine.py: EnhancedQueryEngine with batch execution (11K)
- report_generator.py: Report generation (14K)
- formatters.py: Markdown, JSON, CSV formatters (16K)
- export_formats.py: Export format implementations (19K)
- export_pipeline.py: Export pipeline orchestration (14K)
- script_generator.py: Script generation (25K)

Supports: JSON, JSONL, CSV, SQLite, Markdown exports
```

#### 5. **generation/** (Generation Engine - Phase 10+)
```
Structure:
- checkpoint_manager.py: Checkpoint/save-state management
- fault_handler.py: Fault handling & recovery
- horizontal_generator.py: Parallel generation
- vertical_generator.py: Sequential depth generation
- world_manager.py: World state management (18K)
- progress_tracker.py: Progress tracking
- temporal_expansion.py: Temporal expansion logic
- variation_strategies.py: Variation generation
- resilience_orchestrator.py: Resilience patterns (23K)
- config_schema.py: Configuration schema (320K! - very large)

Key: Handles generation with fault tolerance and checkpointing
```

#### 6. **andos/** (ANDOS Layer System)
```
Structure:
- layer_computer.py: Core ANDOS algorithm (11K)

Purpose: Solve circular dependencies by computing training layers
Concept: "Crystal formation from seeds" - train periphery to core
Exports: compute_andos_layers, validate_andos_layers, build_interaction_graph
```

#### 7. **monitoring/** (Monitoring & Explanation)
```
Structure:
- db_inspector.py: Database inspection (7K)
- llm_explainer.py: LLM-powered explanation (6K)
- monitor_runner.py: Monitoring runner (12K, executable)
- stream_parser.py: Stream parsing (7K)
- config.py: Configuration
- prompts/: Prompt templates

Purpose: Real-time monitoring and LLM-powered explanation
```

#### 8. **metadata/** (Metadata & Tracking)
```
Structure:
- tracking.py: Mechanism tracking (3.6K)
- run_tracker.py: Run tracking (25K)
- run_summarizer.py: Run summarization (14K)
- narrative_exporter.py: Narrative export (29K)
- coverage_matrix.py: Coverage tracking (11K)
- logfire_setup.py: Logfire instrumentation setup

Purpose: Track mechanisms, runs, coverage, and export narratives
Database: runs.db for persistent tracking
```

#### 9. **oxen_integration/** & **e2e_workflows/**
Currently minimal modules (likely for external integrations)

---

## 2. Packaging Configuration

### Current Configuration (pyproject.toml)
```toml
[tool.poetry]
name = "timepoint-daedalus"
version = "0.1.0"
description = "Temporal entity simulation with LLM-driven training and tensor compression"
authors = ["Sean McDonald <sean@example.com>"]
packages = [{include = "*.py"}]

[tool.poetry.scripts]
timepoint = "cli:main"  # CLI entry point
```

### Issues for PyPI Packaging
1. **Flat structure**: All root-level modules need organization into a package
2. **Package specification**: `packages = [{include = "*.py"}]` is incorrect for proper structure
3. **Entry point**: `cli:main` expects a `main()` function in cli.py
4. **Dependencies**: Mixed in requirements.txt AND pyproject.toml
5. **Test files**: 80+ test files in root (should move to tests/ directory)

### Dependencies (from pyproject.toml)

**Core Dependencies:**
- Python 3.10+
- hydra-core (1.3.2+)
- omegaconf (2.3.0+)
- pydantic (2.10.0+)

**LLM & AI:**
- instructor (1.7.0+)
- openai (1.57.0+) - or httpx for OpenRouter
- langgraph (0.2.62+)

**Database & ORM:**
- sqlmodel (0.0.22+)

**Scientific Computing:**
- numpy (2.2.1+)
- scipy (1.15.0+)
- scikit-learn (1.6.1+)

**Serialization & Format:**
- msgspec (0.19.0+)

**Web Framework:**
- fastapi (0.118.0+)
- uvicorn (0.37.0+)
- grpcio (1.68.1+)

**Network Analysis:**
- networkx (3.4.2+)

**Security:**
- bleach (6.0.0+)

**Development (Dev Group):**
- pytest (7.3.1+)
- pytest-asyncio (0.21.0+)
- pytest-cov (4.1.0+)
- pytest-mock (3.11.1+)
- hypothesis (6.122.3+)
- ruff (0.8.5+)
- black (24.10.0+)
- mypy (1.14.1+)

**Optional Dependencies:**
- reportlab (PDF export)
- logfire (instrumentation)
- oxenai (data management)
- python-dotenv (environment config)

---

## 3. Entry Points & Key Interfaces

### CLI Entry Point
**Location**: `/home/user/timepoint-daedalus/cli.py`
**Entry Point**: `timepoint = "cli:main"` (defined in pyproject.toml)
**Framework**: Hydra-based configuration

Key Functions:
- Autopilot mode execution
- Steering configuration
- Tensor compression
- Report generation

### Main Interfaces

#### 1. Natural Language to Config
```python
from nl_interface import NLConfigGenerator

generator = NLConfigGenerator()
config, confidence = generator.generate_config(
    "Simulate a board meeting..."
)
```

#### 2. Simulation Execution
```python
from orchestrator import simulate_event
from llm_v2 import LLMClient
from storage import GraphStore

llm = LLMClient()
store = GraphStore("sqlite:///simulations.db")
result = simulate_event(config['scenario'], llm, store)
```

#### 3. Query & Report
```python
from reporting import EnhancedQueryEngine, ReportGenerator

query_engine = EnhancedQueryEngine()
relationships = query_engine.summarize_relationships(world_id)
```

#### 4. Monitoring
```python
from monitoring import db_inspector, llm_explainer

# Database inspection
inspector = db_inspector.DatabaseInspector()

# LLM explanations
explainer = llm_explainer.LLMExplainer()
```

---

## 4. Test Organization

### Current Structure
- 80+ test files in root directory
- conftest.py with shared fixtures
- pytest.ini with configuration

### Test Categories
1. **Unit Tests**: Core module functionality
2. **Integration Tests**: Multi-module interaction
3. **System Tests**: Full pipeline tests
4. **E2E Tests**: Complete workflow tests
5. **Mechanism Tests**: Individual mechanism validation

**Test Coverage**: Configured for 80%+ minimum (CLAUDE.md)

---

## 5. Configuration & Data

### Hydra Configuration
- **Location**: `/home/user/timepoint-daedalus/conf/`
- **Format**: YAML configuration files
- **Purpose**: Runtime configuration management

### Databases
- `timepoint.db` - Main simulation database (SQLite)
- `metadata/runs.db` - Run tracking database
- Backups available for recovery

### Datasets
- **Location**: `/home/user/timepoint-daedalus/datasets/`
- Various dataset directories for training/evaluation

### Logging
- **Location**: `/home/user/timepoint-daedalus/logs/`
- Multiple directories for different log types

---

## 6. Key Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | 50+ core modules + 80+ tests |
| Root-Level Modules | ~15 main modules + 10+ sub-packages |
| Sub-Packages with __init__.py | 10 packages |
| Largest Module | workflows.py (33K tokens) |
| Second Largest | config_schema.py (320KB!) |
| Total Dependencies | 40+ packages |
| Python Version | 3.10+ |
| Test Files | 80+ |
| Documentation Files | 5+ (README, MECHANICS, HANDOFF, etc.) |

---

## 7. Recommended PyPI Package Structure

For publishing as "timepoint" on PyPI, recommend this reorganization:

```
timepoint-daedalus/
├── pyproject.toml              # Updated for PyPI
├── README.md
├── LICENSE
├── src/
│   └── timepoint/              # Package root
│       ├── __init__.py         # Version, main exports
│       ├── cli/                # CLI module (move from root)
│       │   ├── __init__.py
│       │   └── main.py         # From cli.py
│       ├── core/               # Core functionality
│       │   ├── __init__.py
│       │   ├── orchestrator.py # From root
│       │   ├── schemas.py      # From root
│       │   ├── storage.py      # From root
│       │   └── ...
│       ├── llm/                # LLM-related modules
│       │   ├── __init__.py
│       │   ├── service.py      # llm_service/
│       │   └── ...
│       ├── nl_interface/       # Keep as is (already good structure)
│       ├── workflows/          # Keep as is
│       ├── reporting/          # Keep as is
│       ├── generation/         # Keep as is
│       └── ... [other packages]
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── architecture.md         # From MECHANICS.md
│   └── ...
└── examples/
    └── [demo scripts reorganized]
```

---

## Summary

**Timepoint-Daedalus** is a comprehensive, production-ready system for:
- Natural language simulation generation
- LLM-driven temporal entity simulation
- Adaptive fidelity with tensor compression
- Query execution and reporting
- Fault-tolerant generation with checkpointing

For PyPI packaging as "timepoint":
1. Restructure into `src/timepoint/` layout
2. Move root-level modules into proper sub-packages
3. Reorganize tests into `tests/` directory
4. Update pyproject.toml with proper package specification
5. Ensure all imports use `from timepoint import ...` pattern
