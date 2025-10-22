# Development Plan - Timepoint-Daedalus v2.0

**Last Updated:** October 21, 2025
**Phase:** Core Mechanisms Complete → Production Infrastructure
**Status:** PHASE 1 COMPLETE (17/17 mechanisms) ✅ | PHASE 2 PLANNING 📋

---

## Executive Summary

**Phase 1 (COMPLETE):** All 17 core simulation mechanisms implemented and tested
**Phase 2 (IN PROGRESS):** Production infrastructure for synthetic data generation, reporting, and user workflows

### What Changed: From Research Prototype → Production Platform

Timepoint-Daedalus has evolved from a mechanism demonstration system to a **production-grade simulation and synthetic data generation platform**. The next phase adds:

1. **Generation Infrastructure** - Horizontal (variations) and Vertical (temporal depth) synthetic data generation
2. **Reporting Infrastructure** - Multi-format exports (JSON, Markdown, CSV) with query-driven report generation
3. **Natural Language Interface** - LLM-powered config generation from natural language descriptions
4. **World Management** - Isolated simulation "worlds" for clean separation of demo/user/test data
5. **Production Tooling** - Progress tracking, fault handling, checkpoint/resume for long-running generations

---

## Phase 1: Core Mechanisms ✅ COMPLETE

### ✅ All 17 Mechanisms Implemented (100%)

**Codebase:** 47 Python files, 19,591 lines
**Test Suite:** 200+ tests across 24 files
**E2E Test Status:** 13/13 passing (100%)

**Mechanisms:**
- ✅ M1: Heterogeneous fidelity temporal graphs
- ✅ M2: Progressive training
- ✅ M3: Causal temporal chains
- ✅ M4: Physics validation (5 validators)
- ✅ M5: Query resolution (17 tests)
- ✅ M6: Exposure event tracking
- ✅ M7: TTM tensor compression
- ✅ M8: Embodied states (PhysicalTensor + CognitiveTensor)
- ✅ M9: On-demand generation (23 tests)
- ✅ M10: Scene entities (Environment, Atmosphere, Crowd)
- ✅ M11: Dialog synthesis with validators
- ✅ M12: Counterfactual branching
- ✅ M13: Multi-entity synthesis
- ✅ M14: Circadian patterns
- ✅ M15: Entity prospection
- ✅ M16: Animistic entities (6 types)
- ✅ M17: Modal temporal causality (5 modes)

**Performance Benchmarks:**
- Small (5 entities, 5 timepoints): ~18s, ~$0.80 (46.7% cost reduction)
- Medium (10 entities, 5 timepoints): 30-90s, $1.50-$8
- Large (100 entities, 20 timepoints): <5min, ~$30
- 95% token cost reduction vs naive approach

---

## Phase 2: Production Infrastructure 📋 IN PROGRESS

### Architecture Principles

**1. Clean Separation of Concerns**
```
Core Simulation (Phase 1)    → No changes, stable foundation
Generation Layer (Phase 2)   → New modules only, zero disruption
Reporting Layer (Phase 2)    → Consumes core outputs
User Interface (Phase 2)     → CLI/API wraps generation + reporting
Demo System (Phase 2)        → Uses core, doesn't extend it
```

**2. Flywheel Development**
- Each component requires E2E tests before merge
- No breaking changes to existing code (Phase 1 remains untouched)
- Iterative: Build → Test → Document → Ship

**3. World Isolation**
```
worlds/
├── demo/           # Pre-configured demonstration scenarios
├── test/           # Test suite data (isolated from user data)
└── user-{id}/      # User production data (configurable isolation)
```

**4. Configurable Architecture**
- User chooses isolation level: separate DBs vs single DB with world_id
- User chooses interfaces: CLI only, API only, or both
- User chooses NL interface: enabled/disabled based on API key availability

---

## Sprint 1: Synthetic Data Generation Infrastructure 🔄

**Goal:** Enable horizontal (variations) and vertical (temporal depth) data generation with progress tracking and fault tolerance

**Estimated Time:** 2-3 weeks
**Dependencies:** None (builds on Phase 1)
**Risk Level:** LOW (new modules only)

### 1.1: World Management System ⏳

**Files to Create:**
```
generation/
├── __init__.py
├── world_manager.py        # World isolation and partitioning
└── config_schema.py        # SimulationConfig validation (Pydantic)
```

**Implementation Checklist:**

- [ ] **WorldManager class** (generation/world_manager.py)
  - [ ] `create_world(world_id, isolation_mode)` - Create isolated namespace
  - [ ] `get_world(world_id)` - Retrieve world metadata
  - [ ] `list_worlds()` - List all available worlds
  - [ ] `delete_world(world_id)` - Clean up world data
  - [ ] Support 3 isolation modes:
    - [ ] `separate_db` - Each world gets its own SQLite file
    - [ ] `shared_db_partitioned` - Single DB, world_id on all tables
    - [ ] `hybrid` - Separate DBs for production, shared for demos/tests
  - [ ] Schema migrations for world_id columns (if shared_db mode)

- [ ] **SimulationConfig schema** (generation/config_schema.py)
  - [ ] Pydantic model for simulation configuration
  - [ ] Validation rules for entity counts, timepoint ranges, etc.
  - [ ] JSON schema export for documentation
  - [ ] Example configs embedded as class attributes

**Test Requirements:**
- [ ] **test_world_manager.py** (15 tests minimum)
  - [ ] Test world creation in all 3 isolation modes
  - [ ] Test world listing and retrieval
  - [ ] Test world deletion and cleanup
  - [ ] Test isolation enforcement (no cross-world data leaks)
  - [ ] Test concurrent world access
  - [ ] Test world metadata persistence

**E2E Requirement:**
- [ ] **test_e2e_world_isolation.py**
  - [ ] Create 2 worlds, generate data in each, verify complete isolation
  - [ ] Switch isolation modes dynamically, verify data integrity
  - [ ] Test cleanup doesn't affect other worlds

**Documentation:**
- [ ] Docstrings for all public methods
- [ ] Example usage in module docstring
- [ ] Architecture decision record (ADR) for isolation strategy

**Acceptance Criteria:**
- ✅ Can create/delete worlds in all 3 isolation modes
- ✅ Worlds are completely isolated (no data leaks)
- ✅ All tests pass (unit + E2E)
- ✅ Zero changes to Phase 1 code

---

### 1.2: Horizontal Data Generation (Variations) ⏳

**Files to Create:**
```
generation/
├── horizontal_generator.py  # Generate N variations of same scenario
└── variation_strategies.py  # Strategies for parameter variation
```

**Implementation Checklist:**

- [ ] **HorizontalGenerator class** (generation/horizontal_generator.py)
  - [ ] `generate_variations(base_config, count, variation_params)`
  - [ ] Variation strategies:
    - [ ] `vary_personalities` - Different personality traits for entities
    - [ ] `vary_starting_conditions` - Different initial states
    - [ ] `vary_outcomes` - Different decision outcomes
    - [ ] `vary_relationships` - Different initial relationship states
    - [ ] `vary_knowledge` - Different initial knowledge distributions
  - [ ] Ensure variations are meaningfully different (not trivial changes)
  - [ ] Parallel generation option (multi-threading for batch jobs)
  - [ ] Deduplication detection (reject near-identical variations)

- [ ] **VariationStrategy base class** (generation/variation_strategies.py)
  - [ ] Abstract interface for variation generation
  - [ ] Concrete implementations:
    - [ ] PersonalityVariation - Vary Big Five traits
    - [ ] KnowledgeVariation - Vary initial exposure events
    - [ ] RelationshipVariation - Vary initial trust/alignment
    - [ ] OutcomeVariation - Vary key decision points
  - [ ] Validation: Ensure variations stay within plausible bounds

**Test Requirements:**
- [ ] **test_horizontal_generation.py** (20 tests minimum)
  - [ ] Test each variation strategy independently
  - [ ] Test combination of multiple variation strategies
  - [ ] Test deduplication (reject identical variations)
  - [ ] Test parallel generation (thread safety)
  - [ ] Test variation quality (meaningfully different outcomes)
  - [ ] Test batch generation (100+ variations)

**E2E Requirement:**
- [ ] **test_e2e_horizontal_generation.py**
  - [ ] Generate 100 variations of "board meeting" scenario
  - [ ] Verify all 100 are unique (different outcomes/dialogs)
  - [ ] Verify cost scales linearly (no redundant work)
  - [ ] Verify all data stored correctly in database
  - [ ] Export to JSONL format for ML training
  - [ ] Load exported data and verify integrity

**Documentation:**
- [ ] USER-GUIDE.md section on horizontal generation
- [ ] Example configs for common use cases
- [ ] Cost estimation guide (tokens/$ per variation)

**Acceptance Criteria:**
- ✅ Can generate 100+ variations reliably
- ✅ Variations are meaningfully different
- ✅ Parallel generation works correctly
- ✅ E2E test passes with real LLM
- ✅ Zero changes to Phase 1 code

---

### 1.3: Vertical Data Generation (Temporal Depth) ⏳

**Files to Create:**
```
generation/
├── vertical_generator.py    # Generate deep temporal context
└── temporal_expansion.py    # Timepoint expansion strategies
```

**Implementation Checklist:**

- [ ] **VerticalGenerator class** (generation/vertical_generator.py)
  - [ ] `generate_temporal_depth(base_config, before_count, after_count)`
  - [ ] Progressive training mode:
    - [ ] Start entities at TENSOR_ONLY for early timepoints
    - [ ] Elevate resolution as they approach critical moment
    - [ ] Train to FULL_DETAIL at peak moment
    - [ ] Optionally downgrade after peak
  - [ ] Causal chain enforcement:
    - [ ] Each timepoint must follow causally from previous
    - [ ] Exposure events propagate correctly through time
    - [ ] No temporal paradoxes (validate with M4 validators)
  - [ ] Compression optimization:
    - [ ] Use TTM tensors for non-critical timepoints
    - [ ] Lazy elevation on query (if explored later)

- [ ] **TemporalExpansion strategies** (generation/temporal_expansion.py)
  - [ ] `expand_before(scenario, count)` - Generate lead-up timepoints
  - [ ] `expand_after(scenario, count)` - Generate consequence timepoints
  - [ ] `expand_around(scenario, before, after)` - Both directions
  - [ ] Narrative arc shaping:
    - [ ] Rising action → climax → falling action
    - [ ] Tension/stakes increase approaching critical moment
    - [ ] Resolution and consequences after critical moment

**Test Requirements:**
- [ ] **test_vertical_generation.py** (20 tests minimum)
  - [ ] Test temporal expansion (before/after/around)
  - [ ] Test progressive training (resolution elevation)
  - [ ] Test causal chain integrity
  - [ ] Test exposure event propagation
  - [ ] Test compression optimization
  - [ ] Test narrative arc shaping

**E2E Requirement:**
- [ ] **test_e2e_vertical_generation.py**
  - [ ] Generate "Jefferson Dinner" with 5 before + 5 after timepoints
  - [ ] Verify causal chain is valid (no paradoxes)
  - [ ] Verify entities start low-res, peak at dinner, decrease after
  - [ ] Verify exposure events propagate correctly
  - [ ] Query entities at different timepoints, verify knowledge state
  - [ ] Verify cost savings from progressive training

**Documentation:**
- [ ] USER-GUIDE.md section on vertical generation
- [ ] Example: "Building a world model around a key moment"
- [ ] Comparison: Horizontal vs Vertical generation use cases

**Acceptance Criteria:**
- ✅ Can generate deep temporal context (10+ timepoints)
- ✅ Progressive training reduces cost vs uniform high-res
- ✅ Causal chain validation passes
- ✅ E2E test passes with real LLM
- ✅ Zero changes to Phase 1 code

---

### 1.4: Progress Tracking & Fault Handling ⏳

**Files to Create:**
```
generation/
├── progress_tracker.py      # Real-time progress reporting
├── fault_handler.py         # Retry logic, error recovery
└── checkpoint_manager.py    # Save/resume for long runs
```

**Implementation Checklist:**

- [ ] **ProgressTracker class** (generation/progress_tracker.py)
  - [ ] Real-time progress bar (tqdm integration)
  - [ ] Metrics tracking:
    - [ ] Entities generated (count, success rate)
    - [ ] Timepoints generated (count, success rate)
    - [ ] Tokens consumed (total, rate, cost estimate)
    - [ ] Time elapsed, time remaining (ETA)
    - [ ] LLM call failures and retries
  - [ ] Streaming updates (for long-running jobs)
  - [ ] Summary report at completion
  - [ ] Export progress log to JSON

- [ ] **FaultHandler class** (generation/fault_handler.py)
  - [ ] Retry logic with exponential backoff
  - [ ] Max retries configurable per error type
  - [ ] Graceful degradation:
    - [ ] If LLM fails, use minimal entity (TENSOR_ONLY)
    - [ ] If validation fails, log warning but continue
    - [ ] If non-critical component fails, skip and continue
  - [ ] Error classification:
    - [ ] Retryable (rate limits, transient failures)
    - [ ] Non-retryable (invalid config, auth errors)
  - [ ] Error aggregation and reporting

- [ ] **CheckpointManager class** (generation/checkpoint_manager.py)
  - [ ] Auto-save checkpoints every N entities/timepoints
  - [ ] Resume from last checkpoint on failure
  - [ ] Checkpoint metadata:
    - [ ] Generation progress (entities/timepoints completed)
    - [ ] Random seed (for reproducibility)
    - [ ] Configuration snapshot
    - [ ] Cost/token counters
  - [ ] Checkpoint cleanup (delete old checkpoints)

**Test Requirements:**
- [ ] **test_progress_tracker.py** (10 tests)
  - [ ] Test progress updates
  - [ ] Test ETA calculation
  - [ ] Test cost tracking
  - [ ] Test export to JSON

- [ ] **test_fault_handler.py** (15 tests)
  - [ ] Test retry logic (exponential backoff)
  - [ ] Test graceful degradation
  - [ ] Test error classification
  - [ ] Test max retries enforcement

- [ ] **test_checkpoint_manager.py** (15 tests)
  - [ ] Test checkpoint creation
  - [ ] Test resume from checkpoint
  - [ ] Test checkpoint cleanup
  - [ ] Test checkpoint corruption handling

**E2E Requirement:**
- [ ] **test_e2e_fault_recovery.py**
  - [ ] Start 100-variation generation
  - [ ] Inject LLM failure at variation 50
  - [ ] Verify checkpoint saved at variation 49
  - [ ] Resume from checkpoint
  - [ ] Verify completes all 100 variations
  - [ ] Verify no duplicate work (variations 1-49 not regenerated)

**Documentation:**
- [ ] USER-GUIDE.md section on long-running jobs
- [ ] Best practices for fault tolerance
- [ ] Checkpoint management guide

**Acceptance Criteria:**
- ✅ Progress tracking works for 100+ item generations
- ✅ Fault recovery works (resume from checkpoint)
- ✅ E2E test passes with simulated failures
- ✅ Zero changes to Phase 1 code

---

### 1.5: Sprint 1 Integration & Testing ⏳

**Integration Checklist:**

- [ ] **Unified generation pipeline**
  - [ ] `generate()` method orchestrates: world creation → generation → storage → progress tracking
  - [ ] Supports both horizontal and vertical modes
  - [ ] Automatic checkpoint management
  - [ ] Automatic fault handling

- [ ] **Configuration-driven execution**
  - [ ] JSON config fully specifies generation job
  - [ ] Config validation before execution
  - [ ] Config templates for common scenarios

**E2E Sprint Test:**
- [ ] **test_e2e_sprint1_full_stack.py**
  - [ ] Create new world
  - [ ] Generate 50 horizontal variations
  - [ ] Generate 1 vertical expansion (10 timepoints)
  - [ ] Verify all data in database
  - [ ] Query both datasets
  - [ ] Export both to JSONL
  - [ ] Verify cost tracking
  - [ ] Clean up world

**Documentation:**
- [ ] Sprint 1 completion summary
- [ ] Performance benchmarks
- [ ] Known limitations

**Acceptance Criteria:**
- ✅ All 1.1-1.4 components integrated
- ✅ E2E full-stack test passes
- ✅ Documentation complete
- ✅ Zero breaking changes to Phase 1
- ✅ Ready for Sprint 2 (Reporting)

---

## Sprint 2: Reporting Infrastructure 📊

**Goal:** Multi-format export (JSON, Markdown, CSV) with query-driven report generation

**Estimated Time:** 1-2 weeks
**Dependencies:** Sprint 1 complete
**Risk Level:** LOW (new modules only)

### 2.1: Query Engine Enhancement ⏳

**Files to Create:**
```
reporting/
├── __init__.py
├── query_engine.py         # Enhanced query capabilities
└── query_templates.py      # Pre-built query templates
```

**Implementation Checklist:**

- [ ] **EnhancedQueryEngine class** (reporting/query_engine.py)
  - [ ] Extends existing QueryInterface
  - [ ] Batch query execution:
    - [ ] `execute_batch(queries)` - Run multiple queries efficiently
    - [ ] Shared context caching (don't reload entities multiple times)
  - [ ] Aggregation queries:
    - [ ] `summarize_relationships()` - Relationship matrix
    - [ ] `knowledge_flow_graph()` - Who learned what from whom
    - [ ] `timeline_summary()` - Key events and decisions
    - [ ] `entity_comparison()` - Side-by-side entity analysis
  - [ ] Query result caching (configurable TTL)

- [ ] **QueryTemplate library** (reporting/query_templates.py)
  - [ ] Pre-built templates for common reports:
    - [ ] "What happened?" - Narrative summary
    - [ ] "Who knew what?" - Knowledge provenance
    - [ ] "How did relationships change?" - Relationship evolution
    - [ ] "What if...?" - Counterfactual analysis
  - [ ] Template parameterization (fill in entity/timepoint names)
  - [ ] Template composition (combine multiple templates)

**Test Requirements:**
- [ ] **test_query_engine.py** (15 tests)
  - [ ] Test batch query execution
  - [ ] Test aggregation queries
  - [ ] Test result caching
  - [ ] Test template execution

**E2E Requirement:**
- [ ] **test_e2e_query_batch.py**
  - [ ] Load "Jefferson Dinner" simulation
  - [ ] Run 10 queries in batch
  - [ ] Verify results match individual queries
  - [ ] Verify performance improvement (shared caching)

**Acceptance Criteria:**
- ✅ Batch queries work correctly
- ✅ Aggregation queries provide useful insights
- ✅ E2E test passes
- ✅ Zero changes to Phase 1 code

---

### 2.2: Report Generation ⏳

**Files to Create:**
```
reporting/
├── report_generator.py     # Main report generation logic
├── templates/
│   ├── summary.md.jinja2
│   ├── relationships.md.jinja2
│   ├── knowledge_flow.md.jinja2
│   └── timeline.md.jinja2
└── formatters.py           # Output format handlers
```

**Implementation Checklist:**

- [ ] **ReportGenerator class** (reporting/report_generator.py)
  - [ ] `generate_summary_report(world_id)` - High-level overview
  - [ ] `generate_relationship_report(world_id)` - Relationship analysis
  - [ ] `generate_knowledge_report(world_id)` - Knowledge flow visualization
  - [ ] `generate_timeline_report(world_id)` - Chronological event list
  - [ ] Template-based rendering (Jinja2)
  - [ ] Multi-format output: Markdown, JSON, CSV
  - [ ] Configurable verbosity levels

- [ ] **OutputFormatter classes** (reporting/formatters.py)
  - [ ] `MarkdownFormatter` - Human-readable reports
  - [ ] `JSONFormatter` - Machine-readable exports
  - [ ] `CSVFormatter` - Tabular data (entities, relationships)
  - [ ] `HTMLFormatter` - Optional: Web-viewable reports

**Test Requirements:**
- [ ] **test_report_generator.py** (20 tests)
  - [ ] Test each report type (summary, relationships, knowledge, timeline)
  - [ ] Test each output format (MD, JSON, CSV)
  - [ ] Test template rendering
  - [ ] Test empty data handling

**E2E Requirement:**
- [ ] **test_e2e_reporting.py**
  - [ ] Generate "Board Meeting" simulation
  - [ ] Generate all 4 report types
  - [ ] Export each in all 3 formats (12 files total)
  - [ ] Verify report accuracy against database
  - [ ] Verify markdown is human-readable
  - [ ] Verify JSON is valid and parseable
  - [ ] Verify CSV imports into spreadsheet correctly

**Documentation:**
- [ ] USER-GUIDE.md section on reporting
- [ ] Report template customization guide
- [ ] Example reports for each scenario type

**Acceptance Criteria:**
- ✅ All 4 report types generate correctly
- ✅ All 3 output formats work
- ✅ E2E test passes
- ✅ Reports are accurate and useful
- ✅ Zero changes to Phase 1 code

---

### 2.3: Export Pipeline ⏳

**Files to Create:**
```
reporting/
├── export_pipeline.py      # Batch export orchestration
└── export_formats.py       # Format-specific exporters
```

**Implementation Checklist:**

- [ ] **ExportPipeline class** (reporting/export_pipeline.py)
  - [ ] `export_world(world_id, format, destination)` - Export entire world
  - [ ] `export_dataset(world_id, format, destination)` - Export as ML dataset
  - [ ] Supported formats:
    - [ ] `jsonl` - One entity/timepoint per line (ML training)
    - [ ] `json` - Full structured export
    - [ ] `csv` - Multiple CSVs (entities.csv, timepoints.csv, dialogs.csv)
    - [ ] `sqlite` - Database snapshot (for archival)
  - [ ] Incremental export (only new data since last export)
  - [ ] Compression options (gzip, bzip2)

- [ ] **Dataset formatters** (reporting/export_formats.py)
  - [ ] `JSONLExporter` - ML training format
    - [ ] Each line: complete dialog with full context
    - [ ] Include entity states, relationships, decisions
    - [ ] Metadata: cost, token count, generation time
  - [ ] `CSVExporter` - Relational tables
    - [ ] entities.csv, timepoints.csv, dialogs.csv, relationships.csv
    - [ ] Foreign key references preserved
  - [ ] `DatabaseExporter` - SQLite snapshot
    - [ ] Full database copy
    - [ ] Vacuum and optimize
    - [ ] Include schema version for migrations

**Test Requirements:**
- [ ] **test_export_pipeline.py** (15 tests)
  - [ ] Test each export format
  - [ ] Test incremental export
  - [ ] Test compression
  - [ ] Test large dataset export (100+ entities)

**E2E Requirement:**
- [ ] **test_e2e_export.py**
  - [ ] Generate 100-variation dataset
  - [ ] Export to JSONL (ML training)
  - [ ] Export to CSV (analysis)
  - [ ] Export to SQLite (archival)
  - [ ] Verify all exports are complete
  - [ ] Verify JSONL loads into pandas correctly
  - [ ] Verify SQLite snapshot can be queried

**Acceptance Criteria:**
- ✅ All export formats work correctly
- ✅ Can export 100+ item datasets reliably
- ✅ E2E test passes
- ✅ Zero changes to Phase 1 code

---

### 2.4: Sprint 2 Integration & Testing ⏳

**E2E Sprint Test:**
- [ ] **test_e2e_sprint2_full_stack.py**
  - [ ] Generate dataset (50 variations + 1 vertical)
  - [ ] Run batch queries
  - [ ] Generate all 4 report types
  - [ ] Export in all formats (JSONL, JSON, CSV, SQLite)
  - [ ] Verify report accuracy
  - [ ] Verify export integrity

**Documentation:**
- [ ] Sprint 2 completion summary
- [ ] Reporting best practices
- [ ] Export format comparison

**Acceptance Criteria:**
- ✅ All 2.1-2.3 components integrated
- ✅ E2E full-stack test passes
- ✅ Documentation complete
- ✅ Zero breaking changes to Phase 1
- ✅ Ready for Sprint 3 (Natural Language Interface)

---

## Sprint 3: Natural Language Interface 🗣️

**Goal:** LLM-powered config generation from natural language descriptions

**Estimated Time:** 1-2 weeks
**Dependencies:** Sprint 1 + 2 complete
**Risk Level:** MEDIUM (LLM reliability considerations)

### 3.1: NL to Config Translation ⏳

**Files to Create:**
```
nl_interface/
├── __init__.py
├── nl_to_config.py         # Natural language parser
├── config_validator.py     # Validate generated configs
└── prompts.py              # System prompts for LLM
```

**Implementation Checklist:**

- [ ] **NLConfigGenerator class** (nl_interface/nl_to_config.py)
  - [ ] `generate_config(natural_language_description)` → SimulationConfig
  - [ ] Uses Llama 405B via OpenRouter
  - [ ] Few-shot prompting with example configs
  - [ ] Structured output (Pydantic schema validation)
  - [ ] Fallback handling if LLM fails:
    - [ ] Retry with temperature adjustment
    - [ ] Return error with helpful suggestions
  - [ ] Confidence scoring (how well NL matches generated config)

- [ ] **ConfigValidator class** (nl_interface/config_validator.py)
  - [ ] `validate_config(config)` - Check config is well-formed
  - [ ] Constraint checking:
    - [ ] Entity count reasonable (1-1000)
    - [ ] Timepoint count reasonable (1-100)
    - [ ] Resolution levels valid
    - [ ] Temporal mode valid
    - [ ] Outputs are achievable
  - [ ] Semantic validation:
    - [ ] Historical dates are plausible
    - [ ] Entity roles match scenario
    - [ ] Relationships are coherent
  - [ ] Suggest fixes for invalid configs

- [ ] **Prompt templates** (nl_interface/prompts.py)
  - [ ] System prompt for config generation
  - [ ] Few-shot examples:
    - [ ] "Simulate the Apollo 13 crisis..." → config
    - [ ] "Generate 100 board meeting variations..." → config
    - [ ] "Model the Constitutional Convention..." → config
  - [ ] Error recovery prompts
  - [ ] Clarification prompts (if input ambiguous)

**Test Requirements:**
- [ ] **test_nl_to_config.py** (20 tests)
  - [ ] Test valid NL descriptions → configs
  - [ ] Test ambiguous descriptions (should ask for clarification)
  - [ ] Test invalid descriptions (should error gracefully)
  - [ ] Test config validation
  - [ ] Test confidence scoring

**E2E Requirement:**
- [ ] **test_e2e_nl_interface.py**
  - [ ] Input: "Simulate a tech startup board meeting where the CEO proposes an acquisition. 5 board members, 3 hours, focus on dialog and relationships."
  - [ ] Verify config generated correctly
  - [ ] Execute generated config
  - [ ] Verify simulation runs successfully
  - [ ] Verify outputs match NL request (dialog + relationships)

**Documentation:**
- [ ] USER-GUIDE.md section on natural language interface
- [ ] Best practices for NL descriptions
- [ ] Examples: Good vs ambiguous descriptions
- [ ] Limitations and edge cases

**Acceptance Criteria:**
- ✅ Can generate valid configs from NL
- ✅ Config validation works
- ✅ E2E test passes with real LLM
- ✅ Handles ambiguous/invalid input gracefully
- ✅ Zero changes to Phase 1 code

---

### 3.2: Interactive Refinement ⏳

**Files to Create:**
```
nl_interface/
├── interactive_refiner.py  # Interactive config refinement
└── clarification_engine.py # Ask user for clarifications
```

**Implementation Checklist:**

- [ ] **InteractiveRefiner class** (nl_interface/interactive_refiner.py)
  - [ ] After generating config, show user and ask for refinements
  - [ ] User can:
    - [ ] Adjust entity count
    - [ ] Modify timepoint structure
    - [ ] Change output formats
    - [ ] Add/remove specific entities or events
  - [ ] Iterative refinement (multiple rounds if needed)
  - [ ] Preview mode: Show what would be generated before running

- [ ] **ClarificationEngine class** (nl_interface/clarification_engine.py)
  - [ ] Detect ambiguities in NL description
  - [ ] Ask targeted clarification questions:
    - [ ] "How many entities do you want?" (if not specified)
    - [ ] "What time period?" (if ambiguous)
    - [ ] "What outputs do you need?" (if not specified)
  - [ ] Multiple-choice questions (easier than free-form)
  - [ ] Smart defaults (suggest based on scenario type)

**Test Requirements:**
- [ ] **test_interactive_refiner.py** (10 tests)
  - [ ] Test refinement workflow
  - [ ] Test preview mode
  - [ ] Test clarification prompts

**E2E Requirement:**
- [ ] **test_e2e_interactive_nl.py**
  - [ ] Ambiguous input: "Simulate a meeting"
  - [ ] System asks clarifications (what kind? how many people? etc.)
  - [ ] User provides answers
  - [ ] System generates refined config
  - [ ] User previews and approves
  - [ ] Execution proceeds

**Acceptance Criteria:**
- ✅ Interactive refinement works
- ✅ Clarification questions are helpful
- ✅ E2E test passes
- ✅ Zero changes to Phase 1 code

---

### 3.3: Sprint 3 Integration & Testing ⏳

**E2E Sprint Test:**
- [ ] **test_e2e_sprint3_full_stack.py**
  - [ ] NL input: Complex scenario description
  - [ ] Config generation (with clarifications if needed)
  - [ ] Interactive refinement
  - [ ] Execution (horizontal or vertical)
  - [ ] Reporting (all formats)
  - [ ] Export (ML dataset)
  - [ ] Verify end-to-end accuracy

**Documentation:**
- [ ] Sprint 3 completion summary
- [ ] NL interface best practices
- [ ] Example workflows

**Acceptance Criteria:**
- ✅ All 3.1-3.2 components integrated
- ✅ E2E full-stack test passes
- ✅ Documentation complete
- ✅ Zero breaking changes to Phase 1
- ✅ Ready for Sprint 4 (User Interface)

---

## Sprint 4: User Interface (CLI + API) 🖥️

**Goal:** Production-ready CLI and optional API for programmatic access

**Estimated Time:** 1-2 weeks
**Dependencies:** Sprint 1 + 2 + 3 complete
**Risk Level:** LOW (wrapper layer only)

### 4.1: User CLI ⏳

**Files to Create:**
```
cli_user.py                 # Main user CLI entry point
```

**Implementation Checklist:**

- [ ] **User CLI commands** (cli_user.py)
  - [ ] `timepoint generate <config.json>` - Run simulation from config
  - [ ] `timepoint generate-nl "<description>"` - Run from natural language
  - [ ] `timepoint world list` - List all worlds
  - [ ] `timepoint world create <world-id>` - Create new world
  - [ ] `timepoint world delete <world-id>` - Delete world
  - [ ] `timepoint query <world-id> "<query>"` - Query simulation
  - [ ] `timepoint report <world-id> --type summary` - Generate reports
  - [ ] `timepoint export <world-id> --format jsonl` - Export data
  - [ ] `timepoint status <job-id>` - Check generation progress
  - [ ] Interactive mode: `timepoint interactive` (REPL for queries)

- [ ] **CLI features:**
  - [ ] Progress bars for long-running operations
  - [ ] Colorized output (success=green, error=red, warning=yellow)
  - [ ] JSON output mode (for scripting): `--json`
  - [ ] Verbose mode: `--verbose` or `-v`
  - [ ] Dry-run mode: `--dry-run` (preview without executing)
  - [ ] Configuration file support: `~/.timepoint/config.yaml`

**Test Requirements:**
- [ ] **test_cli_user.py** (25 tests)
  - [ ] Test each command
  - [ ] Test argument parsing
  - [ ] Test error handling
  - [ ] Test JSON output mode
  - [ ] Test dry-run mode

**E2E Requirement:**
- [ ] **test_e2e_cli.py**
  - [ ] Run full workflow via CLI:
    ```bash
    timepoint world create test-world
    timepoint generate-nl "Simulate a board meeting with 5 people"
    timepoint query test-world "What decisions were made?"
    timepoint report test-world --type summary
    timepoint export test-world --format jsonl
    timepoint world delete test-world
    ```
  - [ ] Verify all commands succeed
  - [ ] Verify outputs are correct

**Documentation:**
- [ ] USER-GUIDE.md section on CLI usage
- [ ] Command reference (all commands documented)
- [ ] Workflow examples (common tasks)

**Acceptance Criteria:**
- ✅ All CLI commands work
- ✅ E2E test passes
- ✅ Documentation complete
- ✅ User-friendly error messages
- ✅ Zero changes to Phase 1 code

---

### 4.2: REST API (Optional) ⏳

**Files to Create:**
```
api_server.py               # FastAPI application
api/
├── __init__.py
├── routers/
│   ├── generation.py       # Generation endpoints
│   ├── worlds.py           # World management endpoints
│   ├── queries.py          # Query endpoints
│   └── reports.py          # Reporting endpoints
└── models.py               # API request/response models
```

**Implementation Checklist:**

- [ ] **FastAPI application** (api_server.py)
  - [ ] OpenAPI documentation (auto-generated)
  - [ ] Authentication (API key based)
  - [ ] Rate limiting
  - [ ] CORS support
  - [ ] WebSocket support (for progress streaming)

- [ ] **API endpoints:**
  - [ ] `POST /api/v1/worlds` - Create world
  - [ ] `GET /api/v1/worlds` - List worlds
  - [ ] `DELETE /api/v1/worlds/{world_id}` - Delete world
  - [ ] `POST /api/v1/generate` - Start generation job
  - [ ] `GET /api/v1/jobs/{job_id}` - Check job status
  - [ ] `GET /api/v1/jobs/{job_id}/progress` - Stream progress (WebSocket)
  - [ ] `POST /api/v1/query` - Execute query
  - [ ] `POST /api/v1/reports` - Generate report
  - [ ] `GET /api/v1/exports/{world_id}` - Export data

**Test Requirements:**
- [ ] **test_api_server.py** (30 tests)
  - [ ] Test each endpoint
  - [ ] Test authentication
  - [ ] Test rate limiting
  - [ ] Test WebSocket streaming
  - [ ] Test error responses

**E2E Requirement:**
- [ ] **test_e2e_api.py**
  - [ ] Full workflow via API (same as CLI E2E)
  - [ ] Verify OpenAPI spec is valid
  - [ ] Verify WebSocket progress streaming works

**Documentation:**
- [ ] API reference (OpenAPI/Swagger)
- [ ] Authentication guide
- [ ] Rate limits and quotas
- [ ] Example client code (Python, curl)

**Acceptance Criteria:**
- ✅ All API endpoints work
- ✅ E2E test passes
- ✅ OpenAPI docs are accurate
- ✅ Authentication and rate limiting work
- ✅ Zero changes to Phase 1 code

---

### 4.3: Sprint 4 Integration & Testing ⏳

**E2E Sprint Test:**
- [ ] **test_e2e_sprint4_full_stack.py**
  - [ ] Test same workflow via CLI and API
  - [ ] Verify both produce identical results
  - [ ] Test CLI → API interoperability (export via CLI, import via API)

**Documentation:**
- [ ] Sprint 4 completion summary
- [ ] CLI vs API comparison
- [ ] Deployment guide

**Acceptance Criteria:**
- ✅ CLI and API both complete
- ✅ E2E tests pass for both
- ✅ Documentation complete
- ✅ Zero breaking changes to Phase 1
- ✅ Ready for Sprint 5 (Demo System)

---

## Sprint 5: Demo System 🎬

**Goal:** Batteries-included demos with pre-configured scenarios

**Estimated Time:** 3-5 days
**Dependencies:** Sprint 1-4 complete
**Risk Level:** LOW (uses infrastructure, doesn't extend it)

### 5.1: Demo Infrastructure ⏳

**Files to Create:**
```
demos/
├── run_demo.py             # Demo runner (uses core, doesn't extend)
├── demo.sh                 # Bash wrapper for convenience
├── configs/
│   ├── jefferson-dinner.json
│   ├── board-meeting.json
│   ├── paul-revere.json
│   ├── constitutional-convention.json
│   └── research-lab.json
├── README.md               # Demo documentation
└── outputs/                # Generated artifacts (gitignored)
```

**Implementation Checklist:**

- [ ] **Demo runner** (demos/run_demo.py)
  - [ ] Load config from `configs/`
  - [ ] Create temporary demo world (isolated from user data)
  - [ ] Run simulation
  - [ ] Generate reports automatically
  - [ ] Export to `outputs/`
  - [ ] Print summary to terminal
  - [ ] Cleanup option (delete demo world after viewing)

- [ ] **Bash wrapper** (demos/demo.sh)
  ```bash
  ./demo.sh jefferson-dinner              # Run demo
  ./demo.sh board-meeting --interactive   # Run + query mode
  ./demo.sh --list                        # List available demos
  ```

- [ ] **Demo configs:**
  - [ ] **jefferson-dinner.json** - Historical, 5 entities, dialog-focused
  - [ ] **board-meeting.json** - Modern, 5 entities, decision-making
  - [ ] **paul-revere.json** - Action/movement, animistic (horse)
  - [ ] **constitutional-convention.json** - Large-scale, counterfactuals
  - [ ] **research-lab.json** - Knowledge-intensive, collaboration

**Test Requirements:**
- [ ] **test_demos.py** (10 tests)
  - [ ] Test each demo runs successfully
  - [ ] Test demo cleanup
  - [ ] Test demo isolation (doesn't affect user data)

**E2E Requirement:**
- [ ] **test_e2e_all_demos.py**
  - [ ] Run all 5 demos
  - [ ] Verify all complete successfully
  - [ ] Verify reports generated
  - [ ] Verify outputs exported
  - [ ] Verify demo worlds cleaned up

**Documentation:**
- [ ] demos/README.md - Demo quick start
- [ ] USER-GUIDE.md - Link to demos as entry point
- [ ] Each demo config includes explanation comments

**Acceptance Criteria:**
- ✅ All 5 demos work out-of-the-box
- ✅ Demos are isolated from user data
- ✅ E2E test passes
- ✅ Documentation is clear
- ✅ Zero changes to Phase 1 code

---

### 5.2: Sprint 5 Integration & Testing ⏳

**E2E Sprint Test:**
- [ ] **test_e2e_sprint5_demos.py**
  - [ ] New user runs `./demo.sh --list`
  - [ ] User runs `./demo.sh jefferson-dinner`
  - [ ] User sees simulation complete
  - [ ] User reads generated report
  - [ ] User understands what Timepoint does
  - [ ] User wants to create their own simulation

**Documentation:**
- [ ] Sprint 5 completion summary
- [ ] Demo showcase (screenshots/output examples)

**Acceptance Criteria:**
- ✅ Demos are production-ready
- ✅ E2E test passes
- ✅ Documentation complete
- ✅ Zero breaking changes to Phase 1
- ✅ Phase 2 COMPLETE 🎉

---

## Phase 2 Complete: Success Metrics

### Technical Metrics
- [ ] All 5 sprints complete
- [ ] All E2E tests passing
- [ ] Zero breaking changes to Phase 1
- [ ] Documentation complete (USER-GUIDE.md)
- [ ] Performance benchmarks updated

### User Experience Metrics
- [ ] User can generate simulations from NL in < 5 minutes
- [ ] User can generate 100-variation datasets reliably
- [ ] User can export data in multiple formats
- [ ] User can run demos without any setup
- [ ] User has clear documentation for all features

### Code Quality Metrics
- [ ] Test coverage > 80% for new code
- [ ] All E2E tests run in < 10 minutes
- [ ] No flaky tests
- [ ] Clean separation: Phase 1 untouched
- [ ] Modular architecture (each sprint is independent)

---

## Post-Phase 2: Optional Enhancements

### Performance Optimization (Optional)
- [ ] Parallel entity generation (multi-threading)
- [ ] LLM request batching (reduce API calls)
- [ ] Database query optimization
- [ ] Caching strategies for repeated queries

### Advanced Features (Optional)
- [ ] Web UI (replace CLI)
- [ ] Collaborative editing (multi-user scenarios)
- [ ] Version control for simulations (git-like diffs)
- [ ] Marketplace for scenario configs

### Integration (Optional)
- [ ] Hugging Face integration (publish datasets)
- [ ] Weights & Biases logging (track experiments)
- [ ] MLflow integration (model training workflows)
- [ ] Jupyter notebook support (exploratory analysis)

---

## Risk Mitigation

### Technical Risks

**Risk: LLM reliability (Sprint 3 NL interface)**
- Mitigation: Fallback to manual config generation
- Mitigation: Retry logic with temperature adjustment
- Mitigation: Comprehensive prompt engineering and testing

**Risk: Long-running job failures (Sprint 1)**
- Mitigation: Checkpoint/resume functionality
- Mitigation: Comprehensive fault handling
- Mitigation: E2E tests with simulated failures

**Risk: Database corruption with world isolation**
- Mitigation: Transaction-based world operations
- Mitigation: Backup/restore functionality
- Mitigation: Thorough isolation testing

### Process Risks

**Risk: Breaking existing code (Phase 1)**
- Mitigation: Zero changes to Phase 1 modules
- Mitigation: Comprehensive regression testing
- Mitigation: Git branch protection (require tests to pass)

**Risk: Scope creep**
- Mitigation: Flywheel development (ship each sprint independently)
- Mitigation: Clear acceptance criteria for each component
- Mitigation: Mark optional enhancements as "Post-Phase 2"

**Risk: Documentation falling behind**
- Mitigation: Documentation is acceptance criteria (not optional)
- Mitigation: USER-GUIDE.md updated every sprint
- Mitigation: Examples required for each feature

---

## Timeline Estimate

**Sprint 1 (Generation):** 2-3 weeks
**Sprint 2 (Reporting):** 1-2 weeks
**Sprint 3 (NL Interface):** 1-2 weeks
**Sprint 4 (User Interface):** 1-2 weeks
**Sprint 5 (Demos):** 3-5 days

**Total Phase 2:** 6-10 weeks (1.5-2.5 months)

**Milestones:**
- [ ] Sprint 1 complete: Can generate synthetic datasets
- [ ] Sprint 2 complete: Can export and report on data
- [ ] Sprint 3 complete: Can use NL to generate configs
- [ ] Sprint 4 complete: Production CLI/API ready
- [ ] Sprint 5 complete: Demos ready for users
- [ ] Phase 2 complete: Timepoint is production-ready platform

---

## Questions?

- **Technical Specs:** See `MECHANICS.md` (Phase 1 complete)
- **User Guide:** See `USER-GUIDE.md` (Phase 2, to be written)
- **API Reference:** See `API.md` (Phase 2, to be written)
- **Current Status:** This file (PLAN.md)

---

**Next Action:** Begin Sprint 1.1 (World Management System)
