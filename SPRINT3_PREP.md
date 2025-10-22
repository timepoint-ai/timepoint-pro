# Sprint 3 Preparation: Natural Language Interface

**Status**: Ready to begin
**Dependencies**: ✅ Sprint 1 Complete, ✅ Sprint 2 Complete
**Risk Level**: MEDIUM (LLM reliability considerations)
**Estimated Time**: 1-2 weeks

---

## Sprint 2 Completion Proof

### Test Results
- **Total E2E Tests**: 45/45 passing ✅
  - Phase 1 E2E: 13/13 ✅
  - Sprint 1 E2E: 16/16 ✅
  - Sprint 2 E2E: 16/16 ✅

### Components Delivered (Sprint 2)
1. **Query Engine Enhancement** (reporting/query_engine.py)
   - Batch query execution with caching
   - Aggregation queries (relationships, knowledge flow, timeline, entity comparison)
   - Cache hit rate tracking

2. **Report Generation** (reporting/report_generator.py, formatters.py)
   - 4 report types (summary, relationships, knowledge, entity_comparison)
   - 3 output formats (Markdown, JSON, CSV)
   - Template-based rendering

3. **Export Pipeline** (reporting/export_pipeline.py, export_formats.py)
   - 5 export formats (JSON, JSONL, CSV, SQLite, Markdown)
   - Compression support (gzip, bz2)
   - World package export with metadata
   - Batch export orchestration

### Integration Demo
✅ All components importable and working (demo_sprint2_integration.py)
✅ Zero breaking changes to Phase 1
✅ Full workflow validated (query → report → export)

---

## Sprint 3 Overview

**Goal**: Enable users to generate simulation configurations from natural language descriptions without writing JSON manually.

### Key Components

#### 3.1: NL to Config Translation
Convert natural language descriptions into validated SimulationConfig objects.

**Example Input**:
```
"Simulate the Apollo 13 crisis with the 3 astronauts and mission control.
Focus on decision-making under pressure. I want 10 timepoints covering
the initial explosion through safe return."
```

**Expected Output**:
```json
{
  "scenario": "Apollo 13 Crisis",
  "entities": [
    {"name": "Jim Lovell", "role": "Commander"},
    {"name": "Jack Swigert", "role": "Command Module Pilot"},
    {"name": "Fred Haise", "role": "Lunar Module Pilot"},
    {"name": "Gene Kranz", "role": "Flight Director"}
  ],
  "timepoint_count": 10,
  "start_time": "1970-04-13T19:00:00",
  "temporal_mode": "pearl",
  "focus": ["decision_making", "stress_responses"],
  "outputs": ["dialog", "decisions", "relationships"]
}
```

#### 3.2: Interactive Refinement
Allow users to iteratively refine generated configs through clarifying questions.

**Workflow**:
1. User provides NL description (possibly ambiguous)
2. System generates initial config
3. System detects ambiguities and asks clarifying questions
4. User provides answers
5. System refines config
6. User previews and approves
7. Execution proceeds

---

## Technical Architecture

### LLM Integration Strategy

**Primary Model**: Llama 405B Instruct (via OpenRouter)
- Reasons: Best at structured output, instruction following
- Fallback: Llama 70B Instruct (faster, cheaper for retries)

**Prompt Engineering Approach**:
```
System: You are a configuration generator for historical/scenario simulations.
Convert natural language descriptions into valid JSON configs.

Few-shot examples:
1. "Board meeting with 5 executives" → {config}
2. "The Constitutional Convention of 1787" → {config}
3. "Tech startup pitch to investors" → {config}

User description: {user_input}

Return only valid JSON matching this schema:
{SimulationConfig.schema_json()}
```

### Validation Pipeline

```
NL Input
   ↓
LLM Generation
   ↓
Pydantic Validation (SimulationConfig schema)
   ↓
Semantic Validation (ConfigValidator)
   ↓
Confidence Scoring
   ↓
[If low confidence] → Clarification Questions
   ↓
User Approval
   ↓
Execute Simulation
```

### Error Handling

**LLM Failures**:
- Retry with temperature adjustment (0.7 → 0.3 → 0.1)
- Retry with simpler prompt
- Fall back to manual config entry
- Provide helpful error message with suggestions

**Invalid Configs**:
- Validate with Pydantic schema
- Semantic validation (dates, entity counts, etc.)
- Suggest fixes (e.g., "Entity count should be 1-100")
- Allow user to edit JSON directly

**Ambiguous Input**:
- Detect missing required fields
- Ask targeted clarification questions
- Provide smart defaults based on scenario type
- Show confidence score to user

---

## Implementation Plan

### Phase 3.1: NL to Config Translation (Week 1)

**Files to Create**:
```
nl_interface/
├── __init__.py
├── nl_to_config.py         # Core NL→Config logic
├── config_validator.py     # Semantic validation
└── prompts.py              # LLM prompts and templates
```

**Day 1-2**: Core NLConfigGenerator
- LLM integration with OpenRouter
- Basic prompt template
- Pydantic schema validation
- Initial few-shot examples

**Day 3-4**: ConfigValidator
- Constraint checking (entity counts, dates, etc.)
- Semantic validation (coherence checks)
- Fix suggestions

**Day 5**: Testing & Refinement
- Unit tests (20+ tests)
- E2E test with real LLM
- Confidence scoring implementation

### Phase 3.2: Interactive Refinement (Week 2)

**Files to Create**:
```
nl_interface/
├── interactive_refiner.py  # Interactive refinement loop
└── clarification_engine.py # Ambiguity detection and questions
```

**Day 6-7**: InteractiveRefiner
- Config preview and approval workflow
- Iterative refinement loop
- Parameter adjustment interface

**Day 8-9**: ClarificationEngine
- Ambiguity detection
- Question generation
- Smart defaults

**Day 10**: Integration & Testing
- E2E Sprint 3 test
- Documentation
- Demo scripts

---

## Test Strategy

### Unit Tests (30+ tests)
- `test_nl_to_config.py`: Config generation from various inputs
- `test_config_validator.py`: Validation and fix suggestions
- `test_interactive_refiner.py`: Refinement workflow
- `test_clarification_engine.py`: Question generation

### Integration Tests
- `test_e2e_nl_interface.py`: Full workflow with real LLM
- `test_e2e_interactive_nl.py`: Interactive refinement with clarifications
- `test_e2e_sprint3_full_stack.py`: NL → Config → Generate → Report → Export

### Test Scenarios
1. **Simple**: "Board meeting with 5 people"
2. **Historical**: "The Constitutional Convention of 1787"
3. **Complex**: "Apollo 13 crisis with timeline and stress modeling"
4. **Ambiguous**: "Simulate a meeting" (requires clarifications)
5. **Invalid**: Nonsensical input (should error gracefully)

---

## Success Criteria

### Functional Requirements
- ✅ Can generate valid configs from NL descriptions
- ✅ Config validation catches invalid inputs
- ✅ Interactive refinement allows user corrections
- ✅ Clarification questions are helpful and targeted
- ✅ Handles ambiguous input gracefully
- ✅ Error messages are clear and actionable

### Quality Requirements
- ✅ 30+ unit tests passing
- ✅ 3+ E2E tests passing with real LLM
- ✅ Confidence scoring helps users assess quality
- ✅ Generated configs execute successfully
- ✅ Documentation complete with examples

### Integration Requirements
- ✅ Zero breaking changes to Phase 1
- ✅ Zero breaking changes to Sprint 1 & 2
- ✅ All previous E2E tests still pass (45/45)
- ✅ CLI integration ready for Sprint 4

---

## Known Risks & Mitigations

### Risk: LLM Hallucinations
**Mitigation**:
- Strong Pydantic schema validation
- Semantic validation layer
- Confidence scoring
- User approval before execution
- Fallback to manual config

### Risk: Ambiguous Natural Language
**Mitigation**:
- Clarification engine
- Smart defaults
- Preview mode
- Examples in documentation

### Risk: Cost of LLM Calls
**Mitigation**:
- Cache common configs
- Use cheaper model for retries
- Limit max retries
- Provide option to skip NL interface

### Risk: Integration Complexity
**Mitigation**:
- Clean interface boundaries
- Comprehensive testing
- Optional feature (can still use JSON configs)
- Gradual rollout

---

## Documentation Plan

### USER-GUIDE.md Updates
- Section: "Natural Language Interface"
- Subsection: "Writing Good NL Descriptions"
- Subsection: "Interactive Refinement"
- Subsection: "Troubleshooting Config Generation"

### Examples to Include
1. Simple board meeting
2. Historical scenario (Jefferson Dinner)
3. Complex multi-entity scenario
4. Ambiguous input → clarification flow
5. Bad input → error handling

### Best Practices
- Be specific about entity counts
- Specify time period if historical
- Indicate focus areas (dialog, decisions, relationships)
- Specify desired outputs
- Provide context for scenario type

---

## Ready to Begin?

**Pre-flight Checklist**:
- ✅ Sprint 1 complete and tested
- ✅ Sprint 2 complete and tested
- ✅ All E2E tests passing (45/45)
- ✅ Zero breaking changes verified
- ✅ Architecture documented
- ✅ Test strategy defined
- ✅ Risk mitigation planned

**Next Action**: Create `nl_interface/__init__.py` and begin Sprint 3.1

---

**Estimated Completion**: 2 weeks
**Test Target**: 30+ unit tests, 3+ E2E tests
**Integration Target**: Zero breaking changes, all existing tests pass
