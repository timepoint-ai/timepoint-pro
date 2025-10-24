# Mechanism Coverage Strategy

## Current Status: 7/17 Tracked (41.2%) - Phase 7.5 Complete ✅

### ✅ **Phases Completed:**

**Phase 7.5: Test Reliability Improvements** ✅ COMPLETE
- **Objective**: Achieve >90% test reliability before Phase 8 tracking infrastructure
- **Fixed 8 Critical Bugs**:
  1. M12 schema mismatch - entity.timepoint column missing
  2. M9 NoneType handling in generate_entity_on_demand()
  3. M9 role inference prioritization (event context first)
  4. M9 cache bypass for missing entities
  5. M9 generate any missing entity (not just target_entity)
  6. M5 query_count increment moved before cache check
  7. M13 enumerate subscripting bug in detect_contradictions()
  8. M13 datetime JSON serialization (2 locations in workflows.py)
- **Removed ALL Mock Objects** from test_phase3_dialog_multi_entity.py
- **Test Reliability**: Improved from **71.7%** → **90.6%** ✅
- **Mechanism Results**:
  - **M5 Query Resolution**: 16/17 (94.1%) → **17/17 (100%)** ✅
  - **M9 On-Demand Generation**: 18/23 (78.3%) → **21/23 (91.3%)** ✅
  - **M12 Counterfactual**: 1/2 (50%) → **2/2 (100%)** ✅
  - **M13 Multi-Entity**: 4/11 (36.4%) → **8/11 (72.7%)** ✅
- **Files Modified**: 5 files (test_branching_integration.py, query_interface.py, workflows.py, test_phase3_dialog_multi_entity.py, test_m5_query_resolution.py)
- **Result**: Exceeded 90% test reliability target, ready for Phase 8 tracking infrastructure

**Phase 6: Critical Bug Fixes** ✅ COMPLETE
- **Fixed 3 Critical Blockers**:
  1. UNIQUE constraint violations in storage.py (entity upsert logic)
  2. JSON markdown code fence wrapping in LLM responses
  3. OpenRouter response dict access (changed from `.choices` to `["choices"]`)
- **M5 Tests**: Improved from 12/17 (70.6%) → **16/17 (94.1%)** ✅
- **M9 Tests**: **17/23 (73.9%)** - on-demand generation partially working
- **M12 Tests**: 1/2 (50%) - schema mismatch with `entity.timepoint` column
- **M13 Tests**: 3/11 (27.3%) - mock object configuration issues
- **Result**: Major improvement in M5 mechanism reliability, but 7/17 coverage unchanged (no new mechanisms tracked)

### ✅ **Previous Phases:**

**Phase 1: Entity Lifecycle Fix** ✅
- **Problem**: Entity metadata from orchestrator was destroyed during training workflow
- **Fix**: Modified `workflows.py:72-126` (`aggregate_populations`) to preserve existing metadata while merging new LLM data
- **Result**: Entities now retain physical_tensor, circadian, prospection, consciousness attributes through workflow

**Phase 2: Mock Infrastructure Removal** ✅
- **Scope**: Removed ALL mock/dry-run code from `llm.py` and `llm_v2.py`
- **Result**: System now enforces real OpenRouter LLM calls everywhere
- **Files Modified**:
  - `llm_v2.py` - Removed dry_run parameter, mock methods, ALLOW_MOCK_MODE checks
  - `llm.py` - Removed dry_run parameter, mock methods, added API key validation

**Phase 3: Documentation Cleanup** ✅
- Aligned README.md and technical docs with real-LLM-only architecture
- Removed references to mock/dry-run modes

**Phase 4: Test Infrastructure Fix** ✅
- Fixed all dry_run parameter usages in 12 files (test files + core files)
- Resolved AttributeError risks from Phase 2 changes
- All tests now use real OpenRouter API integration

**Phase 5: Comprehensive Mechanism Testing** ✅
- Ran 5 template-based E2E tests
- Ran 5 pytest mechanism test suites
- **Result**: Added M8, improved from 6/17 to 7/17 tracked mechanisms

### 📊 **Currently Tracked Mechanisms (7/17):**

| Mechanism | Name | Location | Firings |
|-----------|------|----------|---------|
| **M1** | Entity Lifecycle Management | `orchestrator.py:829, 918` | 42 |
| **M3** | Graph Construction & Eigenvector Centrality | `orchestrator.py:766` | 21 |
| **M4** | Tensor Transformation & Embedding | `validation.py:150` | 87 |
| **M7** | Causal Chain Generation | `workflows.py:2214` | 10 |
| **M8** | Vertical Timepoint Expansion ⭐NEW | `workflows.py:533` | 4 |
| **M11** | Dialog Synthesis | `workflows.py:648` | 25 |
| **M17** | Metadata Tracking System | `orchestrator.py:1032` | 21 |

### 🔍 **Phase 5 Gap Analysis - Why 10 Mechanisms Didn't Fire:**

| Mechanism | Status | Root Cause | Recommended Fix |
|-----------|--------|------------|-----------------|
| **M2** | Exposure Event Tracking | ❌ Not Triggered | Requires multi-timepoint workflows with knowledge propagation. Template tests hit errors before M2 could fire. |
| **M5** | Query Resolution with Lazy Elevation | ❌ LLMClient.client AttributeError | `query_interface.py` expects `llm.client` attribute that doesn't exist in `llm_v2.LLMClient`. Tests passed but mechanism never fired due to error handling. |
| **M6** | Knowledge Propagation | ❌ Not Triggered | Depends on M2 firing first. Cascade failure from multi-timepoint errors. |
| **M9** | On-Demand Entity Generation | ❌ LLMClient.client AttributeError | Same issue as M5 - code uses `llm.client` which doesn't exist in llm_v2. |
| **M10** | Scene-Level Entity Management | ❌ Not Triggered | Test execution incomplete - tests passed but queries never reached the mechanism. |
| **M12** | Counterfactual Branching | ❌ SQLite UNIQUE Constraint | Test failed with `sqlite3.IntegrityError: UNIQUE constraint failed: entity.entity_id` |
| **M13** | Multi-Entity Synthesis | ❌ GraphStore API Mismatch | `GraphStore.get_exposure_events() got an unexpected keyword argument 'limit'` |
| **M14** | Cost-Aware Resolution Adjustment | ❌ Not Triggered | Mechanism exists but hospital_crisis template didn't trigger circadian/cost-aware logic. |
| **M15** | Oxen Integration for Fine-Tuning | ❌ Template Error | detective_prospection template failed with ISO format error before M15 could execute. |
| **M16** | Template-Based Scenario Execution | ❌ Not Triggered | kami_shrine completed but M16 animistic entity tracking didn't fire (decorator placement issue?). |

---

## 🔧 **Phase 6: Bug Fix Results**

### ✅ **Fixed Issues (3/3 Critical Blockers Resolved):**

**Issue #1: UNIQUE Constraint Violations** ✅ FIXED
- **Problem**: `sqlite3.IntegrityError: UNIQUE constraint failed: entity.entity_id` when saving entities
- **Root Cause**: Entity schema has both `id` (primary key) and `entity_id` (unique field). `session.merge()` didn't recognize detached entities without `id`, causing INSERT instead of UPDATE
- **Fix**: Rewrote `storage.py:19-51` to query by `entity_id` first, then UPDATE existing or INSERT new
- **Result**: M5 elevation tests now pass, entity saves work correctly

**Issue #2: JSON Markdown Wrapping** ✅ FIXED
- **Problem**: LLM responses wrapped in ```json ``` code blocks broke JSON parsing
- **Root Cause**: OpenRouter returns markdown-formatted responses that need cleaning
- **Fix**: Created `strip_markdown_json()` helper in `query_interface.py:24-36`, applied at lines 143 and 1255
- **Result**: Query intent parsing now works reliably

**Issue #3: Response Dict Access** ✅ FIXED
- **Problem**: `AttributeError: 'dict' object has no attribute 'choices'`
- **Root Cause**: OpenRouterClient returns dict, not OpenAI-style object
- **Fix**: Changed to dict access pattern in `resolution_engine.py:281` and `llm.py:390`
- **Result**: Resolution engine knowledge enrichment now works

### 📊 **Test Results After Phase 7.5:**

| Test Suite | Phase 6 | Phase 7.5 | Improvement | Status |
|------------|---------|-----------|-------------|--------|
| **M5 Query Resolution** | 16/17 (94.1%) | **17/17 (100%)** | +5.9% | ✅ PERFECT |
| **M9 On-Demand Generation** | 18/23 (78.3%) | **21/23 (91.3%)** | +13% | ✅ Excellent |
| **M12 Counterfactual Branching** | 1/2 (50%) | **2/2 (100%)** | +50% | ✅ PERFECT |
| **M13 Multi-Entity Synthesis** | 4/11 (36.4%) | **8/11 (72.7%)** | +36.3% | ✅ Major improvement |
| **OVERALL TEST RELIABILITY** | **38/53 (71.7%)** | **48/53 (90.6%)** | +18.9% | ✅ TARGET EXCEEDED |

### ✅ **All Phase 7 Issues Resolved:**

**M5**: ✅ Query count cache issue fixed
- Moved increment before cache check in `query_interface.py:343-348`
- Now **17/17 (100%)** passing

**M9**: ✅ 5 critical bugs fixed
- NoneType handling in generate_entity_on_demand()
- Role inference prioritization (event context first)
- Cache bypass for missing entities
- Generate any missing entity (not just target_entity)
- Now **21/23 (91.3%)** passing

**M12**: ✅ Schema issue resolved
- Added database rebuild in test setup
- entity.timepoint column now exists
- Now **2/2 (100%)** passing

**M13**: ✅ Mock removal + real bug fixes
- Removed ALL Mock objects
- Fixed 3 datetime JSON serialization bugs
- Fixed argument unpacking bug
- Fixed LLMClient initialization
- Now **8/11 (72.7%)** passing with REAL implementations

### 📋 **Critical Issues Discovered:**

1. **LLMClient Architecture Mismatch**: `llm_v2.LLMClient` doesn't expose `.client` attribute, but `query_interface.py` and other files expect it
2. **Database Constraint Violations**: UNIQUE constraints failing during entity updates (M12)
3. **API Signature Mismatches**: `GraphStore.get_exposure_events()` signature changed (M13)
4. **Template Configuration Errors**: ISO datetime format issues in detective_prospection and board_meeting templates
5. **Missing Tensor Attributes**: Entities created without tensor attributes, causing workflow steps to skip

---

## Mechanism Coverage Matrix

### Category 1: Entity State & Validation (6 mechanisms)

| ID | Mechanism | Location | Status | How to Track |
|----|-----------|----------|--------|--------------|
| M4 | Physics Validation | `validation.py:150` | ✅ Tracked | Auto-tracked via validation workflow |
| M6 | TTM Tensor Compression | `tensors.py:24` | ⏳ Needs testing | Run workflow with entities that have tensor attributes |
| M8 | Embodied States (Pain/Illness) | `workflows.py:533` | ⏳ Needs testing | Use hospital_crisis template with pain_level/fever |
| M11 | Dialog Synthesis | `workflows.py:648` | ⏳ Needs testing | Enable dialog synthesis in workflow |
| M14 | Circadian Patterns | `validation.py:527` | ⏳ Needs testing | Use circadian_config in templates |
| M15 | Entity Prospection | `workflows.py:1186` | ⏳ Needs testing | Use detective_prospection template |

### Category 2: Query Interface (5 mechanisms)

| ID | Mechanism | Location | Status | How to Track |
|----|-----------|----------|--------|--------------|
| M5 | Query Resolution | `query_interface.py:336, 648` | ⏳ Needs testing | pytest test_m5_query_resolution.py |
| M9 | On-Demand Generation | `query_interface.py:1197, 1204` | ⏳ Needs testing | pytest test_m9_on_demand_generation.py |
| M10 | Scene Queries | via `query_interface.py` | ⏳ Needs testing | pytest test_scene_queries.py |
| M12 | Counterfactual Branching | `workflows.py:1419` | ⏳ Needs testing | pytest test_branching_integration.py |
| M13 | Multi-Entity Synthesis | `workflows.py:966` | ⏳ Needs testing | pytest test_phase3_dialog_multi_entity.py |

### Category 3: Workflow & Temporal (4 mechanisms)

| ID | Mechanism | Location | Status | How to Track |
|----|-----------|----------|--------|--------------|
| M1 | Heterogeneous Fidelity | `orchestrator.py:829, 918` | ✅ Tracked | Auto-tracked via orchestrator |
| M2 | Progressive Training | `workflows.py:200` | ⏳ Needs testing | Auto-tracked via multi-timepoint workflows |
| M7 | Causal Temporal Chains | `workflows.py:2214` | ⏳ Needs testing | Use board_meeting or jefferson_dinner (multi-TP) |
| M17 | Modal Temporal Causality | `orchestrator.py:1032` | ✅ Tracked | Auto-tracked via orchestrator |

### Category 4: Special Entity Types (2 mechanisms)

| ID | Mechanism | Location | Status | How to Track |
|----|-----------|----------|--------|--------------|
| M3 | Exposure Event Tracking | `orchestrator.py:766` | ✅ Tracked | Auto-tracked via orchestrator |
| M16 | Animistic Entities | `workflows.py:2018` | ⏳ Needs testing | Use kami_shrine template (animism_level > 0) |

---

## Testing Strategy

### **Next Phase: Test Infrastructure (Phase 4)**

**Verify all tests work with real LLM only:**
1. Check test files don't have mock dependencies
2. Ensure API key setup in all tests
3. Fix any tests broken by Phase 2 changes
4. Verify `run_all_mechanism_tests.py` works end-to-end

### **Target Phase: 17/17 Coverage (Phase 5)**

**Step 1: Run Template-Based Tests**
```bash
python3.10 run_all_mechanism_tests.py
```
**Templates:**
- `hospital_crisis` → M8 (pain/illness), M14 (circadian)
- `kami_shrine` → M16 (animistic entities)
- `detective_prospection` → M15 (prospection)
- `board_meeting` → M7 (temporal chains)
- `jefferson_dinner` → M3, M7 (multi-entity)

**Expected:** M1, M2, M3, M4, M6, M7, M8, M11, M14, M15, M16, M17 = ~12/17

**Step 2: Run pytest Mechanism Tests**
```bash
pytest test_m5_query_resolution.py -v
pytest test_m9_on_demand_generation.py -v
pytest test_scene_queries.py -v
pytest test_branching_integration.py -v
pytest test_phase3_dialog_multi_entity.py -v
```

**Expected:** +M5, M9, M10, M12, M13 = **17/17 total** ✅

---

## Template-to-Mechanism Mapping

### Existing Templates (in config_schema.py)

| Template | Timepoints | Mechanisms |
|----------|------------|------------|
| `jefferson_dinner` | 1 | M1, M3, M17 ✅ |
| `board_meeting` | 3 | M1, M3, M7, M17 |
| `hospital_crisis` | 3 | M8, M14 (pain, circadian) |
| `kami_shrine` | 1 | M16 (animistic) |
| `detective_prospection` | 1 | M15 (prospection) |

---

## Future Phase: LLM Consolidation (Phase 6)

**After 17/17 achieved, consolidate llm.py and llm_v2.py:**
1. Choose llm_v2.py as winner (better architecture)
2. Remove all legacy fallback code
3. Update ~36 files with import changes
4. Delete llm.py
5. Rename llm_v2.py → llm.py
6. Re-verify 17/17 coverage

**Why After Coverage:** Lower risk to refactor from validated working state

---

## Success Criteria

- ✅ Phase 1: Entity Lifecycle Fix - Infrastructure fixed, metadata preservation working
- ✅ Phase 2: Mock Infrastructure Removal - All mocks removed, real LLM enforcement
- ✅ Phase 3: Documentation Cleanup - Aligned with real-LLM-only architecture
- ✅ Phase 4: Test Infrastructure Fix - All dry_run dependencies removed from tests
- ✅ Phase 5: Comprehensive Mechanism Testing - **7/17 tracked (41.2%)**
  - **Success**: Added M8, improved from baseline 6/17
  - **Blockers Identified**: LLMClient.client attribute mismatch, database constraints, API signature changes
- ⏳ **Phase 6 (NEXT)**: Fix critical issues and achieve 17/17 coverage
  - Priority 1: Fix LLMClient.client attribute access (M5, M9)
  - Priority 2: Fix GraphStore API signatures (M13)
  - Priority 3: Fix database UNIQUE constraints (M12)
  - Priority 4: Fix template ISO format errors (M15)
  - Priority 5: Verify decorator placements (M16, M14)
- ⏳ Phase 7: LLM Consolidation - Single LLM client implementation

**Current Goal**: Fix 5 critical blockers, then re-run Phase 5 to achieve 17/17 ✅

---

## Next Steps

### Immediate Actions (Phase 6):

1. **Fix LLMClient Architecture** (`query_interface.py`, `workflows.py`):
   - Add `.client` property to `llm_v2.LLMClient` OR
   - Update all code to use `llm` directly instead of `llm.client`

2. **Fix GraphStore API** (`workflows.py:648`):
   - Update `synthesize_dialog` to match `GraphStore.get_exposure_events()` signature
   - Remove `limit` parameter or add it to GraphStore method

3. **Fix Database Constraints** (`workflows.py`, entity save logic):
   - Use UPDATE instead of INSERT for existing entities
   - Add proper upsert logic to prevent UNIQUE constraint violations

4. **Fix Template Configuration** (`generation/config_schema.py`):
   - Fix ISO datetime format in detective_prospection template
   - Fix ISO datetime format in board_meeting template

5. **Verify Decorator Placements**:
   - Check M16 decorator in animistic entity workflows
   - Check M14 decorator in circadian/cost-aware resolution logic

**After Fixes**: Re-run `python3.10 run_all_mechanism_tests.py` and pytest suite to achieve 17/17 ✅
