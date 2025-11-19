# Dashboard Pages: Data Expectations and Gaps Analysis

## Executive Summary

The Timepoint Dashboard consists of 6 pages with varying completeness. Three critical pages (index.qmd, screenplay.qmd, dialogs.qmd) have significant gaps between what they attempt to display and what data is actually generated/available.

---

## 1. INDEX.QMD - Main Run Details Page

### What It Tries to Display

#### Metadata Section
- Run ID
- Template ID  
- Status
- Causal Mode
- Cost (USD)
- Started/Completed timestamps
- Duration in seconds

#### Simulation Stats
- Entities Created
- Timepoints Created
- LLM Calls
- Tokens Used

#### Mechanisms Used
- Badge display of mechanism names with usage counts

#### Executive Summary
- Narrative markdown summary

#### Timeline Visualization
- Interactive vis-timeline showing timepoints
- Hover tooltips with:
  - Timepoint ID
  - Event description
  - Timestamp
  - Entities present
  - Dialog turn count
  - Importance score
- Detail panel with full timepoint information including character list and dialog snippets

#### Network Visualization
- Cytoscape.js entity relationship graph showing:
  - Entity nodes
  - Relationship edges
  - Relationship labels

#### Metrics Tab
- Radar chart of mechanism coverage

#### Detailed Analysis Tabs
1. **Resolution Assignments Table**: Entity ID, resolution, timepoint, timestamp
2. **Validations Table**: Validator name, passed/failed status, message, timestamp
3. **Mechanism Timeline Table**: Mechanism, function name, context, timestamp

### API Endpoints Called

1. `/api/run/{run_id}` - Get run metadata and details
2. `/api/narrative/{run_id}` - Get full narrative with timepoints, characters, dialogs
3. Static `dashboard_data.json` - Fallback for local display

### Data Fields Expected from API

**From `/api/run/{run_id}` (RunDetails model):**
- run_id, template_id, started_at, completed_at, causal_mode
- max_entities, max_timepoints, entities_created, timepoints_created
- training_examples, cost_usd, llm_calls, tokens_used
- duration_seconds, status, error_message
- mechanism_usage: Array of {mechanism, function_name, timestamp, context}
- resolution_assignments: Array of {entity_id, resolution, timepoint_id, timestamp}
- validations: Array of {validator_name, passed, message, timestamp}
- schema_version, fidelity_distribution

**From `/api/narrative/{run_id}`:**
- executive_summary (markdown string)
- timepoints array with:
  - timepoint_id
  - timestamp
  - event_description
  - entities_present OR entities (list of entity names)
  - dialog_turn_count
  - importance (0-1 float)
- characters array with:
  - entity_id
  - relationships (object mapping entity IDs to relationship strings)
- dialogs array with:
  - timepoint_id/timepoint
  - speaker
  - text/content
  - (optional) context

### Data It Expects That Might Not Be Generated

**Critical Gaps:**
1. **Narrative JSON Not Exported**: Page checks for `displayRun?.narrative?.executive_summary` but many runs don't have exported narrative JSON. Fallback message shows: "*Narrative JSON file not found. Run has data in database but needs export.*"
   
2. **Dialog Details in Timeline**: Detail panel tries to show dialogs from `dialogs.filter(d => d.timepoint_id === tp.timepoint_id)` but narrative may lack full dialog array or proper timepoint_id linkage

3. **Character Relationships**: Network visualization expects `char.relationships` object but this may not be generated

4. **Importance Scores**: Timeline tries to display `tp.importance` which may not be populated

### Missing Visualizations/Components

1. **PORTAL Mode Visualization**: Mentioned in about.qmd as "Coming Soon" - backward causality paths and pivot points not visualized
2. **Run Comparison**: Not implemented
3. **Export Tools**: PNG/SVG exports mentioned as "Coming Soon"
4. **Timeline Edit Mode**: Timeline is read-only, no editing capability
5. **Validation Warnings**: Validations displayed but no visual highlights for failed validations in main view

---

## 2. RUNS.QMD - Run Selection Page

### What It Tries to Display

#### Filters Sidebar (Left)
- Template selector (fetched from API)
- Status filter (hardcoded: All, completed, running, failed)
- Causal Mode filter (hardcoded: All, STANDARD, PORTAL)
- Date range filters (from/to)
- Cost range filters (0-100 USD)
- Entity count filter (min)
- Timepoint count filter (min)
- Mechanism filter (comma-separated input)
- Run ID search
- Data completeness checkbox

#### Run List Table
- Data completeness indicator (✓ or ○)
- Run ID (clickable)
- Template
- Started date/time
- Cost
- Entity count
- Timepoint count
- Status badge

#### Status Summary
- Total runs shown
- Count of complete vs incomplete runs

### API Endpoints Called

1. `/api/templates` - List available templates for dropdown
2. `/api/runs?{filters}` - Paginated list with filtering/sorting

### Data Fields Expected from API

**From `/api/templates`:**
- templates: List[str]

**From `/api/runs`:**
- runs: Array of RunListItem
  - run_id, template_id, started_at, completed_at
  - causal_mode, entities_created, timepoints_created
  - cost_usd, status, duration_seconds, error_message
  - mechanisms_used (dict)
- total, page, limit, pages

### Data Quality Notes

**Good Design:**
- Has logic to detect complete vs incomplete data:
  ```
  hasCompleteData = (run) => {
    return run.status === 'completed' &&
           run.cost_usd > 0 &&
           run.entities_created > 0 &&
           run.timepoints_created > 2
  }
  ```
- This prevents users from clicking runs without visualization data

**Potential Issues:**
1. **Mechanism Filter**: Uses comma-separated input but this is complex UX
2. **Date Filters**: Page uses `dateFromFilter.toISOString()` but may not properly handle timezone conversions
3. **Static Causal Mode Values**: Hardcoded values (STANDARD, PORTAL) - if new modes added, UI needs update

---

## 3. ANALYTICS.QMD - Meta-Analytics Page

### What It Tries to Display

#### Overview Metrics (4 cards)
- Total Runs
- Total Cost
- Success Rate (%)
- Average Cost

#### Run Statistics (4 cards)
- Completed runs count
- Failed runs count
- Total entities
- Total timepoints

#### Cost Analysis Chart
- Line/bar combo chart showing:
  - Total cost per day (last 30 days)
  - Run count per day (secondary axis)

#### Template Performance Chart
- Horizontal bar chart of top templates by usage count

#### Causal Mode Distribution
- Donut chart showing count of runs per causal mode

#### Mechanism Co-Occurrence Network
- Table of top 20 mechanism pairs with co-occurrence counts

### API Endpoints Called

1. `/api/meta-analytics` - All analytics data

### Data Fields Expected from API

**From `/api/meta-analytics` (MetaAnalytics model):**
- total_runs, total_cost, avg_cost
- total_entities, total_timepoints, avg_duration
- completed_runs, failed_runs, success_rate
- top_templates: Array of {template_id, count}
- cost_over_time: Array of {date, total_cost, run_count}
- mechanism_co_occurrence: Array of {mechanism1, mechanism2, co_occurrence}
- causal_mode_distribution: Array of {causal_mode, count}

### Data Quality

**Status:** Complete and well-designed
- All metrics are derived from aggregated database queries
- No reliance on narrative JSON exports
- Graceful fallbacks for empty data

---

## 4. SCREENPLAY.QMD - Screenplay Viewer

### What It Tries to Display

#### Navigation
- Scene selector (dropdown with all scenes)
- Character filter (dropdown with all unique characters)
- Search dialog box

#### Statistics
- Scene count
- Character count
- Dialog line count

#### Main Screenplay Viewer
- Full Fountain screenplay formatted with proper CSS
- Scene headings (uppercase, bordered)
- Action descriptions
- Character names (indented 40%)
- Dialogue (indented 25%)
- Parenthetical (indented 30%, italic)
- Transitions (right-aligned)

#### Filters Applied
- Scene selection filters to specific scene
- Character filter shows only scenes with that character
- Search text highlights matches with 3-token context

#### Download Button
- Exports screenplay as `.fountain` file

### API Endpoints Called

1. `/api/screenplay/{run_id}` - Returns raw Fountain screenplay text

### Data Fields Expected

**From `/api/screenplay/{run_id}`:**
- Plain text in Fountain screenplay format with tokens:
  - SCENE_HEADING
  - ACTION
  - CHARACTER
  - DIALOGUE
  - PARENTHETICAL
  - TRANSITION

### Critical Gaps

**MAJOR ISSUE:**
1. **Screenplay Not Generated**: The API endpoint fetches from filesystem:
   ```python
   screenplay_path = template_dir / f"screenplay_{timestamp}.fountain"
   ```
   But screenplay generation is not integrated into the simulation engine. Many runs won't have screenplay files.
   
2. **No Fallback to Generated Screenplay**: Unlike narrative, there's no fallback to generate from database data

3. **Fountain.js Parsing Incomplete**: Parser extracts tokens but may not handle all Fountain edge cases

### Missing Features

1. **Screenplay Generation**: Needs LLM generation of screenplay from narrative timepoints and dialogs
2. **Fountain Export Validation**: No check that exported Fountain is valid
3. **Scene Thumbnails**: No visual preview of scenes
4. **Comparison Mode**: Can't compare screenplays from different runs
5. **Highlighting Search Matches**: Search finds text but doesn't visually highlight in context

---

## 5. DIALOGS.QMD - Dialog Viewer

### What It Tries to Display

#### Filters Sidebar
- Character filter (dropdown with unique speakers)
- Timepoint filter (dropdown with unique timepoint_ids)
- Location filter (dropdown with unique locations)
- Search text input (searches dialog content, speaker, context)
- Sort options: Chronological, By Character, By Location

#### Statistics
- Character count
- Timepoint count
- Location count

#### Dialog Cards
Each card shows:
- Speaker badge
- Timepoint badge (if available)
- Location badge (if available)
- Timestamp (if available)
- Dialog content (main text)
- Context section (if available)
- Emotional state (if available)

#### Timeline View
- Grouped by timepoint
- Shows count of dialogs per timepoint
- Shows speakers in that timepoint

#### Export Button
- Exports filtered dialogs as CSV with columns:
  - Speaker, Timepoint, Location, Dialog, Context, Emotion

### API Endpoints Called

1. `/api/dialogs/{run_id}` - Returns narrative with dialogs array

### Data Fields Expected from API

**From `/api/dialogs/{run_id}`:**
- Returns narrative JSON, extracts `dialogs` array
- Each dialog object should have:
  - speaker (required)
  - dialog OR text OR content (main content)
  - timepoint_id (for grouping, preferred over `timepoint`)
  - location (optional)
  - timestamp (optional)
  - context (optional)
  - emotional_state (optional)

### Critical Gaps

**MAJOR ISSUE:**
1. **Dialogs Not in Database**: Dialogs are only stored in narrative JSON exports, not in the SQLite database. If narrative wasn't exported, no dialogs available.

2. **Field Name Inconsistency**: Code checks for:
   - `d.timepoint_id` OR `d.timepoint`
   - `d.dialog` OR `d.content` 
   - This suggests uncertainty about actual field names in exported data

3. **Emotional State Field**: Displays `dialog.emotional_state` but unclear if this is actually generated by the simulation engine

4. **Location Field**: Used for filtering but unclear if populated in most runs

### Data Quality Issues

**From the code:**
```javascript
if (searchQuery) {
  const searchLower = searchQuery.toLowerCase();
  filtered = filtered.filter(d =>
    (d.dialog && d.dialog.toLowerCase().includes(searchLower)) ||
    (d.speaker && d.speaker.toLowerCase().includes(searchLower)) ||
    (d.context && d.context.toLowerCase().includes(searchLower))
  );
}
```

This shows defensive coding - checking multiple possible field names suggests inconsistency in data generation.

### Missing Features

1. **Speaker Relationship Visualization**: No graph showing which characters interact
2. **Sentiment Analysis**: No visualization of emotional arcs
3. **Turn Count Metrics**: No stats on how much each character speaks
4. **Dialog Quality Metrics**: No indication of dialog coherence or relevance
5. **Speaker Portraits**: No way to distinguish between similar speaker names
6. **Conversation Threads**: No grouping of multi-turn exchanges

---

## 6. ABOUT.QMD - About Page

### Status
**Complete and informational.** No data fetching, static markdown content.

Provides:
- System description
- Feature list (implemented vs coming soon)
- Technical stack information
- Usage instructions
- Data sources
- Architecture documentation
- Mechanism reference

---

## Summary Table: Data Completeness

| Page | API Dependency | Narrative Dependency | Status | Critical Issues |
|------|----------------|----------------------|--------|-----------------|
| runs.qmd | YES (/api/runs, /api/templates) | NO | COMPLETE | None |
| index.qmd | YES (/api/run, /api/narrative) | YES | PARTIAL | Narrative missing, Network graph depends on narrative |
| screenplay.qmd | YES (/api/screenplay) | IMPLICIT (file-based) | INCOMPLETE | Screenplay files rarely exist, no fallback generation |
| dialogs.qmd | YES (/api/dialogs) | YES (extracts from narrative) | INCOMPLETE | Dialogs only in narrative JSON, field name inconsistencies |
| analytics.qmd | YES (/api/meta-analytics) | NO | COMPLETE | None |
| about.qmd | NO | NO | COMPLETE | N/A |

---

## Critical Data Generation Gaps

### 1. Narrative JSON Export
**Status:** Sometimes missing

Many runs complete successfully but narrative JSON is never exported. This breaks:
- Executive summary display
- Timeline with full timepoint data
- Character network visualization
- Dialog viewer
- Screenplay viewer (indirectly)

**Root Cause:** Narrative export needs to be integrated into simulation completion workflow

**Files Expected:** `datasets/{template}/narrative_{timestamp}.json`

### 2. Screenplay Generation
**Status:** Not generated

Screenplay files are never created. The system expects:
- Files at: `datasets/{template}/screenplay_{timestamp}.fountain`
- Format: Valid Fountain screenplay format
- Content: Scene-based screenplay generated from narrative

**Root Cause:** No screenplay generation implemented in simulation engine or post-processing

### 3. Dialog Metadata Consistency
**Status:** Inconsistent field names

Different runs may have different field names:
- `dialog` vs `text` vs `content`
- `timepoint_id` vs `timepoint`

**Root Cause:** Multiple data sources (different export formats) or inconsistent narrative generation

### 4. Character Relationships
**Status:** Not reliably generated

Network visualization expects `char.relationships` object mapping entity IDs to relationship types, but this data may not be generated

**Root Cause:** Relationships require LLM analysis of interactions across timepoints

### 5. Importance Scores
**Status:** Not generated

Timeline expects `tp.importance` (0-1 float) to indicate timepoint significance, but not present in synthetic narratives

**Root Cause:** Importance scoring requires post-simulation analysis

### 6. Emotional State in Dialogs
**Status:** Unclear if generated

Dialog viewer displays `emotional_state` field but unclear if this is:
- Generated for each dialog turn
- Computed from LLM analysis
- Just a placeholder

---

## Recommendations by Priority

### Priority 1: Critical Path Fixes
1. **Ensure Narrative JSON Export**: Modify simulation to always export narrative JSON at completion
   - Location: `datasets/{template}/narrative_{timestamp}.json`
   - Include: timepoints array, characters array, executive summary, dialogs array
   - Fields must be consistent across all exports

2. **Standardize Dialog Fields**: All dialogs must have:
   - `speaker` (string)
   - `dialog` (string, main content)
   - `timepoint_id` (string, linking to timepoints)
   - `location` (string, optional but consistent when present)
   - `context` (string, optional)

3. **Generate Character Relationships**: Post-process narrative to add relationship map:
   - Analyze dialog interactions between characters
   - Create `characters[].relationships` object

### Priority 2: Enhancement Features
1. **Add Importance Scoring**: Generate importance scores for each timepoint based on:
   - Number of entities involved
   - Dialog turn count
   - Mechanism usage count

2. **Implement Screenplay Generation**: Create LLM-driven screenplay generation:
   - Convert narrative timepoints to Fountain scenes
   - Generate stage directions and dialog formatting
   - Export to valid Fountain format

3. **Add Emotional State Analysis**: Generate emotional_state for dialogs:
   - Use LLM to classify emotional tone
   - Options: positive, negative, neutral, conflicted, etc.

### Priority 3: UI Improvements
1. **Visual Indicators for Missing Data**: 
   - Toast/badge showing "Narrative export pending"
   - Disable visualization tabs if data missing

2. **Better Search in Dialogs**:
   - Highlight matching text
   - Show context around matches

3. **PORTAL Mode Visualization**:
   - Add special handling for backward causality visualization
   - Show branching/pivot points

4. **Run Comparison Tool**:
   - Side-by-side timeline comparison
   - Mechanism usage comparison

---

## Data Flow Diagram

```
Simulation Run (Database)
├── runs table (metadata)
├── mechanism_usage table
├── resolution_assignments table
├── validations table
└── [MISSING] FULL NARRATIVE DATA

Generated Files (Optional, Often Missing)
├── narrative_{timestamp}.json
│   ├── executive_summary
│   ├── timepoints[]
│   ├── characters[]
│   ├── dialogs[]
│   └── relationships
└── screenplay_{timestamp}.fountain

API Endpoints
├── /api/runs → From database (COMPLETE)
├── /api/run/{id} → From database (COMPLETE)
├── /api/narrative/{id} → File-based (INCOMPLETE)
├── /api/screenplay/{id} → File-based (INCOMPLETE)
└── /api/dialogs/{id} → From narrative JSON (INCOMPLETE)

Dashboard Pages
├── runs.qmd → Uses /api/runs (WORKS)
├── index.qmd → Uses /api/run + /api/narrative (PARTIAL)
├── screenplay.qmd → Uses /api/screenplay (RARELY WORKS)
├── dialogs.qmd → Uses /api/dialogs (DEPENDS ON NARRATIVE)
└── analytics.qmd → Uses /api/meta-analytics (WORKS)
```

---

## Field Reference: What Each Dashboard Page Needs

### index.qmd Required Fields
```json
{
  "run_id": "string",
  "template_id": "string", 
  "status": "string",
  "causal_mode": "string",
  "cost_usd": float,
  "entities_created": int,
  "timepoints_created": int,
  "llm_calls": int,
  "tokens_used": int,
  "mechanism_usage": [
    {"mechanism": "string", "function_name": "string", "context": "string", "timestamp": "string"}
  ],
  "resolution_assignments": [
    {"entity_id": "string", "resolution": "string", "timepoint_id": "string", "timestamp": "string"}
  ],
  "validations": [
    {"validator_name": "string", "passed": bool, "message": "string", "timestamp": "string"}
  ],
  "narrative": {
    "executive_summary": "string",
    "timepoints": [
      {
        "timepoint_id": "string",
        "timestamp": "string",
        "event_description": "string",
        "entities_present": ["string"],
        "dialog_turn_count": int,
        "importance": float
      }
    ],
    "characters": [
      {
        "entity_id": "string",
        "relationships": {"entity_id": "relationship_type"}
      }
    ],
    "dialogs": [
      {
        "timepoint_id": "string",
        "speaker": "string",
        "dialog": "string"
      }
    ]
  }
}
```

### screenplay.qmd Required Fields
```text
Fountain format screenplay with proper syntax:
- Scene headings (INT./EXT. LOCATION - TIME)
- Action descriptions
- Character names (centered)
- Dialogue (indented)
- Parentheticals (indented, italicized)
- Transitions (right-aligned)
```

### dialogs.qmd Required Fields
```json
{
  "dialogs": [
    {
      "speaker": "string",
      "dialog": "string",
      "timepoint_id": "string",
      "location": "string (optional)",
      "timestamp": "string (optional)",
      "context": "string (optional)",
      "emotional_state": "string (optional)"
    }
  ]
}
```

---

## Testing Checklist

- [ ] Load a run and verify all metadata displays
- [ ] Check if narrative JSON exists and displays (if not, verify error message)
- [ ] Verify character network displays (requires narrative with relationships)
- [ ] Check if screenplay exists (if not, verify error message)
- [ ] Load dialogs and verify all fields render correctly
- [ ] Test filters on all pages
- [ ] Verify analytics page loads without narrative dependency
- [ ] Test CSV export from dialogs page
- [ ] Test Fountain download from screenplay page

