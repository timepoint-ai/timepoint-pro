#!/usr/bin/env python3.10
"""Pre-generate JSON data file for dashboard."""

import json
from utils import get_recent_runs, load_run

print("Loading Timepoint simulation data...")

# Get recent runs with actual costs (more likely to have complete narratives)
all_runs = get_recent_runs(limit=50)
recent_runs = [r for r in all_runs if r.get('cost_usd', 0) > 0][:20]

print(f"Found {len(recent_runs)} recent runs with costs")

# Get the most recent run with narrative AND timeline data
run_with_narrative = None
for run in recent_runs:
    full_run = load_run(run['run_id'])
    if full_run.get('has_narrative') and full_run.get('narrative'):
        # Fix: Map 'timeline' to 'timepoints' for dashboard compatibility
        narrative = full_run['narrative']
        if 'timeline' in narrative and 'timepoints' not in narrative:
            narrative['timepoints'] = narrative['timeline']

        # Check if it has actual timeline data
        if len(narrative.get('timepoints', [])) > 0:
            run_with_narrative = full_run
            print(f"✓ Found run with {len(narrative['timepoints'])} timepoints")
            break

if not run_with_narrative and recent_runs:
    # Fallback to any recent run
    run_with_narrative = load_run(recent_runs[0]['run_id'])
    if run_with_narrative.get('narrative'):
        # Apply the timeline -> timepoints fix
        narrative = run_with_narrative['narrative']
        if 'timeline' in narrative and 'timepoints' not in narrative:
            narrative['timepoints'] = narrative['timeline']

# Export to JSON file
output = {
    'recent_runs': recent_runs,
    'selected_run': run_with_narrative
}

with open('dashboard_data.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"✓ Exported data to dashboard_data.json")
if run_with_narrative:
    print(f"✓ Selected run: {run_with_narrative['run_id']}")
    print(f"  - Template: {run_with_narrative['template_id']}")
    print(f"  - Cost: ${run_with_narrative.get('cost_usd', 0):.3f}")
    print(f"  - Has narrative: {run_with_narrative.get('has_narrative')}")
    if run_with_narrative.get('narrative'):
        chars = len(run_with_narrative['narrative'].get('characters', []))
        tps = len(run_with_narrative['narrative'].get('timepoints', []))
        print(f"  - Characters: {chars}")
        print(f"  - Timepoints: {tps}")
    print(f"  - Mechanisms: {len(run_with_narrative.get('mechanisms_used', {}))}")
