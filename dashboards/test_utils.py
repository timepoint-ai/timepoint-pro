#!/usr/bin/env python3.10
"""Test script for dashboard data utilities."""

from utils import get_recent_runs, get_most_recent_run, load_run

print('Testing Timepoint Dashboard Data Loader...\n')

# Test 1: Get recent runs
print('1. Loading recent runs:')
runs = get_recent_runs(limit=5)
print(f'   Found {len(runs)} runs')
for run in runs[:3]:
    cost = run.get('cost_usd', 0)
    print(f'   - {run["run_id"]}: {run["template_id"]} (${cost:.3f})')

# Test 2: Get most recent run
print('\n2. Most recent run:')
latest = get_most_recent_run()
print(f'   {latest}')

# Test 3: Load full run data
if latest:
    print(f'\n3. Loading full data for {latest}:')
    run_data = load_run(latest)
    print(f'   - Has narrative: {run_data.get("has_narrative")}')
    print(f'   - Has screenplay: {run_data.get("has_screenplay")}')
    if run_data.get('narrative'):
        chars = run_data["narrative"].get("characters", [])
        tps = run_data["narrative"].get("timepoints", [])
        print(f'   - Characters: {len(chars)}')
        print(f'   - Timepoints: {len(tps)}')

print('\n✓ All tests passed!')
