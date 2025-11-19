"""
Timepoint Dashboard Data Utilities

Loads simulation run data from SQLite database and narrative JSON files.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class TimepointDataLoader:
    """Load and parse Timepoint simulation data."""

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize data loader.

        Args:
            base_path: Root directory of timepoint-daedalus project.
                      Defaults to parent of this file.
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent

        self.base_path = Path(base_path)
        self.db_path = self.base_path / "metadata" / "runs.db"
        self.datasets_path = self.base_path / "datasets"

    def get_mechanism_usage(self, run_id: str, conn: sqlite3.Connection) -> Dict[str, int]:
        """
        Get mechanism usage counts for a specific run.

        Args:
            run_id: Run ID
            conn: Open SQLite connection

        Returns:
            Dict mapping mechanism names to usage counts
        """
        cursor = conn.cursor()
        query = """
        SELECT mechanism, COUNT(*) as count
        FROM mechanism_usage
        WHERE run_id = ?
        GROUP BY mechanism
        """
        cursor.execute(query, (run_id,))
        rows = cursor.fetchall()

        return {row[0]: row[1] for row in rows}

    def get_recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Query SQLite for recent simulation runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of run metadata dicts with keys:
            - run_id, template_id, started_at, completed_at
            - entities_created, timepoints_created, cost_usd
            - status, causal_mode, mechanisms_used
        """
        if not self.db_path.exists():
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
        SELECT
            run_id,
            template_id,
            started_at,
            completed_at,
            entities_created,
            timepoints_created,
            cost_usd,
            status,
            causal_mode,
            fidelity_distribution
        FROM runs
        ORDER BY started_at DESC
        LIMIT ?
        """

        cursor.execute(query, (limit,))
        rows = cursor.fetchall()

        runs = []
        for row in rows:
            run_dict = dict(row)

            # Get mechanism usage from separate table
            run_dict['mechanisms_used'] = self.get_mechanism_usage(run_dict['run_id'], conn)

            # Parse JSON fields
            if run_dict.get('fidelity_distribution'):
                try:
                    run_dict['fidelity_distribution'] = json.loads(run_dict['fidelity_distribution'])
                except json.JSONDecodeError:
                    run_dict['fidelity_distribution'] = {}

            runs.append(run_dict)

        conn.close()
        return runs

    def load_narrative(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Load narrative JSON for a specific run.

        Args:
            run_id: Run ID (e.g., "run_20251103_105234_3a83d054")

        Returns:
            Narrative data dict with keys:
            - run_id, template_id, executive_summary
            - characters (list of entities)
            - timepoints (list of temporal events)
            - mechanisms (dict of mechanism usage)

            Returns None if narrative not found.
        """
        # Find narrative file matching run_id
        # Format: datasets/{template}/narrative_{timestamp}.json
        # Extract timestamp from run_id (e.g., "20251103_105234")

        try:
            # run_id format: run_YYYYMMDD_HHMMSS_hash
            parts = run_id.split('_')
            if len(parts) < 3:
                return None

            timestamp = f"{parts[1]}_{parts[2]}"  # YYYYMMDD_HHMMSS

            # Search all dataset directories for matching narrative
            for template_dir in self.datasets_path.iterdir():
                if not template_dir.is_dir():
                    continue

                # Look for narrative_{timestamp}.json
                narrative_path = template_dir / f"narrative_{timestamp}.json"
                if narrative_path.exists():
                    with open(narrative_path, 'r') as f:
                        return json.load(f)

            return None

        except Exception as e:
            print(f"Error loading narrative for {run_id}: {e}")
            return None

    def load_screenplay(self, run_id: str) -> Optional[str]:
        """
        Load Fountain screenplay for a specific run.

        Args:
            run_id: Run ID (e.g., "run_20251103_105234_3a83d054")

        Returns:
            Fountain screenplay content as string, or None if not found.
        """
        try:
            # Extract timestamp from run_id
            parts = run_id.split('_')
            if len(parts) < 3:
                return None

            timestamp = f"{parts[1]}_{parts[2]}"

            # Search all dataset directories for matching screenplay
            for template_dir in self.datasets_path.iterdir():
                if not template_dir.is_dir():
                    continue

                screenplay_path = template_dir / f"screenplay_{timestamp}.fountain"
                if screenplay_path.exists():
                    with open(screenplay_path, 'r') as f:
                        return f.read()

            return None

        except Exception as e:
            print(f"Error loading screenplay for {run_id}: {e}")
            return None

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary for a single run.

        Combines SQLite metadata with narrative JSON data.

        Args:
            run_id: Run ID

        Returns:
            Dict with all available data for the run
        """
        # Get database record
        runs = self.get_recent_runs(limit=100)
        db_record = next((r for r in runs if r['run_id'] == run_id), None)

        if not db_record:
            return {'run_id': run_id, 'error': 'Run not found in database'}

        # Load narrative
        narrative = self.load_narrative(run_id)

        # Load screenplay
        screenplay = self.load_screenplay(run_id)

        return {
            **db_record,
            'narrative': narrative,
            'screenplay': screenplay,
            'has_narrative': narrative is not None,
            'has_screenplay': screenplay is not None
        }

    def export_for_observable(self, run_id: str) -> str:
        """
        Export run data as JSON for Observable JS consumption.

        Args:
            run_id: Run ID

        Returns:
            JSON string with all run data
        """
        data = self.get_run_summary(run_id)
        return json.dumps(data, indent=2, default=str)


# Convenience functions for use in Quarto notebooks
def get_recent_runs(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent runs (convenience wrapper)."""
    loader = TimepointDataLoader()
    return loader.get_recent_runs(limit)


def load_run(run_id: str) -> Dict[str, Any]:
    """Load full run data (convenience wrapper)."""
    loader = TimepointDataLoader()
    return loader.get_run_summary(run_id)


def get_most_recent_run() -> Optional[str]:
    """Get the most recent run ID."""
    runs = get_recent_runs(limit=1)
    if runs:
        return runs[0]['run_id']
    return None
