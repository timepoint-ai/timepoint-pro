"""
Progress Tracking for Data Generation

Provides real-time progress reporting, metrics tracking, and summary reports
for long-running data generation jobs.
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import time


@dataclass
class GenerationMetrics:
    """Metrics for generation progress"""
    entities_generated: int = 0
    entities_failed: int = 0
    timepoints_generated: int = 0
    timepoints_failed: int = 0
    tokens_consumed: int = 0
    llm_calls_successful: int = 0
    llm_calls_failed: int = 0
    llm_retries: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime to ISO format
        if self.start_time:
            data["start_time"] = self.start_time.isoformat()
        if self.end_time:
            data["end_time"] = self.end_time.isoformat()
        return data


class ProgressTracker:
    """
    Track progress of data generation jobs with real-time updates.

    Example:
        tracker = ProgressTracker(
            total_entities=100,
            total_timepoints=5,
            enable_progress_bar=True
        )

        tracker.start()
        for i in range(100):
            tracker.update_entity_generated()
            tracker.update_tokens(500)
        tracker.complete()

        stats = tracker.get_summary()
        print(f"Generated {stats['entities_generated']} entities in {stats['duration_seconds']}s")
    """

    def __init__(
        self,
        total_entities: int = 0,
        total_timepoints: int = 0,
        enable_progress_bar: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Args:
            total_entities: Expected total entities to generate
            total_timepoints: Expected total timepoints to generate
            enable_progress_bar: Whether to display progress bar (requires tqdm)
            progress_callback: Optional callback for progress updates
        """
        self.total_entities = total_entities
        self.total_timepoints = total_timepoints
        self.enable_progress_bar = enable_progress_bar
        self.progress_callback = progress_callback

        self.metrics = GenerationMetrics()
        self._progress_bar = None
        self._last_update_time = None

    def start(self):
        """Start tracking"""
        self.metrics.start_time = datetime.utcnow()
        self._last_update_time = time.time()

        if self.enable_progress_bar:
            try:
                from tqdm import tqdm
                total = self.total_entities + self.total_timepoints
                self._progress_bar = tqdm(total=total, desc="Generating")
            except ImportError:
                # tqdm not installed, disable progress bar
                self.enable_progress_bar = False

        self._notify_update()

    def update_entity_generated(self, count: int = 1):
        """Record successful entity generation"""
        self.metrics.entities_generated += count
        if self._progress_bar:
            self._progress_bar.update(count)
        self._notify_update()

    def update_entity_failed(self, count: int = 1):
        """Record failed entity generation"""
        self.metrics.entities_failed += count
        self._notify_update()

    def update_timepoint_generated(self, count: int = 1):
        """Record successful timepoint generation"""
        self.metrics.timepoints_generated += count
        if self._progress_bar:
            self._progress_bar.update(count)
        self._notify_update()

    def update_timepoint_failed(self, count: int = 1):
        """Record failed timepoint generation"""
        self.metrics.timepoints_failed += count
        self._notify_update()

    def update_tokens(self, count: int):
        """Record token consumption"""
        self.metrics.tokens_consumed += count
        self._notify_update()

    def update_llm_call_success(self):
        """Record successful LLM call"""
        self.metrics.llm_calls_successful += 1
        self._notify_update()

    def update_llm_call_failure(self):
        """Record failed LLM call"""
        self.metrics.llm_calls_failed += 1
        self._notify_update()

    def update_llm_retry(self):
        """Record LLM retry attempt"""
        self.metrics.llm_retries += 1
        self._notify_update()

    def complete(self):
        """Mark generation as complete"""
        self.metrics.end_time = datetime.utcnow()
        if self._progress_bar:
            self._progress_bar.close()
        self._notify_update()

    def _notify_update(self):
        """Notify progress callback if set"""
        if self.progress_callback:
            self.progress_callback(self.get_current_state())

    def get_current_state(self) -> Dict[str, Any]:
        """Get current progress state"""
        state = self.metrics.to_dict()

        # Calculate progress percentages
        if self.total_entities > 0:
            state["entity_progress_percent"] = (
                self.metrics.entities_generated / self.total_entities * 100
            )
        else:
            state["entity_progress_percent"] = 0.0

        if self.total_timepoints > 0:
            state["timepoint_progress_percent"] = (
                self.metrics.timepoints_generated / self.total_timepoints * 100
            )
        else:
            state["timepoint_progress_percent"] = 0.0

        # Calculate overall progress
        total_items = self.total_entities + self.total_timepoints
        completed_items = self.metrics.entities_generated + self.metrics.timepoints_generated
        if total_items > 0:
            state["overall_progress_percent"] = completed_items / total_items * 100
        else:
            state["overall_progress_percent"] = 0.0

        # Calculate ETA
        if self.metrics.start_time and completed_items > 0:
            elapsed = (datetime.utcnow() - self.metrics.start_time).total_seconds()
            rate = completed_items / elapsed  # items per second
            remaining_items = total_items - completed_items
            if rate > 0:
                eta_seconds = remaining_items / rate
                state["eta_seconds"] = eta_seconds
            else:
                state["eta_seconds"] = None
        else:
            state["eta_seconds"] = None

        return state

    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete summary statistics.

        Returns:
            Dictionary with generation summary:
                - All metrics from GenerationMetrics
                - Success rates
                - Token cost estimates
                - Duration
                - ETA (if in progress)
        """
        summary = self.get_current_state()

        # Calculate success rates
        total_entity_attempts = self.metrics.entities_generated + self.metrics.entities_failed
        if total_entity_attempts > 0:
            summary["entity_success_rate"] = (
                self.metrics.entities_generated / total_entity_attempts
            )
        else:
            summary["entity_success_rate"] = 1.0

        total_timepoint_attempts = (
            self.metrics.timepoints_generated + self.metrics.timepoints_failed
        )
        if total_timepoint_attempts > 0:
            summary["timepoint_success_rate"] = (
                self.metrics.timepoints_generated / total_timepoint_attempts
            )
        else:
            summary["timepoint_success_rate"] = 1.0

        total_llm_calls = self.metrics.llm_calls_successful + self.metrics.llm_calls_failed
        if total_llm_calls > 0:
            summary["llm_success_rate"] = (
                self.metrics.llm_calls_successful / total_llm_calls
            )
        else:
            summary["llm_success_rate"] = 1.0

        # Calculate duration
        if self.metrics.start_time:
            if self.metrics.end_time:
                duration = self.metrics.end_time - self.metrics.start_time
            else:
                duration = datetime.utcnow() - self.metrics.start_time
            summary["duration_seconds"] = duration.total_seconds()
        else:
            summary["duration_seconds"] = 0.0

        # Estimate cost (rough approximation: $0.002 per 1K tokens)
        cost_per_1k_tokens = 0.002
        summary["estimated_cost_usd"] = (
            self.metrics.tokens_consumed / 1000 * cost_per_1k_tokens
        )

        # Token rate
        if summary["duration_seconds"] > 0:
            summary["tokens_per_second"] = (
                self.metrics.tokens_consumed / summary["duration_seconds"]
            )
        else:
            summary["tokens_per_second"] = 0.0

        return summary

    def export_to_json(self, output_path: str):
        """
        Export progress log to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        summary = self.get_summary()
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def reset(self):
        """Reset all metrics"""
        self.metrics = GenerationMetrics()
        self._last_update_time = None
        if self._progress_bar:
            self._progress_bar.close()
            self._progress_bar = None
