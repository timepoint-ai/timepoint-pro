"""
Temporal Expansion Strategies for Vertical Data Generation

Provides strategies for expanding temporal depth:
- expand_before: Generate lead-up timepoints
- expand_after: Generate consequence timepoints
- expand_around: Both directions
- Narrative arc shaping (rising action → climax → falling action)
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from copy import deepcopy


class TemporalExpansionStrategy(ABC):
    """Abstract base class for temporal expansion strategies"""

    @abstractmethod
    def expand(
        self,
        base_config: Dict[str, Any],
        direction: str,
        count: int
    ) -> Dict[str, Any]:
        """
        Expand temporal structure in specified direction.

        Args:
            base_config: Base configuration dict
            direction: "before", "after", or "around"
            count: Number of timepoints to add

        Returns:
            Modified configuration with expanded temporal structure
        """
        pass


class NarrativeArcExpansion(TemporalExpansionStrategy):
    """
    Expand timepoints following narrative arc structure.

    Creates rising action → climax → falling action pattern.
    """

    def __init__(self):
        self.arc_patterns = {
            "rising_action": {
                "tension_progression": [0.3, 0.5, 0.7, 0.85, 0.95],
                "stakes_progression": [0.4, 0.6, 0.75, 0.9, 1.0],
                "resolution_levels": ["scene", "graph", "dialog", "dialog", "full_detail"]
            },
            "climax": {
                "tension": 1.0,
                "stakes": 1.0,
                "resolution": "full_detail"
            },
            "falling_action": {
                "tension_progression": [0.8, 0.6, 0.4, 0.2, 0.1],
                "stakes_progression": [0.9, 0.7, 0.5, 0.3, 0.2],
                "resolution_levels": ["full_detail", "dialog", "graph", "scene", "scene"]
            }
        }

    def expand(
        self,
        base_config: Dict[str, Any],
        direction: str,
        count: int
    ) -> Dict[str, Any]:
        config = deepcopy(base_config)

        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["temporal_expansion"] = {
            "strategy": "narrative_arc",
            "direction": direction,
            "expansion_count": count
        }

        # Update timepoint configuration
        if direction == "before":
            config["timepoints"]["before_count"] = count
            config["metadata"]["narrative_structure"] = "rising_action"
        elif direction == "after":
            config["timepoints"]["after_count"] = count
            config["metadata"]["narrative_structure"] = "falling_action"
        elif direction == "around":
            # For "around", check if counts already set (from expand_temporal_depth)
            # Otherwise split evenly
            timepoints = config.get("timepoints", {})
            if ("before_count" in timepoints and timepoints["before_count"] > 0 and
                "after_count" in timepoints and timepoints["after_count"] > 0):
                # Already set by caller with non-zero values - don't override
                before = timepoints["before_count"]
                after = timepoints["after_count"]
            else:
                before = count // 2
                after = count - before
                config["timepoints"]["before_count"] = before
                config["timepoints"]["after_count"] = after
            config["metadata"]["narrative_structure"] = "complete_arc"

        return config


class ProgressiveTrainingExpansion(TemporalExpansionStrategy):
    """
    Expand with progressive resolution training.

    Entities start at low resolution and elevate as they approach
    the critical moment, then optionally downgrade after.
    """

    def __init__(self, peak_resolution: str = "full_detail"):
        self.peak_resolution = peak_resolution
        self.resolution_progression = [
            "tensor_only",
            "scene",
            "graph",
            "dialog",
            "full_detail"
        ]

    def expand(
        self,
        base_config: Dict[str, Any],
        direction: str,
        count: int
    ) -> Dict[str, Any]:
        config = deepcopy(base_config)

        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["temporal_expansion"] = {
            "strategy": "progressive_training",
            "direction": direction,
            "expansion_count": count,
            "peak_resolution": self.peak_resolution
        }

        # Create resolution schedule
        if direction == "before":
            # Gradually increase resolution approaching critical moment
            config["timepoints"]["before_count"] = count
            schedule = self._create_ascending_schedule(count)
            config["metadata"]["resolution_schedule_before"] = schedule

        elif direction == "after":
            # Gradually decrease resolution after critical moment
            config["timepoints"]["after_count"] = count
            schedule = self._create_descending_schedule(count)
            config["metadata"]["resolution_schedule_after"] = schedule

        elif direction == "around":
            # Check if counts already set with non-zero values
            timepoints = config.get("timepoints", {})
            if ("before_count" in timepoints and timepoints["before_count"] > 0 and
                "after_count" in timepoints and timepoints["after_count"] > 0):
                before = timepoints["before_count"]
                after = timepoints["after_count"]
            else:
                before = count // 2
                after = count - before
                config["timepoints"]["before_count"] = before
                config["timepoints"]["after_count"] = after
            config["metadata"]["resolution_schedule_before"] = self._create_ascending_schedule(before)
            config["metadata"]["resolution_schedule_after"] = self._create_descending_schedule(after)

        return config

    def _create_ascending_schedule(self, count: int) -> List[str]:
        """Create resolution schedule that ascends to peak"""
        if count == 0:
            return []

        peak_idx = self.resolution_progression.index(self.peak_resolution)
        schedule = []

        for i in range(count):
            # Map i (0 to count-1) to resolution index (0 to peak_idx)
            progress = (i + 1) / count
            res_idx = int(progress * peak_idx)
            schedule.append(self.resolution_progression[res_idx])

        return schedule

    def _create_descending_schedule(self, count: int) -> List[str]:
        """Create resolution schedule that descends from peak"""
        if count == 0:
            return []

        peak_idx = self.resolution_progression.index(self.peak_resolution)
        schedule = []

        for i in range(count):
            # Map i (0 to count-1) to resolution index (peak_idx to 0)
            progress = i / count
            res_idx = peak_idx - int(progress * peak_idx)
            schedule.append(self.resolution_progression[res_idx])

        return schedule


class CausalChainExpansion(TemporalExpansionStrategy):
    """
    Expand ensuring strict causal chain integrity.

    Each timepoint must follow causally from previous.
    Exposure events propagate correctly through time.
    """

    def expand(
        self,
        base_config: Dict[str, Any],
        direction: str,
        count: int
    ) -> Dict[str, Any]:
        config = deepcopy(base_config)

        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["temporal_expansion"] = {
            "strategy": "causal_chain",
            "direction": direction,
            "expansion_count": count,
            "enforce_causality": True
        }

        # Mark that causal validation is required
        config["metadata"]["require_causal_validation"] = True

        if direction == "before":
            config["timepoints"]["before_count"] = count
            # Create causal chain structure
            config["metadata"]["causal_chain_before"] = [
                {"timepoint_index": i, "requires_parent": i > 0}
                for i in range(count)
            ]

        elif direction == "after":
            config["timepoints"]["after_count"] = count
            config["metadata"]["causal_chain_after"] = [
                {"timepoint_index": i, "requires_parent": True}
                for i in range(count)
            ]

        elif direction == "around":
            # Check if counts already set with non-zero values
            timepoints = config.get("timepoints", {})
            if ("before_count" in timepoints and timepoints["before_count"] > 0 and
                "after_count" in timepoints and timepoints["after_count"] > 0):
                before = timepoints["before_count"]
                after = timepoints["after_count"]
            else:
                before = count // 2
                after = count - before
                config["timepoints"]["before_count"] = before
                config["timepoints"]["after_count"] = after
            config["metadata"]["causal_chain_before"] = [
                {"timepoint_index": i, "requires_parent": i > 0}
                for i in range(before)
            ]
            config["metadata"]["causal_chain_after"] = [
                {"timepoint_index": i, "requires_parent": True}
                for i in range(after)
            ]

        return config


class TemporalExpander:
    """
    Main class for vertical temporal expansion.

    Example:
        expander = TemporalExpander()

        # Expand with narrative arc
        config = expander.expand_temporal_depth(
            base_config=SimulationConfig.example_board_meeting(),
            strategy="narrative_arc",
            before_count=3,
            after_count=2
        )
    """

    def __init__(self):
        self.strategies = {
            "narrative_arc": NarrativeArcExpansion(),
            "progressive_training": ProgressiveTrainingExpansion(),
            "causal_chain": CausalChainExpansion()
        }

    def expand_temporal_depth(
        self,
        base_config: Any,  # SimulationConfig
        strategy: str = "progressive_training",
        before_count: int = 0,
        after_count: int = 0
    ) -> Any:  # SimulationConfig
        """
        Expand temporal depth of a simulation.

        Args:
            base_config: Base SimulationConfig
            strategy: Expansion strategy name
            before_count: Timepoints to add before critical moment
            after_count: Timepoints to add after critical moment

        Returns:
            SimulationConfig with expanded temporal structure

        Raises:
            ValueError: If strategy not recognized
        """
        if strategy not in self.strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(self.strategies.keys())}"
            )

        # Convert to dict for manipulation
        config_dict = base_config.to_dict()

        expansion_strategy = self.strategies[strategy]

        # Determine direction and apply expansion
        if before_count > 0 and after_count > 0:
            # For "around", pre-set the counts in config_dict before calling expand
            config_dict["timepoints"]["before_count"] = before_count
            config_dict["timepoints"]["after_count"] = after_count
            expanded_dict = expansion_strategy.expand(config_dict, "around", before_count + after_count)
        elif before_count > 0:
            expanded_dict = expansion_strategy.expand(config_dict, "before", before_count)
        elif after_count > 0:
            expanded_dict = expansion_strategy.expand(config_dict, "after", after_count)
        else:
            # No expansion
            return base_config

        # Import here to avoid circular dependency
        from .config_schema import SimulationConfig
        return SimulationConfig.from_dict(expanded_dict)

    def expand_before(
        self,
        base_config: Any,
        count: int,
        strategy: str = "progressive_training"
    ) -> Any:
        """Expand before critical moment"""
        return self.expand_temporal_depth(
            base_config, strategy, before_count=count, after_count=0
        )

    def expand_after(
        self,
        base_config: Any,
        count: int,
        strategy: str = "progressive_training"
    ) -> Any:
        """Expand after critical moment"""
        return self.expand_temporal_depth(
            base_config, strategy, before_count=0, after_count=count
        )

    def expand_around(
        self,
        base_config: Any,
        before_count: int,
        after_count: int,
        strategy: str = "progressive_training"
    ) -> Any:
        """Expand in both directions"""
        return self.expand_temporal_depth(
            base_config, strategy, before_count, after_count
        )

    def get_available_strategies(self) -> List[str]:
        """Get list of available expansion strategies"""
        return list(self.strategies.keys())

    def register_strategy(self, name: str, strategy: TemporalExpansionStrategy):
        """Register custom expansion strategy"""
        if not isinstance(strategy, TemporalExpansionStrategy):
            raise TypeError(
                f"{strategy} must inherit from TemporalExpansionStrategy"
            )
        self.strategies[name] = strategy
