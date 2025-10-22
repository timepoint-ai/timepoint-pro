"""
Simulation Configuration Schema with Pydantic Validation

Provides validated configuration for simulation generation with:
- Entity configuration
- Timepoint structure
- Temporal mode settings
- Output specifications
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ResolutionLevel(str, Enum):
    """Entity resolution levels"""
    TENSOR_ONLY = "tensor_only"
    SCENE = "scene"
    GRAPH = "graph"
    DIALOG = "dialog"
    TRAINED = "trained"
    FULL_DETAIL = "full_detail"


class TemporalMode(str, Enum):
    """Temporal causality modes"""
    PEARL = "pearl"
    DIRECTORIAL = "directorial"
    NONLINEAR = "nonlinear"
    BRANCHING = "branching"
    CYCLICAL = "cyclical"


class EntityConfig(BaseModel):
    """Configuration for entity generation"""
    count: int = Field(ge=1, le=1000, description="Number of entities to generate")
    types: List[str] = Field(default=["human"], description="Entity types to include")
    initial_resolution: ResolutionLevel = Field(
        default=ResolutionLevel.TENSOR_ONLY,
        description="Starting resolution level for entities"
    )
    animism_level: int = Field(
        ge=0, le=6, default=0,
        description="Animistic entity inclusion level (0=humans only, 6=all types)"
    )

    @field_validator('types')
    @classmethod
    def validate_entity_types(cls, v):
        valid_types = {"human", "animal", "building", "object", "abstract", "ai", "any", "kami"}
        if not all(t in valid_types for t in v):
            raise ValueError(f"Invalid entity type. Must be one of: {valid_types}")
        return v


class TimepointConfig(BaseModel):
    """Configuration for timepoint structure"""
    count: int = Field(ge=1, le=100, default=1, description="Number of timepoints to generate")
    start_time: Optional[datetime] = Field(
        default=None,
        description="Start time for simulation (defaults to now)"
    )
    resolution: str = Field(
        default="hour",
        description="Temporal resolution: year, month, day, hour, minute"
    )
    before_count: int = Field(
        ge=0, le=50, default=0,
        description="Number of timepoints to generate before critical moment"
    )
    after_count: int = Field(
        ge=0, le=50, default=0,
        description="Number of timepoints to generate after critical moment"
    )

    @field_validator('resolution')
    @classmethod
    def validate_resolution(cls, v):
        valid_resolutions = {"year", "month", "day", "hour", "minute", "second"}
        if v not in valid_resolutions:
            raise ValueError(f"Invalid resolution. Must be one of: {valid_resolutions}")
        return v


class TemporalConfig(BaseModel):
    """Configuration for temporal causality mode"""
    mode: TemporalMode = Field(
        default=TemporalMode.PEARL,
        description="Temporal causality mode"
    )
    # Directorial mode settings
    narrative_arc: Optional[str] = Field(
        default=None,
        description="Narrative arc for directorial mode: rising_action, climax, falling_action"
    )
    dramatic_tension: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="Dramatic tension level for directorial mode"
    )
    # Cyclical mode settings
    cycle_length: Optional[int] = Field(
        default=None,
        description="Cycle length for cyclical mode"
    )
    prophecy_accuracy: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="Prophecy accuracy for cyclical mode"
    )
    # Branching mode settings
    enable_counterfactuals: bool = Field(
        default=False,
        description="Enable counterfactual branch generation"
    )


class OutputConfig(BaseModel):
    """Configuration for output generation"""
    formats: List[str] = Field(
        default=["json"],
        description="Output formats: json, jsonl, csv, markdown, sqlite"
    )
    include_dialogs: bool = Field(
        default=True,
        description="Generate dialog synthesis"
    )
    include_relationships: bool = Field(
        default=True,
        description="Track relationship evolution"
    )
    include_knowledge_flow: bool = Field(
        default=True,
        description="Track knowledge propagation"
    )
    export_ml_dataset: bool = Field(
        default=False,
        description="Export in ML training format (JSONL)"
    )

    @field_validator('formats')
    @classmethod
    def validate_formats(cls, v):
        valid_formats = {"json", "jsonl", "csv", "markdown", "sqlite", "html"}
        if not all(f in valid_formats for f in v):
            raise ValueError(f"Invalid format. Must be one of: {valid_formats}")
        return v


class VariationConfig(BaseModel):
    """Configuration for horizontal variation generation"""
    enabled: bool = Field(
        default=False,
        description="Enable variation generation"
    )
    count: int = Field(
        ge=1, le=1000, default=1,
        description="Number of variations to generate"
    )
    strategies: List[str] = Field(
        default=["vary_personalities"],
        description="Variation strategies to apply"
    )
    deduplication_threshold: float = Field(
        ge=0.0, le=1.0, default=0.9,
        description="Similarity threshold for deduplication (1.0 = identical)"
    )

    @field_validator('strategies')
    @classmethod
    def validate_strategies(cls, v):
        valid_strategies = {
            "vary_personalities",
            "vary_starting_conditions",
            "vary_outcomes",
            "vary_relationships",
            "vary_knowledge"
        }
        if not all(s in valid_strategies for s in v):
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        return v


class SimulationConfig(BaseModel):
    """
    Complete simulation configuration with validation.

    This is the root configuration object that defines all aspects
    of a simulation generation job.

    Example:
        config = SimulationConfig(
            scenario_description="Simulate the Constitutional Convention of 1787",
            world_id="constitutional_convention",
            entities=EntityConfig(count=10, types=["human"]),
            timepoints=TimepointConfig(count=5, resolution="day"),
            temporal=TemporalConfig(mode=TemporalMode.PEARL),
            outputs=OutputConfig(formats=["json", "markdown"])
        )
    """
    # Core identification
    scenario_description: str = Field(
        description="Natural language description of scenario to simulate"
    )
    world_id: str = Field(
        description="Unique identifier for this simulation world"
    )

    # Component configurations
    entities: EntityConfig = Field(
        default_factory=lambda: EntityConfig(),
        description="Entity generation configuration"
    )
    timepoints: TimepointConfig = Field(
        default_factory=lambda: TimepointConfig(),
        description="Timepoint structure configuration"
    )
    temporal: TemporalConfig = Field(
        default_factory=lambda: TemporalConfig(),
        description="Temporal causality configuration"
    )
    outputs: OutputConfig = Field(
        default_factory=lambda: OutputConfig(),
        description="Output generation configuration"
    )
    variations: VariationConfig = Field(
        default_factory=lambda: VariationConfig(),
        description="Variation generation configuration"
    )

    # Optional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this simulation"
    )

    # Execution settings
    max_cost_usd: Optional[float] = Field(
        default=None,
        description="Maximum cost in USD (None = unlimited)"
    )
    enable_checkpoints: bool = Field(
        default=True,
        description="Enable checkpoint/resume functionality"
    )
    checkpoint_interval: int = Field(
        ge=1, default=10,
        description="Save checkpoint every N entities/timepoints"
    )

    @model_validator(mode='after')
    def validate_config_coherence(self):
        """Validate that configuration is internally coherent"""
        # If variations are enabled, we need at least one strategy
        if self.variations.enabled and not self.variations.strategies:
            raise ValueError("Variation generation requires at least one strategy")

        # Temporal expansion requires at least one timepoint
        if (self.timepoints.before_count + self.timepoints.after_count) > 0:
            if self.timepoints.count < 1:
                raise ValueError("Temporal expansion requires at least one base timepoint")

        # Directorial mode requires narrative arc
        if self.temporal.mode == TemporalMode.DIRECTORIAL and not self.temporal.narrative_arc:
            self.temporal.narrative_arc = "rising_action"  # Set default

        # Cyclical mode requires cycle length
        if self.temporal.mode == TemporalMode.CYCLICAL and not self.temporal.cycle_length:
            self.temporal.cycle_length = self.timepoints.count  # Set default

        return self

    @classmethod
    def example_board_meeting(cls) -> "SimulationConfig":
        """Example configuration for a board meeting scenario"""
        return cls(
            scenario_description="Simulate a tech startup board meeting where CEO proposes an acquisition",
            world_id="board_meeting_example",
            entities=EntityConfig(count=5, types=["human"]),
            timepoints=TimepointConfig(count=3, resolution="hour"),
            temporal=TemporalConfig(mode=TemporalMode.PEARL),
            outputs=OutputConfig(
                formats=["json", "markdown"],
                include_dialogs=True,
                include_relationships=True
            )
        )

    @classmethod
    def example_jefferson_dinner(cls) -> "SimulationConfig":
        """Example configuration for the Jefferson Dinner scenario"""
        return cls(
            scenario_description="Simulate the 1790 Compromise Dinner between Jefferson, Hamilton, and Madison",
            world_id="jefferson_dinner",
            entities=EntityConfig(count=3, types=["human"]),
            timepoints=TimepointConfig(
                count=1,
                before_count=2,
                after_count=2,
                resolution="hour"
            ),
            temporal=TemporalConfig(mode=TemporalMode.PEARL),
            outputs=OutputConfig(
                formats=["json", "markdown"],
                include_dialogs=True,
                include_relationships=True,
                include_knowledge_flow=True
            )
        )

    @classmethod
    def example_variations(cls) -> "SimulationConfig":
        """Example configuration for generating variations"""
        return cls(
            scenario_description="Generate variations of a negotiation scenario",
            world_id="negotiation_variations",
            entities=EntityConfig(count=4, types=["human"]),
            timepoints=TimepointConfig(count=2, resolution="hour"),
            temporal=TemporalConfig(mode=TemporalMode.PEARL),
            outputs=OutputConfig(
                formats=["jsonl"],
                export_ml_dataset=True
            ),
            variations=VariationConfig(
                enabled=True,
                count=100,
                strategies=["vary_personalities", "vary_outcomes"]
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Create from dictionary"""
        return cls(**data)

    def estimate_cost(self) -> Dict[str, float]:
        """
        Estimate generation cost based on configuration.

        Returns:
            Dictionary with cost estimates:
                - min_usd: Minimum expected cost
                - max_usd: Maximum expected cost
                - tokens_estimated: Estimated token count
        """
        # Rough cost estimation based on entity count and resolution
        entity_count = self.entities.count
        timepoint_count = self.timepoints.count + self.timepoints.before_count + self.timepoints.after_count

        # Base tokens per entity-timepoint pair
        resolution_tokens = {
            ResolutionLevel.TENSOR_ONLY: 200,
            ResolutionLevel.SCENE: 2000,
            ResolutionLevel.GRAPH: 5000,
            ResolutionLevel.DIALOG: 10000,
            ResolutionLevel.TRAINED: 50000,
            ResolutionLevel.FULL_DETAIL: 50000
        }

        base_tokens = resolution_tokens.get(self.entities.initial_resolution, 10000)
        total_tokens = entity_count * timepoint_count * base_tokens

        # Apply variation multiplier
        if self.variations.enabled:
            total_tokens *= self.variations.count

        # Cost per million tokens (rough estimate)
        cost_per_m_tokens = 10.0  # $10 per million tokens

        min_cost = (total_tokens * 0.5 * cost_per_m_tokens) / 1_000_000
        max_cost = (total_tokens * 1.5 * cost_per_m_tokens) / 1_000_000

        return {
            "min_usd": round(min_cost, 2),
            "max_usd": round(max_cost, 2),
            "tokens_estimated": total_tokens
        }
