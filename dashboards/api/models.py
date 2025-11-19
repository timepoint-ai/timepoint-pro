"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class RunListItem(BaseModel):
    """Single run in list response."""
    run_id: str
    template_id: str
    started_at: str
    completed_at: Optional[str] = None
    causal_mode: str
    entities_created: int = 0
    timepoints_created: int = 0
    cost_usd: float = 0.0
    status: str
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    mechanisms_used: Dict[str, int] = Field(default_factory=dict)


class PaginatedRunsResponse(BaseModel):
    """Paginated response for run listing."""
    runs: List[RunListItem]
    total: int
    page: int
    limit: int
    pages: int


class RunDetails(BaseModel):
    """Full details for a single run."""
    run_id: str
    template_id: str
    started_at: str
    completed_at: Optional[str] = None
    causal_mode: str
    max_entities: Optional[int] = None
    max_timepoints: Optional[int] = None
    entities_created: int = 0
    timepoints_created: int = 0
    training_examples: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    tokens_used: int = 0
    duration_seconds: Optional[float] = None
    status: str
    error_message: Optional[str] = None
    mechanism_usage: List[Dict[str, Any]] = Field(default_factory=list)
    resolution_assignments: List[Dict[str, Any]] = Field(default_factory=list)
    validations: List[Dict[str, Any]] = Field(default_factory=list)
    schema_version: Optional[str] = None
    fidelity_distribution: Optional[Dict[str, Any]] = None
    fidelity_strategy_json: Optional[Dict[str, Any]] = None
    token_budget_compliance: Optional[float] = None


class MetaAnalytics(BaseModel):
    """Aggregate analytics across all runs."""
    total_runs: int
    total_cost: float
    avg_cost: float
    total_entities: int
    total_timepoints: int
    avg_duration: Optional[float] = None
    completed_runs: int
    failed_runs: int
    success_rate: float
    top_templates: List[Dict[str, Any]]
    cost_over_time: List[Dict[str, Any]]
    mechanism_co_occurrence: List[Dict[str, Any]]
    causal_mode_distribution: List[Dict[str, Any]]


class TemplatesResponse(BaseModel):
    """List of all templates."""
    templates: List[str]


class MechanismsResponse(BaseModel):
    """Mechanism usage counts."""
    mechanisms: Dict[str, int]
