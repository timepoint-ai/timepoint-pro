"""
FastAPI server for Timepoint Dashboard.

Provides REST API for querying runs, analytics, narratives, and screenplays.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from typing import List, Optional
import math

from models import (
    PaginatedRunsResponse,
    RunListItem,
    RunDetails,
    MetaAnalytics,
    TemplatesResponse,
    MechanismsResponse
)
from db import TimepointDB

# Initialize FastAPI app
app = FastAPI(
    title="Timepoint Dashboard API",
    description="REST API for querying Timepoint simulation runs",
    version="1.0.0"
)

# Enable CORS for Quarto frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = TimepointDB()


@app.get("/", tags=["Health"])
async def root():
    """API health check."""
    return {"status": "ok", "message": "Timepoint Dashboard API"}


@app.get("/api/runs", response_model=PaginatedRunsResponse, tags=["Runs"])
async def list_runs(
    template: Optional[str] = Query(None, description="Filter by template ID"),
    status: Optional[str] = Query(None, description="Filter by status (completed, running, failed)"),
    date_from: Optional[str] = Query(None, description="Filter by start date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter by end date (ISO format)"),
    min_cost: Optional[float] = Query(None, description="Minimum cost filter"),
    max_cost: Optional[float] = Query(None, description="Maximum cost filter"),
    causal_mode: Optional[str] = Query(None, description="Filter by causal mode"),
    mechanisms: Optional[str] = Query(None, description="Comma-separated list of mechanisms (e.g., 'M1,M5,M17')"),
    min_entities: Optional[int] = Query(None, description="Minimum entities created"),
    min_timepoints: Optional[int] = Query(None, description="Minimum timepoints created"),
    sort_by: str = Query("started_at", description="Sort field"),
    order: str = Query("DESC", description="Sort order (ASC or DESC)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page")
):
    """
    List all runs with filtering, sorting, and pagination.

    Supports comprehensive filtering by template, status, date range,
    cost range, causal mode, mechanisms, and entity/timepoint counts.
    """
    # Parse mechanisms
    mechanism_list = mechanisms.split(',') if mechanisms else None

    # Query database
    results, total = db.query_runs(
        template=template,
        status=status,
        date_from=date_from,
        date_to=date_to,
        min_cost=min_cost,
        max_cost=max_cost,
        causal_mode=causal_mode,
        mechanisms=mechanism_list,
        min_entities=min_entities,
        min_timepoints=min_timepoints,
        sort_by=sort_by,
        order=order,
        page=page,
        limit=limit
    )

    # Calculate total pages
    pages = math.ceil(total / limit) if total > 0 else 0

    # Convert to Pydantic models
    runs = [RunListItem(**run) for run in results]

    return PaginatedRunsResponse(
        runs=runs,
        total=total,
        page=page,
        limit=limit,
        pages=pages
    )


@app.get("/api/run/{run_id}", response_model=RunDetails, tags=["Runs"])
async def get_run(run_id: str):
    """
    Get full details for a specific run.

    Includes mechanism usage, resolution assignments, and validations.
    """
    run = db.get_run_details(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return RunDetails(**run)


@app.get("/api/narrative/{run_id}", tags=["Content"])
async def get_narrative(run_id: str):
    """
    Get narrative JSON for a specific run.

    Includes characters, timepoints, dialogs, and executive summary.
    """
    narrative = db.get_narrative(run_id)

    if not narrative:
        raise HTTPException(status_code=404, detail=f"Narrative for run {run_id} not found")

    return narrative


@app.get("/api/screenplay/{run_id}", response_class=PlainTextResponse, tags=["Content"])
async def get_screenplay(run_id: str):
    """
    Get Fountain screenplay for a specific run.

    Returns raw Fountain format text.
    """
    screenplay = db.get_screenplay(run_id)

    if not screenplay:
        raise HTTPException(status_code=404, detail=f"Screenplay for run {run_id} not found")

    return screenplay


@app.get("/api/templates", response_model=TemplatesResponse, tags=["Metadata"])
async def list_templates():
    """
    Get list of all unique template IDs.

    Useful for populating filter dropdowns.
    """
    templates = db.get_templates()
    return TemplatesResponse(templates=templates)


@app.get("/api/mechanisms", response_model=MechanismsResponse, tags=["Metadata"])
async def list_mechanisms():
    """
    Get all mechanisms with total usage counts.

    Returns dict of {mechanism_name: usage_count}.
    """
    mechanisms = db.get_mechanisms()
    return MechanismsResponse(mechanisms=mechanisms)


@app.get("/api/meta-analytics", response_model=MetaAnalytics, tags=["Analytics"])
async def get_meta_analytics():
    """
    Get aggregate analytics across all runs.

    Includes:
    - Total runs, cost, entities, timepoints
    - Success rate
    - Top templates
    - Cost over time
    - Mechanism co-occurrence
    - Causal mode distribution
    """
    analytics = db.get_meta_analytics()
    return MetaAnalytics(**analytics)


@app.get("/api/dialogs/{run_id}", tags=["Content"])
async def get_dialogs(run_id: str):
    """
    Get all dialogs for a specific run.

    Extracts dialogs from narrative JSON if available.
    """
    narrative = db.get_narrative(run_id)

    if not narrative:
        raise HTTPException(status_code=404, detail=f"Narrative for run {run_id} not found")

    dialogs = narrative.get('dialogs', [])

    return {
        "run_id": run_id,
        "dialog_count": len(dialogs),
        "dialogs": dialogs
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Timepoint Dashboard API on http://localhost:8000")
    print("📖 API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
