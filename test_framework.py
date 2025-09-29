# ============================================================================
# test_framework.py - Pytest integration with autopilot
# ============================================================================
import pytest
from hypothesis import given, strategies as st
from datetime import datetime
import numpy as np
import networkx as nx
import logging

from storage import GraphStore
from llm import LLMClient
from graph import create_test_graph, create_timeline_graph
from schemas import Entity, ResolutionLevel
from tensors import TensorCompressor
from validation import Validator
from workflows import create_entity_training_workflow, WorkflowState

# Configure logging
logger = logging.getLogger(__name__)


@pytest.fixture
def graph_store(verbose_mode):
    logger.debug("Creating in-memory GraphStore")
    store = GraphStore("sqlite:///:memory:")
    logger.debug("GraphStore created successfully")
    return store

@pytest.fixture
def llm_client_dry_run(verbose_mode):
    logger.debug("Creating LLMClient in dry-run mode")
    client = LLMClient(api_key="test", base_url="http://test", dry_run=True)
    logger.debug("LLMClient created successfully")
    return client

@pytest.fixture
def test_graph(verbose_mode):
    logger.debug("Creating test graph with 10 entities")
    graph = create_test_graph(n_entities=10, seed=42)
    logger.debug(f"Test graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph

@pytest.fixture
def test_timeline(verbose_mode):
    logger.debug("Creating timeline graph from 1750-01-01 to 1750-12-31")
    start = datetime(1750, 1, 1)
    end = datetime(1750, 12, 31)
    timeline = create_timeline_graph(start, end, resolution="day")
    logger.debug(f"Timeline graph created: {timeline.number_of_nodes()} timepoints")
    return timeline

# Unit tests
def test_tensor_compression(verbose_mode):
    logger.info("Starting test_tensor_compression")
    
    # Create 2D tensor with enough samples for PCA
    logger.debug("Creating random tensor: shape (10, 32)")
    tensor = np.random.randn(10, 32)  # 10 samples, 32 features
    logger.debug(f"Tensor created with shape: {tensor.shape}, mean: {tensor.mean():.4f}, std: {tensor.std():.4f}")
    
    logger.debug("Compressing with PCA (n_components=8)")
    compressed_pca = TensorCompressor.compress(tensor, "pca", n_components=8)
    logger.debug(f"PCA compression result length: {len(compressed_pca)}")
    
    logger.debug("Compressing with SVD (n_components=8)")
    compressed_svd = TensorCompressor.compress(tensor, "svd", n_components=8)
    logger.debug(f"SVD compression result length: {len(compressed_svd)}")
    
    assert len(compressed_pca) <= 80  # Flattened result
    assert len(compressed_svd) <= 80
    logger.info("✓ test_tensor_compression passed")

def test_entity_storage(graph_store, verbose_mode):
    logger.info("Starting test_entity_storage")
    
    logger.debug("Creating test entity")
    entity = Entity(
        entity_id="test_entity",
        entity_type="person",
        training_count=0,
        query_count=0
    )
    logger.debug(f"Entity created: {entity.entity_id} ({entity.entity_type})")
    
    logger.debug("Saving entity to GraphStore")
    saved = graph_store.save_entity(entity)
    logger.debug(f"Entity saved with ID: {saved.id}")
    
    logger.debug("Loading entity from GraphStore")
    loaded = graph_store.get_entity("test_entity")
    logger.debug(f"Entity loaded: {loaded.entity_id if loaded else 'None'}")
    
    assert loaded is not None
    assert loaded.entity_id == "test_entity"
    logger.info("✓ test_entity_storage passed")

def test_validation_registry(verbose_mode):
    logger.info("Starting test_validation_registry")
    
    logger.debug("Creating entity with metadata")
    entity = Entity(
        entity_id="test",
        entity_type="person",
        entity_metadata={"energy_budget": 100, "knowledge_state": ["fact_1"]}
    )
    logger.debug(f"Entity metadata: {entity.entity_metadata}")
    
    logger.debug("Setting up validation context")
    context = {"exposure_history": ["fact_1", "fact_2"], "interactions": [30, 40]}
    logger.debug(f"Context: {context}")
    
    logger.debug("Running all validators")
    violations = Validator.validate_all(entity, context)
    logger.debug(f"Validation complete: {len(violations)} violations found")
    
    if violations:
        for v in violations:
            logger.debug(f"  - [{v['severity']}] {v['validator']}: {v['message']}")
    else:
        logger.debug("  No violations detected")
    
    assert isinstance(violations, list)
    logger.info("✓ test_validation_registry passed")

# Property-based tests
@given(st.integers(min_value=5, max_value=100))
def test_graph_creation_property(n_entities):
    # Note: hypothesis controls arguments, so we use logger directly
    logger.info(f"Starting test_graph_creation_property with n_entities={n_entities}")
    
    logger.debug(f"Creating graph with {n_entities} entities")
    graph = create_test_graph(n_entities=n_entities, seed=42)
    logger.debug(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    assert len(graph.nodes()) == n_entities
    # Graph may not always be connected for small sizes, just check it has edges
    assert graph.number_of_edges() > 0 or n_entities == 1
    logger.info(f"✓ test_graph_creation_property passed for n_entities={n_entities}")

# Integration test
def test_full_workflow(graph_store, llm_client_dry_run, test_graph, verbose_mode):
    logger.info("Starting test_full_workflow (integration test)")
    
    # Setup
    logger.debug("Creating entity training workflow")
    workflow = create_entity_training_workflow(llm_client_dry_run, graph_store)
    logger.debug("Workflow created successfully")
    
    logger.debug("Setting up initial workflow state")
    initial_state = WorkflowState(
        graph=test_graph,
        entities=[],
        timepoint="2025-01-01T00:00:00",
        resolution=ResolutionLevel.TENSOR_ONLY,
        violations=[],
        results={}
    )
    logger.debug(f"Initial state: {len(initial_state['entities'])} entities, "
                f"{initial_state['graph'].number_of_nodes()} graph nodes")
    
    # Execute workflow
    logger.debug("Executing workflow...")
    final_state = workflow.invoke(initial_state)
    logger.debug("Workflow execution complete")
    
    logger.debug(f"Final state: {len(final_state.get('violations', []))} violations")
    logger.debug(f"Results keys: {list(final_state.get('results', {}).keys())}")
    
    # Assertions
    assert "results" in final_state
    assert "violations" in final_state
    assert final_state["graph"] is not None
    logger.info("✓ test_full_workflow passed")
