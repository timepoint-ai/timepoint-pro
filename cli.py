# ============================================================================
# cli.py - Hydra CLI for autopilot and steering
# ============================================================================
import hydra
from omegaconf import DictConfig
from pathlib import Path
from datetime import datetime

# Import all required modules
from storage import GraphStore
from llm import LLMClient
from workflows import create_entity_training_workflow, WorkflowState
from graph import create_test_graph
from evaluation import EvaluationMetrics
from schemas import ResolutionLevel


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration"""
    
    # Initialize components
    store = GraphStore(cfg.database.url)
    llm_client = LLMClient(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        dry_run=cfg.llm.dry_run
    )
    
    # Run autopilot mode
    if cfg.mode == "autopilot":
        run_autopilot(cfg, store, llm_client)
    elif cfg.mode == "evaluate":
        run_evaluation(cfg, store, llm_client)
    elif cfg.mode == "train":
        run_training(cfg, store, llm_client)
    else:
        print(f"Unknown mode: {cfg.mode}")

def run_autopilot(cfg: DictConfig, store: GraphStore, llm_client: LLMClient):
    """Autopilot self-testing mode"""
    print(f"Running autopilot in {cfg.autopilot.depth} mode")
    
    # Create test scenarios
    graph_sizes = cfg.autopilot.graph_sizes
    results = []
    
    for size in graph_sizes:
        graph = create_test_graph(n_entities=size, seed=cfg.seed)
        workflow = create_entity_training_workflow(llm_client, store)
        
        state = WorkflowState(
            graph=graph,
            entities=[],
            timepoint=datetime.now().isoformat(),
            resolution=ResolutionLevel.TENSOR_ONLY,
            violations=[],
            results={}
        )
        
        final_state = workflow.invoke(state)
        
        results.append({
            "graph_size": size,
            "violations": len(final_state["violations"]),
            "cost": llm_client.cost
        })
    
    # Report results
    print("\nAutopilot Results:")
    for result in results:
        print(f"  Graph size {result['graph_size']}: {result['violations']} violations, ${result['cost']:.2f}")
    
    return results

def run_evaluation(cfg: DictConfig, store: GraphStore, llm_client: LLMClient):
    """Run evaluation metrics"""
    evaluator = EvaluationMetrics(store)
    
    # Load test entities
    entities = []
    for i in range(10):
        entity = store.get_entity(f"entity_{i}")
        if entity:
            entities.append(entity)
    
    # Compute metrics
    for entity in entities:
        coherence = evaluator.temporal_coherence_score(entity, [datetime.now()])
        consistency = evaluator.knowledge_consistency_score(entity, {"exposure_history": []})
        plausibility = evaluator.biological_plausibility_score(entity, [])
        
        print(f"{entity.entity_id}: coherence={coherence:.2f}, consistency={consistency:.2f}, plausibility={plausibility:.2f}")

def run_training(cfg: DictConfig, store: GraphStore, llm_client: LLMClient):
    """Run entity training workflow"""
    graph = create_test_graph(n_entities=cfg.training.graph_size, seed=cfg.seed)
    workflow = create_entity_training_workflow(llm_client, store)
    
    state = WorkflowState(
        graph=graph,
        entities=[],
        timepoint=datetime.now().isoformat(),
        resolution=ResolutionLevel(cfg.training.target_resolution),
        violations=[],
        results={}
    )
    
    final_state = workflow.invoke(state)
    
    print(f"Training complete: {len(final_state['violations'])} violations")
    print(f"Total cost: ${llm_client.cost:.2f}")
    print(f"Tokens used: {llm_client.token_count}")

if __name__ == "__main__":
    main()