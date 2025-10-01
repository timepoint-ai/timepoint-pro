# ============================================================================
# validation.py - Validation framework with plugin registry
# ============================================================================
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
import numpy as np

from schemas import Entity, ExposureEvent
from temporal_chain import validate_temporal_reference

class Validator(ABC):
    """Base validator with plugin registry"""
    _validators = {}
    
    @classmethod
    def register(cls, name: str, severity: str = "ERROR"):
        def decorator(func: Callable):
            cls._validators[name] = {"func": func, "severity": severity}
            return func
        return decorator
    
    @classmethod
    def validate_all(cls, entity: Entity, context: Dict) -> List[Dict]:
        violations = []
        for name, validator in cls._validators.items():
            result = validator["func"](entity, context)
            if not result["valid"]:
                violations.append({
                    "validator": name,
                    "severity": validator["severity"],
                    "message": result["message"]
                })
        return violations

@Validator.register("information_conservation", "ERROR")
def validate_information_conservation(entity: Entity, context: Dict, store=None) -> Dict:
    """Validate knowledge ⊆ exposure history"""
    # If store is provided, query actual exposure events from database
    if store:
        exposure_events = store.get_exposure_events(entity.entity_id)
        exposure = set(event.information for event in exposure_events)
    else:
        # Fallback to context-based validation for backward compatibility
        exposure = set(context.get("exposure_history", []))

    knowledge = set(entity.entity_metadata.get("knowledge_state", []))

    unknown = knowledge - exposure
    if unknown:
        return {"valid": False, "message": f"Entity knows about {unknown} without exposure"}
    return {"valid": True, "message": "Information conservation satisfied"}

@Validator.register("energy_budget", "WARNING")
def validate_energy_budget(entity: Entity, context: Dict) -> Dict:
    """Validate interaction costs ≤ capacity"""
    budget = entity.entity_metadata.get("energy_budget", 100)

    # Count interactions at this timepoint based on knowledge items added
    current_knowledge = set(entity.entity_metadata.get("knowledge_state", []))
    previous_knowledge = set(context.get("previous_knowledge", []) or [])
    new_knowledge_count = len(current_knowledge - previous_knowledge)

    # Each knowledge item represents some cognitive effort
    expenditure = new_knowledge_count * 5  # Base cost per knowledge item

    if expenditure > budget * 1.2:  # Allow 20% temporary excess
        return {"valid": False, "message": f"Energy expenditure {expenditure} exceeds budget {budget}"}
    return {"valid": True, "message": "Energy budget satisfied"}

@Validator.register("behavioral_inertia", "WARNING")
def validate_behavioral_inertia(entity: Entity, context: Dict) -> Dict:
    """Validate personality drift is gradual"""
    if "previous_personality" not in context or not context["previous_personality"]:
        return {"valid": True, "message": "No previous state to compare"}

    current = np.array(entity.entity_metadata.get("personality_traits", []))
    previous = np.array(context["previous_personality"])

    if len(current) == 0 or len(previous) == 0:
        return {"valid": True, "message": "Personality data not available"}

    # Handle different length arrays (take minimum length)
    min_len = min(len(current), len(previous))
    current = current[:min_len]
    previous = previous[:min_len]

    drift = np.linalg.norm(current - previous)
    if drift > 1.0:  # Threshold for significant personality change
        return {"valid": False, "message": f"Personality drift {drift:.2f} exceeds threshold 1.0"}
    return {"valid": True, "message": "Behavioral inertia satisfied"}

@Validator.register("biological_constraints", "ERROR")
def validate_biological_constraints(entity: Entity, context: Dict) -> Dict:
    """Validate age-dependent capabilities"""
    age = entity.entity_metadata.get("age", 0)
    action = context.get("action", "")

    if age > 100 and "physical_labor" in action:
        return {"valid": False, "message": f"Entity age {age} incompatible with physical labor"}
    if age < 18 and age > 50 and "childbirth" in action:
        return {"valid": False, "message": f"Entity age {age} incompatible with childbirth"}

    return {"valid": True, "message": "Biological constraints satisfied"}

@Validator.register("network_flow", "WARNING")
def validate_network_flow(entity: Entity, context: Dict) -> Dict:
    """Validate that influence/status changes propagate through relationship graph edges"""
    graph = context.get("graph")
    if not graph or entity.entity_id not in graph:
        return {"valid": True, "message": "No graph available for network flow validation"}

    # Get current knowledge as a proxy for "influence" or "status"
    current_knowledge = set(entity.entity_metadata.get("knowledge_state", []))
    previous_knowledge = set(context.get("previous_knowledge", []) or [])

    # Check for new knowledge acquisition
    new_knowledge = current_knowledge - previous_knowledge
    if not new_knowledge:
        return {"valid": True, "message": "No new knowledge to validate network flow"}

    # Check if entity has connections to sources of this knowledge
    connected_entities = set(graph.neighbors(entity.entity_id))
    # Note: Don't include self for network flow validation - self-knowledge is allowed

    # Get knowledge from connected entities (simplified - in real implementation,
    # we'd need to track knowledge propagation through time)
    connected_knowledge = set()
    for connected_id in connected_entities:
        if connected_id in context.get("all_entity_knowledge", {}):
            connected_knowledge.update(context["all_entity_knowledge"][connected_id])

    # Check if new knowledge could have come from connected entities
    unexplained_knowledge = new_knowledge - connected_knowledge

    # Allow some knowledge to come from events/exposure (not just direct connections)
    exposure_knowledge = set(context.get("exposure_history", []))
    truly_unexplained = unexplained_knowledge - exposure_knowledge

    # DEBUG: Uncomment for validation debugging
    # print(f"DEBUG {entity.entity_id}: new={new_knowledge}, connected={connected_entities}, connected_knowledge={connected_knowledge}, unexplained={unexplained_knowledge}, exposure={exposure_knowledge}, truly_unexplained={truly_unexplained}")

    if truly_unexplained:
        return {
            "valid": False,
            "message": f"Entity gained knowledge {list(truly_unexplained)} without network connections or exposure"
        }

    return {"valid": True, "message": "Network flow validation satisfied"}

@Validator.register("temporal_causality", "ERROR")
def validate_temporal_causality(entity: Entity, context: Dict) -> Dict:
    """Validate that entity knowledge follows causal temporal constraints"""
    store = context.get("store")
    timepoint_id = context.get("timepoint_id")

    if not store or not timepoint_id:
        return {"valid": True, "message": "Insufficient context for temporal causality validation"}

    # Check each knowledge item for temporal validity
    knowledge_state = entity.entity_metadata.get("knowledge_state", [])
    invalid_items = []

    for knowledge_item in knowledge_state:
        validation = validate_temporal_reference(entity.entity_id, knowledge_item, timepoint_id, store)
        if not validation["valid"]:
            invalid_items.append(knowledge_item)

    if invalid_items:
        return {
            "valid": False,
            "message": f"Entity has knowledge {invalid_items} that violates temporal causality"
        }

    return {"valid": True, "message": "Temporal causality satisfied"}