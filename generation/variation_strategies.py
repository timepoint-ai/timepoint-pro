"""
Variation Strategies for Horizontal Data Generation

Provides strategies for generating meaningful variations of simulation scenarios:
- PersonalityVariation: Vary Big Five personality traits
- KnowledgeVariation: Vary initial exposure events
- RelationshipVariation: Vary initial trust/alignment
- OutcomeVariation: Vary key decision points
- StartingConditionVariation: Vary initial entity states
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import random
from copy import deepcopy


class VariationStrategy(ABC):
    """
    Abstract base class for variation strategies.

    Each strategy defines how to create a meaningful variation
    of a base simulation configuration.
    """

    @abstractmethod
    def apply(
        self,
        base_config: Dict[str, Any],
        variation_index: int,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Apply variation to base configuration.

        Args:
            base_config: Base simulation configuration
            variation_index: Index of this variation (0-based)
            random_seed: Random seed for reproducibility

        Returns:
            Modified configuration dict
        """
        pass

    @abstractmethod
    def get_description(self, variation_index: int) -> str:
        """Get human-readable description of this variation"""
        pass


class PersonalityVariation(VariationStrategy):
    """
    Vary personality traits (Big Five) for entities.

    Modifies:
    - Openness to experience (0.0-1.0)
    - Conscientiousness (0.0-1.0)
    - Extraversion (0.0-1.0)
    - Agreeableness (0.0-1.0)
    - Neuroticism (0.0-1.0)
    """

    def __init__(self, magnitude: float = 0.3):
        """
        Args:
            magnitude: How much to vary traits (0.0-1.0)
        """
        self.magnitude = magnitude

    def apply(
        self,
        base_config: Dict[str, Any],
        variation_index: int,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        if random_seed is not None:
            random.seed(random_seed + variation_index)

        config = deepcopy(base_config)

        # Add personality variation metadata
        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["variation_strategy"] = "personality"
        config["metadata"]["variation_index"] = variation_index

        # Generate varied personality profiles
        num_entities = config.get("entities", {}).get("count", 1)
        config["metadata"]["personality_variations"] = []

        for i in range(num_entities):
            # Base personality (neutral)
            base_traits = [0.5, 0.5, 0.5, 0.5, 0.5]

            # Apply random variation
            varied_traits = [
                max(0.0, min(1.0, trait + random.uniform(-self.magnitude, self.magnitude)))
                for trait in base_traits
            ]

            config["metadata"]["personality_variations"].append({
                "entity_index": i,
                "openness": varied_traits[0],
                "conscientiousness": varied_traits[1],
                "extraversion": varied_traits[2],
                "agreeableness": varied_traits[3],
                "neuroticism": varied_traits[4]
            })

        return config

    def get_description(self, variation_index: int) -> str:
        return f"Personality variation #{variation_index} (Big Five traits varied)"


class KnowledgeVariation(VariationStrategy):
    """
    Vary initial knowledge distribution across entities.

    Modifies:
    - Which entities know which information initially
    - Information accessibility patterns
    - Knowledge asymmetries
    """

    def __init__(self, knowledge_pool_size: int = 10):
        """
        Args:
            knowledge_pool_size: Number of knowledge items to distribute
        """
        self.knowledge_pool_size = knowledge_pool_size

    def apply(
        self,
        base_config: Dict[str, Any],
        variation_index: int,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        if random_seed is not None:
            random.seed(random_seed + variation_index)

        config = deepcopy(base_config)

        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["variation_strategy"] = "knowledge"
        config["metadata"]["variation_index"] = variation_index

        # Create knowledge pool
        knowledge_pool = [f"knowledge_item_{i}" for i in range(self.knowledge_pool_size)]

        # Distribute knowledge to entities
        num_entities = config.get("entities", {}).get("count", 1)
        config["metadata"]["knowledge_distributions"] = []

        for i in range(num_entities):
            # Each entity gets random subset of knowledge
            num_items = random.randint(1, self.knowledge_pool_size)
            entity_knowledge = random.sample(knowledge_pool, num_items)

            config["metadata"]["knowledge_distributions"].append({
                "entity_index": i,
                "knowledge_items": entity_knowledge,
                "knowledge_count": len(entity_knowledge)
            })

        return config

    def get_description(self, variation_index: int) -> str:
        return f"Knowledge variation #{variation_index} (initial knowledge distribution)"


class RelationshipVariation(VariationStrategy):
    """
    Vary initial relationship states between entities.

    Modifies:
    - Trust levels between pairs
    - Emotional bonds
    - Power dynamics
    - Belief alignment
    """

    def __init__(self, magnitude: float = 0.4):
        """
        Args:
            magnitude: How much to vary relationships (0.0-1.0)
        """
        self.magnitude = magnitude

    def apply(
        self,
        base_config: Dict[str, Any],
        variation_index: int,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        if random_seed is not None:
            random.seed(random_seed + variation_index)

        config = deepcopy(base_config)

        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["variation_strategy"] = "relationships"
        config["metadata"]["variation_index"] = variation_index

        # Generate relationship matrix
        num_entities = config.get("entities", {}).get("count", 1)
        relationships = []

        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                # Base relationship (neutral)
                trust = 0.5
                emotional_bond = 0.0
                power_dynamic = 0.0
                belief_alignment = 0.5

                # Apply variation
                relationships.append({
                    "entity_a": i,
                    "entity_b": j,
                    "trust_level": max(0.0, min(1.0, trust + random.uniform(-self.magnitude, self.magnitude))),
                    "emotional_bond": max(-1.0, min(1.0, emotional_bond + random.uniform(-self.magnitude, self.magnitude))),
                    "power_dynamic": max(-1.0, min(1.0, power_dynamic + random.uniform(-self.magnitude * 0.5, self.magnitude * 0.5))),
                    "belief_alignment": max(0.0, min(1.0, belief_alignment + random.uniform(-self.magnitude, self.magnitude)))
                })

        config["metadata"]["initial_relationships"] = relationships

        return config

    def get_description(self, variation_index: int) -> str:
        return f"Relationship variation #{variation_index} (trust, bonds, alignment)"


class OutcomeVariation(VariationStrategy):
    """
    Vary key decision points to produce different outcomes.

    Modifies:
    - Decision biases
    - Risk tolerance
    - Decision confidence thresholds
    - Patience levels
    """

    def __init__(self, magnitude: float = 0.3):
        """
        Args:
            magnitude: How much to vary decision parameters (0.0-1.0)
        """
        self.magnitude = magnitude

    def apply(
        self,
        base_config: Dict[str, Any],
        variation_index: int,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        if random_seed is not None:
            random.seed(random_seed + variation_index)

        config = deepcopy(base_config)

        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["variation_strategy"] = "outcomes"
        config["metadata"]["variation_index"] = variation_index

        # Generate decision parameter variations
        num_entities = config.get("entities", {}).get("count", 1)
        decision_params = []

        for i in range(num_entities):
            decision_params.append({
                "entity_index": i,
                "risk_tolerance": max(0.0, min(1.0, 0.5 + random.uniform(-self.magnitude, self.magnitude))),
                "decision_confidence": max(0.0, min(1.0, 0.8 + random.uniform(-self.magnitude * 0.5, self.magnitude * 0.5))),
                "patience_threshold": max(0.0, min(100.0, 50.0 + random.uniform(-self.magnitude * 50, self.magnitude * 50))),
                "social_engagement": max(0.0, min(1.0, 0.8 + random.uniform(-self.magnitude, self.magnitude)))
            })

        config["metadata"]["decision_parameters"] = decision_params

        return config

    def get_description(self, variation_index: int) -> str:
        return f"Outcome variation #{variation_index} (decision parameters)"


class StartingConditionVariation(VariationStrategy):
    """
    Vary initial physical and cognitive states of entities.

    Modifies:
    - Energy budgets
    - Emotional states (valence, arousal)
    - Health status
    - Stress levels
    """

    def __init__(self, magnitude: float = 0.25):
        """
        Args:
            magnitude: How much to vary starting conditions (0.0-1.0)
        """
        self.magnitude = magnitude

    def apply(
        self,
        base_config: Dict[str, Any],
        variation_index: int,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        if random_seed is not None:
            random.seed(random_seed + variation_index)

        config = deepcopy(base_config)

        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["variation_strategy"] = "starting_conditions"
        config["metadata"]["variation_index"] = variation_index

        # Generate starting state variations
        num_entities = config.get("entities", {}).get("count", 1)
        starting_states = []

        for i in range(num_entities):
            starting_states.append({
                "entity_index": i,
                "energy_budget": max(20.0, min(100.0, 80.0 + random.uniform(-self.magnitude * 60, self.magnitude * 60))),
                "emotional_valence": max(-1.0, min(1.0, random.uniform(-self.magnitude, self.magnitude))),
                "emotional_arousal": max(0.0, min(1.0, 0.3 + random.uniform(-self.magnitude, self.magnitude))),
                "health_status": max(0.5, min(1.0, 1.0 - random.uniform(0, self.magnitude * 0.5))),
                "pain_level": max(0.0, min(1.0, random.uniform(0, self.magnitude * 0.3)))
            })

        config["metadata"]["starting_states"] = starting_states

        return config

    def get_description(self, variation_index: int) -> str:
        return f"Starting condition variation #{variation_index} (energy, emotion, health)"


class VariationStrategyFactory:
    """Factory for creating variation strategies"""

    _strategies = {
        "vary_personalities": PersonalityVariation,
        "vary_knowledge": KnowledgeVariation,
        "vary_relationships": RelationshipVariation,
        "vary_outcomes": OutcomeVariation,
        "vary_starting_conditions": StartingConditionVariation
    }

    @classmethod
    def create(cls, strategy_name: str, **kwargs) -> VariationStrategy:
        """
        Create a variation strategy by name.

        Args:
            strategy_name: Name of strategy to create
            **kwargs: Arguments to pass to strategy constructor

        Returns:
            VariationStrategy instance

        Raises:
            ValueError: If strategy name not recognized
        """
        if strategy_name not in cls._strategies:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(cls._strategies.keys())}"
            )

        return cls._strategies[strategy_name](**kwargs)

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names"""
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """
        Register a custom variation strategy.

        Args:
            name: Name for the strategy
            strategy_class: Class implementing VariationStrategy
        """
        if not issubclass(strategy_class, VariationStrategy):
            raise TypeError(f"{strategy_class} must inherit from VariationStrategy")
        cls._strategies[name] = strategy_class
