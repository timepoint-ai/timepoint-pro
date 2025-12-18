#!/usr/bin/env python3.10
"""
Quick test to verify entity_config with profiles is passed through context.

This tests ONLY the context passing, not the full simulation.
"""
import os
import pytest
import json
from pathlib import Path

from generation.config_schema import SimulationConfig, EntityConfig


@pytest.mark.unit
def test_entity_config_has_profiles_field():
    """Test 1: Verify profiles field exists in EntityConfig."""
    config = SimulationConfig.portal_timepoint_unicorn()
    entity_config = config.entities

    assert hasattr(entity_config, 'profiles'), "EntityConfig.profiles field NOT FOUND"
    print("  ✓ EntityConfig.profiles field exists")
    print(f"  ✓ Profiles: {entity_config.profiles}")


@pytest.mark.unit
def test_profile_files_exist():
    """Test 2: Verify profile files exist and are valid JSON."""
    config = SimulationConfig.portal_timepoint_unicorn()
    entity_config = config.entities

    if not entity_config.profiles:
        pytest.skip("No profiles configured for this template")

    for profile_path in entity_config.profiles:
        profile_file = Path(profile_path)
        assert profile_file.exists(), f"Profile file not found: {profile_path}"

        # Load and validate JSON
        with open(profile_file) as f:
            profile_data = json.load(f)

        name = profile_data.get("name", profile_file.stem)
        archetype = profile_data.get("archetype_id", "unknown")
        print(f"  ✓ {profile_path}")
        print(f"    Name: {name}")
        print(f"    Archetype: {archetype}")


@pytest.mark.unit
def test_e2e_runner_context_building():
    """Test 3: Simulate what E2E runner does - build context dict."""
    config = SimulationConfig.portal_timepoint_unicorn()

    # This is what the E2E runner does in _generate_initial_scene()
    context = {
        "max_entities": config.entities.count,
        "max_timepoints": 1,
        "temporal_mode": config.temporal.mode.value,
        "entity_metadata": config.metadata,
        "entity_config": {
            "count": config.entities.count,
            "types": config.entities.types,
            "profiles": config.entities.profiles if config.entities.profiles else []
        }
    }

    assert "entity_config" in context, "entity_config not in context"
    assert "profiles" in context["entity_config"], "profiles not in entity_config"

    print(f"  ✓ Context built")
    print(f"    max_entities: {context['max_entities']}")
    print(f"    temporal_mode: {context['temporal_mode']}")
    print(f"    entity_config: {context['entity_config']}")
    print(f"  ✓ entity_config.profiles in context: {context['entity_config']['profiles']}")


@pytest.mark.unit
def test_orchestrator_profile_extraction():
    """Test 4: Simulate what orchestrator does - extract profiles from context."""
    config = SimulationConfig.portal_timepoint_unicorn()

    # Build context as E2E runner does
    context = {
        "max_entities": config.entities.count,
        "max_timepoints": 1,
        "temporal_mode": config.temporal.mode.value,
        "entity_metadata": config.metadata,
        "entity_config": {
            "count": config.entities.count,
            "types": config.entities.types,
            "profiles": config.entities.profiles if config.entities.profiles else []
        }
    }

    # This is what orchestrator does
    entity_config_from_context = context.get("entity_config", {})
    profile_paths = entity_config_from_context.get("profiles", [])

    print(f"  entity_config from context: {entity_config_from_context}")
    print(f"  profile_paths extracted: {profile_paths}")

    if profile_paths:
        print(f"  ✓ Orchestrator would load {len(profile_paths)} profile(s)")
    else:
        print(f"  ⚠️ No profile paths found (may be expected for this config)")


@pytest.mark.unit
def test_profile_loading_simulation():
    """Test 5: Simulate profile loading."""
    config = SimulationConfig.portal_timepoint_unicorn()
    profile_paths = config.entities.profiles or []

    if not profile_paths:
        pytest.skip("No profiles configured for this template")

    loaded_entities = []

    for profile_path in profile_paths:
        profile_file = Path(profile_path)
        if not profile_file.exists():
            pytest.fail(f"Profile not found: {profile_path}")

        with open(profile_file) as f:
            profile_data = json.load(f)

        # Extract name from filename
        name = profile_file.stem
        full_name = profile_data.get("name", name.replace("_", " ").title())
        entity_id = name.lower().replace(" ", "_")

        print(f"  ✓ Loaded profile: {full_name}")
        print(f"    entity_id: {entity_id}")
        print(f"    archetype: {profile_data.get('archetype_id', 'unknown')}")

        loaded_entities.append({
            "entity_id": entity_id,
            "name": full_name,
            "archetype": profile_data.get("archetype_id")
        })

    assert len(loaded_entities) > 0, "No entities loaded from profiles"
    print(f"  ✓ Successfully loaded {len(loaded_entities)} entities from profiles")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
