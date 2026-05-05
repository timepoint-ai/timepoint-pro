"""
Regression tests for portal_strategy.py — guard against re-introduction of
hardcoded fake score sentinels (e.g. importance_score=0.5) silently emitted
to consumers.

Background
----------
Three fields used to be hardcoded with a TODO:

    "importance_score": 0.5,  # TODO: compute from state
    "state_complexity": 0.5,  # TODO: compute from state
    "pivot_detected": False,  # TODO: detect pivot points

Consumers (`temporal_agent.determine_next_step_fidelity_and_time`) treated
these as real signals to choose fidelity/temporal granularity, which meant
every call branched identically. This test fixes that contract:

  * `importance_score` and `state_complexity` are EITHER computed from real
    state signals OR omitted (so consumer's `dict.get(..., default)` fires).
  * `pivot_detected` is always a real boolean derived from state.
  * Output varies with input (no constant 0.5).
  * Source no longer contains the old fake-score literal pattern.

These are pure-function tests: no LLM, no DB, no network. They construct
PortalStrategy via `__new__` to bypass `__init__` (which requires real
llm_client/store) because the methods under test only use class-level
constants and self-free logic.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from schemas import Entity
from workflows.portal_strategy import PortalState, PortalStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_strategy() -> PortalStrategy:
    """
    Construct a PortalStrategy without running __init__.

    The signal-computation helpers under test are self-contained
    (they only read class-level constants and the passed-in state),
    so we don't need a real config/llm_client/store. Avoiding __init__
    keeps these tests truly hermetic — no LLM, no DB, no network.
    """
    return PortalStrategy.__new__(PortalStrategy)


def _state(
    description: str = "",
    entities: list[Entity] | None = None,
    world_state: dict | None = None,
    plausibility_score: float = 0.0,
) -> PortalState:
    return PortalState(
        year=2030,
        month=1,
        description=description,
        entities=entities or [],
        world_state=world_state or {},
        plausibility_score=plausibility_score,
    )


# ---------------------------------------------------------------------------
# Source-level guards
# ---------------------------------------------------------------------------


PORTAL_STRATEGY_SRC = (
    Path(__file__).resolve().parent.parent / "workflows" / "portal_strategy.py"
).read_text()


def test_source_does_not_contain_old_fake_score_literals():
    """
    The exact old fake-score lines must not reappear. This is a literal-text
    guard against future regressions (e.g. someone reverting the fix).
    """
    forbidden_patterns = [
        r'"importance_score":\s*0\.5\s*,\s*#\s*TODO',
        r'"state_complexity":\s*0\.5\s*,\s*#\s*TODO',
        r'"pivot_detected":\s*False\s*,\s*#\s*TODO',
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, PORTAL_STRATEGY_SRC), (
            f"Forbidden hardcoded fake-score literal re-introduced: /{pat}/"
        )


# ---------------------------------------------------------------------------
# importance_score: computed-real OR absent (never 0.5 sentinel)
# ---------------------------------------------------------------------------


class TestImportanceScore:
    def test_returns_none_for_empty_state(self):
        """Empty state has no signal -> field should be omitted (None)."""
        s = _bare_strategy()
        result = s._compute_state_importance(_state())
        assert result is None

    def test_uses_plausibility_score_when_set(self):
        s = _bare_strategy()
        result = s._compute_state_importance(_state(plausibility_score=0.8))
        assert result is not None
        assert 0.0 <= result <= 1.0
        # Should reflect the real plausibility signal, not be 0.5
        assert result == pytest.approx(0.8)

    def test_pivot_keywords_boost_importance(self):
        s = _bare_strategy()
        plain = s._compute_state_importance(_state(description="quiet quarter"))
        pivot = s._compute_state_importance(
            _state(description="Founded the company; raised Series A funding")
        )
        # Plain has no signal -> None; pivot has signal
        assert plain is None
        assert pivot is not None and pivot > 0.5

    def test_key_events_contribute(self):
        s = _bare_strategy()
        result = s._compute_state_importance(
            _state(world_state={"key_events": ["product launch", "first hire"]})
        )
        assert result is not None
        assert 0.0 < result <= 1.0

    def test_varies_with_input(self):
        """Two distinct rich states must not collapse to the same number."""
        s = _bare_strategy()
        a = s._compute_state_importance(
            _state(description="raised funding", plausibility_score=0.9)
        )
        b = s._compute_state_importance(
            _state(description="quiet operations", plausibility_score=0.3)
        )
        assert a is not None and b is not None
        assert a != b
        assert a > b

    def test_never_returns_bare_zero_point_five(self):
        """
        Across many varied inputs, the score must never collapse to exactly
        0.5 (the old fake sentinel). 0.5 is allowed if computed from real
        signals — this test ensures it's not the *default*.
        """
        s = _bare_strategy()
        # Empty state -> None (not 0.5)
        assert s._compute_state_importance(_state()) is None
        # Single weak signal -> not 0.5
        r = s._compute_state_importance(_state(plausibility_score=0.7))
        assert r != 0.5


# ---------------------------------------------------------------------------
# state_complexity: computed-real OR absent
# ---------------------------------------------------------------------------


class TestStateComplexity:
    def test_returns_none_for_empty_state(self):
        s = _bare_strategy()
        assert s._compute_state_complexity(_state()) is None

    def test_grows_with_entity_count(self):
        s = _bare_strategy()
        e1 = [Entity(entity_id=f"e{i}", entity_type="person") for i in range(2)]
        e8 = [Entity(entity_id=f"e{i}", entity_type="person") for i in range(8)]
        c1 = s._compute_state_complexity(_state(entities=e1))
        c8 = s._compute_state_complexity(_state(entities=e8))
        assert c1 is not None and c8 is not None
        assert c8 > c1

    def test_grows_with_world_state_keys(self):
        s = _bare_strategy()
        small = s._compute_state_complexity(_state(world_state={"a": 1}))
        big = s._compute_state_complexity(
            _state(world_state={k: v for k, v in zip("abcdef", range(6), strict=False)})
        )
        assert small is not None and big is not None
        assert big > small

    def test_in_unit_interval(self):
        s = _bare_strategy()
        e = [Entity(entity_id=f"e{i}", entity_type="person") for i in range(50)]
        c = s._compute_state_complexity(
            _state(
                description="x" * 5000,
                entities=e,
                world_state={f"k{i}": i for i in range(50)},
            )
        )
        assert c is not None
        assert 0.0 <= c <= 1.0

    def test_varies_with_input(self):
        s = _bare_strategy()
        a = s._compute_state_complexity(
            _state(description="brief", world_state={"a": 1})
        )
        b = s._compute_state_complexity(
            _state(
                description="x" * 200,
                world_state={"a": 1, "b": 2, "c": 3, "d": 4},
            )
        )
        assert a is not None and b is not None
        assert a != b


# ---------------------------------------------------------------------------
# pivot_detected: always a real boolean (never a sentinel)
# ---------------------------------------------------------------------------


class TestPivotDetection:
    def test_empty_state_not_pivot(self):
        s = _bare_strategy()
        assert s._detect_pivot_in_state(_state()) is False

    def test_keyword_in_description_is_pivot(self):
        s = _bare_strategy()
        assert (
            s._detect_pivot_in_state(_state(description="The team pivoted to enterprise"))
            is True
        )

    def test_key_event_is_pivot(self):
        s = _bare_strategy()
        assert (
            s._detect_pivot_in_state(
                _state(world_state={"key_events": ["product launch"]})
            )
            is True
        )

    def test_many_entity_changes_is_pivot(self):
        s = _bare_strategy()
        assert (
            s._detect_pivot_in_state(
                _state(world_state={"entity_changes": {"a": 1, "b": 2, "c": 3, "d": 4}})
            )
            is True
        )

    def test_unrelated_state_is_not_pivot(self):
        s = _bare_strategy()
        assert (
            s._detect_pivot_in_state(
                _state(description="business as usual; no notable events")
            )
            is False
        )

    def test_returns_real_bool_not_truthy_sentinel(self):
        """Must be exactly True/False, not 0.5 or some other proxy."""
        s = _bare_strategy()
        out = s._detect_pivot_in_state(_state(description="raised Series B"))
        assert isinstance(out, bool)
        assert out is True


# ---------------------------------------------------------------------------
# Integration: the context dict assembled at the call-site
# ---------------------------------------------------------------------------


class TestContextDictDoesNotEmitFakes:
    """
    Replicate the exact context-dict assembly used inside
    _explore_reverse_chronological() and verify it never emits the old
    hardcoded sentinels.
    """

    def _assemble(self, state: PortalState) -> dict:
        s = _bare_strategy()
        ctx: dict = {"entities": state.entities}
        importance = s._compute_state_importance(state)
        if importance is not None:
            ctx["importance_score"] = importance
        complexity = s._compute_state_complexity(state)
        if complexity is not None:
            ctx["state_complexity"] = complexity
        ctx["pivot_detected"] = s._detect_pivot_in_state(state)
        return ctx

    def test_empty_state_omits_score_fields(self):
        ctx = self._assemble(_state())
        # Both score fields stripped — consumer's .get(..., default) handles it
        assert "importance_score" not in ctx
        assert "state_complexity" not in ctx
        assert ctx["pivot_detected"] is False

    def test_rich_state_includes_real_values(self):
        ctx = self._assemble(
            _state(
                description="raised Series A funding",
                world_state={"key_events": ["funding round"], "stage": "growth"},
                plausibility_score=0.85,
            )
        )
        assert "importance_score" in ctx
        assert "state_complexity" in ctx
        # Not the old sentinels
        assert ctx["importance_score"] != 0.5
        assert ctx["state_complexity"] != 0.5
        assert ctx["pivot_detected"] is True

    def test_two_different_states_yield_different_contexts(self):
        a = self._assemble(_state(description="quiet", world_state={"x": 1}))
        b = self._assemble(
            _state(
                description="founded the company",
                world_state={"key_events": ["incorporation"], "y": 2, "z": 3},
                plausibility_score=0.9,
            )
        )
        # The contexts must materially differ — proving output varies with input
        assert a != b
        assert a.get("pivot_detected") != b.get("pivot_detected") or a.get(
            "importance_score"
        ) != b.get("importance_score")
