# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Boundary observer tests

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from scpn_phase_orchestrator.binding.types import BoundaryDef
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver, BoundaryState

# ── Helpers ─────────────────────────────────────────────────────────────


def _soft_def(name: str, variable: str, lower=None, upper=None) -> BoundaryDef:
    return BoundaryDef(
        name=name, variable=variable, lower=lower, upper=upper, severity="soft"
    )


def _hard_def(name: str, variable: str, lower=None, upper=None) -> BoundaryDef:
    return BoundaryDef(
        name=name, variable=variable, lower=lower, upper=upper, severity="hard"
    )


# ── BoundaryState dataclass ─────────────────────────────────────────────


class TestBoundaryState:
    def test_default_empty_lists(self):
        state = BoundaryState()
        assert state.violations == []
        assert state.soft_violations == []
        assert state.hard_violations == []

    def test_mutable_lists_are_independent(self):
        """Two BoundaryState instances must not share list references."""
        a = BoundaryState()
        b = BoundaryState()
        a.violations.append("x")
        assert b.violations == []


# ── Single boundary tests ──────────────────────────────────────────────


class TestSingleBoundary:
    def test_lower_bound_exact_not_violated(self):
        """Value exactly at the lower bound is not a violation."""
        obs = BoundaryObserver([_soft_def("R_floor", "R", lower=0.2)])
        state = obs.observe({"R": 0.2})
        assert state.violations == []

    def test_lower_bound_epsilon_below_violated(self):
        """Value just below the lower bound triggers violation."""
        obs = BoundaryObserver([_soft_def("R_floor", "R", lower=0.2)])
        state = obs.observe({"R": 0.19999})
        assert len(state.violations) == 1

    def test_upper_bound_exact_not_violated(self):
        """Value exactly at the upper bound is not a violation."""
        obs = BoundaryObserver([_hard_def("T_cap", "T", upper=100.0)])
        state = obs.observe({"T": 100.0})
        assert state.violations == []

    def test_upper_bound_epsilon_above_violated(self):
        """Value just above upper bound triggers violation."""
        obs = BoundaryObserver([_hard_def("T_cap", "T", upper=100.0)])
        state = obs.observe({"T": 100.001})
        assert len(state.violations) == 1

    def test_lower_only_high_value_ok(self):
        """Lower-only bound: any value above lower is fine."""
        obs = BoundaryObserver([_soft_def("R_floor", "R", lower=0.2)])
        state = obs.observe({"R": 999.0})
        assert state.violations == []

    def test_upper_only_low_value_ok(self):
        """Upper-only bound: any value below upper is fine."""
        obs = BoundaryObserver([_hard_def("T_cap", "T", upper=100.0)])
        state = obs.observe({"T": -999.0})
        assert state.violations == []

    def test_no_bounds_defined(self):
        """Boundary with both lower=None and upper=None never fires."""
        bdef = BoundaryDef(
            name="X", variable="X", lower=None, upper=None, severity="soft"
        )
        obs = BoundaryObserver([bdef])
        state = obs.observe({"X": 1e6})
        assert state.violations == []


# ── Multi-violation scenarios ──────────────────────────────────────────


class TestMultipleViolations:
    def test_all_violated_simultaneously(self):
        """Three boundaries, all violated at once."""
        defs = [
            _soft_def("R_floor", "R", lower=0.2),
            _hard_def("T_cap", "T", upper=100.0),
            _soft_def("P_band", "P", lower=1.0, upper=10.0),
        ]
        obs = BoundaryObserver(defs)
        state = obs.observe({"R": 0.0, "T": 200.0, "P": 0.0})
        assert len(state.violations) == 3
        assert len(state.soft_violations) == 2
        assert len(state.hard_violations) == 1

    def test_band_violation_below(self):
        """Value below lower of a band boundary."""
        obs = BoundaryObserver([_soft_def("P_band", "P", lower=1.0, upper=10.0)])
        state = obs.observe({"P": 0.5})
        assert len(state.violations) == 1

    def test_band_violation_above(self):
        """Value above upper of a band boundary."""
        obs = BoundaryObserver([_soft_def("P_band", "P", lower=1.0, upper=10.0)])
        state = obs.observe({"P": 11.0})
        assert len(state.violations) == 1

    def test_band_within_ok(self):
        """Value within band → no violation."""
        obs = BoundaryObserver([_soft_def("P_band", "P", lower=1.0, upper=10.0)])
        state = obs.observe({"P": 5.0})
        assert state.violations == []


# ── Step tracking ───────────────────────────────────────────────────────


class TestStepTracking:
    def test_step_defaults_to_zero(self):
        """Without explicit step, internal counter starts at 0."""
        obs = BoundaryObserver([])
        assert obs._step == 0

    def test_step_updates_on_observe(self):
        """Passing step= to observe updates the internal counter."""
        obs = BoundaryObserver([])
        obs.observe({}, step=42)
        assert obs._step == 42

    def test_step_persists_across_calls(self):
        """Step from previous call persists if not overridden."""
        obs = BoundaryObserver([])
        obs.observe({}, step=10)
        obs.observe({})  # No step specified
        assert obs._step == 10


# ── Event bus integration ──────────────────────────────────────────────


class TestEventBus:
    def test_set_event_bus(self):
        """set_event_bus stores the bus for later use."""
        obs = BoundaryObserver([])
        mock_bus = MagicMock()
        obs.set_event_bus(mock_bus)
        assert obs._event_bus is mock_bus

    def test_no_event_bus_no_post(self):
        """Without event bus, violations do not raise errors."""
        obs = BoundaryObserver([_soft_def("R_floor", "R", lower=0.2)])
        # Should not raise even with violations
        state = obs.observe({"R": 0.0})
        assert len(state.violations) == 1

    def test_event_bus_receives_boundary_breach(self):
        """When violations occur and event bus is set, a RegimeEvent is posted."""
        obs = BoundaryObserver([_hard_def("T_cap", "T", upper=100.0)])
        mock_bus = MagicMock()
        obs.set_event_bus(mock_bus)
        obs.observe({"T": 200.0}, step=5)
        mock_bus.post.assert_called_once()
        event = mock_bus.post.call_args[0][0]
        assert event.kind == "boundary_breach"
        assert event.step == 5
        assert "T_cap" in event.detail

    def test_event_bus_not_called_when_no_violations(self):
        """No violations → event bus not called."""
        obs = BoundaryObserver([_hard_def("T_cap", "T", upper=100.0)])
        mock_bus = MagicMock()
        obs.set_event_bus(mock_bus)
        obs.observe({"T": 50.0})
        mock_bus.post.assert_not_called()

    def test_event_bus_detail_joins_multiple_violations(self):
        """Multiple violations are joined with '; ' in the event detail."""
        defs = [
            _soft_def("R_floor", "R", lower=0.2),
            _hard_def("T_cap", "T", upper=100.0),
        ]
        obs = BoundaryObserver(defs)
        mock_bus = MagicMock()
        obs.set_event_bus(mock_bus)
        obs.observe({"R": 0.0, "T": 200.0})
        event = mock_bus.post.call_args[0][0]
        assert "; " in event.detail


# ── Violation message format ────────────────────────────────────────────


class TestViolationMessages:
    def test_message_contains_variable_name(self):
        obs = BoundaryObserver([_soft_def("R_floor", "R", lower=0.2)])
        state = obs.observe({"R": 0.1})
        assert "R=" in state.violations[0]

    def test_message_contains_boundary_name(self):
        obs = BoundaryObserver([_soft_def("R_floor", "R", lower=0.2)])
        state = obs.observe({"R": 0.1})
        assert "R_floor" in state.violations[0]

    def test_message_contains_bounds(self):
        obs = BoundaryObserver([_soft_def("P_band", "P", lower=1.0, upper=10.0)])
        state = obs.observe({"P": 0.5})
        assert "1.0" in state.violations[0] or "1" in state.violations[0]
        assert "10.0" in state.violations[0] or "10" in state.violations[0]


# ── Unknown severity ────────────────────────────────────────────────────


class TestUnknownSeverity:
    def test_unknown_severity_logged_as_warning(self, caplog):
        """Unrecognised severity should log a warning and treat as hard."""
        bdef = BoundaryDef(
            name="X", variable="X", lower=0.0, upper=None, severity="banana"
        )
        obs = BoundaryObserver([bdef])
        log_name = "scpn_phase_orchestrator.monitor.boundaries"
        with caplog.at_level(logging.WARNING, logger=log_name):
            state = obs.observe({"X": -1.0})
        assert len(state.hard_violations) == 1
        assert "banana" in caplog.text

    def test_empty_severity_treated_as_hard(self, caplog):
        """Empty string severity is unknown → hard."""
        bdef = BoundaryDef(name="Y", variable="Y", lower=0.0, upper=None, severity="")
        obs = BoundaryObserver([bdef])
        log_name = "scpn_phase_orchestrator.monitor.boundaries"
        with caplog.at_level(logging.WARNING, logger=log_name):
            state = obs.observe({"Y": -1.0})
        assert len(state.hard_violations) == 1


# ── Empty observer ──────────────────────────────────────────────────────


class TestEmptyObserver:
    def test_no_definitions_no_violations(self):
        obs = BoundaryObserver([])
        state = obs.observe({"R": 0.0, "T": 999.0})
        assert state.violations == []

    def test_empty_values_no_violations(self):
        obs = BoundaryObserver([_soft_def("R_floor", "R", lower=0.2)])
        state = obs.observe({})
        assert state.violations == []


# ── Negative / special float values ─────────────────────────────────────


class TestSpecialValues:
    def test_negative_value_below_negative_lower(self):
        obs = BoundaryObserver([_hard_def("neg", "V", lower=-10.0, upper=-1.0)])
        state = obs.observe({"V": -20.0})
        assert len(state.violations) == 1

    def test_zero_value_within_bounds(self):
        obs = BoundaryObserver([_soft_def("zero_ok", "V", lower=-1.0, upper=1.0)])
        state = obs.observe({"V": 0.0})
        assert state.violations == []

    def test_very_large_value(self):
        obs = BoundaryObserver([_hard_def("cap", "V", upper=1e10)])
        state = obs.observe({"V": 1e20})
        assert len(state.violations) == 1
