# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Foundation-model governor tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.control_barrier import (
    ControlBarrierFilter,
    NeuralBarrier,
)
from scpn_phase_orchestrator.actuation.foundation_model_governor import (
    ADMITTED,
    CONSTRAINED,
    REJECTED,
    FoundationModelGovernor,
    GovernorDecision,
)

_STATE = np.array([0.5])
_DRIFT = np.array([0.0])


def _cbf(
    threshold: float = 0.0,
    gamma: float = 0.5,
    lo: float = -1.0,
    hi: float = 1.0,
) -> ControlBarrierFilter:
    """1-D barrier ``h(x) = x - threshold`` (safe set ``x >= threshold``)."""
    barrier = NeuralBarrier(
        weights=(np.array([[1.0]]),), biases=(np.array([-threshold]),)
    )
    return ControlBarrierFilter(
        barrier=barrier,
        gamma=gamma,
        control_lo=lo,
        control_hi=hi,
        control_effect=np.array([1.0]),
    )


def _governor(**overrides: object) -> FoundationModelGovernor:
    base: dict[str, object] = {"control_lo": -1.0, "control_hi": 1.0, "max_rate": 0.5}
    base.update(overrides)
    return FoundationModelGovernor(**base)  # type: ignore[arg-type]


class TestConfigValidation:
    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="control_hi must be greater"):
            _governor(control_lo=1.0, control_hi=-1.0)

    def test_rejects_non_positive_rate(self) -> None:
        with pytest.raises(ValueError, match="max_rate must be positive"):
            _governor(max_rate=0.0)

    def test_rejects_non_finite_bound(self) -> None:
        with pytest.raises(ValueError, match="control_hi"):
            _governor(control_hi=float("inf"))


class TestEnvelopeStages:
    def test_in_bounds_proposal_is_admitted_unchanged(self) -> None:
        decision = _governor(max_rate=2.0).govern(
            0.3, _STATE, _DRIFT, previous_action=0.2
        )

        assert decision.status == ADMITTED
        assert decision.admitted_action == pytest.approx(0.3)
        assert decision.stages_applied == ()
        assert decision.barrier_value is None

    def test_out_of_bounds_proposal_is_clamped(self) -> None:
        decision = _governor(max_rate=2.0).govern(
            5.0, _STATE, _DRIFT, previous_action=0.9
        )

        assert decision.status == CONSTRAINED
        assert decision.admitted_action == pytest.approx(1.0)
        assert "bounds" in decision.stages_applied

    def test_rate_limit_caps_the_step(self) -> None:
        decision = _governor(max_rate=0.5).govern(
            0.9, _STATE, _DRIFT, previous_action=0.1
        )

        assert decision.status == CONSTRAINED
        assert decision.admitted_action == pytest.approx(0.6)
        assert decision.stages_applied == ("rate_limit",)

    def test_rate_limit_downward(self) -> None:
        decision = _governor(max_rate=0.3).govern(
            -0.9, _STATE, _DRIFT, previous_action=0.2
        )

        assert decision.admitted_action == pytest.approx(-0.1)
        assert "rate_limit" in decision.stages_applied

    def test_bounds_then_rate_limit_compose(self) -> None:
        decision = _governor(max_rate=0.5).govern(
            5.0, _STATE, _DRIFT, previous_action=0.1
        )

        assert decision.stages_applied == ("bounds", "rate_limit")
        assert decision.admitted_action == pytest.approx(0.6)


class TestBarrierStage:
    def test_no_barrier_reports_none(self) -> None:
        decision = _governor(max_rate=2.0).govern(0.3, _STATE, _DRIFT)

        assert decision.barrier_value is None

    def test_safe_state_reports_positive_barrier(self) -> None:
        governor = _governor(max_rate=2.0, barrier_filter=_cbf())
        decision = governor.govern(0.5, _STATE, _DRIFT, previous_action=0.4)

        assert decision.barrier_value == pytest.approx(0.5)
        assert decision.status in (ADMITTED, CONSTRAINED)
        assert not decision.violations

    def test_barrier_projects_unsafe_action(self) -> None:
        # x=0.05, gamma=0.5, drift=-0.5 -> CBF requires u >= 0.475; propose 0.1.
        governor = _governor(max_rate=2.0, barrier_filter=_cbf())
        decision = governor.govern(
            0.1, np.array([0.05]), np.array([-0.5]), previous_action=0.1
        )

        assert "cbf" in decision.stages_applied
        assert decision.admitted_action == pytest.approx(0.475, abs=1e-6)
        assert decision.status == CONSTRAINED

    def test_state_outside_safe_set_is_rejected(self) -> None:
        governor = _governor(max_rate=2.0, barrier_filter=_cbf())
        decision = governor.govern(0.5, np.array([-0.2]), _DRIFT, previous_action=0.3)

        assert decision.status == REJECTED
        assert decision.barrier_value == pytest.approx(-0.2)
        assert any("outside certified safe set" in v for v in decision.violations)


class TestSafetyPredicates:
    def test_passing_predicate_admits(self) -> None:
        governor = _governor(
            max_rate=2.0,
            safety_predicates=(
                ("envelope", lambda a, s: (abs(a) <= 0.5, "too large")),
            ),
        )
        decision = governor.govern(0.3, _STATE, _DRIFT, previous_action=0.2)

        assert decision.status == ADMITTED
        assert not decision.violations

    def test_failing_predicate_rejects_and_holds_previous(self) -> None:
        governor = _governor(
            max_rate=2.0,
            safety_predicates=(
                ("envelope", lambda a, s: (abs(a) <= 0.3, "exceeds 0.3")),
            ),
        )
        decision = governor.govern(0.9, _STATE, _DRIFT, previous_action=0.15)

        assert decision.status == REJECTED
        assert decision.admitted_action == pytest.approx(0.15)
        assert decision.violations == ("envelope: exceeds 0.3",)

    def test_reject_with_neutral_fallback(self) -> None:
        governor = _governor(
            max_rate=2.0,
            hold_on_reject=False,
            safety_predicates=(("deny", lambda a, s: (False, "blocked")),),
        )
        decision = governor.govern(0.9, _STATE, _DRIFT, previous_action=0.15)

        assert decision.status == REJECTED
        assert decision.admitted_action == pytest.approx(0.0)

    def test_reject_fallback_is_clamped_to_bounds(self) -> None:
        governor = _governor(
            max_rate=5.0,
            safety_predicates=(("deny", lambda a, s: (False, "blocked")),),
        )
        # previous_action sits outside the actuator bounds; the held fallback clamps.
        decision = governor.govern(0.2, _STATE, _DRIFT, previous_action=3.0)

        assert decision.admitted_action == pytest.approx(1.0)

    def test_multiple_predicates_accumulate_violations(self) -> None:
        governor = _governor(
            max_rate=2.0,
            safety_predicates=(
                ("a", lambda v, s: (False, "first")),
                ("b", lambda v, s: (False, "second")),
            ),
        )
        decision = governor.govern(0.2, _STATE, _DRIFT, previous_action=0.1)

        assert decision.violations == ("a: first", "b: second")


class TestGovernorDecision:
    def _decision(self) -> GovernorDecision:
        return _governor(max_rate=2.0).govern(0.3, _STATE, _DRIFT, previous_action=0.2)

    def test_audit_record_round_trips(self) -> None:
        record = self._decision().to_audit_record()

        assert record["status"] == ADMITTED
        assert record["proposed_action"] == pytest.approx(0.3)
        assert record["barrier_value"] is None
        assert record["stages_applied"] == []
        assert len(record["content_hash"]) == 64  # type: ignore[arg-type]

    def test_hash_is_deterministic(self) -> None:
        assert self._decision().content_hash == self._decision().content_hash

    def test_hash_changes_with_content(self) -> None:
        other = _governor(max_rate=2.0).govern(0.4, _STATE, _DRIFT, previous_action=0.2)

        assert self._decision().content_hash != other.content_hash


class TestInputValidation:
    def test_rejects_non_real_proposal(self) -> None:
        with pytest.raises(ValueError, match="proposed_action"):
            _governor().govern("fast", _STATE, _DRIFT)  # type: ignore[arg-type]

    def test_rejects_non_finite_previous(self) -> None:
        with pytest.raises(ValueError, match="previous_action"):
            _governor().govern(0.2, _STATE, _DRIFT, previous_action=float("nan"))

    def test_rejects_boolean_state(self) -> None:
        with pytest.raises(ValueError, match="boolean"):
            _governor().govern(0.2, np.array([True]), _DRIFT)

    def test_rejects_complex_state(self) -> None:
        with pytest.raises(ValueError, match="real-valued"):
            _governor().govern(0.2, np.array([1 + 2j]), _DRIFT)

    def test_rejects_non_numeric_state(self) -> None:
        with pytest.raises(ValueError, match="real float array"):
            _governor().govern(0.2, np.array(["a", "b"]), _DRIFT)

    def test_rejects_two_dimensional_state(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            _governor().govern(0.2, np.array([[1.0, 2.0]]), _DRIFT)

    def test_rejects_non_finite_drift(self) -> None:
        with pytest.raises(ValueError, match="drift"):
            _governor().govern(0.2, _STATE, np.array([np.inf]))


class TestPipelineWiring:
    def test_governs_external_drive_into_the_engine(self) -> None:
        """An FM proposes the engine's external drive; the governor gates it."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 8
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(3)
        phases = 0.2 * rng.standard_normal(n)
        omegas = np.zeros(n)
        coupling = 3.0 * np.ones((n, n))
        np.fill_diagonal(coupling, 0.0)
        alpha = np.zeros((n, n))

        # Barrier on the mean-phase margin; governor bounds the drive strength.
        governor = _governor(
            control_lo=0.0,
            control_hi=2.0,
            max_rate=0.5,
            barrier_filter=_cbf(threshold=-1.0, lo=0.0, hi=2.0),
            safety_predicates=(
                ("drive_cap", lambda u, s: (u <= 1.5, "drive too strong")),
            ),
        )

        previous = 0.0
        proposals = [0.4, 0.9, 3.0, 1.2]  # last two trip rate-limit / predicate
        for proposed in proposals:
            margin = float(np.cos(phases).mean())
            decision = governor.govern(
                proposed, np.array([margin]), np.array([0.0]), previous_action=previous
            )
            assert 0.0 <= decision.admitted_action <= 2.0
            assert len(decision.content_hash) == 64
            phases = engine.step(
                phases, omegas, coupling, decision.admitted_action, 0.0, alpha
            )
            assert np.all(np.isfinite(phases))
            previous = decision.admitted_action

        # The over-cap proposal (3.0) must have been constrained, never admitted raw.
        capped = governor.govern(
            3.0, np.array([1.0]), np.array([0.0]), previous_action=1.0
        )
        assert capped.admitted_action <= 1.5 or capped.status == REJECTED
