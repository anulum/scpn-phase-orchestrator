# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for supervisor CBF action admission

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.control_barrier import (
    BarrierCertificate,
    ControlBarrierFilter,
    NeuralBarrier,
)
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.cbf_admission import (
    PolicyCBFAdmissionGate,
    PolicyCBFChannel,
)
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _state(r_values: list[float]) -> UPDEState:
    layers = [LayerState(R=value, psi=0.0) for value in r_values]
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(len(layers)),
        stability_proxy=float(np.mean(r_values)),
        regime_id="nominal",
    )


def _empty_state() -> UPDEState:
    return UPDEState(
        layers=[],
        cross_layer_alignment=np.empty((0, 0)),
        stability_proxy=0.0,
        regime_id="nominal",
    )


def _cbf(threshold: float = 0.0) -> ControlBarrierFilter:
    barrier = NeuralBarrier(
        weights=(np.array([[1.0]], dtype=np.float64),),
        biases=(np.array([-threshold], dtype=np.float64),),
    )
    return ControlBarrierFilter(
        barrier=barrier,
        gamma=0.5,
        control_lo=0.0,
        control_hi=1.0,
        control_effect=np.array([1.0], dtype=np.float64),
    )


def _certificate(cbf: ControlBarrierFilter) -> BarrierCertificate:
    return cbf.verify_forward_invariance(
        np.array([-1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([-0.5], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
        cells_per_axis=8,
        boundary_shell=0.25,
    )


def _gate(
    cbf: ControlBarrierFilter | None = None,
    *,
    previous_action: float = 0.0,
    state_metrics: tuple[str, ...] = ("R_min",),
    drift_bounds: tuple[float, ...] = (-0.5,),
    max_rate: float | None = None,
) -> PolicyCBFAdmissionGate:
    filt = _cbf() if cbf is None else cbf
    return PolicyCBFAdmissionGate(
        (
            PolicyCBFChannel(
                knob="zeta",
                scope="global",
                barrier_filter=filt,
                barrier_certificate=_certificate(filt),
                state_metrics=state_metrics,
                drift_bounds=drift_bounds,
                previous_action=previous_action,
                max_rate=max_rate,
            ),
        )
    )


def test_supervisor_policy_constrains_zeta_through_verified_cbf() -> None:
    policy = SupervisorPolicy(
        RegimeManager(cooldown_steps=0),
        admission_gate=_gate(),
    )

    actions = policy.decide(_state([0.05, 0.2]), BoundaryState())
    zeta = next(action for action in actions if action.knob == "zeta")
    records = policy.last_admission_records

    assert zeta.value == pytest.approx(0.475, abs=1.0e-6)
    assert "CBF constrained" in zeta.justification
    assert len(records) == 1
    assert records[0].status == "constrained"
    assert records[0].proposed_value == pytest.approx(0.1)
    assert records[0].admitted_value == pytest.approx(0.475, abs=1.0e-6)
    assert records[0].smt_artifact.artifact_type == "smt2"
    assert "(check-sat)" in records[0].smt_artifact.text
    assert records[0].smt_artifact_hash in records[0].to_audit_record().values()


def test_supervisor_policy_rejects_action_when_state_is_outside_cbf_set() -> None:
    policy = SupervisorPolicy(
        RegimeManager(cooldown_steps=0),
        admission_gate=_gate(previous_action=0.0),
    )

    actions = policy.decide(_state([-0.2, 0.1]), BoundaryState())
    zeta = next(action for action in actions if action.knob == "zeta")
    record = policy.last_admission_records[0]

    assert zeta.value == pytest.approx(0.0)
    assert "CBF rejected" in zeta.justification
    assert record.status == "rejected"
    assert any("outside certified safe set" in item for item in record.violations)


def test_gate_leaves_unmatched_actions_unchanged_without_audit_noise() -> None:
    gate = _gate()
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.05,
        ttl_s=10.0,
        justification="degraded: boost global coupling",
    )

    result = gate.admit_actions((action,), _state([0.4, 0.5]), BoundaryState())

    assert result.actions == (action,)
    assert result.records == ()


def test_channel_requires_matching_verified_certificate() -> None:
    cbf = _cbf()
    bad_certificate = BarrierCertificate(
        verified=False,
        cells_checked=1,
        boundary_cells=1,
        worst_margin=-1.0,
        boundary_shell=0.25,
        gamma=cbf.gamma,
        filter_digest=cbf.filter_digest,
        verification_digest="bad",
    )

    with pytest.raises(ValueError, match="must be verified"):
        PolicyCBFChannel(
            knob="zeta",
            scope="global",
            barrier_filter=cbf,
            barrier_certificate=bad_certificate,
            state_metrics=("R_min",),
            drift_bounds=(-0.5,),
        )


def test_admission_record_hash_is_deterministic() -> None:
    gate = _gate()
    action = ControlAction(
        knob="zeta",
        scope="global",
        value=0.1,
        ttl_s=5.0,
        justification="critical: increase damping",
    )

    first = gate.admit_actions((action,), _state([0.05, 0.2]), BoundaryState())
    second = gate.admit_actions((action,), _state([0.05, 0.2]), BoundaryState())

    assert first.records[0].content_hash == second.records[0].content_hash
    assert len(first.records[0].content_hash) == 64


def test_result_audit_record_includes_admission_records() -> None:
    gate = _gate()
    action = ControlAction(
        knob="zeta",
        scope="global",
        value=0.1,
        ttl_s=5.0,
        justification="critical: increase damping",
    )

    result = gate.admit_actions((action,), _state([0.05, 0.2]), BoundaryState())
    record = result.to_audit_record()

    assert record["actions"] == 1
    assert isinstance(record["records"], list)
    assert record["records"][0]["status"] == "constrained"


def test_channel_admit_rejects_unmatched_action() -> None:
    cbf = _cbf()
    channel = PolicyCBFChannel(
        knob="zeta",
        scope="global",
        barrier_filter=cbf,
        barrier_certificate=_certificate(cbf),
        state_metrics=("R_min",),
        drift_bounds=(-0.5,),
    )

    with pytest.raises(ValueError, match="does not match"):
        channel.admit(
            ControlAction("K", "global", 0.1, 1.0, "wrong"),
            _state([0.4]),
            BoundaryState(),
        )


def test_channel_max_rate_limits_matching_action() -> None:
    action = ControlAction(
        knob="zeta",
        scope="global",
        value=0.8,
        ttl_s=5.0,
        justification="critical: increase damping",
    )

    result = _gate(max_rate=0.2, drift_bounds=(0.0,)).admit_actions(
        (action,), _state([0.9]), BoundaryState()
    )

    assert result.actions[0].value == pytest.approx(0.2)
    assert result.records[0].stages_applied == ("rate_limit",)


def test_supported_metrics_feed_multidimensional_barrier() -> None:
    barrier = NeuralBarrier(
        weights=(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),),
        biases=(np.array([0.0], dtype=np.float64),),
    )
    cbf = ControlBarrierFilter(
        barrier=barrier,
        gamma=0.5,
        control_lo=0.0,
        control_hi=1.0,
        control_effect=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    certificate = cbf.verify_forward_invariance(
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        cells_per_axis=1,
        boundary_shell=1.0,
    )
    gate = PolicyCBFAdmissionGate(
        (
            PolicyCBFChannel(
                knob="zeta",
                scope="global",
                barrier_filter=cbf,
                barrier_certificate=certificate,
                state_metrics=(
                    "R_mean",
                    "stability_proxy",
                    "violation_count",
                    "hard_violation_count",
                ),
                drift_bounds=(0.0, 0.0, 0.0, 0.0),
            ),
        )
    )
    action = ControlAction("zeta", "global", 1.0e-20, 5.0, "tiny")
    boundary = BoundaryState(violations=["soft"], hard_violations=["hard"])

    result = gate.admit_actions((action,), _state([0.7, 0.9]), boundary)

    assert result.records[0].status == "admitted"
    assert "1e-20" not in result.records[0].smt_artifact.text.lower()


def test_empty_r_mean_metric_uses_zero_state_value() -> None:
    action = ControlAction("zeta", "global", 0.1, 5.0, "critical")

    result = _gate(
        state_metrics=("R_mean",),
        drift_bounds=(0.0,),
    ).admit_actions((action,), _empty_state(), BoundaryState())

    assert result.records[0].barrier_value == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"knob": ""}, "knob"),
        ({"scope": ""}, "scope"),
        ({"state_metrics": ()}, "state_metrics length"),
        ({"drift_bounds": ()}, "drift_bounds length"),
        ({"previous_action": True}, "previous_action"),
        ({"previous_action": float("inf")}, "previous_action"),
        ({"max_rate": 0.0}, "max_rate"),
        ({"drift_bounds": (float("nan"),)}, "drift_bounds"),
    ],
)
def test_channel_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    cbf = _cbf()
    config: dict[str, object] = {
        "knob": "zeta",
        "scope": "global",
        "barrier_filter": cbf,
        "barrier_certificate": _certificate(cbf),
        "state_metrics": ("R_min",),
        "drift_bounds": (-0.5,),
    }
    config.update(kwargs)

    with pytest.raises(ValueError, match=message):
        PolicyCBFChannel(**config)  # type: ignore[arg-type]


def test_channel_rejects_multidimensional_drift_vector() -> None:
    cbf = _cbf()
    with pytest.raises(ValueError, match="one-dimensional"):
        PolicyCBFChannel(
            knob="zeta",
            scope="global",
            barrier_filter=cbf,
            barrier_certificate=_certificate(cbf),
            state_metrics=("R_min",),
            drift_bounds=((0.0,),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("channels", "message"),
    [
        ((), "at least one"),
        (("not-a-channel",), "only PolicyCBFChannel"),
    ],
)
def test_gate_rejects_invalid_channel_sets(
    channels: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PolicyCBFAdmissionGate(channels)  # type: ignore[arg-type]


def test_gate_rejects_duplicate_channels() -> None:
    cbf = _cbf()
    channel = PolicyCBFChannel(
        knob="zeta",
        scope="global",
        barrier_filter=cbf,
        barrier_certificate=_certificate(cbf),
        state_metrics=("R_min",),
        drift_bounds=(-0.5,),
    )

    with pytest.raises(ValueError, match="unique"):
        PolicyCBFAdmissionGate((channel, channel))


def test_gate_rejects_unsupported_state_metric() -> None:
    action = ControlAction("zeta", "global", 0.1, 5.0, "critical")

    with pytest.raises(ValueError, match="unsupported CBF admission metric"):
        _gate(state_metrics=("unknown_metric",)).admit_actions(
            (action,), _state([0.4]), BoundaryState()
        )
