# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CBF admission rate-limit reference tests

"""The CBF admission gate rate-limits against the last admitted value.

``PolicyCBFChannel`` is frozen, and its ``previous_action`` was the rate-limit
reference for every decision. A knob ramped in steps within ``max_rate`` was
held at one step above the constructor value for the whole run, and a small
decrease after an admitted value was reported as constrained.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.control_barrier import (
    ControlBarrierFilter,
    NeuralBarrier,
)
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.cbf_admission import (
    PolicyCBFAdmissionGate,
    PolicyCBFChannel,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

SAFE_STATE = UPDEState(
    layers=[LayerState(R=0.8, psi=0.0), LayerState(R=0.9, psi=0.0)],
    cross_layer_alignment=np.eye(2),
    stability_proxy=0.85,
    regime_id="nominal",
)


def _gate(*, max_rate: float, previous_action: float = 0.0) -> PolicyCBFAdmissionGate:
    """Return a zeta gate over a verified scalar CBF on ``R_min``."""
    barrier = NeuralBarrier(
        weights=(np.array([[1.0]], dtype=np.float64),),
        biases=(np.array([0.0], dtype=np.float64),),
    )
    cbf = ControlBarrierFilter(
        barrier=barrier,
        gamma=0.5,
        control_lo=0.0,
        control_hi=1.0,
        control_effect=np.array([1.0], dtype=np.float64),
    )
    certificate = cbf.verify_forward_invariance(
        np.array([-1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([-0.5], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
        cells_per_axis=8,
        boundary_shell=0.25,
    )
    channel = PolicyCBFChannel(
        knob="zeta",
        scope="global",
        barrier_filter=cbf,
        barrier_certificate=certificate,
        state_metrics=("R_min",),
        drift_bounds=(-0.5,),
        previous_action=previous_action,
        max_rate=max_rate,
    )
    return PolicyCBFAdmissionGate((channel,))


def _admit(gate: PolicyCBFAdmissionGate, value: float) -> tuple[float, str]:
    """Admit one zeta proposal and return the admitted value and status."""
    result = gate.admit_actions(
        [ControlAction("zeta", "global", value, 1.0, "ramp")],
        SAFE_STATE,
        BoundaryState(),
    )
    (record,) = result.records
    return record.admitted_value, record.status


def test_ramp_within_rate_is_admitted_step_by_step() -> None:
    """Each step within ``max_rate`` of the last admitted value is admitted."""
    gate = _gate(max_rate=0.1)

    decisions = [_admit(gate, value) for value in (0.1, 0.2, 0.3, 0.4)]

    assert [value for value, _ in decisions] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert [status for _, status in decisions] == ["admitted"] * 4


def test_jump_is_limited_relative_to_the_last_admitted_value() -> None:
    """A jump is cut to ``max_rate`` from the last admitted value."""
    gate = _gate(max_rate=0.1)
    _admit(gate, 0.1)
    _admit(gate, 0.2)

    value, status = _admit(gate, 0.9)

    assert status == "constrained"
    assert value == pytest.approx(0.3)


def test_small_decrease_after_an_admitted_value_is_admitted() -> None:
    """Moving back within ``max_rate`` of the last admitted value is admitted.

    The barrier requires ``u >= 0.1`` in this state (``R_min = 0.8``, drift
    ``-0.5``, ``gamma = 0.5``), so the decrease stays above that bound.
    """
    gate = _gate(max_rate=0.1)
    _admit(gate, 0.1)
    _admit(gate, 0.2)

    value, status = _admit(gate, 0.15)

    assert status == "admitted"
    assert value == pytest.approx(0.15)


def test_first_decision_uses_the_channel_previous_action() -> None:
    """The channel's ``previous_action`` is the reference for the first call."""
    gate = _gate(max_rate=0.1, previous_action=0.5)

    value, status = _admit(gate, 0.9)

    assert status == "constrained"
    assert value == pytest.approx(0.6)
