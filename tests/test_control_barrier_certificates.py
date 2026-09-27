# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CBF certificate admission tests

"""Exercise certificate binding through the filter and its runtime consumers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.control_barrier import (
    BarrierCertificate,
    ControlBarrierFilter,
    NeuralBarrier,
)
from scpn_phase_orchestrator.actuation.foundation_model_governor import (
    CONSTRAINED,
    FoundationModelGovernor,
)
from scpn_phase_orchestrator.supervisor.cbf_admission import PolicyCBFChannel


@pytest.fixture
def certified_filter() -> tuple[ControlBarrierFilter, BarrierCertificate]:
    """Certify the supported one-state coherence-floor actuator envelope."""
    barrier = NeuralBarrier(weights=(np.array([[1.0]]),), biases=(np.array([-0.3]),))
    filt = ControlBarrierFilter(
        barrier=barrier,
        gamma=0.5,
        control_lo=0.0,
        control_hi=1.0,
        control_effect=np.array([1.0]),
    )
    certificate = filt.verify_forward_invariance(
        np.array([0.0]),
        np.array([1.0]),
        np.array([-0.1]),
        np.array([0.1]),
        cells_per_axis=16,
    )
    assert certificate.verified
    assert certificate.boundary_cells > 0
    return filt, certificate


@pytest.mark.parametrize("consumer", ["filter", "governor", "policy"])
@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("unverified", "barrier_certificate must be verified"),
        ("missing_digest", "barrier_certificate must carry a filter_digest"),
        ("foreign_filter", "barrier_certificate does not match barrier_filter"),
        ("gamma_drift", "barrier_certificate gamma does not match barrier_filter"),
    ],
)
def test_certificate_metadata_refused_at_runtime_admission(
    certified_filter: tuple[ControlBarrierFilter, BarrierCertificate],
    consumer: str,
    damage: str,
    message: str,
) -> None:
    """A retained certificate cannot admit altered or missing binding metadata."""
    filt, certificate = certified_filter
    if damage == "unverified":
        damaged = replace(certificate, verified=False)
    elif damage == "missing_digest":
        damaged = replace(certificate, filter_digest="")
    elif damage == "foreign_filter":
        other = replace(filt, control_hi=0.5)
        damaged = replace(certificate, filter_digest=other.filter_digest)
    else:
        assert damage == "gamma_drift"
        damaged = replace(certificate, gamma=1.0)

    with pytest.raises(ValueError, match=f"^{message}$"):
        if consumer == "filter":
            filt.validate_certificate(damaged)
        elif consumer == "governor":
            FoundationModelGovernor(
                control_lo=0.0,
                control_hi=1.0,
                max_rate=1.0,
                barrier_filter=filt,
                barrier_certificate=damaged,
            )
        else:
            assert consumer == "policy"
            PolicyCBFChannel(
                knob="K",
                scope="global",
                barrier_filter=filt,
                barrier_certificate=damaged,
                state_metrics=("R",),
                drift_bounds=(-0.1,),
            )

    filt.validate_certificate(certificate)
    assert certificate.filter_digest == filt.filter_digest
    assert certificate.gamma == filt.gamma


def test_matching_certificate_admits_barrier_constrained_control(
    certified_filter: tuple[ControlBarrierFilter, BarrierCertificate],
) -> None:
    """The original certificate still permits a real constrained runtime action."""
    filt, certificate = certified_filter
    channel = PolicyCBFChannel(
        knob="K",
        scope="global",
        barrier_filter=filt,
        barrier_certificate=certificate,
        state_metrics=("R",),
        drift_bounds=(-0.1,),
    )
    assert channel.barrier_certificate is certificate
    governor = FoundationModelGovernor(
        control_lo=0.0,
        control_hi=1.0,
        max_rate=1.0,
        barrier_filter=filt,
        barrier_certificate=certificate,
    )
    decision = governor.govern(0.0, np.array([0.35]), np.array([-0.1]))
    assert decision.status == CONSTRAINED
    assert decision.admitted_action == pytest.approx(0.075)
    assert decision.barrier_value == pytest.approx(0.05)
    assert decision.stages_applied == ("cbf",)
    assert not decision.violations
