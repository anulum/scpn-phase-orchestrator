# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — regime classification of non-finite coherence

"""A non-finite mean order parameter is classified as CRITICAL.

A diverged simulation or a failed monitor yields ``R = NaN``. Every threshold
comparison is false for NaN, so the Python ``RegimeManager.evaluate`` fell
through to NOMINAL (or RECOVERY from CRITICAL) and the supervisor saw a healthy
state. The Rust kernel's ``classify_regime_from_summary`` returns Critical for
a non-finite mean; the Python path now agrees.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


def _state(*values: float) -> UPDEState:
    """Return a UPDE state whose layers carry the order parameters ``values``."""
    return UPDEState(
        layers=[LayerState(R=value, psi=0.0) for value in values],
        cross_layer_alignment=np.zeros((len(values), len(values))),
        stability_proxy=0.0,
        regime_id="probe",
    )


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize("current", list(Regime))
def test_non_finite_mean_r_proposes_critical(value: float, current: Regime) -> None:
    """From every regime, a non-finite mean R proposes CRITICAL."""
    manager = RegimeManager()
    manager.force_transition(current)

    assert manager.evaluate(_state(0.9, value), BoundaryState()) is Regime.CRITICAL


def test_finite_readings_keep_their_classification() -> None:
    """Finite readings are classified by the thresholds as before."""
    manager = RegimeManager()

    assert manager.evaluate(_state(0.9, 0.85), BoundaryState()) is Regime.NOMINAL
    assert manager.evaluate(_state(0.5, 0.45), BoundaryState()) is Regime.DEGRADED
    assert manager.evaluate(_state(0.1, 0.2), BoundaryState()) is Regime.CRITICAL


@pytest.mark.parametrize("value", NON_FINITE)
def test_python_and_rust_agree_on_non_finite_mean_r(value: float) -> None:
    """The Python and Rust regime managers give the same verdict."""
    try:
        kernel = importlib.import_module("spo_kernel")
    except ImportError:
        pytest.skip("spo_kernel is not installed")
    rust_manager = kernel.PyRegimeManager(cooldown_steps=0)

    rust_verdict = rust_manager.evaluate([0.9, value], [])
    python_verdict = RegimeManager().evaluate(_state(0.9, value), BoundaryState())

    assert rust_verdict == python_verdict.value == "critical"
