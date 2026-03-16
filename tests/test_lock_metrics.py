# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lock metrics tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.metrics import LayerState, LockSignature, UPDEState


def test_lock_signature_creation():
    sig = LockSignature(source_layer=0, target_layer=1, plv=0.95, mean_lag=0.02)
    assert sig.plv == 0.95
    assert sig.source_layer == 0
    assert sig.target_layer == 1
    assert sig.mean_lag == 0.02


def test_layer_state_r_in_range():
    ls = LayerState(R=0.75, psi=1.2)
    assert 0.0 <= ls.R <= 1.0
    assert ls.psi == 1.2


def test_layer_state_with_signatures():
    sig = LockSignature(source_layer=0, target_layer=1, plv=0.88, mean_lag=0.01)
    ls = LayerState(R=0.5, psi=0.0, lock_signatures={"0_1": sig})
    assert "0_1" in ls.lock_signatures
    assert ls.lock_signatures["0_1"].plv == 0.88


def test_upde_state_construction():
    layers = [
        LayerState(R=0.9, psi=0.1),
        LayerState(R=0.7, psi=0.5),
        LayerState(R=0.3, psi=1.0),
    ]
    alignment = np.eye(3)
    state = UPDEState(
        layers=layers,
        cross_layer_alignment=alignment,
        stability_proxy=0.8,
        regime_id="nominal",
    )
    assert len(state.layers) == 3
    assert state.stability_proxy == 0.8
    assert state.regime_id == "nominal"
    np.testing.assert_array_equal(state.cross_layer_alignment, np.eye(3))


def test_upde_state_empty_layers():
    state = UPDEState(
        layers=[],
        cross_layer_alignment=np.array([]),
        stability_proxy=0.0,
        regime_id="critical",
    )
    assert len(state.layers) == 0
    assert state.regime_id == "critical"
