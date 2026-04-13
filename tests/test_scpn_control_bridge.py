# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SCPN-control bridge tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.scpn_control_bridge import SCPNControlBridge
from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.upde.metrics import LayerState, LockSignature, UPDEState


def _make_state(n_layers: int = 3) -> UPDEState:
    layers = []
    for i in range(n_layers):
        layers.append(
            LayerState(
                R=0.8 + 0.05 * i,
                psi=float(i),
                mean_amplitude=1.0,
                lock_signatures={
                    j: LockSignature(
                        source_layer=i, target_layer=j, plv=0.9, mean_lag=0.01
                    )
                    for j in range(n_layers)
                    if j != i
                },
            )
        )
    return UPDEState(
        layers=layers,
        regime_id="coherent",
        stability_proxy=0.85,
        cross_layer_alignment=np.eye(n_layers),
    )


class TestImportKnm:
    def test_valid_square(self) -> None:
        bridge = SCPNControlBridge({"layers": 4})
        knm = np.ones((4, 4))
        np.fill_diagonal(knm, 0.0)
        cs = bridge.import_knm(knm)
        assert isinstance(cs, CouplingState)
        assert cs.knm.shape == (4, 4)
        assert cs.alpha.shape == (4, 4)
        assert cs.active_template == "scpn_import"
        np.testing.assert_array_equal(cs.alpha, 0.0)

    def test_non_square_raises(self) -> None:
        bridge = SCPNControlBridge({})
        with pytest.raises(ValueError, match="square"):
            bridge.import_knm(np.ones((3, 4)))

    def test_1d_raises(self) -> None:
        bridge = SCPNControlBridge({})
        with pytest.raises(ValueError, match="square"):
            bridge.import_knm(np.ones(5))

    def test_preserves_dtype(self) -> None:
        bridge = SCPNControlBridge({})
        knm_int = np.array([[0, 1], [1, 0]])
        cs = bridge.import_knm(knm_int)
        assert cs.knm.dtype == np.float64


class TestImportOmega:
    def test_valid_positive(self) -> None:
        bridge = SCPNControlBridge({})
        omega = np.array([1.0, 2.0, 3.0])
        result = bridge.import_omega(omega)
        np.testing.assert_array_equal(result, omega)
        assert result.dtype == np.float64

    def test_non_positive_raises(self) -> None:
        bridge = SCPNControlBridge({})
        with pytest.raises(ValueError, match="positive"):
            bridge.import_omega(np.array([1.0, 0.0, 3.0]))

    def test_negative_raises(self) -> None:
        bridge = SCPNControlBridge({})
        with pytest.raises(ValueError, match="positive"):
            bridge.import_omega(np.array([1.0, -1.0]))

    def test_2d_raises(self) -> None:
        bridge = SCPNControlBridge({})
        with pytest.raises(ValueError, match="1-D"):
            bridge.import_omega(np.ones((3, 2)))


class TestExportState:
    def test_keys_present(self) -> None:
        bridge = SCPNControlBridge({})
        state = _make_state(3)
        result = bridge.export_state(state)
        assert "regime" in result
        assert "stability" in result
        assert "layers" in result
        assert "cross_alignment" in result

    def test_regime_matches(self) -> None:
        bridge = SCPNControlBridge({})
        state = _make_state(2)
        result = bridge.export_state(state)
        assert result["regime"] == "coherent"
        assert result["stability"] == pytest.approx(0.85)

    def test_layers_count(self) -> None:
        bridge = SCPNControlBridge({})
        state = _make_state(4)
        result = bridge.export_state(state)
        assert len(result["layers"]) == 4

    def test_layer_fields(self) -> None:
        bridge = SCPNControlBridge({})
        state = _make_state(2)
        result = bridge.export_state(state)
        layer = result["layers"][0]
        assert "R" in layer
        assert "psi" in layer
        assert "locks" in layer

    def test_cross_alignment_serialised(self) -> None:
        bridge = SCPNControlBridge({})
        state = _make_state(3)
        result = bridge.export_state(state)
        assert isinstance(result["cross_alignment"], list)
        assert len(result["cross_alignment"]) == 3
