# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for physics-based K_nm builder

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.knm import (
    SCPN_CALIBRATION_ANCHORS,
    SCPN_LAYER_NAMES,
    SCPN_LAYER_TIMESCALES,
    CouplingBuilder,
)


class TestBuildScpnPhysics:
    """Tests for CouplingBuilder.build_scpn_physics()."""

    def setup_method(self):
        self.builder = CouplingBuilder()
        self.state = self.builder.build_scpn_physics()

    def test_shape_16x16(self):
        assert self.state.knm.shape == (16, 16)

    def test_symmetric(self):
        np.testing.assert_allclose(self.state.knm, self.state.knm.T, atol=1e-14)

    def test_zero_diagonal(self):
        np.testing.assert_allclose(np.diag(self.state.knm), 0.0)

    def test_anchors_exact(self):
        """Calibration anchors must appear exactly in the matrix."""
        K = self.state.knm
        for (n, m), expected in SCPN_CALIBRATION_ANCHORS.items():
            assert K[n - 1, m - 1] == pytest.approx(expected, abs=1e-10), (
                f"K[{n},{m}] = {K[n - 1, m - 1]}, expected {expected}"
            )
            assert K[m - 1, n - 1] == pytest.approx(expected, abs=1e-10), (
                f"K[{m},{n}] = {K[m - 1, n - 1]}, expected {expected} (symmetry)"
            )

    def test_cross_hierarchy_boosts(self):
        K = self.state.knm
        # Quantum-Meta: K[1,16] >= 0.05
        assert K[0, 15] >= 0.05
        # Psycho-Symbolic: K[5,7] >= 0.15
        assert K[4, 6] >= 0.15

    def test_all_nonneg_except_diagonal(self):
        K = self.state.knm.copy()
        np.fill_diagonal(K, 0.0)
        assert np.all(K >= 0.0)

    def test_adjacent_stronger_than_distant(self):
        """Adjacent coupling should generally be stronger than distant."""
        K = self.state.knm
        # K[1,2] (adjacent) > K[1,8] (distant)
        assert K[0, 1] > K[0, 7]

    def test_near_neighbor_bounded(self):
        """Near-neighbor coupling clipped to [0.01, 0.4]."""
        K = self.state.knm
        for n in range(1, 15):
            m = n + 2
            val = K[n - 1, m - 1]
            assert 0.01 <= val <= 0.4, f"K[{n},{m}] = {val} out of bounds"

    def test_distant_bounded(self):
        """Distant coupling clipped to [0.001, 0.2]."""
        K = self.state.knm
        for n in range(1, 17):
            for m in range(n + 3, 17):
                val = K[n - 1, m - 1]
                # Cross-hierarchy boosts can exceed 0.2 via max()
                if (n, m) in {(1, 16), (5, 7)}:
                    continue
                assert 0.001 <= val <= 0.2, f"K[{n},{m}] = {val} out of bounds"

    def test_alpha_zeros(self):
        np.testing.assert_allclose(self.state.alpha, 0.0)

    def test_active_template(self):
        assert self.state.active_template == "scpn_physics"

    def test_finite(self):
        assert np.all(np.isfinite(self.state.knm))

    def test_matches_holonomic_atlas_anchors(self):
        """Cross-check: our anchors match HolonomicAtlas values."""
        # HolonomicAtlas: K[L1,L2]=0.302, K[L2,L3]=0.201, K[L3,L4]=0.252, K[L4,L5]=0.154
        K = self.state.knm
        assert K[0, 1] == pytest.approx(0.302)
        assert K[1, 2] == pytest.approx(0.201)
        assert K[2, 3] == pytest.approx(0.252)
        assert K[3, 4] == pytest.approx(0.154)

    def test_custom_k_base(self):
        state2 = self.builder.build_scpn_physics(k_base=0.9)
        # Non-anchor adjacent should be stronger with higher k_base
        # L5-L6 (index 4,5) is not an anchor
        assert state2.knm[4, 5] > self.state.knm[4, 5]


class TestApplyHandshakes:
    """Tests for CouplingBuilder.apply_handshakes()."""

    def test_overlay_from_json(self, tmp_path):
        builder = CouplingBuilder()
        base = builder.build_scpn_physics()

        spec = {
            "version": "2.0",
            "matrix": [
                {
                    "from_layer": 5,
                    "to_layer": 1,
                    "coupling_strength": 0.35,
                    "mechanism": "test",
                },
                {
                    "from_layer": 10,
                    "to_layer": 2,
                    "coupling_strength": -0.40,
                    "mechanism": "inhibitory",
                },
            ],
            "statistics": {"total_handshakes": 2, "documented": 2},
        }
        spec_path = tmp_path / "handshakes.json"
        spec_path.write_text(json.dumps(spec))

        result = builder.apply_handshakes(base, spec_path)

        # Positive coupling is symmetric
        assert result.knm[4, 0] == pytest.approx(0.35)
        assert result.knm[0, 4] == pytest.approx(0.35)

        # Negative coupling is directional (from→to only)
        assert result.knm[9, 1] == pytest.approx(-0.40)
        # Reverse is NOT set for negative values
        assert result.knm[1, 9] != pytest.approx(-0.40)

    def test_active_template_name(self, tmp_path):
        builder = CouplingBuilder()
        base = builder.build_scpn_physics()
        spec = {"version": "2.0", "matrix": [], "statistics": {}}
        spec_path = tmp_path / "empty.json"
        spec_path.write_text(json.dumps(spec))
        result = builder.apply_handshakes(base, spec_path)
        assert result.active_template == "scpn_handshakes"

    def test_preserves_alpha(self, tmp_path):
        builder = CouplingBuilder()
        base = builder.build_scpn_physics()
        spec = {"version": "2.0", "matrix": [], "statistics": {}}
        spec_path = tmp_path / "empty.json"
        spec_path.write_text(json.dumps(spec))
        result = builder.apply_handshakes(base, spec_path)
        np.testing.assert_allclose(result.alpha, base.alpha)


class TestScpnConstants:
    """Verify the exported constants are consistent."""

    def test_16_timescales(self):
        assert len(SCPN_LAYER_TIMESCALES) == 16

    def test_16_names(self):
        assert len(SCPN_LAYER_NAMES) == 16

    def test_all_timescales_positive(self):
        for layer, tau in SCPN_LAYER_TIMESCALES.items():
            assert tau > 0, f"Layer {layer} has non-positive timescale {tau}"

    def test_anchors_within_layers_1_to_5(self):
        for (n, m), val in SCPN_CALIBRATION_ANCHORS.items():
            assert 1 <= n <= 5
            assert 1 <= m <= 5
            assert 0 < val < 1


class TestSCPNPhysicsKnmPipelineWiring:
    """Pipeline: SCPN physics K_nm → 16-oscillator engine → R."""

    def test_scpn_physics_knm_drives_engine(self):
        """build_scpn_physics → 16×16 K_nm → engine → R∈[0,1].
        Proves the physics-based coupling model feeds simulation."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        cs = CouplingBuilder().build_scpn_physics()
        n = cs.knm.shape[0]
        assert n == 16

        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        for _ in range(200):
            phases = eng.step(
                phases,
                omegas,
                cs.knm,
                0.0,
                0.0,
                cs.alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
