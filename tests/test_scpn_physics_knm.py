# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for physics-based K_nm builder

"""Declared SCPN hierarchy anchors and real handshake overlay admission."""

from __future__ import annotations

import json
from pathlib import Path

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

    def setup_method(self) -> None:
        """Construct a fresh sixteen-layer SCPN snapshot before each assertion."""
        self.builder = CouplingBuilder()
        self.state = self.builder.build_scpn_physics()

    def test_shape_16x16(self) -> None:
        """The fixed SCPN model publishes a sixteen-by-sixteen phase matrix."""
        assert self.state.knm.shape == (16, 16)

    def test_symmetric(self) -> None:
        """Each SCPN weight is identical in the reverse layer direction."""
        np.testing.assert_allclose(self.state.knm, self.state.knm.T, atol=1e-14)

    def test_zero_diagonal(self) -> None:
        """The SCPN hierarchy introduces no self-coupled layers."""
        np.testing.assert_allclose(np.diag(self.state.knm), 0.0)

    def test_anchors_exact(self) -> None:
        """Calibration anchors must appear exactly in the matrix."""
        K = self.state.knm
        for (n, m), expected in SCPN_CALIBRATION_ANCHORS.items():
            assert K[n - 1, m - 1] == pytest.approx(expected, abs=1e-10), (
                f"K[{n},{m}] = {K[n - 1, m - 1]}, expected {expected}"
            )
            assert K[m - 1, n - 1] == pytest.approx(expected, abs=1e-10), (
                f"K[{m},{n}] = {K[m - 1, n - 1]}, expected {expected} (symmetry)"
            )

    def test_cross_hierarchy_boosts(self) -> None:
        """Declared quantum-meta and psycho-symbolic lower bounds are retained."""
        K = self.state.knm
        # Quantum-Meta: K[1,16] >= 0.05
        assert K[0, 15] >= 0.05
        # Psycho-Symbolic: K[5,7] >= 0.15
        assert K[4, 6] >= 0.15

    def test_all_nonneg_except_diagonal(self) -> None:
        """Every off-diagonal SCPN phase weight is non-negative."""
        K = self.state.knm.copy()
        np.fill_diagonal(K, 0.0)
        assert np.all(K >= 0.0)

    def test_adjacent_stronger_than_distant(self) -> None:
        """Adjacent coupling should generally be stronger than distant."""
        K = self.state.knm
        # K[1,2] (adjacent) > K[1,8] (distant)
        assert K[0, 1] > K[0, 7]

    def test_near_neighbor_bounded(self) -> None:
        """Near-neighbor coupling clipped to [0.01, 0.4]."""
        K = self.state.knm
        for n in range(1, 15):
            m = n + 2
            val = K[n - 1, m - 1]
            assert 0.01 <= val <= 0.4, f"K[{n},{m}] = {val} out of bounds"

    def test_distant_bounded(self) -> None:
        """Distant coupling clipped to [0.001, 0.2]."""
        K = self.state.knm
        for n in range(1, 17):
            for m in range(n + 3, 17):
                val = K[n - 1, m - 1]
                # Cross-hierarchy boosts can exceed 0.2 via max()
                if (n, m) in {(1, 16), (5, 7)}:
                    continue
                assert 0.001 <= val <= 0.2, f"K[{n},{m}] = {val} out of bounds"

    def test_alpha_zeros(self) -> None:
        """The hierarchy begins with zero lag on every ordered layer pair."""
        np.testing.assert_allclose(self.state.alpha, 0.0)

    def test_active_template(self) -> None:
        """SCPN construction names the published topology scpn_physics."""
        assert self.state.active_template == "scpn_physics"

    def test_finite(self) -> None:
        """Default SCPN construction publishes only finite binary64 weights."""
        assert np.all(np.isfinite(self.state.knm))

    def test_matches_holonomic_atlas_anchors(self) -> None:
        """Cross-check: our anchors match HolonomicAtlas values."""
        # HolonomicAtlas: K[L1,L2]=0.302, K[L2,L3]=0.201, K[L3,L4]=0.252, K[L4,L5]=0.154
        K = self.state.knm
        assert K[0, 1] == pytest.approx(0.302)
        assert K[1, 2] == pytest.approx(0.201)
        assert K[2, 3] == pytest.approx(0.252)
        assert K[3, 4] == pytest.approx(0.154)

    def test_custom_k_base(self) -> None:
        """Increasing base strength strengthens an unanchored adjacent pair."""
        state2 = self.builder.build_scpn_physics(k_base=0.9)
        # Non-anchor adjacent should be stronger with higher k_base
        # L5-L6 (index 4,5) is not an anchor
        assert state2.knm[4, 5] > self.state.knm[4, 5]


class TestApplyHandshakes:
    """Tests for CouplingBuilder.apply_handshakes()."""

    def test_overlay_from_json(self, tmp_path: Path) -> None:
        """A handshake file applies reciprocal excitation and directed inhibition.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real handshake JSON document.
        """
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

    def test_active_template_name(self, tmp_path: Path) -> None:
        """An empty valid overlay still identifies the handshake topology.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real handshake JSON document.
        """
        builder = CouplingBuilder()
        base = builder.build_scpn_physics()
        spec = {"version": "2.0", "matrix": [], "statistics": {}}
        spec_path = tmp_path / "empty.json"
        spec_path.write_text(json.dumps(spec))
        result = builder.apply_handshakes(base, spec_path)
        assert result.active_template == "scpn_handshakes"

    def test_rejects_self_coupling_entry(self, tmp_path: Path) -> None:
        """A JSON handshake targeting its own layer is refused.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real handshake JSON document.
        """
        builder = CouplingBuilder()
        base = builder.build_scpn_physics()
        spec_path = tmp_path / "self_coupling.json"
        spec_path.write_text(
            json.dumps(
                {
                    "matrix": [
                        {
                            "from_layer": 4,
                            "to_layer": 4,
                            "coupling_strength": 0.2,
                        }
                    ]
                }
            )
        )

        with pytest.raises(ValueError, match="self-coupling"):
            builder.apply_handshakes(base, spec_path)

    def test_preserves_alpha(self, tmp_path: Path) -> None:
        """Overlay publication preserves the source phase-lag values.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real handshake JSON document.
        """
        builder = CouplingBuilder()
        base = builder.build_scpn_physics()
        spec = {"version": "2.0", "matrix": [], "statistics": {}}
        spec_path = tmp_path / "empty.json"
        spec_path.write_text(json.dumps(spec))
        result = builder.apply_handshakes(base, spec_path)
        np.testing.assert_allclose(result.alpha, base.alpha)

    @pytest.mark.parametrize(
        ("entry", "match"),
        [
            (
                {"from_layer": 0, "to_layer": 2, "coupling_strength": 0.1},
                "from_layer",
            ),
            (
                {"from_layer": 1, "to_layer": 17, "coupling_strength": 0.1},
                "to_layer",
            ),
            (
                {"from_layer": 1, "to_layer": 2, "coupling_strength": "bad"},
                "coupling_strength",
            ),
            (
                {"from_layer": True, "to_layer": 2, "coupling_strength": 0.1},
                "from_layer",
            ),
        ],
    )
    def test_rejects_malformed_entries(
        self, tmp_path: Path, entry: dict[str, object], match: str
    ) -> None:
        """Invalid layer identifiers, missing fields and bad strengths refuse overlays.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real handshake JSON document.
        entry : dict[str, object]
            Malformed JSON handshake entry whose original values are preserved.
        match : str
            Expected field or violated invariant in the admission diagnostic.
        """
        builder = CouplingBuilder()
        base = builder.build_scpn_physics()
        spec_path = tmp_path / "bad_handshakes.json"
        spec_path.write_text(json.dumps({"matrix": [entry]}))

        with pytest.raises(ValueError, match=match):
            builder.apply_handshakes(base, spec_path)

    def test_rejects_non_finite_json_constants(self, tmp_path: Path) -> None:
        """Non-standard JSON NaN and Infinity cannot become phase weights.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real handshake JSON document.
        """
        builder = CouplingBuilder()
        base = builder.build_scpn_physics()
        spec_path = tmp_path / "bad_handshakes.json"
        spec_path.write_text(
            '{"matrix":[{"from_layer":1,"to_layer":2,"coupling_strength":NaN}]}',
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="finite JSON"):
            builder.apply_handshakes(base, spec_path)


class TestScpnConstants:
    """Verify the exported constants are consistent."""

    def test_16_timescales(self) -> None:
        """The exported hierarchy has one timescale for each of its sixteen layers."""
        assert len(SCPN_LAYER_TIMESCALES) == 16

    def test_16_names(self) -> None:
        """Every declared hierarchy layer has a public display name."""
        assert len(SCPN_LAYER_NAMES) == 16

    def test_all_timescales_positive(self) -> None:
        """All exported default timescales satisfy positive-duration admission."""
        for layer, tau in SCPN_LAYER_TIMESCALES.items():
            assert tau > 0, f"Layer {layer} has non-positive timescale {tau}"

    def test_meta_layer_uses_operational_anchor_not_artificial_subsecond_scale(
        self,
    ) -> None:
        """The meta-layer retains its declared one-second operational anchor."""
        assert SCPN_LAYER_TIMESCALES[16] == pytest.approx(1.0)

    def test_anchors_within_layers_1_to_5(self) -> None:
        """Unit-bounded calibration anchors connect only layers one through five."""
        for (n, m), val in SCPN_CALIBRATION_ANCHORS.items():
            assert 1 <= n <= 5
            assert 1 <= m <= 5
            assert 0 < val < 1


class TestSCPNPhysicsKnmPipelineWiring:
    """Pipeline: SCPN physics K_nm → 16-oscillator engine → R."""

    def test_scpn_physics_knm_drives_engine(self) -> None:
        """build_scpn_physics → 16×16 K_nm → engine → R∈[0,1].

        Proves the physics-based coupling model feeds simulation.
        """
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        cs = CouplingBuilder().build_scpn_physics()
        n = cs.knm.shape[0]
        assert n == 16

        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = np.asarray(rng.uniform(0, 2 * np.pi, n), dtype=np.float64)
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
