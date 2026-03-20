# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Coverage gap tests — exercise remaining uncovered branches

from __future__ import annotations

import logging

import numpy as np
import pytest

from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    BindingSpec,
    BoundaryDef,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ObjectivePartition,
    OscillatorFamily,
)

# ---------------------------------------------------------------------------
# Item 6: audit/replay.py — SL replay warning + amplitude paths
# ---------------------------------------------------------------------------


class TestStuartLandauReplayPaths:
    """Cover verify_determinism_sl_chained branches: amplitude concat,
    legacy mu format, and missing-field warning skip."""

    @pytest.fixture()
    def replay_engine(self, tmp_path):
        from scpn_phase_orchestrator.audit.replay import ReplayEngine

        log = tmp_path / "audit.jsonl"
        log.write_text("")
        return ReplayEngine(log)

    @pytest.fixture()
    def sl_engine(self):
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        return StuartLandauEngine(n_oscillators=2, dt=0.01)

    def _make_sl_step(self, engine, state, omegas, mu, knm, knm_r, alpha):
        return engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

    def test_amplitude_field_path(self, replay_engine, sl_engine):
        """Cover lines 157-160 (amplitudes in curr) and 193-196 (amplitudes in nxt)."""
        n = 2
        omegas = [1.0, 1.0]
        mu = [0.5, 0.5]
        knm = np.zeros((n, n)).tolist()
        knm_r = np.zeros((n, n)).tolist()
        alpha = np.zeros((n, n)).tolist()
        phases0 = [0.1, 0.2]
        amps0 = [0.7, 0.8]
        state0 = np.array(phases0 + amps0)
        nxt_state = sl_engine.step(
            state0,
            np.array(omegas),
            np.array(mu),
            np.array(knm).reshape(n, n),
            np.array(knm_r).reshape(n, n),
            0.0,
            0.0,
            np.array(alpha).reshape(n, n),
        )
        entries = [
            {
                "phases": phases0,
                "amplitudes": amps0,
                "omegas": omegas,
                "mu": mu,
                "knm": np.array(knm).ravel().tolist(),
                "knm_r": np.array(knm_r).ravel().tolist(),
                "alpha": np.array(alpha).ravel().tolist(),
            },
            {
                "phases": nxt_state[:n].tolist(),
                "amplitudes": nxt_state[n:].tolist(),
                "omegas": omegas,
                "mu": mu,
                "knm": np.array(knm).ravel().tolist(),
                "knm_r": np.array(knm_r).ravel().tolist(),
                "alpha": np.array(alpha).ravel().tolist(),
            },
        ]
        ok, verified = replay_engine.verify_determinism_sl_chained(
            sl_engine,
            entries,
        )
        assert ok
        assert verified == 1

    def test_missing_amplitude_fields_skips(self, replay_engine, sl_engine, caplog):
        """Cover lines 165-166: warning + continue when neither amplitudes nor mu."""
        entries = [
            {
                "phases": [0.1, 0.2],
                "omegas": [1.0, 1.0],
                "knm": [0.0] * 4,
                "alpha": [0.0] * 4,
            },
            {
                "phases": [0.3, 0.4],
                "omegas": [1.0, 1.0],
                "knm": [0.0] * 4,
                "alpha": [0.0] * 4,
            },
        ]
        with caplog.at_level(logging.WARNING):
            ok, verified = replay_engine.verify_determinism_sl_chained(
                sl_engine,
                entries,
            )
        assert ok
        assert verified == 0

    def test_next_entry_without_amplitudes(self, replay_engine, sl_engine):
        """Cover line 194 branch: nxt has no 'amplitudes' key (legacy)."""
        n = 2
        omegas = [1.0, 1.0]
        phases0 = [0.1, 0.2]
        amps0 = [0.7, 0.8]
        np.zeros((n, n)).tolist()
        np.zeros((n, n)).tolist()
        np.zeros((n, n)).tolist()
        mu_val = [0.5, 0.5]
        state0 = np.array(phases0 + amps0)
        nxt_state = sl_engine.step(
            state0,
            np.array(omegas),
            np.array(mu_val),
            np.zeros((n, n)),
            np.zeros((n, n)),
            0.0,
            0.0,
            np.zeros((n, n)),
        )
        entries = [
            {
                "phases": phases0,
                "amplitudes": amps0,
                "omegas": omegas,
                "mu": mu_val,
                "knm": np.zeros(n * n).tolist(),
                "knm_r": np.zeros(n * n).tolist(),
                "alpha": np.zeros(n * n).tolist(),
            },
            {
                "phases": nxt_state.tolist(),
                "omegas": omegas,
                "mu": mu_val,
                "knm": np.zeros(n * n).tolist(),
                "alpha": np.zeros(n * n).tolist(),
            },
        ]
        ok, verified = replay_engine.verify_determinism_sl_chained(
            sl_engine,
            entries,
        )
        assert verified == 1


# ---------------------------------------------------------------------------
# Item 7: binding/types.py — get_omegas ValueError on length mismatch
# ---------------------------------------------------------------------------


class TestGetOmegasLengthMismatch:
    def test_omegas_length_mismatch_raises(self):
        spec = BindingSpec(
            name="test",
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[
                HierarchyLayer(
                    name="L1",
                    index=0,
                    oscillator_ids=["o1", "o2"],
                    omegas=[1.0],  # length 1, but 2 oscillators
                ),
            ],
            oscillator_families={},
            coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
            drivers=DriverSpec(physical={}, informational={}, symbolic={}),
            objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
            boundaries=[],
            actuators=[],
        )
        with pytest.raises(ValueError, match="omegas length 1 != oscillator count 2"):
            spec.get_omegas()


# ---------------------------------------------------------------------------
# Item 8: binding/validator.py:82 — boundary lower > upper
# ---------------------------------------------------------------------------


class TestValidatorBoundaryInverted:
    def test_boundary_lower_gt_upper(self):
        from scpn_phase_orchestrator.binding.validator import validate_binding_spec

        # BoundaryDef.__post_init__ prevents lower >= upper at construction,
        # so bypass it via object.__setattr__ on a frozen dataclass.
        bdef = BoundaryDef(
            name="bad_bound",
            variable="R",
            lower=0.2,
            upper=0.8,
            severity="hard",
        )
        object.__setattr__(bdef, "lower", 0.8)
        object.__setattr__(bdef, "upper", 0.2)
        spec = BindingSpec(
            name="test",
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[
                HierarchyLayer(name="L1", index=0, oscillator_ids=["o1"]),
            ],
            oscillator_families={},
            coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
            drivers=DriverSpec(physical={}, informational={}, symbolic={}),
            objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
            boundaries=[bdef],
            actuators=[],
        )
        errors = validate_binding_spec(spec)
        assert any("lower (0.8)" in e and "must be <= upper (0.2)" in e for e in errors)


# ---------------------------------------------------------------------------
# Item 9: monitor/session_start.py:87 — low initial coherence warning
# ---------------------------------------------------------------------------


class TestSessionStartLowCoherence:
    def test_low_coherence_warning(self):
        from scpn_phase_orchestrator.imprint.state import ImprintState
        from scpn_phase_orchestrator.monitor.session_start import check_session_start

        n_osc = 4
        # Widely spread phases → low R
        phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        imprint = ImprintState(m_k=np.ones(n_osc), last_update=0.0)
        report = check_session_start([], phases, imprint, n_osc)
        assert any("Low initial coherence" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Item 10: oscillators/init_phases.py — channel fallback paths
# ---------------------------------------------------------------------------


class TestInitPhasesFallbacks:
    def test_unknown_channel_random_fallback(self):
        """Cover line 75: channel not P/I/S → random uniform."""
        from scpn_phase_orchestrator.oscillators.init_phases import (
            extract_initial_phases,
        )

        spec = BindingSpec(
            name="test",
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[
                HierarchyLayer(
                    name="L1",
                    index=0,
                    oscillator_ids=["o1"],
                    family="exotic",
                ),
            ],
            oscillator_families={
                "exotic": OscillatorFamily(
                    channel="X",
                    extractor_type="hilbert",
                    config={},
                ),
            },
            coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
            drivers=DriverSpec(physical={}, informational={}, symbolic={}),
            objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
            boundaries=[],
            actuators=[],
        )
        omegas = np.array([1.0])
        phases = extract_initial_phases(spec, omegas)
        assert phases.shape == (1,)
        assert 0.0 <= phases[0] < 2 * np.pi

    def test_resolve_channel_no_families(self):
        """Cover line 96: empty families → default 'P'."""
        from scpn_phase_orchestrator.oscillators.init_phases import _resolve_channel

        assert _resolve_channel(None, {}, 0) == "P"

    def test_get_n_states_no_symbolic(self):
        """Cover line 106: no symbolic family → default 4."""
        from scpn_phase_orchestrator.oscillators.init_phases import _get_n_states

        families = {
            "phys": OscillatorFamily(
                channel="P",
                extractor_type="hilbert",
                config={},
            ),
        }
        assert _get_n_states(families) == 4


# ---------------------------------------------------------------------------
# Item 11: server.py:113 — non-amplitude-mode step path
# ---------------------------------------------------------------------------


class TestServerNonAmplitudeStep:
    def test_step_without_amplitude_mode(self):
        """Cover line 113: the else branch when amplitude_mode is False."""
        from scpn_phase_orchestrator.server import SimulationState

        spec = BindingSpec(
            name="test",
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[
                HierarchyLayer(name="L1", index=0, oscillator_ids=["o1", "o2"]),
            ],
            oscillator_families={
                "phys": OscillatorFamily(
                    channel="P",
                    extractor_type="hilbert",
                    config={},
                ),
            },
            coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
            drivers=DriverSpec(physical={}, informational={}, symbolic={}),
            objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
            boundaries=[],
            actuators=[
                ActuatorMapping(
                    name="K_g",
                    knob="K",
                    scope="global",
                    limits=(0.0, 1.0),
                ),
            ],
        )
        sim = SimulationState(spec)
        assert not sim.amplitude_mode
        result = sim.step()
        assert result["step"] == 1
        assert "R_global" in result


# ---------------------------------------------------------------------------
# Item 12: upde/engine.py — NaN validation + RK45 exhaust-retry path
# ---------------------------------------------------------------------------


class TestUPDEEngineEdges:
    def test_alpha_nan_raises(self):
        """Cover line 110: alpha contains NaN/Inf."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(2, dt=0.01)
        phases = np.array([0.1, 0.2])
        omegas = np.array([1.0, 1.0])
        knm = np.zeros((2, 2))
        alpha = np.array([[0.0, float("nan")], [0.0, 0.0]])
        with pytest.raises(ValueError, match="alpha contains NaN"):
            eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

    def test_rk45_exhaust_retries(self):
        """Cover lines 258-264: RK45 exhausting max_reject iterations.

        Use extreme coupling to force large error norms on every retry.
        """
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(4, dt=1.0, method="rk45", atol=1e-15, rtol=1e-15)
        phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        omegas = np.array([100.0, 200.0, 300.0, 400.0])
        knm = np.full((4, 4), 1000.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((4, 4))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Item 14: upde/order_params.py — empty array guards
# ---------------------------------------------------------------------------


class TestOrderParamsEmptyGuards:
    def test_order_parameter_empty(self):
        """Cover line 25: empty phases → (0.0, 0.0)."""
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        r, psi = compute_order_parameter(np.array([]))
        assert r == 0.0
        assert psi == 0.0

    def test_plv_empty(self):
        """Cover line 44: empty arrays → 0.0."""
        from scpn_phase_orchestrator.upde.order_params import compute_plv

        assert compute_plv(np.array([]), np.array([1.0])) == 0.0
        assert compute_plv(np.array([1.0]), np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Item 15: upde/pac.py:23 — n_bins < 2 guard
# ---------------------------------------------------------------------------


class TestPACGuard:
    def test_modulation_index_nbins_lt_2(self):
        """Cover line 23: n_bins < 2 → 0.0."""
        from scpn_phase_orchestrator.upde.pac import modulation_index

        theta = np.array([0.1, 0.2, 0.3])
        amp = np.array([1.0, 2.0, 3.0])
        assert modulation_index(theta, amp, n_bins=1) == 0.0
        assert modulation_index(theta, amp, n_bins=0) == 0.0


# ---------------------------------------------------------------------------
# Item 16: upde/stuart_landau.py — NaN validation + RK45 exhaust
# ---------------------------------------------------------------------------


class TestStuartLandauEdges:
    def test_state_nan_raises(self):
        """Cover line 170: state contains NaN."""
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        eng = StuartLandauEngine(2, dt=0.01)
        state = np.array([0.1, 0.2, float("nan"), 0.8])
        omegas = np.array([1.0, 1.0])
        mu = np.array([0.5, 0.5])
        knm = np.zeros((2, 2))
        knm_r = np.zeros((2, 2))
        alpha = np.zeros((2, 2))
        with pytest.raises(ValueError, match="state contains NaN"):
            eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

    def test_omegas_nan_raises(self):
        """Cover line 172."""
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        eng = StuartLandauEngine(2, dt=0.01)
        state = np.array([0.1, 0.2, 0.7, 0.8])
        omegas = np.array([float("nan"), 1.0])
        mu = np.array([0.5, 0.5])
        knm = np.zeros((2, 2))
        knm_r = np.zeros((2, 2))
        alpha = np.zeros((2, 2))
        with pytest.raises(ValueError, match="omegas contain NaN"):
            eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

    def test_mu_nan_raises(self):
        """Cover line 174."""
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        eng = StuartLandauEngine(2, dt=0.01)
        state = np.array([0.1, 0.2, 0.7, 0.8])
        omegas = np.array([1.0, 1.0])
        mu = np.array([float("inf"), 0.5])
        knm = np.zeros((2, 2))
        knm_r = np.zeros((2, 2))
        alpha = np.zeros((2, 2))
        with pytest.raises(ValueError, match="mu contains NaN"):
            eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

    def test_knm_nan_raises(self):
        """Cover line 176."""
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        eng = StuartLandauEngine(2, dt=0.01)
        state = np.array([0.1, 0.2, 0.7, 0.8])
        omegas = np.array([1.0, 1.0])
        mu = np.array([0.5, 0.5])
        knm = np.array([[0.0, float("nan")], [0.0, 0.0]])
        knm_r = np.zeros((2, 2))
        alpha = np.zeros((2, 2))
        with pytest.raises(ValueError, match="knm contains NaN"):
            eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

    def test_knm_r_nan_raises(self):
        """Cover line 178."""
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        eng = StuartLandauEngine(2, dt=0.01)
        state = np.array([0.1, 0.2, 0.7, 0.8])
        omegas = np.array([1.0, 1.0])
        mu = np.array([0.5, 0.5])
        knm = np.zeros((2, 2))
        knm_r = np.array([[0.0, float("inf")], [0.0, 0.0]])
        alpha = np.zeros((2, 2))
        with pytest.raises(ValueError, match="knm_r contains NaN"):
            eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

    def test_alpha_nan_raises(self):
        """Cover line 180."""
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        eng = StuartLandauEngine(2, dt=0.01)
        state = np.array([0.1, 0.2, 0.7, 0.8])
        omegas = np.array([1.0, 1.0])
        mu = np.array([0.5, 0.5])
        knm = np.zeros((2, 2))
        knm_r = np.zeros((2, 2))
        alpha = np.array([[0.0, float("nan")], [0.0, 0.0]])
        with pytest.raises(ValueError, match="alpha contains NaN"):
            eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

    def test_epsilon_nan_raises(self):
        """Cover line 182."""
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        eng = StuartLandauEngine(2, dt=0.01)
        state = np.array([0.1, 0.2, 0.7, 0.8])
        omegas = np.array([1.0, 1.0])
        mu = np.array([0.5, 0.5])
        knm = np.zeros((2, 2))
        knm_r = np.zeros((2, 2))
        alpha = np.zeros((2, 2))
        with pytest.raises(ValueError, match="epsilon must be finite"):
            eng.step(
                state,
                omegas,
                mu,
                knm,
                knm_r,
                0.0,
                0.0,
                alpha,
                epsilon=float("nan"),
            )

    def test_rk45_exhaust_retries(self):
        """Cover lines 264-268: SL RK45 exhausting max_reject iterations."""
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        eng = StuartLandauEngine(4, dt=1.0, method="rk45", atol=1e-15, rtol=1e-15)
        state = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 1.0, 1.0, 1.0, 1.0])
        omegas = np.array([100.0, 200.0, 300.0, 400.0])
        mu = np.array([10.0, 10.0, 10.0, 10.0])
        knm = np.full((4, 4), 1000.0)
        np.fill_diagonal(knm, 0.0)
        knm_r = np.full((4, 4), 1000.0)
        np.fill_diagonal(knm_r, 0.0)
        alpha = np.zeros((4, 4))
        result = eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)
        assert result.shape == (8,)
