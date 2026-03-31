# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Behavioural tests for edge cases across modules

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
# Stuart-Landau replay determinism with amplitude tracking
# ---------------------------------------------------------------------------


class TestStuartLandauReplayDeterminism:
    """Verify that SL replay engine reproduces exact state trajectories,
    including amplitude fields, under both new and legacy log formats."""

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

    def test_amplitude_replay_matches_forward_step(self, replay_engine, sl_engine):
        """Chained SL replay with amplitude fields must reproduce the engine's
        forward integration to machine precision."""
        n = 2
        omegas = [1.0, 2.0]
        mu = [0.5, 0.3]
        knm = np.array([[0.0, 0.1], [0.1, 0.0]])
        knm_r = np.array([[0.0, 0.05], [0.05, 0.0]])
        alpha = np.zeros((n, n))
        phases0 = [0.1, 0.8]
        amps0 = [0.7, 0.5]
        state0 = np.array(phases0 + amps0)

        nxt_state = sl_engine.step(
            state0,
            np.array(omegas),
            np.array(mu),
            knm,
            knm_r,
            0.0,
            0.0,
            alpha,
        )

        entries = [
            {
                "phases": phases0,
                "amplitudes": amps0,
                "omegas": omegas,
                "mu": mu,
                "knm": knm.ravel().tolist(),
                "knm_r": knm_r.ravel().tolist(),
                "alpha": alpha.ravel().tolist(),
            },
            {
                "phases": nxt_state[:n].tolist(),
                "amplitudes": nxt_state[n:].tolist(),
                "omegas": omegas,
                "mu": mu,
                "knm": knm.ravel().tolist(),
                "knm_r": knm_r.ravel().tolist(),
                "alpha": alpha.ravel().tolist(),
            },
        ]

        ok, verified = replay_engine.verify_determinism_sl_chained(sl_engine, entries)
        assert ok, "Replay must match forward integration exactly"
        assert verified == 1, "Exactly one transition should be verified"

        # Cross-validate: amplitudes must evolve toward sqrt(mu) under subcritical
        amps_next = nxt_state[n:]
        for i in range(n):
            equilibrium = np.sqrt(max(mu[i], 0.0))
            dist_before = abs(amps0[i] - equilibrium)
            dist_after = abs(amps_next[i] - equilibrium)
            assert dist_after <= dist_before + 1e-10, (
                f"Osc {i}: should converge toward "
                f"sqrt(mu)={equilibrium:.3f}, "
                f"dist {dist_before:.4f}→{dist_after:.4f}"
            )

    def test_missing_amplitude_fields_warns_and_skips(
        self,
        replay_engine,
        sl_engine,
        caplog,
    ):
        """Entries without amplitude or mu fields must emit a warning and
        skip verification — not silently pass or crash."""
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
        assert ok, "Should pass (nothing to fail on)"
        assert verified == 0, "No steps should be verified when fields are missing"
        assert any("missing" in msg.lower() for msg in caplog.messages), (
            "Must log a warning about missing amplitude fields"
        )

    def test_legacy_format_without_separate_amplitudes(
        self,
        replay_engine,
        sl_engine,
    ):
        """Legacy logs store full SL state [θ; r] in 'phases' with 'mu' present
        but no 'amplitudes' key. Verify replay handles this format correctly."""
        n = 2
        omegas = [1.0, 1.0]
        mu_val = [0.5, 0.5]
        phases0 = [0.1, 0.2]
        amps0 = [0.7, 0.8]
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
                "knm": [0.0] * (n * n),
                "knm_r": [0.0] * (n * n),
                "alpha": [0.0] * (n * n),
            },
            {
                # Legacy: no 'amplitudes' key, full state in 'phases'
                "phases": nxt_state.tolist(),
                "omegas": omegas,
                "mu": mu_val,
                "knm": [0.0] * (n * n),
                "alpha": [0.0] * (n * n),
            },
        ]
        ok, verified = replay_engine.verify_determinism_sl_chained(sl_engine, entries)
        assert verified == 1, "Legacy format must be replayed successfully"
        assert ok, "Legacy replay must match forward integration"

    def test_multi_step_chain_accumulates_correctly(self, replay_engine, sl_engine):
        """Chain 5 consecutive steps and verify all are deterministically replayed.
        This catches off-by-one errors in state handoff between steps."""
        n = 2
        omegas = np.array([1.5, 0.8])
        mu = np.array([0.4, 0.6])
        knm = np.array([[0.0, 0.2], [0.2, 0.0]])
        knm_r = np.zeros((n, n))
        alpha = np.zeros((n, n))
        state = np.array([0.0, np.pi, 0.5, 0.5])
        entries = []

        for _ in range(6):
            entries.append(
                {
                    "phases": state[:n].tolist(),
                    "amplitudes": state[n:].tolist(),
                    "omegas": omegas.tolist(),
                    "mu": mu.tolist(),
                    "knm": knm.ravel().tolist(),
                    "knm_r": knm_r.ravel().tolist(),
                    "alpha": alpha.ravel().tolist(),
                }
            )
            state = sl_engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

        ok, verified = replay_engine.verify_determinism_sl_chained(sl_engine, entries)
        assert ok, "All 5 transitions must replay deterministically"
        assert verified == 5, f"Expected 5 verified steps, got {verified}"


# ---------------------------------------------------------------------------
# BindingSpec omega validation
# ---------------------------------------------------------------------------


class TestBindingSpecOmegaValidation:
    """Verify that BindingSpec.get_omegas() enforces length consistency
    between declared oscillator_ids and explicit omega lists."""

    def _make_spec(self, oscillator_ids, omegas=None):
        return BindingSpec(
            name="test",
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[
                HierarchyLayer(
                    name="L1",
                    index=0,
                    oscillator_ids=oscillator_ids,
                    omegas=omegas,
                )
            ],
            oscillator_families={},
            coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
            drivers=DriverSpec(physical={}, informational={}, symbolic={}),
            objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
            boundaries=[],
            actuators=[],
        )

    def test_length_mismatch_raises_with_exact_counts(self):
        """Error message must report both the omegas length and oscillator count."""
        spec = self._make_spec(["o1", "o2", "o3"], omegas=[1.0])
        with pytest.raises(ValueError, match="omegas length 1 != oscillator count 3"):
            spec.get_omegas()

    def test_matching_lengths_returns_correct_array(self):
        """Happy path: omegas length matches oscillator count."""
        spec = self._make_spec(["o1", "o2"], omegas=[3.14, 2.71])
        result = spec.get_omegas()
        np.testing.assert_allclose(result, [3.14, 2.71])

    def test_no_omegas_returns_default(self):
        """No omegas specified → returns default per oscillator."""
        spec = self._make_spec(["o1", "o2", "o3"])
        result = spec.get_omegas()
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Boundary validator: inverted limits detection
# ---------------------------------------------------------------------------


class TestBoundaryInvertedLimitsValidation:
    """Verify that the binding validator catches inverted boundary limits
    and related constraint violations."""

    def _make_spec_with_boundary(self, lower, upper, severity="hard"):
        bdef = BoundaryDef(
            name="test_bound",
            variable="R",
            lower=min(lower, upper),
            upper=max(lower, upper),
            severity=severity,
        )
        # Force inversion via frozen dataclass bypass
        if lower > upper:
            object.__setattr__(bdef, "lower", lower)
            object.__setattr__(bdef, "upper", upper)
        return BindingSpec(
            name="test",
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[HierarchyLayer(name="L1", index=0, oscillator_ids=["o1"])],
            oscillator_families={},
            coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
            drivers=DriverSpec(physical={}, informational={}, symbolic={}),
            objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
            boundaries=[bdef],
            actuators=[],
        )

    def test_inverted_boundary_reports_error(self):
        """lower > upper must produce a validation error with both values reported."""
        from scpn_phase_orchestrator.binding.validator import validate_binding_spec

        spec = self._make_spec_with_boundary(lower=0.8, upper=0.2)
        errors = validate_binding_spec(spec)
        assert any(
            "lower (0.8)" in e and "must be <= upper (0.2)" in e for e in errors
        ), f"Expected inverted-limits error, got: {errors}"

    def test_valid_boundary_no_errors(self):
        """Correct boundaries produce no validation errors related to limits."""
        from scpn_phase_orchestrator.binding.validator import validate_binding_spec

        spec = self._make_spec_with_boundary(lower=0.2, upper=0.8)
        errors = validate_binding_spec(spec)
        limit_errors = [e for e in errors if "lower" in e and "upper" in e]
        assert len(limit_errors) == 0, (
            f"Valid boundary should not produce errors: {limit_errors}"
        )


# ---------------------------------------------------------------------------
# Session start coherence check
# ---------------------------------------------------------------------------


class TestSessionStartCoherenceAnalysis:
    """Verify that session_start checks correctly identify low vs high
    initial coherence and produce appropriate warnings."""

    def _run_check(self, phases):
        from scpn_phase_orchestrator.imprint.state import ImprintState
        from scpn_phase_orchestrator.monitor.session_start import check_session_start

        n = len(phases)
        imprint = ImprintState(m_k=np.ones(n), last_update=0.0)
        return check_session_start([], np.array(phases), imprint, n)

    def test_near_chaos_phases_produce_low_coherence_warning(self):
        """Uniformly spread phases (R ≈ 0) must trigger the warning."""
        phases = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        report = self._run_check(phases)
        assert any("Low initial coherence" in w for w in report.warnings)
        assert report.initial_r < 0.05, (
            f"R={report.initial_r:.4f} should be near 0 for uniformly spread phases"
        )

    def test_synchronised_phases_no_warning(self):
        """Nearly identical phases (R ≈ 1) must NOT trigger the warning."""
        phases = [0.01, 0.02, 0.015, 0.005]
        report = self._run_check(phases)
        low_coh = [w for w in report.warnings if "Low initial" in w]
        assert low_coh == [], f"Sync phases → no warning: {low_coh}"
        assert report.initial_r > 0.99, (
            f"R={report.initial_r:.4f} should be near 1 for nearly identical phases"
        )

    def test_moderate_coherence_no_warning(self):
        """R just above 0.05 threshold must not trigger the warning."""
        # Two clusters: [0, 0.1] and [0.3, 0.4] — R well above 0.05
        phases = [0.0, 0.1, 0.3, 0.4]
        report = self._run_check(phases)
        assert report.initial_r > 0.05
        assert not any("Low initial coherence" in w for w in report.warnings)

    def test_imprint_size_mismatch_error(self):
        """If imprint dimension doesn't match oscillator count, report error."""
        from scpn_phase_orchestrator.imprint.state import ImprintState
        from scpn_phase_orchestrator.monitor.session_start import check_session_start

        phases = np.array([0.1, 0.2, 0.3])
        imprint = ImprintState(m_k=np.ones(5), last_update=0.0)  # 5 != 3
        report = check_session_start([], phases, imprint, 3)
        assert not report.passed
        assert any("mismatch" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# Initial phase extraction: channel fallback behaviour
# ---------------------------------------------------------------------------


class TestInitPhasesChannelResolution:
    """Verify that initial phase extraction correctly resolves channels
    and falls back to uniform random for unknown channels."""

    def _make_spec(self, channel, family_name="test_fam"):
        return BindingSpec(
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
                    family=family_name,
                )
            ],
            oscillator_families={
                family_name: OscillatorFamily(
                    channel=channel,
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

    def test_unknown_channel_falls_back_to_uniform(self):
        """Channel 'X' (not P/I/S) must produce phases in [0, 2π)."""
        from scpn_phase_orchestrator.oscillators.init_phases import (
            extract_initial_phases,
        )

        spec = self._make_spec("X")
        phases = extract_initial_phases(spec, np.array([1.0, 1.0]))
        assert phases.shape == (2,)
        assert np.all(phases >= 0.0) and np.all(phases < 2 * np.pi), (
            f"Phases must be in [0, 2π), got {phases}"
        )

    def test_physical_channel_produces_valid_phases(self):
        """Channel 'P' must produce phases in [0, 2π)."""
        from scpn_phase_orchestrator.oscillators.init_phases import (
            extract_initial_phases,
        )

        spec = self._make_spec("P")
        phases = extract_initial_phases(spec, np.array([5.0, 5.0]))
        assert phases.shape == (2,)
        assert np.all(phases >= 0.0) and np.all(phases < 2 * np.pi)

    def test_resolve_channel_empty_families_defaults_to_P(self):
        """With no families defined, channel resolution must default to 'P'."""
        from scpn_phase_orchestrator.oscillators.init_phases import _resolve_channel

        assert _resolve_channel(None, {}, 0) == "P"
        assert _resolve_channel(None, {}, 5) == "P"

    def test_get_n_states_no_symbolic_family_defaults_to_4(self):
        """Without any symbolic family, n_states must default to 4."""
        from scpn_phase_orchestrator.oscillators.init_phases import _get_n_states

        families = {
            "phys": OscillatorFamily(channel="P", extractor_type="hilbert", config={}),
        }
        assert _get_n_states(families) == 4

    def test_get_n_states_from_symbolic_config(self):
        """Symbolic family with explicit n_states must use that value."""
        from scpn_phase_orchestrator.oscillators.init_phases import _get_n_states

        families = {
            "sym": OscillatorFamily(
                channel="S",
                extractor_type="symbolic",
                config={"n_states": 8},
            ),
        }
        assert _get_n_states(families) == 8


# ---------------------------------------------------------------------------
# SimulationState: non-amplitude (Kuramoto-only) step
# ---------------------------------------------------------------------------


class TestSimulationStateKuramotoStep:
    """Verify that SimulationState in non-amplitude mode (pure Kuramoto)
    produces physically correct phase evolution."""

    @pytest.fixture()
    def sim(self):
        from scpn_phase_orchestrator.server import SimulationState

        spec = BindingSpec(
            name="test",
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[HierarchyLayer(name="L1", index=0, oscillator_ids=["o1", "o2"])],
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
                    name="K_g", knob="K", scope="global", limits=(0.0, 1.0)
                ),
            ],
        )
        return SimulationState(spec)

    def test_non_amplitude_mode_flag(self, sim):
        """Non-amplitude spec must produce Kuramoto mode, not Stuart-Landau."""
        assert not sim.amplitude_mode

    def test_step_returns_valid_state(self, sim):
        """Single step must return step counter and R_global in [0, 1]."""
        result = sim.step()
        assert result["step"] == 1
        assert "R_global" in result
        assert 0.0 <= result["R_global"] <= 1.0

    def test_multi_step_advances_phase(self, sim):
        """10 consecutive steps must advance the state — R should vary,
        step counter must increment monotonically."""
        results = [sim.step() for _ in range(10)]
        steps = [r["step"] for r in results]
        assert steps == list(range(1, 11)), "Step counter must increment by 1 each call"
        r_values = [r["R_global"] for r in results]
        # With coupling, R should not be exactly constant (phases evolve)
        assert not all(r == r_values[0] for r in r_values), (
            "R_global should vary across steps under coupling"
        )

    def test_step_r_bounded(self, sim):
        """R must stay in [0, 1] across 50 integration steps."""
        for _ in range(50):
            result = sim.step()
            r = result["R_global"]
            s = result["step"]
            assert 0.0 <= r <= 1.0, f"R={r:.4f} at step {s}"


# ---------------------------------------------------------------------------
# UPDE engine: NaN validation and RK45 edge cases
# ---------------------------------------------------------------------------


class TestUPDEEngineInputValidation:
    """Verify that UPDEEngine rejects invalid inputs with clear error messages
    and that RK45 handles extreme conditions gracefully."""

    def test_alpha_nan_raises_with_message(self):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="alpha contains NaN"):
            eng.step(
                np.array([0.1, 0.2]),
                np.array([1.0, 1.0]),
                np.zeros((2, 2)),
                0.0,
                0.0,
                np.array([[0.0, float("nan")], [0.0, 0.0]]),
            )

    def test_alpha_inf_also_rejected(self):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="alpha contains NaN"):
            eng.step(
                np.array([0.1, 0.2]),
                np.array([1.0, 1.0]),
                np.zeros((2, 2)),
                0.0,
                0.0,
                np.array([[0.0, float("inf")], [0.0, 0.0]]),
            )

    def test_rk45_extreme_coupling_remains_finite(self):
        """RK45 with extremely tight tolerances and large coupling must
        exhaust retries gracefully (fallback to Euler) and return finite phases."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(4, dt=1.0, method="rk45", atol=1e-15, rtol=1e-15)
        phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        omegas = np.array([100.0, 200.0, 300.0, 400.0])
        knm = np.full((4, 4), 1000.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((4, 4))

        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result)), (
            f"RK45 fallback must return finite phases: {result}"
        )
        # Phases must have changed (omegas are non-zero)
        assert not np.allclose(result, phases), (
            "Extreme conditions should still advance phases"
        )


# ---------------------------------------------------------------------------
# Order parameters: empty and degenerate input guards
# ---------------------------------------------------------------------------


class TestOrderParameterEdgeCases:
    """Verify order parameter functions handle edge cases with defined behaviour,
    not just "doesn't crash"."""

    def test_empty_phases_returns_zero_r_and_psi(self):
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        r, psi = compute_order_parameter(np.array([]))
        assert r == 0.0 and psi == 0.0, "Empty phases must give R=0, Ψ=0"

    def test_single_phase_returns_r_one(self):
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        r, psi = compute_order_parameter(np.array([1.23]))
        assert abs(r - 1.0) < 1e-10, "Single oscillator must have R=1"
        assert abs(psi - 1.23) < 1e-10, "Single oscillator Ψ must equal its phase"

    def test_plv_empty_either_side_returns_zero(self):
        from scpn_phase_orchestrator.upde.order_params import compute_plv

        assert compute_plv(np.array([]), np.array([1.0])) == 0.0
        assert compute_plv(np.array([1.0]), np.array([])) == 0.0

    def test_plv_identical_phases_returns_one(self):
        from scpn_phase_orchestrator.upde.order_params import compute_plv

        phases = np.array([0.5, 0.5, 0.5, 0.5])
        plv = compute_plv(phases, phases)
        assert abs(plv - 1.0) < 1e-10, "PLV of identical phases must be 1"

    def test_plv_anti_phase_returns_one(self):
        """PLV measures consistency of phase difference, not alignment.
        Anti-phase (π offset) has a consistent difference → PLV = 1."""
        from scpn_phase_orchestrator.upde.order_params import compute_plv

        a = np.array([0.0, 0.5, 1.0, 1.5])
        b = a + np.pi
        plv = compute_plv(a, b)
        assert abs(plv - 1.0) < 1e-10, f"Anti-phase PLV should be 1, got {plv:.4f}"


# ---------------------------------------------------------------------------
# PAC modulation index: n_bins guard and value range
# ---------------------------------------------------------------------------


class TestPACModulationIndexGuards:
    """Verify PAC modulation_index rejects invalid n_bins and produces
    bounded results for valid inputs."""

    def test_nbins_below_2_returns_zero(self):
        from scpn_phase_orchestrator.upde.pac import modulation_index

        theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        amp = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        assert modulation_index(theta, amp, n_bins=1) == 0.0
        assert modulation_index(theta, amp, n_bins=0) == 0.0

    def test_uniform_amplitude_gives_low_mi(self):
        """Constant amplitude across all phases → no modulation → MI near 0."""
        from scpn_phase_orchestrator.upde.pac import modulation_index

        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        amp = np.ones(200)
        mi = modulation_index(theta, amp, n_bins=18)
        assert mi < 0.1, f"Uniform amplitude should give MI near 0, got {mi:.4f}"

    def test_mi_in_unit_interval(self):
        """MI must always be in [0, 1] regardless of input."""
        from scpn_phase_orchestrator.upde.pac import modulation_index

        rng = np.random.default_rng(42)
        theta = rng.uniform(0, 2 * np.pi, 500)
        amp = rng.exponential(1.0, 500)
        mi = modulation_index(theta, amp, n_bins=18)
        assert 0.0 <= mi <= 1.0, f"MI must be in [0, 1], got {mi:.4f}"


# ---------------------------------------------------------------------------
# Stuart-Landau engine: comprehensive input validation
# ---------------------------------------------------------------------------


class TestStuartLandauInputValidation:
    """Verify that StuartLandauEngine rejects every type of invalid input
    with the correct error message and field name."""

    @pytest.fixture()
    def engine(self):
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        return StuartLandauEngine(2, dt=0.01)

    @pytest.fixture()
    def valid_args(self):
        return {
            "state": np.array([0.1, 0.2, 0.7, 0.8]),
            "omegas": np.array([1.0, 1.0]),
            "mu": np.array([0.5, 0.5]),
            "knm": np.zeros((2, 2)),
            "knm_r": np.zeros((2, 2)),
            "zeta": 0.0,
            "psi": 0.0,
            "alpha": np.zeros((2, 2)),
        }

    @pytest.mark.parametrize(
        "field,bad_value,error_pattern",
        [
            ("state", np.array([0.1, 0.2, float("nan"), 0.8]), "state contains NaN"),
            ("omegas", np.array([float("nan"), 1.0]), "omegas contain NaN"),
            ("mu", np.array([float("inf"), 0.5]), "mu contains NaN"),
            ("knm", np.array([[0.0, float("nan")], [0.0, 0.0]]), "knm contains NaN"),
            (
                "knm_r",
                np.array([[0.0, float("inf")], [0.0, 0.0]]),
                "knm_r contains NaN",
            ),
            (
                "alpha",
                np.array([[0.0, float("nan")], [0.0, 0.0]]),
                "alpha contains NaN",
            ),
        ],
    )
    def test_nan_in_field_raises_valueerror(
        self, engine, valid_args, field, bad_value, error_pattern
    ):
        """Each numeric input field must be validated for NaN/Inf."""
        args = dict(valid_args)
        args[field] = bad_value
        with pytest.raises(ValueError, match=error_pattern):
            engine.step(**args)

    def test_epsilon_nan_raises(self, engine, valid_args):
        """Non-finite epsilon must be rejected."""
        with pytest.raises(ValueError, match="epsilon must be finite"):
            engine.step(**valid_args, epsilon=float("nan"))

    def test_epsilon_inf_raises(self, engine, valid_args):
        """Infinite epsilon must also be rejected."""
        with pytest.raises(ValueError, match="epsilon must be finite"):
            engine.step(**valid_args, epsilon=float("inf"))

    def test_valid_inputs_produce_finite_output(self, engine, valid_args):
        """Valid inputs → finite state vector of correct size."""
        result = engine.step(**valid_args)
        assert result.shape == (4,), f"SL state should be 2*N=4, got {result.shape}"
        assert np.all(np.isfinite(result)), (
            f"Valid inputs should give finite output: {result}"
        )

    def test_rk45_extreme_parameters_stays_finite(self):
        """SL RK45 with extreme coupling that exhausts retry budget must
        still return finite results via Euler fallback."""
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
        assert np.all(np.isfinite(result)), "Fallback must produce finite state"
        # Amplitudes must remain non-negative (physical constraint for Stuart-Landau)
        amplitudes = result[4:]
        assert np.all(amplitudes >= -1e-10), (
            f"SL amplitudes must be non-negative, got {amplitudes}"
        )
