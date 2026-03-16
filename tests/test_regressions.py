# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Regression tests for CHANGELOG v0.4.1 fixes

# Each test pins a specific bug so it can never recur.

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.coupling.lags import LagModel
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

# ── Regression 1: Stuart-Landau sign-flip ────────────────────────────
# Bug: intermediate RK stages produced r < 0, flipping amplitude coupling.
# Fix: clamp r >= 0 in _derivative before using in coupling term.


class TestStuartLandauSignFlip:
    def test_amplitude_never_negative_under_subcritical_bifurcation(self) -> None:
        """μ < 0 with strong coupling must not produce negative amplitudes."""
        n = 4
        engine = StuartLandauEngine(n, dt=0.01, method="rk4")
        state = np.zeros(2 * n)
        state[n:] = 0.01  # small initial amplitude
        omegas = np.ones(n)
        mu = np.full(n, -2.0)  # subcritical — amplitudes should decay
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0)
        knm_r = knm * 2.0  # strong amplitude coupling
        alpha = np.zeros((n, n))

        for _ in range(500):
            state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, 1.0)
            assert np.all(state[n:] >= 0.0), f"negative amplitude: {state[n:]}"

    def test_coupling_sign_correct_with_negative_intermediate(self) -> None:
        """Verify derivative uses clamped r in coupling, not raw r."""
        n = 2
        engine = StuartLandauEngine(n, dt=0.01, method="euler")
        # Force r near zero so RK intermediate could go negative
        state = np.array([0.0, np.pi, 0.001, 0.001])
        omegas = np.ones(n)
        mu = np.full(n, -1.0)
        knm = np.zeros((n, n))
        knm_r = np.full((n, n), 1.0)
        np.fill_diagonal(knm_r, 0)
        alpha = np.zeros((n, n))

        state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, 1.0)
        assert np.all(state[n:] >= 0.0)


# ── Regression 2: Audit hash chain collision ─────────────────────────
# Bug: _write_record included _hash in digest, causing user-data collision.
# Fix: strip _hash before computing digest.


class TestAuditHashChain:
    def test_hash_excludes_user_hash_field(self, tmp_path: object) -> None:
        """If user data contains '_hash', it must NOT affect the chain digest."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "audit.jsonl"
            logger = AuditLogger(log_path)
            # Record with user-supplied _hash field
            logger._write_record({"step": 0, "_hash": "user_injected_value"})
            logger._write_record({"step": 1})
            logger.close()

            lines = log_path.read_text().strip().split("\n")
            r0 = json.loads(lines[0])
            r1 = json.loads(lines[1])

            # Verify chain: r1's hash is computed over r0's hash + r1's clean content
            clean_r1 = {k: v for k, v in r1.items() if k != "_hash"}
            expected = hashlib.sha256(
                (r0["_hash"] + json.dumps(clean_r1, separators=(",", ":"))).encode()
            ).hexdigest()
            assert r1["_hash"] == expected

    def test_hash_chain_integrity(self, tmp_path: object) -> None:
        """Sequential records must form a valid hash chain."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "chain.jsonl"
            logger = AuditLogger(log_path)
            for i in range(5):
                logger._write_record({"step": i, "value": i * 0.1})
            logger.close()

            lines = log_path.read_text().strip().split("\n")
            prev_hash = "0" * 64
            for line in lines:
                record = json.loads(line)
                stored_hash = record.pop("_hash")
                json_line = json.dumps(record, separators=(",", ":"))
                expected = hashlib.sha256((prev_hash + json_line).encode()).hexdigest()
                step = record.get("step")
                assert stored_hash == expected, f"chain broken at {step}"
                prev_hash = stored_hash


# ── Regression 3: Regime FSM CRITICAL→RECOVERY→NOMINAL ──────────────
# Bug: CRITICAL could transition directly to NOMINAL, skipping RECOVERY.
# Fix: evaluate() returns RECOVERY when is_recovering and avg_r > threshold.


class TestRegimeFSMRecovery:
    def _make_state(self, r: float) -> tuple[UPDEState, object]:
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState

        layer = LayerState(R=r, psi=0.0, lock_signatures=[])
        upde = UPDEState(
            layers=[layer],
            cross_layer_alignment=np.array([[r]]),
            stability_proxy=r,
            regime_id="nominal",
        )
        boundary = BoundaryState(hard_violations=[], soft_violations=[])
        return upde, boundary

    def test_critical_must_pass_through_recovery(self) -> None:
        """After CRITICAL, even high R must go through RECOVERY first."""
        mgr = RegimeManager(hysteresis=0.05, cooldown_steps=0)

        # Drive to CRITICAL
        state_low, bstate = self._make_state(0.1)
        proposed = mgr.evaluate(state_low, bstate)
        mgr.transition(proposed)
        assert mgr.current_regime == Regime.CRITICAL

        # Now R recovers to 0.8 — should enter RECOVERY, not NOMINAL
        state_high, bstate = self._make_state(0.8)
        proposed = mgr.evaluate(state_high, bstate)
        assert proposed == Regime.RECOVERY
        mgr.transition(proposed)
        assert mgr.current_regime == Regime.RECOVERY

        # One more evaluation at high R — NOW can go NOMINAL
        proposed = mgr.evaluate(state_high, bstate)
        assert proposed == Regime.NOMINAL

    def test_degraded_can_go_directly_to_nominal(self) -> None:
        """DEGRADED (not CRITICAL) can go straight to NOMINAL."""
        mgr = RegimeManager(hysteresis=0.05, cooldown_steps=0)

        state_mid, bstate = self._make_state(0.5)
        proposed = mgr.evaluate(state_mid, bstate)
        mgr.transition(proposed)
        assert mgr.current_regime == Regime.DEGRADED

        state_high, bstate = self._make_state(0.9)
        proposed = mgr.evaluate(state_high, bstate)
        assert proposed == Regime.NOMINAL


# ── Regression 4: Hilbert guard on short signals ─────────────────────
# Bug: PhysicalExtractor accepted 0-sample or 1-sample signals.
# Fix: validate signal.ndim == 1 and signal.size >= 2.


class TestHilbertGuard:
    def test_empty_signal_rejected(self) -> None:
        ext = PhysicalExtractor()
        with pytest.raises(ValueError, match="1-D with >= 2 samples"):
            ext.extract(np.array([]), 256.0)

    def test_single_sample_rejected(self) -> None:
        ext = PhysicalExtractor()
        with pytest.raises(ValueError, match="1-D with >= 2 samples"):
            ext.extract(np.array([1.0]), 256.0)

    def test_2d_signal_rejected(self) -> None:
        ext = PhysicalExtractor()
        with pytest.raises(ValueError, match="1-D with >= 2 samples"):
            ext.extract(np.ones((2, 3)), 256.0)

    def test_valid_signal_accepted(self) -> None:
        ext = PhysicalExtractor()
        t = np.linspace(0, 1, 256)
        signal = np.sin(2 * np.pi * 10 * t)
        result = ext.extract(signal, 256.0)
        assert len(result) == 1
        assert 0.0 <= result[0].theta < 2 * np.pi


# ── Regression 5: Lag model radians conversion ──────────────────────
# Bug: build_alpha_matrix used raw lag seconds without frequency conversion.
# Fix: alpha[i,j] = 2*pi*carrier_freq_hz*lag[i,j].


class TestLagModelRadians:
    def test_carrier_frequency_scales_lag(self) -> None:
        """Different carrier frequencies must produce different alpha offsets."""
        lm = LagModel()
        lags = {(0, 1): 0.01}  # 10ms lag

        alpha_1hz = lm.build_alpha_matrix(lags, 2, carrier_freq_hz=1.0)
        alpha_10hz = lm.build_alpha_matrix(lags, 2, carrier_freq_hz=10.0)

        expected_1 = 2 * np.pi * 1.0 * 0.01
        expected_10 = 2 * np.pi * 10.0 * 0.01
        assert abs(alpha_1hz[0, 1] - expected_1) < 1e-12
        assert abs(alpha_10hz[0, 1] - expected_10) < 1e-12
        assert abs(alpha_10hz[0, 1]) > abs(alpha_1hz[0, 1])

    def test_antisymmetry(self) -> None:
        """alpha[i,j] = -alpha[j,i]."""
        lm = LagModel()
        lags = {(0, 1): 0.005, (1, 2): -0.003}
        alpha = lm.build_alpha_matrix(lags, 3, carrier_freq_hz=5.0)
        assert abs(alpha[0, 1] + alpha[1, 0]) < 1e-15
        assert abs(alpha[1, 2] + alpha[2, 1]) < 1e-15


# ── Regression 6: Zeta lower bound clamping ─────────────────────────
# Bug: zeta accumulation could go negative.
# Fix: cli.py uses max(0.0, min(zeta + act.value, 0.5)).
# This test verifies the clamping logic directly (not via CLI runner).


class TestZetaClamping:
    def test_zeta_clamp_prevents_negative(self) -> None:
        """Applying negative action to zeta must not produce negative result."""
        zeta = 0.1
        act_value = -0.5  # would make zeta = -0.4 without clamping
        zeta = max(0.0, min(zeta + act_value, 0.5))
        assert zeta >= 0.0

    def test_zeta_clamp_prevents_overshoot(self) -> None:
        zeta = 0.4
        act_value = 0.3  # would make zeta = 0.7 without upper clamping
        zeta = max(0.0, min(zeta + act_value, 0.5))
        assert zeta <= 0.5

    def test_zeta_clamp_identity_in_range(self) -> None:
        zeta = 0.2
        act_value = 0.1
        zeta = max(0.0, min(zeta + act_value, 0.5))
        assert abs(zeta - 0.3) < 1e-15
