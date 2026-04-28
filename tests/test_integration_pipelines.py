# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Integration pipeline tests

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.coupling import CouplingBuilder
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Imprint → Coupling modulation → UPDE pipeline
# ---------------------------------------------------------------------------


class TestImprintCouplingUPDEPipeline:
    """Full pipeline: ImprintModel modulates K_nm, which is fed into
    UPDEEngine. Verifies that the imprint has a measurable, directional
    effect on coherence dynamics."""

    def test_imprint_shifts_coherence(self):
        """Boosted coupling via imprint must produce different R than baseline."""
        n = 8
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)

        cs = CouplingBuilder().build(n, base_strength=0.45, decay_alpha=0.3)
        alpha = np.zeros((n, n))

        # Baseline run
        eng = UPDEEngine(n, dt=0.01)
        p_base = phases.copy()
        for _ in range(200):
            p_base = eng.step(p_base, omegas, cs.knm, 0.0, 0.0, alpha)
        r_base, _ = compute_order_parameter(p_base)

        # Imprinted run: 2x exposure → knm rows scaled by (1 + m_k)
        imprint = ImprintModel(decay_rate=0.0, saturation=5.0)
        state = ImprintState(m_k=np.zeros(n), last_update=0.0)
        state = imprint.update(state, np.ones(n) * 2.0, dt=1.0)
        knm_boosted = imprint.modulate_coupling(cs.knm.copy(), state)

        eng2 = UPDEEngine(n, dt=0.01)
        p_imp = phases.copy()
        for _ in range(200):
            p_imp = eng2.step(p_imp, omegas, knm_boosted, 0.0, 0.0, alpha)
        r_imp, _ = compute_order_parameter(p_imp)

        assert r_imp != pytest.approx(r_base, abs=1e-4), (
            f"Imprint must shift R: base={r_base:.4f}, imprinted={r_imp:.4f}"
        )

    def test_stronger_imprint_stronger_coupling(self):
        """Higher imprint exposure → stronger coupling → R should tend toward 1."""
        n = 8
        rng = np.random.default_rng(7)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.zeros(n)  # pure coupling dynamics

        cs = CouplingBuilder().build(n, base_strength=1.0, decay_alpha=0.1)
        alpha = np.zeros((n, n))
        imprint = ImprintModel(decay_rate=0.0, saturation=100.0)

        r_by_exposure = []
        for exposure_level in [0.0, 2.0, 10.0]:
            state = ImprintState(m_k=np.full(n, exposure_level), last_update=0.0)
            knm = imprint.modulate_coupling(cs.knm.copy(), state)
            eng = UPDEEngine(n, dt=0.01)
            p = phases.copy()
            for _ in range(500):
                p = eng.step(p, omegas, knm, 0.0, 0.0, alpha)
            r, _ = compute_order_parameter(p)
            r_by_exposure.append(r)

        # With zero omegas and increasing coupling, R should increase
        assert r_by_exposure[-1] >= r_by_exposure[0] - 0.1, (
            f"Stronger coupling should not decrease R: {r_by_exposure}"
        )

    def test_imprint_mu_modulation_affects_sl(self):
        """ImprintModel.modulate_mu must scale bifurcation parameter,
        changing Stuart-Landau amplitude dynamics."""
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        n = 4
        mu_base = np.array([0.3, 0.3, 0.3, 0.3])
        imprint = ImprintModel(decay_rate=0.0, saturation=10.0)
        state = ImprintState(m_k=np.array([0.0, 1.0, 2.0, 5.0]), last_update=0.0)
        mu_boosted = imprint.modulate_mu(mu_base, state)

        # mu_boosted[i] = mu_base[i] * (1 + m_k[i])
        np.testing.assert_allclose(mu_boosted, [0.3, 0.6, 0.9, 1.8])

        # Higher mu → larger equilibrium amplitude (sqrt(mu))
        eng = StuartLandauEngine(n, dt=0.01)
        init_state = np.array([0.0, 0.5, 1.0, 1.5, 0.1, 0.1, 0.1, 0.1])
        result = init_state.copy()
        for _ in range(500):
            result = eng.step(
                result,
                np.ones(n),
                mu_boosted,
                np.zeros((n, n)),
                np.zeros((n, n)),
                0.0,
                0.0,
                np.zeros((n, n)),
            )
        amplitudes = result[n:]
        # Oscillator 3 (mu=1.8) should have larger amplitude than oscillator 0 (mu=0.3)
        assert amplitudes[3] > amplitudes[0], (
            f"Higher mu should produce larger amplitude: {amplitudes}"
        )


# ---------------------------------------------------------------------------
# AuditLogger → ReplayEngine round-trip
# ---------------------------------------------------------------------------


class TestAuditLoggerReplayRoundTrip:
    """Write audit log → read back via ReplayEngine. Verify exact data
    preservation and hash chain integrity across the pipeline."""

    def test_step_entry_exact_round_trip(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        layers = [LayerState(R=0.85, psi=1.23), LayerState(R=0.42, psi=3.14)]
        state = UPDEState(
            layers=layers,
            cross_layer_alignment=np.zeros((2, 2)),
            stability_proxy=0.77,
            regime_id="degraded",
        )
        with AuditLogger(log_file) as logger:
            logger.log_step(step=0, upde_state=state, actions=[])

        replay = ReplayEngine(log_file)
        entries = replay.load()
        assert len(entries) == 1
        e = entries[0]
        assert e["step"] == 0
        assert e["regime"] == "degraded"
        assert e["stability"] == pytest.approx(0.77)
        assert e["layers"][0]["R"] == pytest.approx(0.85)
        assert e["layers"][1]["psi"] == pytest.approx(3.14)

    def test_event_entry_round_trip(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        with AuditLogger(log_file) as logger:
            logger.log_event("regime_change", {"from": "nominal", "to": "critical"})

        entries = ReplayEngine(log_file).load()
        assert entries[0]["event"] == "regime_change"
        assert entries[0]["from"] == "nominal"
        assert entries[0]["to"] == "critical"

    def test_replay_reconstructs_upde_state(self, tmp_path):
        """ReplayEngine.replay_step must reconstruct UPDEState with
        correct layer R, psi, and regime_id."""
        log_file = tmp_path / "audit.jsonl"
        state = UPDEState(
            layers=[LayerState(R=0.65, psi=2.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.5,
            regime_id="nominal",
        )
        with AuditLogger(log_file) as logger:
            logger.log_step(0, state, [])

        replay = ReplayEngine(log_file)
        entries = replay.load()
        reconstructed = replay.replay_step(entries[0])
        assert pytest.approx(0.65) == reconstructed.layers[0].R
        assert reconstructed.regime_id == "nominal"

    def test_multi_step_hash_chain_survives_round_trip(self, tmp_path):
        """Write 5 steps, read back, verify hash chain is unbroken."""
        log_file = tmp_path / "audit.jsonl"
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.6,
            regime_id="nominal",
        )
        with AuditLogger(log_file) as logger:
            for i in range(5):
                logger.log_step(i, state, [])

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 5

        prev_hash = "0" * 64
        for line in lines:
            record = json.loads(line)
            clean = {k: v for k, v in record.items() if k != "_hash"}
            json_line = json.dumps(clean, separators=(",", ":"), sort_keys=True)
            expected = hashlib.sha256((prev_hash + json_line).encode()).hexdigest()
            assert record["_hash"] == expected, "Hash chain broken in round-trip"
            prev_hash = record["_hash"]

    def test_full_state_replay_determinism(self, tmp_path):
        """Log phases/omegas/knm/alpha, replay via UPDEEngine, verify
        the replayed state matches the logged next state."""
        n = 4
        log_file = tmp_path / "audit.jsonl"
        engine = UPDEEngine(n, dt=0.01)

        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.array([1.0, 1.5, 0.5, 2.0])
        knm = np.array(
            [
                [0.0, 0.3, 0.1, 0.0],
                [0.3, 0.0, 0.0, 0.2],
                [0.1, 0.0, 0.0, 0.1],
                [0.0, 0.2, 0.1, 0.0],
            ]
        )
        alpha = np.zeros((n, n))

        next_phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

        state0 = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.5,
            regime_id="nominal",
        )
        state1 = UPDEState(
            layers=[LayerState(R=0.6, psi=0.1)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.5,
            regime_id="nominal",
        )

        with AuditLogger(log_file) as logger:
            logger.log_step(
                0, state0, [], phases=phases, omegas=omegas, knm=knm, alpha=alpha
            )
            logger.log_step(
                1, state1, [], phases=next_phases, omegas=omegas, knm=knm, alpha=alpha
            )

        replay = ReplayEngine(log_file)
        entries = replay.load()
        step_entries = [e for e in entries if "phases" in e]

        ok = replay.verify_determinism(engine, step_entries)
        assert ok, "Replay must reproduce the engine's forward step exactly"

    def test_actions_preserved_in_log(self, tmp_path):
        """ControlActions must survive the log→replay cycle."""
        log_file = tmp_path / "audit.jsonl"
        actions = [
            ControlAction(
                knob="K", scope="global", value=0.3, ttl_s=5.0, justification="boost"
            ),
        ]
        state = UPDEState(
            layers=[LayerState(R=0.8, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.9,
            regime_id="nominal",
        )
        with AuditLogger(log_file) as logger:
            logger.log_step(0, state, actions)

        entries = ReplayEngine(log_file).load()
        assert len(entries[0]["actions"]) == 1
        assert entries[0]["actions"][0]["knob"] == "K"
        assert entries[0]["actions"][0]["value"] == 0.3
        assert entries[0]["actions"][0]["justification"] == "boost"
