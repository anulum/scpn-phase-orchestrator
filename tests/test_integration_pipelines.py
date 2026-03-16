# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Integration pipeline tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.coupling import CouplingBuilder
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


class TestImprintCouplingUPDEPipeline:
    """Imprint → modulate_coupling → UPDEEngine.step: imprinted coupling shifts R."""

    def test_imprint_shifts_coherence(self) -> None:
        n = 8
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)

        cs = CouplingBuilder().build(n, base_strength=0.45, decay_alpha=0.3)
        alpha = np.zeros((n, n))
        engine = UPDEEngine(n, dt=0.01)

        # Baseline run
        p_base = phases.copy()
        for _ in range(200):
            p_base = engine.step(p_base, omegas, cs.knm, 0.0, 0.0, alpha)
        r_base, _ = compute_order_parameter(p_base)

        # Imprinted run: boost coupling via imprint
        imprint = ImprintModel(decay_rate=0.0, saturation=5.0)
        state = ImprintState(m_k=np.zeros(n), last_update=0.0)
        exposure = np.ones(n) * 2.0
        state = imprint.update(state, exposure, dt=1.0)
        knm_boosted = imprint.modulate_coupling(cs.knm.copy(), state)

        p_imp = phases.copy()
        engine2 = UPDEEngine(n, dt=0.01)
        for _ in range(200):
            p_imp = engine2.step(p_imp, omegas, knm_boosted, 0.0, 0.0, alpha)
        r_imp, _ = compute_order_parameter(p_imp)

        # Boosted coupling should produce different coherence
        assert r_imp != pytest.approx(r_base, abs=1e-6), (
            f"R_imprint={r_imp:.4f} should differ from R_base={r_base:.4f}"
        )


class TestAuditLoggerReplayRoundTrip:
    """AuditLogger → ReplayEngine: write then read back, verify exact preservation."""

    def test_round_trip(self, tmp_path) -> None:
        log_file = tmp_path / "audit.jsonl"

        layers = [
            LayerState(R=0.85, psi=1.23),
            LayerState(R=0.42, psi=3.14),
        ]
        state = UPDEState(
            layers=layers,
            cross_layer_alignment=np.zeros((2, 2)),
            stability_proxy=0.77,
            regime_id="degraded",
        )

        with AuditLogger(log_file) as logger:
            logger.log_step(step=0, upde_state=state, actions=[])
            logger.log_event("test_event", {"key": "value"})

        replay = ReplayEngine(log_file)
        entries = replay.load()
        assert len(entries) == 2

        step_entry = entries[0]
        assert step_entry["step"] == 0
        assert step_entry["regime"] == "degraded"
        assert step_entry["stability"] == pytest.approx(0.77)
        assert len(step_entry["layers"]) == 2
        assert step_entry["layers"][0]["R"] == pytest.approx(0.85)
        assert step_entry["layers"][1]["psi"] == pytest.approx(3.14)

        event_entry = entries[1]
        assert event_entry["event"] == "test_event"
        assert event_entry["key"] == "value"

        # ReplayEngine can reconstruct UPDEState from step entry
        reconstructed = replay.replay_step(step_entry)
        assert len(reconstructed.layers) == 2
        assert pytest.approx(0.85) == reconstructed.layers[0].R
        assert reconstructed.regime_id == "degraded"

    def test_context_manager_closes_file(self, tmp_path) -> None:
        log_file = tmp_path / "ctx.jsonl"
        with AuditLogger(log_file) as logger:
            logger.log_event("ping", {})
        # File should be readable after context exit
        assert log_file.read_text(encoding="utf-8").strip() != ""
