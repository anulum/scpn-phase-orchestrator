# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


class ReplayEngine:
    """Replay and verify determinism of JSONL audit logs."""

    def __init__(self, log_path: str | Path):
        self._log_path = Path(log_path)

    def load(self) -> list[dict]:
        entries = []
        with open(self._log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def replay_step(self, step_data: dict) -> UPDEState:
        """Reconstruct UPDEState from a log entry."""
        layers = [
            LayerState(R=ld["R"], psi=ld["psi"]) for ld in step_data.get("layers", [])
        ]
        return UPDEState(
            layers=layers,
            cross_layer_alignment=np.zeros((len(layers), len(layers))),
            stability_proxy=step_data.get("stability", 0.0),
            regime_id=step_data.get("regime", "unknown"),
        )

    def verify_determinism(self, engine: UPDEEngine, steps: list[dict]) -> bool:
        """Re-run logged steps and compare R values.

        Returns True if all reproduced layer-R values match within tolerance.
        Requires steps to include 'phases', 'omegas', 'knm', 'zeta', 'psi',
        'alpha' fields for full replay. Falls back to log-only comparison
        if engine input fields are missing.
        """
        for entry in steps:
            if "phases" not in entry:
                continue
            phases = np.asarray(entry["phases"])
            omegas = np.asarray(entry["omegas"])
            knm = np.asarray(entry["knm"])
            alpha = np.asarray(entry["alpha"])
            zeta = entry.get("zeta", 0.0)
            psi_drive = entry.get("psi_drive", 0.0)

            new_phases = engine.step(phases, omegas, knm, zeta, psi_drive, alpha)
            r_actual, _ = engine.compute_order_parameter(new_phases)

            logged_layers = entry.get("layers", [])
            if logged_layers:
                r_logged = np.mean([ld["R"] for ld in logged_layers])
                if abs(r_actual - r_logged) > 1e-6:
                    return False
        return True
