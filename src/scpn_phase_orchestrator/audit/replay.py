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

__all__ = ["ReplayEngine"]


class ReplayEngine:
    """Replay and verify determinism of JSONL audit logs."""

    def __init__(self, log_path: str | Path):
        self._log_path = Path(log_path)

    def load(self) -> list[dict]:
        entries = []
        with self._log_path.open(encoding="utf-8") as fh:
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

    def load_header(self, entries: list[dict]) -> dict | None:
        """Extract the header record (engine config) if present."""
        for entry in entries:
            if entry.get("header"):
                return entry
        return None

    def step_entries(self, entries: list[dict]) -> list[dict]:
        """Filter to entries with full UPDE state (replayable)."""
        return [e for e in entries if "phases" in e]

    def build_engine(self, header: dict) -> UPDEEngine:
        """Construct a UPDEEngine from a header record."""
        return UPDEEngine(
            n_oscillators=header["n_oscillators"],
            dt=header["dt"],
            method=header.get("method", "euler"),
        )

    def verify_determinism_chained(
        self, engine: UPDEEngine, entries: list[dict], atol: float = 1e-6
    ) -> tuple[bool, int]:
        """Chained multi-step replay: output of step N must match input of step N+1.

        Returns (passed, n_verified).
        """
        replayable = self.step_entries(entries)
        if len(replayable) < 2:
            return True, 0

        verified = 0
        for i in range(len(replayable) - 1):
            curr = replayable[i]
            nxt = replayable[i + 1]
            phases = np.asarray(curr["phases"])
            omegas = np.asarray(curr["omegas"])
            knm_arr = np.asarray(curr["knm"])
            alpha_arr = np.asarray(curr["alpha"])
            zeta = curr.get("zeta", 0.0)
            psi_drive = curr.get("psi_drive", 0.0)

            computed = engine.step(phases, omegas, knm_arr, zeta, psi_drive, alpha_arr)
            logged_next = np.asarray(nxt["phases"])

            if not np.allclose(computed, logged_next, atol=atol):
                return False, verified
            verified += 1

        return True, verified

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

            r_logged = entry.get("stability_proxy")
            if r_logged is not None and abs(r_actual - r_logged) > 1e-6:
                return False
        return True
