# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Deterministic replay engine

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

_log = logging.getLogger(__name__)

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

    def build_engine(self, header: dict) -> UPDEEngine | StuartLandauEngine:
        """Construct engine from header (UPDE or Stuart-Landau)."""
        if header.get("amplitude_mode"):
            return StuartLandauEngine(
                n_oscillators=header["n_oscillators"],
                dt=header["dt"],
                method=header.get("method", "euler"),
            )
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

    @staticmethod
    def verify_integrity(entries: list[dict]) -> tuple[bool, int]:
        """Verify the SHA256 hash chain of audit log entries.

        Returns (all_valid, n_verified).  Legacy logs without ``_hash``
        fields return (True, 0).
        """
        prev = "0" * 64
        verified = 0
        for entry in entries:
            stored = entry.get("_hash")
            if stored is None:
                continue
            without_hash = {k: v for k, v in entry.items() if k != "_hash"}
            json_line = json.dumps(without_hash, separators=(",", ":"))
            expected = hashlib.sha256((prev + json_line).encode()).hexdigest()
            if expected != stored:
                return False, verified
            prev = stored
            verified += 1
        return True, verified

    def verify_determinism_sl_chained(
        self,
        engine: StuartLandauEngine,
        entries: list[dict],
        atol: float = 1e-6,
    ) -> tuple[bool, int]:
        """Chained multi-step replay for Stuart-Landau engine.

        Supports two log formats:
        - New format: separate 'phases' (N) + 'amplitudes' (N) fields.
        - Legacy format: 'phases' holds the full SL state (2N) with 'mu' present.
        When neither mu nor amplitudes are present, skips with a warning.
        Returns (passed, n_verified).
        """
        replayable = self.step_entries(entries)
        if len(replayable) < 2:
            return True, 0

        verified = 0
        for i in range(len(replayable) - 1):
            curr = replayable[i]
            nxt = replayable[i + 1]

            omegas = np.asarray(curr["omegas"])
            n = len(omegas)

            if "amplitudes" in curr:
                state = np.concatenate(
                    [np.asarray(curr["phases"]), np.asarray(curr["amplitudes"])]
                )
            elif "mu" in curr:
                # Legacy: full SL state [theta; r] stored in 'phases'
                state = np.asarray(curr["phases"])
            else:
                _log.warning("SL replay step %d: amplitude fields missing, skipping", i)
                continue

            knm_flat = np.asarray(curr["knm"])
            alpha_flat = np.asarray(curr["alpha"])
            zeta = curr.get("zeta", 0.0)
            psi_drive = curr.get("psi_drive", 0.0)

            knm_arr = knm_flat.reshape(n, n) if knm_flat.ndim == 1 else knm_flat
            alpha_arr = alpha_flat.reshape(n, n) if alpha_flat.ndim == 1 else alpha_flat

            mu = np.asarray(curr.get("mu", np.zeros(n)))
            knm_r_flat = np.asarray(curr.get("knm_r", np.zeros(n * n)))
            knm_r = knm_r_flat.reshape(n, n) if knm_r_flat.ndim == 1 else knm_r_flat
            epsilon = curr.get("epsilon", 1.0)

            computed = engine.step(
                state,
                omegas,
                mu,
                knm_arr,
                knm_r,
                zeta,
                psi_drive,
                alpha_arr,
                epsilon=epsilon,
            )

            if "amplitudes" in nxt:
                logged_next = np.concatenate(
                    [np.asarray(nxt["phases"]), np.asarray(nxt["amplitudes"])]
                )
            else:
                logged_next = np.asarray(nxt["phases"])

            if not np.allclose(computed, logged_next, atol=atol):
                return False, verified
            verified += 1

        return True, verified

    def verify_determinism(self, engine: UPDEEngine, steps: list[dict]) -> bool:
        """Re-run logged steps and compare global order parameter R.

        Requires steps to include 'phases', 'omegas', 'knm', 'zeta', 'psi',
        'alpha' fields for full replay. Compares replayed global R against
        logged 'R' (or 'r_global') field.
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

            # Compare against logged global R (same quantity as compute_order_parameter)
            r_logged = entry.get("R") or entry.get("r_global")
            if r_logged is not None and abs(r_actual - r_logged) > 1e-6:
                return False
        return True
