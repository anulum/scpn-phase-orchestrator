# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit event logger

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.audit.stream import EventStreamWriter
from scpn_phase_orchestrator.exceptions import AuditError
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["AuditLogger"]


class AuditLogger:
    """Append-only JSONL audit log for UPDE simulation steps."""

    def __init__(self, path: str | Path, *, event_stream: str | Path | None = None):
        self._path = Path(path)
        self._prev_hash = self._load_previous_hash()
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)
        self._event_stream = (
            EventStreamWriter(event_stream) if event_stream is not None else None
        )

    def _load_previous_hash(self) -> str:
        if not self._path.exists() or self._path.stat().st_size == 0:
            return "0" * 64
        previous = "0" * 64
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                stored = record.get("_hash")
                if isinstance(stored, str) and len(stored) == 64:
                    previous = stored
        return previous

    def _write_record(self, record: dict) -> None:
        clean = {k: v for k, v in record.items() if k != "_hash"}
        json_line = json.dumps(clean, separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256((self._prev_hash + json_line).encode()).hexdigest()
        self._prev_hash = digest
        stored = {**clean, "_hash": digest}
        self._fh.write(json.dumps(stored, sort_keys=True) + "\n")
        if self._event_stream is not None:
            self._event_stream.write(stored)

    def log_header(
        self,
        *,
        n_oscillators: int,
        dt: float,
        method: str = "euler",
        seed: int | None = None,
        amplitude_mode: bool = False,
        binding_config: dict[str, object] | None = None,
        binding_summary: dict[str, object] | None = None,
    ) -> None:
        """Engine configuration record for replay reconstruction."""
        record: dict = {
            "header": True,
            "n_oscillators": n_oscillators,
            "dt": dt,
            "method": method,
        }
        if seed is not None:
            record["seed"] = seed
        if amplitude_mode:
            record["amplitude_mode"] = True

        if binding_summary is None:
            binding_summary = binding_config
        if binding_summary is not None:
            record["binding_summary"] = binding_summary
        if binding_config is not None:
            record["binding_config"] = binding_config
        self._write_record(record)

    def log_step(
        self,
        step: int,
        upde_state: UPDEState,
        actions: list[ControlAction],
        *,
        phases: NDArray[np.floating] | None = None,
        omegas: NDArray[np.floating] | None = None,
        knm: NDArray[np.floating] | None = None,
        alpha: NDArray[np.floating] | None = None,
        zeta: float = 0.0,
        psi_drive: float = 0.0,
        amplitudes: NDArray[np.floating] | None = None,
        mu: NDArray[np.floating] | None = None,
        knm_r: NDArray[np.floating] | None = None,
        epsilon: float | None = None,
        channel_runtime: dict[str, object] | None = None,
    ) -> None:
        """Write one simulation step to the audit log with optional full state."""
        record = {
            "ts": time.time(),
            "step": step,
            "regime": upde_state.regime_id,
            "stability": upde_state.stability_proxy,
            "layers": [{"R": ls.R, "psi": ls.psi} for ls in upde_state.layers],
            "actions": [
                {
                    "knob": a.knob,
                    "scope": a.scope,
                    "value": a.value,
                    "ttl_s": a.ttl_s,
                    "justification": a.justification,
                }
                for a in actions
            ],
        }
        if phases is not None:
            if omegas is None or knm is None or alpha is None:
                msg = "omegas, knm, alpha required when phases is provided"
                raise AuditError(msg)
            record["phases"] = phases.tolist()
            record["omegas"] = omegas.tolist()
            record["knm"] = knm.tolist()
            record["alpha"] = alpha.tolist()
            record["zeta"] = zeta
            record["psi_drive"] = psi_drive
        if amplitudes is not None:
            record["amplitudes"] = amplitudes.tolist()
        if mu is not None:
            record["mu"] = mu.tolist()
        if knm_r is not None:
            record["knm_r"] = knm_r.tolist()
        if epsilon is not None:
            record["epsilon"] = epsilon
        if channel_runtime is not None:
            record["channel_runtime"] = channel_runtime
        self._write_record(record)

    def log_event(self, event_type: str, data: dict) -> None:
        """Write a named event with arbitrary data to the audit log."""
        record = {"ts": time.time(), "event": event_type, **data}
        self._write_record(record)

    def close(self) -> None:
        """Flush and close the audit log file handle."""
        self._fh.flush()
        self._fh.close()
        if self._event_stream is not None:
            self._event_stream.close()

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
