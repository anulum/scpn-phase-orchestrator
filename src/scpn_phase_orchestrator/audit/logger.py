# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["AuditLogger"]


class AuditLogger:
    """Append-only JSONL audit log for UPDE simulation steps."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)

    def log_header(
        self,
        *,
        n_oscillators: int,
        dt: float,
        method: str = "euler",
        seed: int | None = None,
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
        self._fh.write(json.dumps(record) + "\n")

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
    ) -> None:
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
                raise ValueError(msg)
            record["phases"] = phases.tolist()
            record["omegas"] = omegas.tolist()
            record["knm"] = knm.tolist()
            record["alpha"] = alpha.tolist()
            record["zeta"] = zeta
            record["psi_drive"] = psi_drive
        self._fh.write(json.dumps(record) + "\n")

    def log_event(self, event_type: str, data: dict) -> None:
        record = {"ts": time.time(), "event": event_type, **data}
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
