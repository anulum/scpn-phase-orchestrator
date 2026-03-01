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

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.upde.metrics import UPDEState


class AuditLogger:
    """Append-only JSONL audit log for UPDE simulation steps."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._fh = self._path.open("a", encoding="utf-8")

    def log_step(
        self, step: int, upde_state: UPDEState, actions: list[ControlAction]
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
