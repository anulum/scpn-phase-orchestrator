# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit event logger

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.audit.stream import EventStreamWriter
from scpn_phase_orchestrator.exceptions import AuditError
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["AuditLogger"]

_AUDIT_SCHEMA_VERSION = 1
_DEFAULT_STREAM_ID = "spo-audit-jsonl"
_ZERO_HASH = "0" * 64
_SIGNATURE_ALGORITHM = "HMAC-SHA256"


class AuditLogger:
    """Append-only JSONL audit log for UPDE simulation steps."""

    def __init__(self, path: str | Path, *, event_stream: str | Path | None = None):
        self._path = Path(path)
        self._prev_hash, self._sequence = self._load_previous_state()
        self._audit_key = os.environ.get("SPO_AUDIT_KEY")
        self._stream_id = _DEFAULT_STREAM_ID
        if self._audit_key is not None and self._audit_key == "":
            msg = "SPO_AUDIT_KEY must not be empty"
            raise AuditError(msg)
        if self._audit_key is not None and self._sequence > 0:
            self._ensure_existing_stream_is_signed()
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)
        self._event_stream = (
            EventStreamWriter(event_stream) if event_stream is not None else None
        )

    def _load_previous_state(self) -> tuple[str, int]:
        if not self._path.exists() or self._path.stat().st_size == 0:
            return _ZERO_HASH, 0
        previous = _ZERO_HASH
        sequence = 0
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                stored = record.get("_hash")
                if isinstance(stored, str) and len(stored) == 64:
                    previous = stored
                stored_sequence = record.get("_audit_sequence")
                if isinstance(stored_sequence, int) and stored_sequence > sequence:
                    sequence = stored_sequence
                else:
                    sequence += 1
        return previous, sequence

    def _ensure_existing_stream_is_signed(self) -> None:
        with self._path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if not isinstance(record.get("_signature"), dict):
                    msg = (
                        "SPO_AUDIT_KEY configured but existing audit log "
                        f"contains unsigned record at line {line_number}"
                    )
                    raise AuditError(msg)

    def _write_record(self, record: dict) -> None:
        clean = {k: v for k, v in record.items() if k != "_hash"}
        if self._audit_key is not None:
            clean = self._attach_signature_metadata(clean)
        else:
            clean["_audit_mode"] = "unsigned-development"
            clean["_payload_hash"] = _payload_hash(clean)
        json_line = json.dumps(clean, separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256((self._prev_hash + json_line).encode()).hexdigest()
        self._prev_hash = digest
        stored = {**clean, "_hash": digest}
        self._fh.write(json.dumps(stored, sort_keys=True) + "\n")
        if self._event_stream is not None:
            self._event_stream.write(stored)

    def _attach_signature_metadata(self, clean: dict) -> dict:
        self._sequence += 1
        payload = _payload_without_audit_metadata(clean)
        payload_hash = _payload_hash(payload)
        timestamp_ns = time.time_ns()
        key = self._audit_key
        if key is None:
            msg = "audit key missing during signature construction"
            raise AuditError(msg)
        key_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        signature_metadata = {
            "algorithm": _SIGNATURE_ALGORITHM,
            "key_id": key_id,
        }
        audit_mode = "hmac-signed"
        signing_material = {
            "audit_mode": audit_mode,
            "metadata": signature_metadata,
            "payload_hash": payload_hash,
            "previous_event_hash": self._prev_hash,
            "schema_version": _AUDIT_SCHEMA_VERSION,
            "sequence": self._sequence,
            "stream_id": self._stream_id,
            "timestamp_unix_ns": timestamp_ns,
        }
        signature = hmac.new(
            key.encode(),
            json.dumps(
                signing_material, separators=(",", ":"), sort_keys=True
            ).encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            **clean,
            "_audit_schema_version": _AUDIT_SCHEMA_VERSION,
            "_audit_mode": audit_mode,
            "_audit_sequence": self._sequence,
            "_audit_stream_id": self._stream_id,
            "_audit_timestamp_unix_ns": timestamp_ns,
            "_payload_hash": payload_hash,
            "_previous_hash": self._prev_hash,
            "_signature": {
                **signature_metadata,
                "value": signature,
            },
        }

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


def _payload_without_audit_metadata(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in {"_hash", "_signature"}
        and not key.startswith("_audit_")
        and key not in {"_payload_hash", "_previous_hash"}
    }


def _payload_hash(record: dict) -> str:
    payload = _payload_without_audit_metadata(record)
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload_json.encode()).hexdigest()
