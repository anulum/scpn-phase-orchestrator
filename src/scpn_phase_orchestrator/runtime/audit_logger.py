# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit event logger

"""Append-only JSONL logger for replayable SPO runs.

`AuditLogger` records headers, simulation steps, supervisor actions, and named
events with a SHA-256 hash chain. When `SPO_AUDIT_KEY` is configured, existing
unsigned streams are rejected and new records carry HMAC metadata so downstream
replay and reporting can verify provenance before trusting an audit trail.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.exceptions import AuditError
from scpn_phase_orchestrator.runtime.audit_signing import (
    SIGNATURE_ALGORITHM,
    key_id_for_secret,
)
from scpn_phase_orchestrator.runtime.audit_stream import (
    EventStreamWriter,
    read_event_stream,
)
from scpn_phase_orchestrator.runtime.audit_stream import (
    verify_event_stream_integrity as verify_event_stream_events,
)
from scpn_phase_orchestrator.runtime.network_security import is_production_mode
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["AuditLogger", "AuditStreamIntegrityResult"]

FloatArray = NDArray[np.float64]

_AUDIT_SCHEMA_VERSION = 1
_DEFAULT_STREAM_ID = "spo-audit-jsonl"
_ZERO_HASH = "0" * 64


@dataclass(frozen=True, slots=True)
class AuditStreamIntegrityResult:
    """Close-time integrity summary for a protobuf audit event stream.

    Parameters
    ----------
    event_stream_path : str
        Filesystem path to the verified protobuf audit event stream.
    ok : bool
        Whether payload digests, sequence numbers, hash links, and required
        signatures verified.
    verified_events : int
        Number of consecutive events verified before the first failure, or all
        events when ``ok`` is ``True``.
    """

    event_stream_path: str
    ok: bool
    verified_events: int

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit-stream integrity record.

        Returns
        -------
        dict[str, object]
            A JSON-safe integrity summary for downstream reports.
        """
        return {
            "event_stream_path": self.event_stream_path,
            "ok": self.ok,
            "verified_events": self.verified_events,
        }


class AuditLogger:
    """Append-only JSONL audit log for UPDE simulation steps."""

    def __init__(self, path: str | Path, *, event_stream: str | Path | None = None):
        if not isinstance(path, (str, Path)):
            raise AuditError(f"audit path must be str or Path, got {path!r}")
        if event_stream is not None and not isinstance(event_stream, (str, Path)):
            raise AuditError(
                f"event_stream path must be str, Path, or None, got {event_stream!r}"
            )
        if not str(path).strip():
            raise AuditError("audit path must be a non-empty path")
        if event_stream is not None and not str(event_stream).strip():
            raise AuditError("event_stream path must be non-empty when provided")
        self._path = Path(path)
        if self._path.exists() and self._path.is_dir():
            raise AuditError(
                f"audit path must be a file path, got directory {self._path}"
            )
        self._prev_hash, self._sequence = self._load_previous_state()
        self._audit_key = os.environ.get("SPO_AUDIT_KEY")
        self._stream_id = _DEFAULT_STREAM_ID
        if self._audit_key is not None and self._audit_key == "":
            msg = "SPO_AUDIT_KEY must not be empty"
            raise AuditError(msg)
        if self._audit_key is None and is_production_mode("SPO_AUDIT"):
            raise AuditError(
                "SPO_AUDIT_KEY is required in production mode "
                "(SPO_AUDIT_ENV/SPO_AUDIT_PROFILE or SPO_ENV/SPO_PROFILE = "
                "'production'): refusing to write an unsigned, unverifiable audit "
                "trail. Set SPO_AUDIT_KEY to enable HMAC signing."
            )
        if self._audit_key is not None and self._sequence > 0:
            self._ensure_existing_stream_is_signed()
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)
        self._event_stream = (
            EventStreamWriter(event_stream) if event_stream is not None else None
        )
        self._event_stream_integrity: AuditStreamIntegrityResult | None = None

    def _load_previous_state(self) -> tuple[str, int]:
        """Load the previous audit-chain state from disk, else raise."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return _ZERO_HASH, 0
        previous = _ZERO_HASH
        sequence = 0
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = _loads_audit_json(stripped)
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
        """Assert an existing audit stream is signed, else raise."""
        with self._path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = _loads_audit_json(stripped)
                if not isinstance(record.get("_signature"), dict):
                    msg = (
                        "SPO_AUDIT_KEY configured but existing audit log "
                        f"contains unsigned record at line {line_number}"
                    )
                    raise AuditError(msg)

    def _write_record(self, record: dict[str, Any]) -> None:
        """Append an audit record to the JSONL log."""
        clean = {k: v for k, v in record.items() if k != "_hash"}
        if self._audit_key is not None:
            clean = self._attach_signature_metadata(clean)
        else:
            clean["_audit_mode"] = "unsigned-development"
            clean["_previous_hash"] = self._prev_hash
            clean["_payload_hash"] = _payload_hash(clean)
        json_line = _dumps_audit_json(clean, compact=True)
        digest = hashlib.sha256((self._prev_hash + json_line).encode()).hexdigest()
        self._prev_hash = digest
        stored = {**clean, "_hash": digest}
        self._fh.write(_dumps_audit_json(stored) + "\n")
        if self._event_stream is not None:
            self._event_stream.write(stored)

    def _attach_signature_metadata(self, clean: dict[str, Any]) -> dict[str, Any]:
        """Attach signature metadata to an audit record."""
        self._sequence += 1
        payload = _payload_without_audit_metadata(clean)
        payload_hash = _payload_hash(payload)
        timestamp_ns = time.time_ns()
        key = self._audit_key
        if key is None:
            msg = "audit key missing during signature construction"
            raise AuditError(msg)
        key_id = key_id_for_secret(key)
        signature_metadata = {
            "algorithm": SIGNATURE_ALGORITHM,
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
            _dumps_audit_json(signing_material, compact=True).encode(),
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
        control_mode: str = "supervisor_policy",
        binding_config: dict[str, object] | None = None,
        binding_summary: dict[str, object] | None = None,
    ) -> None:
        """Engine configuration record for replay reconstruction.

        Parameters
        ----------
        n_oscillators : int
            Number of oscillators in the system.
        dt : float
            Integration step size.
        method : str
            Integration method (``euler``, ``rk4``, or ``rk45``).
        seed : int | None
            Seed for the deterministic RNG, or ``None``.
        amplitude_mode : bool
            Whether the engine runs in Stuart-Landau amplitude mode.
        control_mode : str
            Live control surface used by the simulation core.
        binding_config : dict[str, object] | None
            Resolved binding configuration, or ``None``.
        binding_summary : dict[str, object] | None
            Resolved binding summary, or ``None``.

        Raises
        ------
        AuditError
            If the audit log cannot be written.
        """
        if isinstance(n_oscillators, bool) or not isinstance(n_oscillators, int):
            raise AuditError(
                f"n_oscillators must be a positive integer, got {n_oscillators!r}"
            )
        if n_oscillators <= 0:
            raise AuditError(
                f"n_oscillators must be a positive integer, got {n_oscillators!r}"
            )
        if (
            isinstance(dt, bool)
            or not isinstance(dt, (int, float))
            or not isfinite(float(dt))
        ):
            raise AuditError(f"dt must be a finite positive real, got {dt!r}")
        if float(dt) <= 0.0:
            raise AuditError(f"dt must be a finite positive real, got {dt!r}")
        if not isinstance(method, str) or not method.strip():
            raise AuditError(f"method must be a non-empty string, got {method!r}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise AuditError(f"seed must be integer or None, got {seed!r}")
        if isinstance(amplitude_mode, bool) is False:
            raise AuditError(f"amplitude_mode must be bool, got {amplitude_mode!r}")
        if not isinstance(control_mode, str) or not control_mode.strip():
            raise AuditError(
                f"control_mode must be a non-empty string, got {control_mode!r}"
            )
        if binding_config is not None and not isinstance(binding_config, dict):
            raise AuditError(
                "binding_config must be dict[str, object] or None, "
                f"got {binding_config!r}"
            )
        if binding_summary is not None and not isinstance(binding_summary, dict):
            raise AuditError(
                "binding_summary must be dict[str, object] or None, "
                f"got {binding_summary!r}"
            )
        record: dict[str, Any] = {
            "header": True,
            "n_oscillators": n_oscillators,
            "dt": dt,
            "method": method,
        }
        if seed is not None:
            record["seed"] = seed
        if amplitude_mode:
            record["amplitude_mode"] = True
        record["control_mode"] = control_mode

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
        phases: FloatArray | None = None,
        omegas: FloatArray | None = None,
        knm: FloatArray | None = None,
        alpha: FloatArray | None = None,
        zeta: float = 0.0,
        psi_drive: float = 0.0,
        amplitudes: FloatArray | None = None,
        mu: FloatArray | None = None,
        knm_r: FloatArray | None = None,
        epsilon: float | None = None,
        channel_runtime: dict[str, object] | None = None,
    ) -> None:
        """Write one simulation step to the audit log with optional full state.

        Parameters
        ----------
        step : int
            Zero-based simulation step index.
        upde_state : UPDEState
            The UPDE state to record or export.
        actions : list[ControlAction]
            The control actions recorded for the step.
        phases : FloatArray | None
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray | None
            Natural frequencies in rad/s, shape ``(N,)``.
        knm : FloatArray | None
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        alpha : FloatArray | None
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
        zeta : float
            External drive strength ``ζ``.
        psi_drive : float
            External drive reference phase in radians.
        amplitudes : FloatArray | None
            Oscillator amplitudes, shape ``(N,)``, or ``None``.
        mu : FloatArray | None
            Per-oscillator linear growth parameters, or ``None``.
        knm_r : FloatArray | None
            Amplitude coupling matrix, shape ``(N, N)``, or ``None``.
        epsilon : float | None
            Stuart-Landau amplitude coupling factor, or ``None``.
        channel_runtime : dict[str, object] | None
            N-channel runtime evidence, or ``None``.

        Raises
        ------
        AuditError
            If the audit log cannot be written.
        """
        if isinstance(step, bool) or not isinstance(step, int):
            raise AuditError(f"step must be a non-negative integer, got {step!r}")
        if step < 0:
            raise AuditError(f"step must be a non-negative integer, got {step!r}")
        if not isinstance(upde_state, UPDEState):
            raise AuditError(f"upde_state must be UPDEState, got {upde_state!r}")
        if not isinstance(actions, list):
            raise AuditError(f"actions must be list[ControlAction], got {actions!r}")
        for idx, action in enumerate(actions):
            if not isinstance(action, ControlAction):
                raise AuditError(
                    f"actions[{idx}] must be ControlAction, got {action!r}"
                )
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
            for name, arr in (
                ("phases", phases),
                ("omegas", omegas),
                ("knm", knm),
                ("alpha", alpha),
            ):
                if not np.isfinite(arr).all():
                    raise AuditError(f"{name} must contain only finite values")
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
            if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
                raise AuditError(f"epsilon must be finite real, got {epsilon!r}")
            if not isfinite(float(epsilon)):
                raise AuditError(f"epsilon must be finite real, got {epsilon!r}")
            record["epsilon"] = epsilon
        if channel_runtime is not None:
            if not isinstance(channel_runtime, dict):
                raise AuditError(
                    "channel_runtime must be dict[str, object] when provided"
                )
            record["channel_runtime"] = channel_runtime
        self._write_record(record)

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Write a named event with arbitrary data to the audit log.

        Parameters
        ----------
        event_type : str
            Named event type, or ``None``.
        data : dict[str, Any]
            Arbitrary JSON-safe event payload.

        Raises
        ------
        AuditError
            If the audit log cannot be written.
        """
        if not isinstance(event_type, str) or not event_type.strip():
            raise AuditError(
                f"event_type must be a non-empty string, got {event_type!r}"
            )
        if not isinstance(data, dict):
            raise AuditError(f"data must be dict[str, object], got {data!r}")
        record = {"ts": time.time(), "event": event_type, **data}
        self._write_record(record)

    def close(self) -> None:
        """Flush and close the audit log file handle."""
        self._fh.flush()
        if self._event_stream is not None:
            if self._event_stream_integrity is None:
                self.verify_event_stream_integrity()
            self._event_stream.close()
        self._fh.close()

    def verify_event_stream_integrity(self) -> AuditStreamIntegrityResult | None:
        """Verify the configured protobuf event stream after flushing writes.

        Returns
        -------
        AuditStreamIntegrityResult | None
            The integrity summary when this logger owns an event stream, else
            ``None`` for JSONL-only audit logs.
        """
        if self._event_stream is None:
            return None
        self._fh.flush()
        self._event_stream.flush()
        path = self._event_stream.path
        events = read_event_stream(path)
        ok, verified_events = verify_event_stream_events(events)
        result = AuditStreamIntegrityResult(
            event_stream_path=str(path),
            ok=ok,
            verified_events=verified_events,
        )
        self._event_stream_integrity = result
        return result

    @property
    def event_stream_integrity(self) -> AuditStreamIntegrityResult | None:
        """Return the most recent event-stream integrity summary, if any.

        Returns
        -------
        AuditStreamIntegrityResult | None
            The last computed event-stream integrity result, or ``None`` when no
            protobuf stream has been verified.
        """
        return self._event_stream_integrity

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()


def _payload_without_audit_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return the payload with audit metadata stripped."""
    return {
        key: value
        for key, value in record.items()
        if key not in {"_hash", "_signature"}
        and not key.startswith("_audit_")
        and key not in {"_payload_hash", "_previous_hash"}
    }


def _payload_hash(record: dict[str, Any]) -> str:
    """Return the canonical hash of a payload."""
    payload = _payload_without_audit_metadata(record)
    payload_json = _dumps_audit_json(payload, compact=True)
    return hashlib.sha256(payload_json.encode()).hexdigest()


def _reject_json_constant(value: str) -> None:
    """Raise if the JSON value is a forbidden constant."""
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _loads_audit_json(payload: str) -> dict[str, Any]:
    """Load and validate an audit JSON payload, else raise."""
    try:
        decoded = json.loads(payload, parse_constant=_reject_json_constant)
    except json.JSONDecodeError:
        raise
    except ValueError as exc:
        raise AuditError("audit payload must contain only finite JSON numbers") from exc
    if not isinstance(decoded, dict):
        raise AuditError("audit payload must be a JSON object")
    return decoded


def _dumps_audit_json(payload: dict[str, Any], *, compact: bool = False) -> str:
    """Return the canonical JSON serialisation of an audit payload."""
    try:
        if compact:
            return json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        return json.dumps(payload, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise AuditError("audit payload must contain only finite JSON values") from exc
