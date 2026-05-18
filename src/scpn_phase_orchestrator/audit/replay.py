# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Deterministic replay engine

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from math import isfinite
from numbers import Real
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.audit.signing import (
    SIGNATURE_ALGORITHM,
    audit_verification_keys,
    key_id_for_secret,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

_log = logging.getLogger(__name__)

__all__ = ["ReplayEngine"]

_AUDIT_SCHEMA_VERSION = 1
_ZERO_HASH = "0" * 64
_ENGINE_METHODS = frozenset(("euler", "rk4", "rk45"))
_UPDE_REPLAY_FIELDS = frozenset(("phases", "omegas", "knm", "alpha"))
_SL_REPLAY_FIELDS = frozenset(("phases", "omegas", "knm", "alpha"))


def _layer_records(step_data: dict) -> list[dict]:
    layers = step_data.get("layers", [])
    if not isinstance(layers, list):
        return []
    return [layer for layer in layers if isinstance(layer, dict)]


def _numeric_value(value: object) -> float:
    if isinstance(value, Real) and not isinstance(value, bool):
        parsed = float(value)
        if isfinite(parsed):
            return parsed
    return 0.0


def _required_header_value(header: dict, field: str) -> object:
    if field not in header:
        raise ValueError(f"audit header missing {field}")
    return header[field]


def _required_header_int(header: dict, field: str) -> int:
    value = _required_header_value(header, field)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError(f"audit header {field} must be a positive integer")


def _required_header_float(header: dict, field: str) -> float:
    value = _required_header_value(header, field)
    parsed = _numeric_value(value)
    if parsed > 0.0:
        return parsed
    raise ValueError(f"audit header {field} must be a positive finite number")


def _header_method(header: dict) -> str:
    value = header.get("method", "euler")
    if isinstance(value, str) and value in _ENGINE_METHODS:
        return value
    allowed = ", ".join(sorted(_ENGINE_METHODS))
    raise ValueError(f"audit header method must be one of: {allowed}")


def _header_amplitude_mode(header: dict) -> bool:
    value = header.get("amplitude_mode", False)
    if isinstance(value, bool):
        return value
    raise ValueError("audit header amplitude_mode must be a boolean")


def _has_fields(entry: dict, fields: frozenset[str]) -> bool:
    return all(field in entry for field in fields)


class ReplayEngine:
    """Replay and verify determinism of JSONL audit logs."""

    def __init__(self, log_path: str | Path):
        self._log_path = Path(log_path)

    def load(self) -> list[dict]:
        """Read and parse all JSONL entries from the audit log file."""
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
            LayerState(
                R=_numeric_value(ld.get("R", 0.0)),
                psi=_numeric_value(ld.get("psi", 0.0)),
            )
            for ld in _layer_records(step_data)
        ]
        return UPDEState(
            layers=layers,
            cross_layer_alignment=np.zeros((len(layers), len(layers))),
            stability_proxy=_numeric_value(step_data.get("stability", 0.0)),
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
        n_oscillators = _required_header_int(header, "n_oscillators")
        dt = _required_header_float(header, "dt")
        method = _header_method(header)
        if _header_amplitude_mode(header):
            return StuartLandauEngine(
                n_oscillators=n_oscillators,
                dt=dt,
                method=method,
            )
        return UPDEEngine(
            n_oscillators=n_oscillators,
            dt=dt,
            method=method,
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
            if not _has_fields(curr, _UPDE_REPLAY_FIELDS) or "phases" not in nxt:
                return False, verified
            try:
                phases = np.asarray(curr["phases"])
                omegas = np.asarray(curr["omegas"])
                knm_arr = np.asarray(curr["knm"])
                alpha_arr = np.asarray(curr["alpha"])
                zeta = curr.get("zeta", 0.0)
                psi_drive = curr.get("psi_drive", 0.0)

                computed = engine.step(
                    phases, omegas, knm_arr, zeta, psi_drive, alpha_arr
                )
                logged_next = np.asarray(nxt["phases"])
            except (TypeError, ValueError):
                return False, verified

            if not np.allclose(computed, logged_next, atol=atol):
                return False, verified
            verified += 1

        return True, verified

    @staticmethod
    def verify_integrity(entries: list[dict]) -> tuple[bool, int]:
        """Verify the SHA256 hash chain of audit log entries.

        Returns (all_valid, n_verified).  Legacy logs without ``_hash``
        fields return (True, 0) unless ``SPO_AUDIT_KEY`` is configured.
        """
        try:
            audit_keys = audit_verification_keys()
        except ValueError:
            return False, 0
        require_signature = bool(audit_keys)
        prev = _ZERO_HASH
        verified = 0
        expected_sequence = 1
        for entry in entries:
            stored = entry.get("_hash")
            if stored is None:
                if require_signature:
                    return False, verified
                continue
            without_hash = {k: v for k, v in entry.items() if k != "_hash"}
            json_line = json.dumps(without_hash, separators=(",", ":"), sort_keys=True)
            expected = hashlib.sha256((prev + json_line).encode()).hexdigest()
            if expected != stored:
                return False, verified
            if require_signature and not _verify_hmac_signature(
                entry,
                audit_keys,
                expected_previous_hash=prev,
                expected_sequence=expected_sequence,
            ):
                return False, verified
            prev = stored
            verified += 1
            expected_sequence += 1
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
            if not _has_fields(curr, _SL_REPLAY_FIELDS) or "phases" not in nxt:
                return False, verified

            try:
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
                    _log.warning(
                        "SL replay step %d: amplitude fields missing, skipping", i
                    )
                    continue

                knm_flat = np.asarray(curr["knm"])
                alpha_flat = np.asarray(curr["alpha"])
                zeta = curr.get("zeta", 0.0)
                psi_drive = curr.get("psi_drive", 0.0)

                knm_arr = knm_flat.reshape(n, n) if knm_flat.ndim == 1 else knm_flat
                alpha_arr = (
                    alpha_flat.reshape(n, n) if alpha_flat.ndim == 1 else alpha_flat
                )

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
            except (TypeError, ValueError):
                return False, verified

            try:
                if "amplitudes" in nxt:
                    logged_next = np.concatenate(
                        [np.asarray(nxt["phases"]), np.asarray(nxt["amplitudes"])]
                    )
                else:
                    logged_next = np.asarray(nxt["phases"])

                if not np.allclose(computed, logged_next, atol=atol):
                    return False, verified
            except (TypeError, ValueError):
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


def _verify_hmac_signature(
    entry: dict,
    audit_keys: dict[str, str],
    *,
    expected_previous_hash: str,
    expected_sequence: int,
) -> bool:
    signature = entry.get("_signature")
    if not isinstance(signature, dict):
        return False
    if signature.get("algorithm") != SIGNATURE_ALGORITHM:
        return False
    signature_value = signature.get("value")
    if not isinstance(signature_value, str) or len(signature_value) != 64:
        return False
    key_id = signature.get("key_id")
    if not isinstance(key_id, str):
        return False
    audit_key = audit_keys.get(key_id)
    if audit_key is None:
        return False
    if key_id != key_id_for_secret(audit_key):
        return False
    if entry.get("_audit_schema_version") != _AUDIT_SCHEMA_VERSION:
        return False
    if entry.get("_audit_mode") != "hmac-signed":
        return False
    if entry.get("_audit_sequence") != expected_sequence:
        return False
    if entry.get("_previous_hash") != expected_previous_hash:
        return False
    stream_id = entry.get("_audit_stream_id")
    timestamp_ns = entry.get("_audit_timestamp_unix_ns")
    payload_hash = entry.get("_payload_hash")
    if not isinstance(stream_id, str) or stream_id == "":
        return False
    if not isinstance(timestamp_ns, int) or timestamp_ns < 0:
        return False
    if not isinstance(payload_hash, str) or len(payload_hash) != 64:
        return False

    payload = _payload_without_audit_metadata(entry)
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    expected_payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    if not hmac.compare_digest(payload_hash, expected_payload_hash):
        return False

    signing_metadata = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": key_id,
    }
    signing_material = {
        "audit_mode": "hmac-signed",
        "metadata": signing_metadata,
        "payload_hash": payload_hash,
        "previous_event_hash": expected_previous_hash,
        "schema_version": _AUDIT_SCHEMA_VERSION,
        "sequence": expected_sequence,
        "stream_id": stream_id,
        "timestamp_unix_ns": timestamp_ns,
    }
    expected_signature = hmac.new(
        audit_key.encode(),
        json.dumps(signing_material, separators=(",", ":"), sort_keys=True).encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(signature_value, expected_signature)


def _payload_without_audit_metadata(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in {"_hash", "_signature"}
        and not key.startswith("_audit_")
        and key not in {"_payload_hash", "_previous_hash"}
    }
