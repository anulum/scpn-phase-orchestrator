# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL trace-result round-trip through the audit stream

"""Seal STL trace results into the SHA-256 audit event stream and replay them.

`STLMonitor.evaluate_result` produces an :class:`STLTraceResult` whose
`to_audit_record` payload is JSON-safe. This module writes those payloads
through :class:`~scpn_phase_orchestrator.runtime.audit_stream.EventStreamWriter`
so each STL verdict becomes a hash-chained, optionally HMAC-signed audit event,
and reconstructs the original :class:`STLTraceResult` objects from the sealed
stream after verifying its integrity. This is the "replay" path: the STL
evidence a run emitted can be recovered, byte-for-byte, from a tamper-evident
log rather than trusted in memory.

A bounded STL operator over a window past the trace end yields a vacuous
robustness of ``+inf``/``-inf``; such a value cannot be represented in the
JSON-backed stream and is rejected at the sealing boundary rather than silently
coerced.
"""

from __future__ import annotations

import math
from pathlib import Path

from scpn_phase_orchestrator.monitor.stl import STLTraceResult
from scpn_phase_orchestrator.runtime.audit_stream import (
    EventStreamWriter,
    read_event_stream,
    verify_event_stream_integrity,
)

STL_AUDIT_EVENT_TYPE = "stl.trace_result"
_STL_RECORD_FIELDS = ("spec", "robustness", "satisfied", "backend")

__all__ = [
    "STL_AUDIT_EVENT_TYPE",
    "append_stl_result",
    "read_stl_results",
    "write_stl_results",
]


def _require_finite_robustness(result: STLTraceResult) -> None:
    """Reject an STL result whose robustness cannot be sealed as JSON.

    Parameters
    ----------
    result : STLTraceResult
        The STL verdict to seal.

    Raises
    ------
    ValueError
        If the robustness is not finite (a bounded vacuous window yields
        ``+inf``/``-inf``, which the JSON-backed SHA-256 stream cannot encode).
    """
    if not math.isfinite(result.robustness):
        raise ValueError(
            "STL robustness must be finite to seal into the audit stream; got "
            f"{result.robustness!r} (a bounded vacuous window yields an infinite "
            "robustness that the JSON-backed SHA-256 event stream cannot encode)"
        )


def append_stl_result(writer: EventStreamWriter, result: STLTraceResult) -> None:
    """Append one STL trace result to an open audit stream.

    Parameters
    ----------
    writer : EventStreamWriter
        An open audit-stream writer.
    result : STLTraceResult
        The STL verdict to seal under the :data:`STL_AUDIT_EVENT_TYPE` event.

    Raises
    ------
    ValueError
        If the robustness is not finite.
    """
    _require_finite_robustness(result)
    writer.write(result.to_audit_record(), event_type=STL_AUDIT_EVENT_TYPE)


def write_stl_results(
    path: str | Path,
    results: list[STLTraceResult],
    *,
    stream_id: str = "spo-stl-audit",
) -> Path:
    """Seal a batch of STL trace results into a fresh or appended stream file.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the audit stream file.
    results : list[STLTraceResult]
        The STL verdicts to seal, in order.
    stream_id : str
        Stream identifier recorded in each event envelope.

    Returns
    -------
    Path
        The stream file path written.

    Raises
    ------
    ValueError
        If any result has a non-finite robustness.
    """
    writer = EventStreamWriter(path, stream_id=stream_id)
    try:
        for result in results:
            append_stl_result(writer, result)
    finally:
        writer.close()
    return writer.path


def _stl_result_from_payload(payload: dict[str, object]) -> STLTraceResult:
    """Reconstruct an STL trace result from a sealed audit payload.

    Parameters
    ----------
    payload : dict[str, object]
        A decoded STL audit event payload.

    Returns
    -------
    STLTraceResult
        The reconstructed verdict.

    Raises
    ------
    ValueError
        If the payload does not carry exactly the STL record fields with the
        expected types.
    """
    if set(payload) != set(_STL_RECORD_FIELDS):
        raise ValueError(
            f"STL audit payload must carry exactly {_STL_RECORD_FIELDS}, "
            f"got keys {sorted(payload)}"
        )
    spec, robustness, satisfied, backend = (
        payload["spec"],
        payload["robustness"],
        payload["satisfied"],
        payload["backend"],
    )
    if not isinstance(spec, str) or not isinstance(backend, str):
        raise ValueError("STL audit payload 'spec' and 'backend' must be strings")
    if not isinstance(satisfied, bool):
        raise ValueError("STL audit payload 'satisfied' must be a boolean")
    if isinstance(robustness, bool) or not isinstance(robustness, (int, float)):
        raise ValueError("STL audit payload 'robustness' must be a real number")
    return STLTraceResult(
        spec=spec,
        robustness=float(robustness),
        satisfied=satisfied,
        backend=backend,
    )


def read_stl_results(path: str | Path) -> list[STLTraceResult]:
    """Replay the STL trace results sealed in an audit stream.

    The stream's payload digests, sequence continuity, hash chain, and (when an
    audit key is configured) signatures are verified before any record is
    reconstructed, so a tampered stream is rejected rather than replayed.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the audit stream file.

    Returns
    -------
    list[STLTraceResult]
        The STL verdicts recovered from the stream, in recorded order.

    Raises
    ------
    ValueError
        If the stream fails integrity verification or an STL payload is
        malformed.
    """
    events = read_event_stream(path)
    ok, _ = verify_event_stream_integrity(events)
    if not ok:
        raise ValueError("STL audit stream failed integrity verification")
    return [
        _stl_result_from_payload(event.payload)
        for event in events
        if event.event_type == STL_AUDIT_EVENT_TYPE
    ]
