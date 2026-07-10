# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance evidence from a twin-confidence score

"""Derive twin-confidence assurance evidence from a serialised confidence score.

The twin-confidence monitor
(:mod:`scpn_phase_orchestrator.monitor.twin_confidence`) scores each digital-twin
tick against a calibrated baseline and produces a
:class:`~scpn_phase_orchestrator.monitor.twin_confidence.TwinConfidenceScore`
whose ``to_audit_record()`` is a JSON-safe mapping (calibrated confidence,
operator status, raw divergences, one-sided z-scores, band flags, backend, and a
content hash). This module maps that record into a single
:class:`~scpn_phase_orchestrator.assurance.evidence.EvidenceItem` in the
``twin_confidence`` category, so a conformity package can attest the live drift
monitoring the deployment ran — the one evidence category the assurance-case
clause map references but no producer previously emitted.

The helper consumes the serialised score (a ``Mapping``), not the
``TwinConfidenceScore`` object, mirroring
:func:`~scpn_phase_orchestrator.assurance.formal_evidence.build_formal_verification_evidence`
and :func:`~scpn_phase_orchestrator.assurance.run_evidence.build_run_evidence`:
the assurance package stays free of the monitor's numeric import chain and can
attest a score persisted to disk. It restates the score verbatim and never
fabricates a confidence value — a record missing a required field or carrying a
confidence outside ``[0, 1]`` is rejected rather than coerced.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from scpn_phase_orchestrator.assurance.evidence import (
    TWIN_CONFIDENCE,
    EvidenceItem,
    build_evidence_item,
)


def _require_non_empty_str(record: Mapping[str, object], key: str) -> str:
    """Return ``record[key]`` as a non-empty string, else raise ``ValueError``."""
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"twin-confidence score field {key!r} must be a non-empty string"
        )
    return value


def _require_unit_confidence(record: Mapping[str, object]) -> float:
    """Return ``record['confidence']`` as a finite float in ``[0, 1]``.

    Parameters
    ----------
    record:
        The serialised twin-confidence score.

    Returns
    -------
    float
        The calibrated confidence value.

    Raises
    ------
    ValueError
        If ``confidence`` is absent, boolean, non-numeric, non-finite, or outside
        the calibrated ``[0, 1]`` range.
    """
    value = record.get("confidence")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("twin-confidence score field 'confidence' must be a number")
    confidence = float(value)
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "twin-confidence score field 'confidence' must be a finite value in [0, 1]"
        )
    return confidence


def build_twin_confidence_evidence(
    score_record: Mapping[str, object],
) -> EvidenceItem:
    """Build a twin-confidence evidence item from a serialised confidence score.

    Parameters
    ----------
    score_record:
        A JSON-safe ``TwinConfidenceScore.to_audit_record()`` mapping. It must
        carry a ``confidence`` in ``[0, 1]``, a non-empty ``status``, and a
        non-empty ``score_hash``.

    Returns
    -------
    EvidenceItem
        A ``twin_confidence`` evidence item whose record is the score verbatim,
        summarising the operator status and calibrated confidence.

    Raises
    ------
    ValueError
        If the score is not a mapping, a required field is missing, or a field has
        the wrong type or an out-of-range value.
    """
    if not isinstance(score_record, Mapping):
        raise ValueError("twin-confidence score must be a mapping")

    confidence = _require_unit_confidence(score_record)
    status = _require_non_empty_str(score_record, "status")
    _require_non_empty_str(score_record, "score_hash")

    summary = (
        f"Twin-confidence score {status!r} at calibrated confidence {confidence:.3f}"
    )
    return build_evidence_item(
        evidence_id="twin-confidence-score",
        category=TWIN_CONFIDENCE,
        summary=summary,
        record=dict(score_record),
    )


__all__ = ["build_twin_confidence_evidence"]
