# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Session-start coherence gate

"""Session-start validation gate for extractor, imprint, and coherence inputs.

The validator checks startup preconditions across extractor quality signals,
imprint availability, and initial coherence metrics before a session is allowed
to proceed. It returns explicit warnings and errors without mutating source
state or triggering actuation, keeping the gate suitable for dry-run previews,
operator review, and fail-closed orchestration handoffs.

The gate is fail-closed on malformed evidence: phase and imprint vectors must
be one-dimensional real numeric arrays with finite entries and the expected
oscillator count, and extractor quality values must be finite floats in
``[0, 1]``. Any violation is recorded as an error and fails the gate rather
than being silently skipped; quality scoring and coherence metrics are only
computed from evidence that passed validation. ``n_osc`` is a caller-supplied
structural parameter, so an invalid ``n_osc`` raises instead of reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

__all__ = ["SessionCoherenceReport", "check_session_start"]

TWO_PI = 2.0 * np.pi
FloatArray: TypeAlias = NDArray[np.float64]

# numpy dtype kinds admissible as phase/imprint evidence: real floats and
# exact integers. Booleans, complex numbers, strings, and objects are not
# phase evidence and must not be silently coerced.
_REAL_KINDS = frozenset("fiu")


@dataclass
class SessionCoherenceReport:
    """Results of the session-start coherence gate check."""

    quality_scores: dict[str, float] = field(default_factory=dict)
    initial_r: float = 0.0
    imprint_level: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    passed: bool = True


def _validate_real_vector(
    name: str, candidate: object, report: SessionCoherenceReport
) -> FloatArray | None:
    """Validate one evidence vector; record an error and return None on failure.

    Parameters
    ----------
    name : str
        Evidence name used in error messages.
    candidate : object
        Value supplied as the evidence vector.
    report : SessionCoherenceReport
        Report collecting validation errors.

    Returns
    -------
    FloatArray | None
        The vector as float64, or None when validation failed.
    """
    if not isinstance(candidate, np.ndarray):
        report.errors.append(
            f"{name} must be a numpy array, got {type(candidate).__name__}"
        )
        report.passed = False
        return None
    if candidate.dtype.kind not in _REAL_KINDS:
        report.errors.append(
            f"{name} has non-real dtype {candidate.dtype!s}; "
            "boolean, complex, text, and object evidence is rejected"
        )
        report.passed = False
        return None
    if candidate.ndim != 1:
        report.errors.append(
            f"{name} must be one-dimensional, got shape {candidate.shape}"
        )
        report.passed = False
        return None
    vector = np.asarray(candidate, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        report.errors.append(f"{name} contains non-finite entries")
        report.passed = False
        return None
    return vector


def _validate_quality_evidence(
    phase_states: list[PhaseState], report: SessionCoherenceReport
) -> bool:
    """Check every extractor quality value is a finite float in ``[0, 1]``.

    Parameters
    ----------
    phase_states : list[PhaseState]
        Extracted states whose quality fields feed the scorer.
    report : SessionCoherenceReport
        Report collecting validation errors.

    Returns
    -------
    bool
        True when all quality values are admissible.
    """
    valid = True
    for ps in phase_states:
        quality = ps.quality
        if (
            isinstance(quality, bool)
            or not isinstance(quality, (int, float))
            or not np.isfinite(quality)
            or not 0.0 <= float(quality) <= 1.0
        ):
            report.errors.append(
                f"Phase state {ps.node_id}: quality must be a finite float "
                f"in [0, 1], got {quality!r}"
            )
            report.passed = False
            valid = False
    return valid


def check_session_start(
    phase_states: list[PhaseState],
    initial_phases: FloatArray,
    imprint_state: ImprintState,
    n_osc: int,
) -> SessionCoherenceReport:
    """Validate extraction quality, imprint consistency, and initial coherence.

    Parameters
    ----------
    phase_states : list[PhaseState]
        extracted states from all configured channels.
    initial_phases : FloatArray
        phase array that will seed the UPDE engine.
    imprint_state : ImprintState
        loaded (or fresh) imprint state.
    n_osc : int
        expected oscillator count; must be a positive int.

    Returns
    -------
    SessionCoherenceReport
        SessionCoherenceReport with pass/fail, quality scores, and diagnostics.

    Raises
    ------
    TypeError
        If ``n_osc`` is not an int (bool excluded).
    ValueError
        If ``n_osc`` is not positive.
    """
    if isinstance(n_osc, bool) or not isinstance(n_osc, int):
        raise TypeError(f"n_osc must be an int, got {type(n_osc).__name__}")
    if n_osc < 1:
        raise ValueError(f"n_osc must be positive, got {n_osc}")

    report = SessionCoherenceReport()
    scorer = PhaseQualityScorer()

    # Quality per channel — only scored when every quality value is admissible;
    # a poisoned quality would silently disable the thresholds below.
    if _validate_quality_evidence(phase_states, report):
        by_channel: dict[str, list[PhaseState]] = {}
        for ps in phase_states:
            by_channel.setdefault(ps.channel, []).append(ps)

        for ch, states in by_channel.items():
            q = scorer.score(states)
            report.quality_scores[ch] = q
            if q < 0.3:
                report.warnings.append(
                    f"Channel {ch}: low quality ({q:.2f}); extraction may be unreliable"
                )

        if scorer.detect_collapse(phase_states):
            report.errors.append(
                "Signal collapse: majority of extractors below threshold"
            )
            report.passed = False

    # Imprint consistency
    m_k = _validate_real_vector("Imprint vector m_k", imprint_state.m_k, report)
    if m_k is not None:
        if m_k.shape[0] != n_osc:
            report.errors.append(f"Imprint size mismatch: {m_k.shape[0]} != {n_osc}")
            report.passed = False
        else:
            report.imprint_level = float(np.mean(m_k))

    # Initial coherence from extracted phases — the seed that drives the UPDE
    # engine, so a malformed or wrong-sized vector fails the gate.
    phases = _validate_real_vector("initial_phases", initial_phases, report)
    if phases is not None:
        if phases.shape[0] != n_osc:
            report.errors.append(
                f"Initial phase size mismatch: {phases.shape[0]} != {n_osc}"
            )
            report.passed = False
        else:
            r, _ = compute_order_parameter(phases)
            report.initial_r = float(r)
            if r < 0.05:
                report.warnings.append(
                    f"Low initial coherence (R={r:.3f}); starting from near-chaos"
                )

    return report
