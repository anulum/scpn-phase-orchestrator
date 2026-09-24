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
oscillator count, the imprint vector must be non-negative (the imprint model
rejects negative accumulation), every extractor record must be a
:class:`PhaseState` whose quality is a finite real in ``[0, 1]`` and whose
amplitude is a finite, non-negative real. Amplitudes are the weights of the
per-channel quality score, so a non-finite amplitude would otherwise turn the
score into NaN and silently suppress the low-quality warning. Python and NumPy
reals are accepted; booleans, complex values, text and other objects are not.
Any violation is recorded as an error and fails the gate rather than being
silently skipped; quality scoring and coherence metrics are only computed from
evidence that passed validation. A session with no extractor records fails as
a signal collapse. ``n_osc`` is a caller-supplied structural parameter (a
Python or NumPy integer, not a boolean), so an invalid ``n_osc`` raises instead
of reporting.
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


def _is_real_scalar(value: object) -> bool:
    """Return whether ``value`` is a Python or NumPy real number, not a boolean."""
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _validate_real_vector(
    name: str,
    candidate: object,
    report: SessionCoherenceReport,
    *,
    non_negative: bool = False,
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
    non_negative : bool, optional
        Also reject negative entries.

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
    if non_negative and np.any(vector < 0.0):
        report.errors.append(f"{name} contains negative entries")
        report.passed = False
        return None
    return vector


def _validate_extractor_evidence(
    phase_states: list[PhaseState], report: SessionCoherenceReport
) -> bool:
    """Check the extractor records the quality scorer consumes.

    Every record must be a :class:`PhaseState`; its quality must be a finite
    real in ``[0, 1]`` and its amplitude, the scorer's weight, a finite real
    that is not negative.

    Parameters
    ----------
    phase_states : list[PhaseState]
        Extracted states whose quality and amplitude fields feed the scorer.
    report : SessionCoherenceReport
        Report collecting validation errors.

    Returns
    -------
    bool
        True when there is at least one record and all are admissible.
    """
    if not phase_states:
        report.errors.append(
            "Signal collapse: no extractor phase states supplied; the session "
            "has no extraction evidence"
        )
        report.passed = False
        return False
    valid = True
    for index, ps in enumerate(phase_states):
        if not isinstance(ps, PhaseState):
            report.errors.append(
                f"Phase state {index}: expected PhaseState, got {type(ps).__name__}"
            )
            report.passed = False
            valid = False
            continue
        quality = ps.quality
        if (
            not _is_real_scalar(quality)
            or not np.isfinite(quality)
            or not 0.0 <= float(quality) <= 1.0
        ):
            report.errors.append(
                f"Phase state {ps.node_id}: quality must be a finite float "
                f"in [0, 1], got {quality!r}"
            )
            report.passed = False
            valid = False
        amplitude = ps.amplitude
        if (
            not _is_real_scalar(amplitude)
            or not np.isfinite(amplitude)
            or float(amplitude) < 0.0
        ):
            report.errors.append(
                f"Phase state {ps.node_id}: amplitude must be a finite, "
                f"non-negative float, got {amplitude!r}"
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
        expected oscillator count; a positive Python or NumPy integer.

    Returns
    -------
    SessionCoherenceReport
        SessionCoherenceReport with pass/fail, quality scores, and diagnostics.

    Raises
    ------
    TypeError
        If ``n_osc`` is not an integer (booleans excluded).
    ValueError
        If ``n_osc`` is not positive.
    """
    if isinstance(n_osc, (bool, np.bool_)) or not isinstance(n_osc, (int, np.integer)):
        raise TypeError(f"n_osc must be an int, got {type(n_osc).__name__}")
    n_osc = int(n_osc)
    if n_osc < 1:
        raise ValueError(f"n_osc must be positive, got {n_osc}")

    report = SessionCoherenceReport()
    scorer = PhaseQualityScorer()

    # Quality per channel — only scored when every quality and amplitude is
    # admissible; a poisoned value would silently disable the thresholds below.
    if _validate_extractor_evidence(phase_states, report):
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
    m_k = _validate_real_vector(
        "Imprint vector m_k", imprint_state.m_k, report, non_negative=True
    )
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
