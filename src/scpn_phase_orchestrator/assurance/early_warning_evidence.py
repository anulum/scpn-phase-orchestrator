# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sealed early-warning assurance evidence

"""Hash-sealed, claim-bounded assurance evidence for an early-warning alarm.

The early-warning detector suite — critical slowing down
(:mod:`~scpn_phase_orchestrator.monitor.critical_slowing_down`), rising
synchronisation (:mod:`~scpn_phase_orchestrator.monitor.synchronisation`), and
ordinal-transition entropy (:mod:`~scpn_phase_orchestrator.monitor.explosive_sync`)
— reads a passive observable and emits a warning record. A fair head-to-head
(``bench/early_warning_leadtime.py``) established that the *detection* is a
commodity: none of these indicators beats the others by a decisive margin. What
is *not* a commodity, and what this module supplies, is the auditable envelope
around the alarm: a content-addressed record that pins **which indicators
contributed and their robust z-scores at the alarm window**, the **provenance**
of the screened signal, the **claim boundary** (this is a review-only technical
artefact, not a clinical/operational/safety decision or a certification), and,
when a ground-truth transition onset is supplied, the **honest lead time** —
including a non-positive lead when the alarm was late.

The record is content-addressed with the same canonical-JSON SHA-256 hashing the
assurance-case bundle and the NERC PRC oscillation evidence use
(:func:`~scpn_phase_orchestrator.assurance._hashing.canonical_record_hash`), so a
sealed alarm can be referenced by a stable digest and any later mutation is
detectable. The capture timestamp and the ground-truth onset are supplied by the
caller (they are properties of the measured event, not wall-clock readings taken
here) so the record is deterministic and reproducible.

:func:`seal_early_warning` is the neutral primitive — it depends only on the
alarm decision, the provenance, and a pre-extracted set of
:class:`EarlyWarningIndicator` contributions, so it seals any present or future
detector (including the real-EEG harness) without importing detector internals.
The three ``seal_*_alarm`` adapters bridge each concrete suite detector's warning
dataclass onto that primitive.

References
----------
* Scheffer et al. 2009, *Nature* 461, 53 — generic early-warning signals for
  critical transitions (the framework the sealed indicators contribute to).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from scpn_phase_orchestrator.monitor.critical_slowing_down import (
        CriticalSlowingDownWarning,
    )
    from scpn_phase_orchestrator.monitor.ensemble_warning import EnsembleWarning
    from scpn_phase_orchestrator.monitor.explosive_sync import ExplosiveSyncWarning
    from scpn_phase_orchestrator.monitor.synchronisation import SynchronisationWarning

__all__ = [
    "DROP",
    "EARLY_WARNING_DISCLAIMER",
    "EARLY_WARNING_FLAGGED",
    "EARLY_WARNING_FRAMEWORK",
    "NO_EARLY_WARNING",
    "RISE",
    "EarlyWarningEvidence",
    "EarlyWarningIndicator",
    "seal_critical_slowing_down_alarm",
    "seal_early_warning",
    "seal_ensemble_alarm",
    "seal_synchronisation_alarm",
    "seal_transition_entropy_alarm",
]

#: Indicator whose alarm direction is a rise above baseline (variance,
#: autocorrelation, order parameter).
RISE = "rise"
#: Indicator whose alarm direction is a drop below baseline (transition entropy).
DROP = "drop"

#: Verdict when the detector raised a sustained early-warning alarm.
EARLY_WARNING_FLAGGED = "early_warning_flagged"
#: Verdict when no sustained alarm was raised.
NO_EARLY_WARNING = "no_early_warning"

#: Early-warning framework the sealed indicators contribute to.
EARLY_WARNING_FRAMEWORK = (
    "Generic early-warning-signals framework (Scheffer et al. 2009, Nature 461:53)"
)

#: Review-only disclaimer carried by every sealed early-warning record.
EARLY_WARNING_DISCLAIMER = (
    "This early-warning evidence record is a technical evidence-mapping artefact. "
    "It seals a passive detector's alarm decision, the contributing indicators and "
    "their robust z-scores, and the provenance of the screened signal; it does not "
    "constitute a clinical, operational, or safety decision, nor a certification of "
    "compliance. Any reported lead time is measured against a caller-supplied "
    "ground-truth onset and holds only for the recorded observable and detector "
    "configuration; a non-positive lead means the alarm was coincident with or "
    "later than the transition."
)


@dataclass(frozen=True, slots=True)
class EarlyWarningIndicator:
    """A single indicator's contribution to an early-warning alarm.

    Attributes
    ----------
    name : str
        Indicator label, e.g. ``variance``, ``lag1_autocorrelation``,
        ``order_parameter``, or ``transition_entropy``.
    direction : str
        :data:`RISE` if the indicator warns by rising above its baseline, or
        :data:`DROP` if it warns by falling below it.
    robust_z : float
        Median / MAD robust z-score of the indicator at the reported window (the
        alarm window if the detector triggered, else the closest approach among
        the post-baseline windows).
    baseline_median : float
        Median of the indicator over the leading baseline windows.
    z_threshold : float
        Robust z-score magnitude at or beyond which the indicator breaches its
        gate.
    breached : bool
        Whether ``robust_z`` crossed the gate in the indicator's alarm direction
        at the reported window.
    """

    name: str
    direction: str
    robust_z: float
    baseline_median: float
    z_threshold: float
    breached: bool

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the indicator contribution.

        Returns
        -------
        dict[str, object]
            The indicator label, alarm direction, robust z-score, baseline
            median, gate threshold, and breach status.
        """
        return {
            "name": self.name,
            "direction": self.direction,
            "robust_z": self.robust_z,
            "baseline_median": self.baseline_median,
            "z_threshold": self.z_threshold,
            "breached": self.breached,
        }


@dataclass(frozen=True, slots=True)
class EarlyWarningEvidence:
    """A hash-sealed, review-only early-warning assurance record.

    Attributes
    ----------
    detector : str
        Detector family label, e.g. ``critical_slowing_down``,
        ``synchronisation``, or ``transition_entropy``.
    observable : str
        The physical quantity the detector read (a bus-frequency variance, a
        cross-channel order parameter, a per-channel phase field, ...).
    signal_source : str
        Provenance identifier of the screened signal (dataset, event, or channel
        set).
    captured_at : str
        Measurement timestamp of the event, supplied by the caller.
    sampling_rate_hz : float
        Sampling rate of the screened signal, in hertz; converts a sample lead
        into seconds.
    window, step : int
        Echoed analysis window length and hop, in samples.
    persistence : int
        Echoed number of consecutive breaching windows required to alarm.
    n_baseline_windows : int
        Number of leading windows the detector fitted its baseline on.
    warning_triggered : bool
        Whether the detector raised a sustained alarm.
    warning_window : int | None
        Index of the first window of the triggering run, or ``None``.
    warning_sample : int | None
        Sample index of the triggering window, or ``None``.
    transition_onset_sample : int | None
        Caller-supplied ground-truth onset sample, or ``None`` when unknown.
    lead_samples : int | None
        ``transition_onset_sample - warning_sample`` when both are known, else
        ``None``. Positive means an early alarm; non-positive means late or
        coincident.
    lead_seconds : float | None
        ``lead_samples / sampling_rate_hz`` when defined, else ``None``.
    lead_is_early : bool
        ``True`` only when the lead is defined and strictly positive.
    indicators : tuple[EarlyWarningIndicator, ...]
        Per-indicator contributions at the reported window.
    verdict : str
        :data:`EARLY_WARNING_FLAGGED` if the detector alarmed, else
        :data:`NO_EARLY_WARNING`.
    framework : str
        The early-warning framework the record maps to.
    disclaimer : str
        The review-only claim boundary.
    content_hash : str
        SHA-256 of the canonical record (excluding this field); computed on
        construction.
    """

    detector: str
    observable: str
    signal_source: str
    captured_at: str
    sampling_rate_hz: float
    window: int
    step: int
    persistence: int
    n_baseline_windows: int
    warning_triggered: bool
    warning_window: int | None
    warning_sample: int | None
    transition_onset_sample: int | None
    lead_samples: int | None
    lead_seconds: float | None
    lead_is_early: bool
    indicators: tuple[EarlyWarningIndicator, ...]
    verdict: str
    framework: str
    disclaimer: str
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Compute the content hash from the canonical evidence payload."""
        object.__setattr__(
            self, "content_hash", canonical_record_hash(self._canonical_payload())
        )

    def _canonical_payload(self) -> dict[str, object]:
        """Return the canonical payload for the sealed record."""
        return {
            "detector": self.detector,
            "observable": self.observable,
            "signal_source": self.signal_source,
            "captured_at": self.captured_at,
            "sampling_rate_hz": self.sampling_rate_hz,
            "window": self.window,
            "step": self.step,
            "persistence": self.persistence,
            "n_baseline_windows": self.n_baseline_windows,
            "warning_triggered": self.warning_triggered,
            "warning_window": self.warning_window,
            "warning_sample": self.warning_sample,
            "transition_onset_sample": self.transition_onset_sample,
            "lead_samples": self.lead_samples,
            "lead_seconds": self.lead_seconds,
            "lead_is_early": self.lead_is_early,
            "indicators": [
                indicator.to_audit_record() for indicator in self.indicators
            ],
            "verdict": self.verdict,
            "framework": self.framework,
            "disclaimer": self.disclaimer,
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the whole sealed record.

        Returns
        -------
        dict[str, object]
            The canonical payload plus the computed ``content_hash``.
        """
        record = self._canonical_payload()
        record["content_hash"] = self.content_hash
        return record


def seal_early_warning(
    *,
    detector: str,
    observable: str,
    signal_source: str,
    captured_at: str,
    sampling_rate_hz: float,
    window: int,
    step: int,
    persistence: int,
    n_baseline_windows: int,
    warning_triggered: bool,
    warning_window: int | None,
    warning_sample: int | None,
    indicators: Sequence[EarlyWarningIndicator],
    transition_onset_sample: int | None = None,
) -> EarlyWarningEvidence:
    """Seal an early-warning alarm into a hash-addressed evidence record.

    This is the neutral primitive: it depends only on the alarm decision, the
    provenance, and a pre-extracted set of indicator contributions, so it seals
    any detector without importing its internals.

    Parameters
    ----------
    detector : str
        Detector family label.
    observable : str
        The physical quantity the detector read.
    signal_source : str
        Provenance identifier of the screened signal.
    captured_at : str
        Measurement timestamp of the event, supplied by the caller.
    sampling_rate_hz : float
        Sampling rate of the screened signal, in hertz (``> 0``).
    window, step, persistence : int
        Echoed analysis parameters; each must be a positive integer.
    n_baseline_windows : int
        Number of leading windows the detector fitted its baseline on; must be a
        positive integer.
    warning_triggered : bool
        Whether the detector raised a sustained alarm.
    warning_window, warning_sample : int | None
        Triggering window and sample indices; both must be present when
        ``warning_triggered`` is true and absent otherwise.
    indicators : Sequence[EarlyWarningIndicator]
        At least one indicator contribution; each ``direction`` must be
        :data:`RISE` or :data:`DROP` and each numeric field finite.
    transition_onset_sample : int | None
        Caller-supplied ground-truth onset sample; enables the lead computation.

    Returns
    -------
    EarlyWarningEvidence
        The hash-sealed, review-only early-warning record.

    Raises
    ------
    ValueError
        If an identifier is empty, a count is not a positive integer, a sample
        index is negative, the alarm flags are inconsistent, the indicators are
        empty or malformed, or the sampling rate is not positive.
    """
    detector_label = _non_empty_str(detector, "detector")
    observable_label = _non_empty_str(observable, "observable")
    source = _non_empty_str(signal_source, "signal_source")
    captured = _non_empty_str(captured_at, "captured_at")
    fs = _positive_real(sampling_rate_hz, "sampling_rate_hz")
    window_int = _positive_int(window, "window")
    step_int = _positive_int(step, "step")
    persistence_int = _positive_int(persistence, "persistence")
    n_baseline = _positive_int(n_baseline_windows, "n_baseline_windows")
    triggered = _bool(warning_triggered, "warning_triggered")
    window_idx = _optional_non_negative_int(warning_window, "warning_window")
    sample_idx = _optional_non_negative_int(warning_sample, "warning_sample")
    onset = _optional_non_negative_int(
        transition_onset_sample, "transition_onset_sample"
    )
    if triggered != (window_idx is not None) or triggered != (sample_idx is not None):
        raise ValueError(
            "warning_window and warning_sample must be present exactly when "
            "warning_triggered is true"
        )
    sealed_indicators = _validate_indicators(indicators)

    lead_samples, lead_seconds, lead_is_early = _lead(onset, sample_idx, fs)
    verdict = EARLY_WARNING_FLAGGED if triggered else NO_EARLY_WARNING

    return EarlyWarningEvidence(
        detector=detector_label,
        observable=observable_label,
        signal_source=source,
        captured_at=captured,
        sampling_rate_hz=fs,
        window=window_int,
        step=step_int,
        persistence=persistence_int,
        n_baseline_windows=n_baseline,
        warning_triggered=triggered,
        warning_window=window_idx,
        warning_sample=sample_idx,
        transition_onset_sample=onset,
        lead_samples=lead_samples,
        lead_seconds=lead_seconds,
        lead_is_early=lead_is_early,
        indicators=sealed_indicators,
        verdict=verdict,
        framework=EARLY_WARNING_FRAMEWORK,
        disclaimer=EARLY_WARNING_DISCLAIMER,
    )


def seal_critical_slowing_down_alarm(
    warning: CriticalSlowingDownWarning,
    *,
    observable: str,
    signal_source: str,
    captured_at: str,
    sampling_rate_hz: float,
    transition_onset_sample: int | None = None,
) -> EarlyWarningEvidence:
    """Seal a critical-slowing-down alarm, pinning both rising indicators.

    The record carries the variance and lag-one autocorrelation contributions —
    the two second-moment indicators of critical slowing down — at the reported
    window, so an auditor sees which indicator carried the alarm.

    Parameters
    ----------
    warning : CriticalSlowingDownWarning
        The detector output to seal.
    observable, signal_source, captured_at : str
        Provenance of the screened signal, forwarded to :func:`seal_early_warning`.
    sampling_rate_hz : float
        Sampling rate of the screened signal, in hertz.
    transition_onset_sample : int | None
        Caller-supplied ground-truth onset sample.

    Returns
    -------
    EarlyWarningEvidence
        The sealed record for the critical-slowing-down alarm.

    Raises
    ------
    ValueError
        If ``warning`` is not a
        :class:`~scpn_phase_orchestrator.monitor.critical_slowing_down.CriticalSlowingDownWarning`
        or the forwarded provenance is invalid.
    """
    from scpn_phase_orchestrator.monitor.critical_slowing_down import (
        CriticalSlowingDownWarning,
    )

    if not isinstance(warning, CriticalSlowingDownWarning):
        raise ValueError("warning must be a CriticalSlowingDownWarning")
    report = _report_window(
        warning.combined_z.tolist(),
        RISE,
        warning.n_baseline_windows,
        warning.warning_window,
    )
    indicators = (
        _indicator_at(
            "variance",
            RISE,
            warning.robust_z_variance.tolist(),
            warning.baseline_variance,
            warning.z_threshold,
            report,
        ),
        _indicator_at(
            "lag1_autocorrelation",
            RISE,
            warning.robust_z_autocorrelation.tolist(),
            warning.baseline_autocorrelation,
            warning.z_threshold,
            report,
        ),
    )
    return seal_early_warning(
        detector="critical_slowing_down",
        observable=observable,
        signal_source=signal_source,
        captured_at=captured_at,
        sampling_rate_hz=sampling_rate_hz,
        window=warning.window,
        step=warning.step,
        persistence=warning.persistence,
        n_baseline_windows=warning.n_baseline_windows,
        warning_triggered=warning.warning_triggered,
        warning_window=warning.warning_window,
        warning_sample=warning.warning_sample,
        indicators=indicators,
        transition_onset_sample=transition_onset_sample,
    )


def seal_synchronisation_alarm(
    warning: SynchronisationWarning,
    *,
    observable: str,
    signal_source: str,
    captured_at: str,
    sampling_rate_hz: float,
    transition_onset_sample: int | None = None,
) -> EarlyWarningEvidence:
    """Seal a rising-synchronisation alarm on the Kuramoto order parameter.

    Parameters
    ----------
    warning : SynchronisationWarning
        The detector output to seal.
    observable, signal_source, captured_at : str
        Provenance of the screened signal, forwarded to :func:`seal_early_warning`.
    sampling_rate_hz : float
        Sampling rate of the screened signal, in hertz.
    transition_onset_sample : int | None
        Caller-supplied ground-truth onset sample.

    Returns
    -------
    EarlyWarningEvidence
        The sealed record for the synchronisation alarm.

    Raises
    ------
    ValueError
        If ``warning`` is not a
        :class:`~scpn_phase_orchestrator.monitor.synchronisation.SynchronisationWarning`
        or the forwarded provenance is invalid.
    """
    from scpn_phase_orchestrator.monitor.synchronisation import SynchronisationWarning

    if not isinstance(warning, SynchronisationWarning):
        raise ValueError("warning must be a SynchronisationWarning")
    report = _report_window(
        warning.robust_z.tolist(),
        RISE,
        warning.n_baseline_windows,
        warning.warning_window,
    )
    indicators = (
        _indicator_at(
            "order_parameter",
            RISE,
            warning.robust_z.tolist(),
            warning.baseline_median,
            warning.z_threshold,
            report,
        ),
    )
    return seal_early_warning(
        detector="synchronisation",
        observable=observable,
        signal_source=signal_source,
        captured_at=captured_at,
        sampling_rate_hz=sampling_rate_hz,
        window=warning.window,
        step=warning.step,
        persistence=warning.persistence,
        n_baseline_windows=warning.n_baseline_windows,
        warning_triggered=warning.warning_triggered,
        warning_window=warning.warning_window,
        warning_sample=warning.warning_sample,
        indicators=indicators,
        transition_onset_sample=transition_onset_sample,
    )


def seal_transition_entropy_alarm(
    warning: ExplosiveSyncWarning,
    *,
    observable: str,
    signal_source: str,
    captured_at: str,
    sampling_rate_hz: float,
    transition_onset_sample: int | None = None,
) -> EarlyWarningEvidence:
    """Seal an ordinal-transition-entropy alarm (a regularisation drop).

    Parameters
    ----------
    warning : ExplosiveSyncWarning
        The detector output to seal.
    observable, signal_source, captured_at : str
        Provenance of the screened signal, forwarded to :func:`seal_early_warning`.
    sampling_rate_hz : float
        Sampling rate of the screened signal, in hertz.
    transition_onset_sample : int | None
        Caller-supplied ground-truth onset sample.

    Returns
    -------
    EarlyWarningEvidence
        The sealed record for the transition-entropy alarm.

    Raises
    ------
    ValueError
        If ``warning`` is not an
        :class:`~scpn_phase_orchestrator.monitor.explosive_sync.ExplosiveSyncWarning`
        or the forwarded provenance is invalid.
    """
    from scpn_phase_orchestrator.monitor.explosive_sync import ExplosiveSyncWarning

    if not isinstance(warning, ExplosiveSyncWarning):
        raise ValueError("warning must be an ExplosiveSyncWarning")
    report = _report_window(
        warning.robust_z.tolist(),
        DROP,
        warning.n_baseline_windows,
        warning.warning_window,
    )
    indicators = (
        _indicator_at(
            "transition_entropy",
            DROP,
            warning.robust_z.tolist(),
            warning.baseline_median,
            warning.z_threshold,
            report,
        ),
    )
    return seal_early_warning(
        detector="transition_entropy",
        observable=observable,
        signal_source=signal_source,
        captured_at=captured_at,
        sampling_rate_hz=sampling_rate_hz,
        window=warning.window,
        step=warning.step,
        persistence=warning.persistence,
        n_baseline_windows=warning.n_baseline_windows,
        warning_triggered=warning.warning_triggered,
        warning_window=warning.warning_window,
        warning_sample=warning.warning_sample,
        indicators=indicators,
        transition_onset_sample=transition_onset_sample,
    )


def seal_ensemble_alarm(
    ensemble: EnsembleWarning,
    *,
    observable: str,
    signal_source: str,
    captured_at: str,
    sampling_rate_hz: float,
    window: int,
    step: int,
    transition_onset_sample: int | None = None,
) -> EarlyWarningEvidence:
    """Seal a fused ensemble alarm, pinning every member's contribution.

    Each fused member becomes an indicator carrying its native robust z-score at
    the reported window, so an auditor sees exactly which detectors drove — or
    failed to drive — the fused decision. The suite is run on one window grid, so
    ``window`` and ``step`` are supplied by the caller that ran it.

    Parameters
    ----------
    ensemble : EnsembleWarning
        The fused decision to seal.
    observable, signal_source, captured_at : str
        Provenance of the screened signal, forwarded to :func:`seal_early_warning`.
    sampling_rate_hz : float
        Sampling rate of the screened signal, in hertz.
    window, step : int
        Analysis window length and hop the suite was run with.
    transition_onset_sample : int | None
        Caller-supplied ground-truth onset sample.

    Returns
    -------
    EarlyWarningEvidence
        The sealed record for the fused ensemble alarm.

    Raises
    ------
    ValueError
        If ``ensemble`` is not an
        :class:`~scpn_phase_orchestrator.monitor.ensemble_warning.EnsembleWarning`
        or the forwarded provenance is invalid.
    """
    from scpn_phase_orchestrator.monitor.ensemble_warning import EnsembleWarning

    if not isinstance(ensemble, EnsembleWarning):
        raise ValueError("ensemble must be an EnsembleWarning")
    indicators = tuple(
        EarlyWarningIndicator(
            name=contribution.name,
            direction=contribution.direction,
            robust_z=contribution.robust_z,
            baseline_median=contribution.baseline_median,
            z_threshold=contribution.z_threshold,
            breached=contribution.breached,
        )
        for contribution in ensemble.contributions
    )
    return seal_early_warning(
        detector=f"ensemble_{ensemble.rule}",
        observable=observable,
        signal_source=signal_source,
        captured_at=captured_at,
        sampling_rate_hz=sampling_rate_hz,
        window=window,
        step=step,
        persistence=ensemble.persistence,
        n_baseline_windows=ensemble.n_baseline_windows,
        warning_triggered=ensemble.warning_triggered,
        warning_window=ensemble.warning_window,
        warning_sample=ensemble.warning_sample,
        indicators=indicators,
        transition_onset_sample=transition_onset_sample,
    )


def _report_window(
    headline_z: Sequence[float],
    direction: str,
    n_baseline_windows: int,
    warning_window: int | None,
) -> int | None:
    """Return the window index whose indicators the record should pin.

    The alarm window when the detector triggered, else the closest approach in
    the alarm direction among the post-baseline windows, or ``None`` when there
    are no post-baseline windows to report.
    """
    if warning_window is not None:
        return warning_window
    n_windows = len(headline_z)
    if n_baseline_windows >= n_windows:
        return None
    tail = list(headline_z[n_baseline_windows:])
    if direction == RISE:
        offset = max(range(len(tail)), key=tail.__getitem__)
    else:
        offset = min(range(len(tail)), key=tail.__getitem__)
    return n_baseline_windows + offset


def _indicator_at(
    name: str,
    direction: str,
    robust_z: Sequence[float],
    baseline_median: float,
    z_threshold: float,
    report_window: int | None,
) -> EarlyWarningIndicator:
    """Return the indicator contribution at ``report_window``.

    When there is no window to report (a baseline-only sweep) the contribution
    is a non-breaching zero, so the sealed record still lists the indicator.
    """
    if report_window is None:
        z_value = 0.0
        breached = False
    elif direction == RISE:
        z_value = float(robust_z[report_window])
        breached = z_value >= z_threshold
    else:
        z_value = float(robust_z[report_window])
        breached = z_value <= -z_threshold
    return EarlyWarningIndicator(
        name=name,
        direction=direction,
        robust_z=z_value,
        baseline_median=float(baseline_median),
        z_threshold=float(z_threshold),
        breached=breached,
    )


def _lead(
    onset: int | None, warning_sample: int | None, sampling_rate_hz: float
) -> tuple[int | None, float | None, bool]:
    """Return the honest lead as ``(samples, seconds, is_early)``.

    The lead is defined only when both the ground-truth onset and the alarm
    sample are known; a non-positive lead (a late or coincident alarm) is
    reported as such rather than suppressed.
    """
    if onset is None or warning_sample is None:
        return None, None, False
    lead_samples = onset - warning_sample
    lead_seconds = lead_samples / sampling_rate_hz
    return lead_samples, lead_seconds, lead_samples > 0


def _validate_indicators(
    indicators: Sequence[EarlyWarningIndicator],
) -> tuple[EarlyWarningIndicator, ...]:
    """Return the indicators as a validated non-empty tuple, else raise."""
    sealed = tuple(indicators)
    if not sealed:
        raise ValueError("indicators must contain at least one contribution")
    for position, indicator in enumerate(sealed):
        if not isinstance(indicator, EarlyWarningIndicator):
            raise ValueError(f"indicators[{position}] must be an EarlyWarningIndicator")
        if indicator.direction not in (RISE, DROP):
            raise ValueError(
                f"indicators[{position}].direction must be {RISE!r} or {DROP!r}"
            )
        for field_name in ("robust_z", "baseline_median", "z_threshold"):
            _finite_real(
                getattr(indicator, field_name),
                f"indicators[{position}].{field_name}",
            )
    return sealed


def _non_empty_str(value: object, name: str) -> str:
    """Return ``value`` as a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_real(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    scalar = _finite_real(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _finite_real(value: object, name: str) -> float:
    """Return ``value`` as a finite real scalar, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar


def _positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _bool(value: object, name: str) -> bool:
    """Return ``value`` as a strict Python bool, else raise ``ValueError``."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def _optional_non_negative_int(value: object, name: str) -> int | None:
    """Return ``value`` as a non-negative int or ``None``, else raise."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{name} must be a non-negative integer or None, got {value!r}"
        )
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result
