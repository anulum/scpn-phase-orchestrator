# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — domain-adaptable early-warning detector suite

"""Domain-adaptable early-warning suite over a neutral phase-observable contract.

The early-warning detectors — critical slowing down
(:mod:`~scpn_phase_orchestrator.monitor.critical_slowing_down`), rising
synchronisation (:mod:`~scpn_phase_orchestrator.monitor.synchronisation`), and
ordinal-transition entropy (:mod:`~scpn_phase_orchestrator.monitor.explosive_sync`)
— read generic arrays, not any one domain. The reason a scalp-EEG seizure, a grid
coherence collapse, and a cardiac arrhythmia can all be screened by the *same*
suite is that each is a synchronisation transition in a population of coupled
oscillators; the only per-domain work is turning that domain's raw signals into
three phase observables. This module makes that the explicit contract.

:class:`SuiteObservables` is the neutral bundle every detector reads: the
per-node instantaneous phases (rising synchronisation), their projection
``sin(phase)`` (ordinal-transition entropy), and the cross-node Kuramoto order
parameter ``R(t) = |⟨e^{iφ}⟩|`` (critical slowing down). A
:class:`DomainObservableAdapter` is anything that turns a domain's raw signal
block into that bundle — the scalp-EEG band-pass/Hilbert/decimation pipeline is
one adapter; a cardiac ECG or grid PMU pipeline is another. Given the bundle,
:func:`run_early_warning_suite` runs all three members and the weighted fusion
under one alarm contract, returning a :class:`SuiteWarnings`, with no knowledge of
where the observables came from.

The suite is passive: it reads observables and emits warning records; it never
actuates. Sealing an alarm into auditable evidence is
:mod:`~scpn_phase_orchestrator.assurance.early_warning_evidence`; calibrating a
matched false-alarm threshold and measuring lead time on a labelled corpus is a
validation harness, not this module.

References
----------
* Scheffer et al. 2009, *Nature* 461, 53 — generic early-warning signals for
  critical transitions.
* Kuramoto 1984, *Chemical Oscillations, Waves, and Turbulence* — the order
  parameter of coupled phase oscillators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    CriticalSlowingDownWarning,
    critical_slowing_down_warning,
)
from scpn_phase_orchestrator.monitor.ensemble_warning import (
    WEIGHTED_RULE,
    EnsembleWarning,
    ensemble_warning,
    member_from_critical_slowing_down,
    member_from_synchronisation,
    member_from_transition_entropy,
)
from scpn_phase_orchestrator.monitor.explosive_sync import (
    ExplosiveSyncWarning,
    explosive_sync_warning,
)
from scpn_phase_orchestrator.monitor.synchronisation import (
    SynchronisationWarning,
    synchronisation_warning,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Mapping

FloatArray = NDArray[np.float64]

#: Detector label whose observable is the cross-node order parameter.
CRITICAL_SLOWING_DOWN = "critical_slowing_down"
#: Detector label whose observable is the per-node phase field.
SYNCHRONISATION = "synchronisation"
#: Detector label whose observable is the per-node ``sin(phase)`` projection.
TRANSITION_ENTROPY = "transition_entropy"
#: Detector label for the weighted fusion of the three members.
ENSEMBLE_WEIGHTED = "ensemble_weighted"
#: Suite detector labels in report order — the three members then the fusion.
SUITE_DETECTORS = (
    CRITICAL_SLOWING_DOWN,
    SYNCHRONISATION,
    TRANSITION_ENTROPY,
    ENSEMBLE_WEIGHTED,
)

_SCALE_FLOOR = 1.0e-12

__all__ = [
    "CRITICAL_SLOWING_DOWN",
    "ENSEMBLE_WEIGHTED",
    "SUITE_DETECTORS",
    "SYNCHRONISATION",
    "TRANSITION_ENTROPY",
    "DomainObservableAdapter",
    "SuiteObservables",
    "SuiteWarnings",
    "observables_from_phases",
    "run_early_warning_suite",
]


@dataclass(frozen=True)
class SuiteObservables:
    """The neutral phase observables every early-warning detector reads.

    Attributes
    ----------
    phases : FloatArray
        Per-node instantaneous phase in radians, shape ``(N, T)`` with at least
        two nodes; the rising-synchronisation input.
    phase_field : FloatArray
        Per-node projection ``sin(phase)``, shape ``(N, T)``; the
        ordinal-transition-entropy input.
    order_parameter : FloatArray
        Cross-node Kuramoto order parameter ``R(t) = |⟨e^{iφ}⟩|`` in ``[0, 1]``,
        shape ``(T,)``; the critical-slowing-down input.
    sampling_rate_hz : float
        Sampling rate of the observables, in hertz; converts a sample lead into
        seconds when an alarm is sealed.
    """

    phases: FloatArray = field(repr=False)
    phase_field: FloatArray = field(repr=False)
    order_parameter: FloatArray = field(repr=False)
    sampling_rate_hz: float

    def __post_init__(self) -> None:
        """Validate the observable shapes and ranges are mutually consistent."""
        phases = _validate_field(self.phases, "phases")
        field_ = _validate_field(self.phase_field, "phase_field")
        order = _validate_series(self.order_parameter, "order_parameter")
        if phases.shape[0] < 2:
            raise ValueError("phases must have at least two nodes for synchrony")
        if field_.shape != phases.shape:
            raise ValueError("phase_field must share the shape of phases")
        if order.shape[0] != phases.shape[1]:
            raise ValueError("order_parameter length must match the phase length")
        if np.any(order < -_SCALE_FLOOR) or np.any(order > 1.0 + _SCALE_FLOOR):
            raise ValueError("order_parameter must lie in [0, 1]")
        _positive_real(self.sampling_rate_hz, "sampling_rate_hz")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "phase_field", field_)
        object.__setattr__(self, "order_parameter", order)
        object.__setattr__(self, "sampling_rate_hz", float(self.sampling_rate_hz))

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the observable field."""
        return int(self.phases.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of samples per node."""
        return int(self.phases.shape[1])


@dataclass(frozen=True)
class SuiteWarnings:
    """The four early-warning records the suite emits over one observable bundle.

    Attributes
    ----------
    critical_slowing_down : CriticalSlowingDownWarning
        The variance / autocorrelation rise on the order parameter.
    synchronisation : SynchronisationWarning
        The order-parameter rise on the per-node phases.
    transition_entropy : ExplosiveSyncWarning
        The ordinal-transition-entropy drop on the ``sin(phase)`` field.
    ensemble : EnsembleWarning
        The weighted fusion of the three members.
    """

    critical_slowing_down: CriticalSlowingDownWarning
    synchronisation: SynchronisationWarning
    transition_entropy: ExplosiveSyncWarning
    ensemble: EnsembleWarning

    def triggered(self) -> dict[str, bool]:
        """Return each detector's alarm verdict keyed by :data:`SUITE_DETECTORS`.

        Returns
        -------
        dict[str, bool]
            One ``label -> warning_triggered`` entry per detector, in
            :data:`SUITE_DETECTORS` order — the three members then the fusion.
        """
        return {
            CRITICAL_SLOWING_DOWN: self.critical_slowing_down.warning_triggered,
            SYNCHRONISATION: self.synchronisation.warning_triggered,
            TRANSITION_ENTROPY: self.transition_entropy.warning_triggered,
            ENSEMBLE_WEIGHTED: self.ensemble.warning_triggered,
        }


@runtime_checkable
class DomainObservableAdapter(Protocol):
    """A domain's bridge from raw signals to :class:`SuiteObservables`.

    An adapter names its domain and turns a raw per-channel signal block into the
    neutral observable bundle the suite reads. The scalp-EEG band-pass / Hilbert /
    decimation pipeline is one adapter; a cardiac ECG or grid PMU pipeline is
    another. Adapters carry their own domain configuration, so the suite stays
    ignorant of the domain.
    """

    @property
    def domain(self) -> str:
        """Return the domain label, e.g. ``scalp_eeg`` or ``cardiac_ecg``."""
        ...

    def observables(self, raw: FloatArray) -> SuiteObservables:
        """Return the neutral observable bundle for one raw recording.

        Parameters
        ----------
        raw : FloatArray
            One raw per-channel recording block in the adapter's native domain
            units, e.g. band-passed scalp-EEG samples or PMU frequency traces.

        Returns
        -------
        SuiteObservables
            The neutral phase-observable bundle the suite reads.
        """
        ...


def observables_from_phases(
    phases: FloatArray, *, sampling_rate_hz: float
) -> SuiteObservables:
    """Build the neutral observable bundle from a per-node phase field.

    Most adapters end at a reconstructed per-node phase; this derives the
    remaining two observables — the ``sin(phase)`` projection and the cross-node
    order parameter — so an adapter need only supply phases and a rate.

    Parameters
    ----------
    phases : FloatArray
        Per-node phase in radians, shape ``(N, T)`` with at least two nodes.
    sampling_rate_hz : float
        Sampling rate of the phases, in hertz.

    Returns
    -------
    SuiteObservables
        The phases, their ``sin`` projection, and the order parameter.

    Raises
    ------
    ValueError
        If the phase field is malformed or has fewer than two nodes.
    """
    field_ = _validate_field(phases, "phases")
    if field_.shape[0] < 2:
        raise ValueError("phases must have at least two nodes for synchrony")
    order = np.abs(np.mean(np.exp(1j * field_), axis=0))
    return SuiteObservables(
        phases=field_,
        phase_field=np.ascontiguousarray(np.sin(field_), dtype=np.float64),
        order_parameter=np.ascontiguousarray(order, dtype=np.float64),
        sampling_rate_hz=sampling_rate_hz,
    )


def run_early_warning_suite(
    observables: SuiteObservables,
    *,
    thresholds: Mapping[str, float],
    relative_gate: float = 0.05,
    window: int = 128,
    step: int = 16,
    baseline_fraction: float = 0.25,
    persistence: int = 2,
) -> SuiteWarnings:
    """Run the three members and the weighted fusion over one observable bundle.

    Each detector reads the observable it is designed for at the supplied
    threshold: critical slowing down the order parameter, rising synchronisation
    the per-node phases, ordinal-transition entropy the ``sin(phase)`` field. The
    fusion is a weighted mean of the members' oriented z-scores. This is the
    domain-neutral core — it does not know which domain produced ``observables``.

    Parameters
    ----------
    observables : SuiteObservables
        The neutral observable bundle.
    thresholds : Mapping[str, float]
        Robust z-score (fused-score for the ensemble) gate per label in
        :data:`SUITE_DETECTORS`.
    relative_gate : float
        Minimum fractional change gate shared by the three members.
    window, step : int
        Analysis window length and hop, in samples.
    baseline_fraction : float
        Leading fraction of windows used to fit each detector's baseline.
    persistence : int
        Consecutive breaching windows required to raise an alarm.

    Returns
    -------
    SuiteWarnings
        The four warning records, aligned on one window grid.

    Raises
    ------
    KeyError
        If ``thresholds`` is missing a detector label.
    ValueError
        If an analysis control is out of range for a detector.
    """
    critical = critical_slowing_down_warning(
        observables.order_parameter[np.newaxis, :],
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=thresholds[CRITICAL_SLOWING_DOWN],
        rise_threshold=relative_gate,
        persistence=persistence,
    )
    synchrony = synchronisation_warning(
        observables.phases,
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=thresholds[SYNCHRONISATION],
        rise_threshold=relative_gate,
        persistence=persistence,
    )
    entropy = explosive_sync_warning(
        observables.phase_field,
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=thresholds[TRANSITION_ENTROPY],
        drop_threshold=relative_gate,
        persistence=persistence,
    )
    fusion = ensemble_warning(
        [
            member_from_critical_slowing_down(critical),
            member_from_synchronisation(synchrony),
            member_from_transition_entropy(entropy),
        ],
        rule=WEIGHTED_RULE,
        fused_threshold=thresholds[ENSEMBLE_WEIGHTED],
        persistence=persistence,
    )
    return SuiteWarnings(
        critical_slowing_down=critical,
        synchronisation=synchrony,
        transition_entropy=entropy,
        ensemble=fusion,
    )


def _validate_field(value: object, name: str) -> FloatArray:
    """Return ``value`` as a validated 2-D finite float field, else raise."""
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real float array") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} shape {raw.shape} must be two-dimensional (N, T)")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_series(value: object, name: str) -> FloatArray:
    """Return ``value`` as a validated 1-D finite float series, else raise."""
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real float array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} shape {raw.shape} must be one-dimensional (T,)")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _positive_real(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive real, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {result}")
    return result
