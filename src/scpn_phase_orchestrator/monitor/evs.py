# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Entrainment Verification Score (EVS)

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.itpc import compute_itpc, itpc_persistence

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

try:
    from spo_kernel import (
        frequency_specificity_rust as _rust_freq_spec,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["EVSMonitor", "EVSResult"]


@dataclass(frozen=True, slots=True)
class EVSResult:
    """Entrainment verification outcome."""

    itpc_value: float
    persistence_score: float
    specificity_ratio: float
    is_entrained: bool


class EVSMonitor:
    """Combines ITPC, persistence, and frequency specificity into
    a single entrainment verification score.

    Three criteria must all pass for ``is_entrained=True``:

    1. Mean ITPC across all time points >= ``itpc_threshold``
    2. ITPC during/after stimulus pause >= ``persistence_threshold``
    3. ITPC at the target frequency / ITPC at a control frequency
       >= ``specificity_threshold``

    The specificity test distinguishes frequency-specific entrainment
    from broadband phase-locking artefacts.
    """

    def __init__(
        self,
        itpc_threshold: float = 0.6,
        persistence_threshold: float = 0.4,
        specificity_threshold: float = 1.5,
    ) -> None:
        self.itpc_threshold = _validate_unit_threshold(
            itpc_threshold, name="itpc_threshold"
        )
        self.persistence_threshold = _validate_unit_threshold(
            persistence_threshold, name="persistence_threshold"
        )
        self.specificity_threshold = _validate_positive_real(
            specificity_threshold, name="specificity_threshold"
        )

    def evaluate(
        self,
        phases_trials: FloatArray,
        pause_indices: list[int] | IntArray,
        target_freq: float,
        control_freq: float,
    ) -> EVSResult:
        """Run the full EVS battery.

        Args:
            phases_trials: shape (n_trials, n_timepoints), phases in
                radians at the *target* frequency.
            pause_indices: time-point indices within/after a stimulus
                pause window.
            target_freq: stimulus frequency (Hz).
            control_freq: non-stimulus control frequency (Hz).

        Returns:
            EVSResult with all three sub-scores and the overall verdict.
        """
        phases = _validate_phase_trials(phases_trials)
        pause_idx = _validate_pause_indices(pause_indices)
        target = _validate_positive_real(target_freq, name="target_freq")
        control = _validate_positive_real(control_freq, name="control_freq")

        itpc_vals = compute_itpc(phases)
        itpc_mean = float(np.mean(itpc_vals)) if itpc_vals.size > 0 else 0.0

        persistence = itpc_persistence(phases, pause_idx)

        specificity = self._frequency_specificity(
            phases,
            target,
            control,
        )

        entrained = (
            itpc_mean >= self.itpc_threshold
            and persistence >= self.persistence_threshold
            and specificity >= self.specificity_threshold
        )

        return EVSResult(
            itpc_value=itpc_mean,
            persistence_score=persistence,
            specificity_ratio=specificity,
            is_entrained=entrained,
        )

    @staticmethod
    def _frequency_specificity(
        phases_trials: FloatArray,
        target_freq: float,
        control_freq: float,
    ) -> float:
        """Ratio of mean ITPC at target vs control frequency.

        Uses phase scaling: phases at the control frequency are obtained
        by rescaling the target-frequency phases by the frequency ratio.
        This models the assumption that the raw signal was band-pass
        filtered at each frequency before phase extraction.
        """
        phases = _validate_phase_trials(phases_trials)
        target = _validate_positive_real(target_freq, name="target_freq")
        control = _validate_positive_real(control_freq, name="control_freq")

        if _HAS_RUST:
            p = np.ascontiguousarray(phases, dtype=np.float64)
            n_trials, n_tp = p.shape
            return float(_rust_freq_spec(p.ravel(), n_trials, n_tp, target, control))

        itpc_target = compute_itpc(phases)
        target_mean = float(np.mean(itpc_target)) if itpc_target.size > 0 else 0.0

        # Phase at control frequency: rescale by freq ratio
        ratio = control / target
        control_phases = phases * ratio
        itpc_control = compute_itpc(control_phases)
        control_mean = float(np.mean(itpc_control)) if itpc_control.size > 0 else 0.0

        if control_mean < 1e-12:
            return float("inf") if target_mean > 0 else 0.0

        return target_mean / control_mean


def _validate_unit_threshold(value: object, *, name: str) -> float:
    threshold = _validate_real(value, name=name)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return threshold


def _validate_positive_real(value: object, *, name: str) -> float:
    scalar = _validate_real(value, name=name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _validate_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real value")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_phase_trials(value: object) -> FloatArray:
    try:
        phases = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases_trials must be a finite 2-D phase matrix") from exc
    if phases.ndim != 2:
        raise ValueError(
            f"phases_trials must be 2-D (trials, timepoints), got shape {phases.shape}"
        )
    if phases.shape[0] == 0 or phases.shape[1] == 0:
        raise ValueError(
            f"phases_trials must contain at least one trial and one timepoint, "
            f"got shape {phases.shape}"
        )
    if not np.all(np.isfinite(phases)):
        raise ValueError("phases_trials must contain only finite values")
    return phases


def _validate_pause_indices(value: list[int] | IntArray) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError("pause_indices must be a 1-D integer index array")
    if raw.dtype == np.bool_:
        raise TypeError("pause_indices must contain integer indices, not booleans")
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("pause_indices must contain integer indices") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("pause_indices must contain finite integer indices")
    if not np.all(numeric == np.floor(numeric)):
        raise TypeError("pause_indices must contain integer indices")
    return numeric.astype(np.int64)
