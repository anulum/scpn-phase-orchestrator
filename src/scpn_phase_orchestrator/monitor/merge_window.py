# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C merge-window monitor

"""Phase-and-space merge-window lock monitor.

The PHA-C moving-frame lane tracks phase ``theta`` and axial position ``z`` for
candidate merger/coalescence events. A merge is accepted only when both the
wrapped phase dispersion and the axial spatial dispersion remain inside their
reviewed tolerances for a configured number of consecutive samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
TWO_PI = 2.0 * np.pi
DEFAULT_PHASE_TOL_RAD = 0.01
DEFAULT_SPATIAL_TOL_M = 0.002
MERGE_WINDOW_TOLERANCE_PROFILE_MULTIPLIERS = {
    "baseline_1x": 1.0,
    "buffer_3x": 3.0,
    "review_5x": 5.0,
}


@dataclass(frozen=True, slots=True)
class MergeWindowToleranceProfile:
    """Resolved phase and spatial tolerances for a PHA-C merge window."""

    name: str
    phase_tol_rad: float
    spatial_tol_m: float
    multiplier: float
    baseline_phase_tol_rad: float
    baseline_spatial_tol_m: float

    def __post_init__(self) -> None:
        name = _validate_profile_name(self.name)
        phase_tol = _validate_tolerance(self.phase_tol_rad, name="phase_tol_rad")
        spatial_tol = _validate_tolerance(self.spatial_tol_m, name="spatial_tol_m")
        multiplier = _validate_positive_scalar(self.multiplier, name="multiplier")
        baseline_phase = _validate_tolerance(
            self.baseline_phase_tol_rad,
            name="baseline_phase_tol_rad",
        )
        baseline_spatial = _validate_tolerance(
            self.baseline_spatial_tol_m,
            name="baseline_spatial_tol_m",
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "phase_tol_rad", phase_tol)
        object.__setattr__(self, "spatial_tol_m", spatial_tol)
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "baseline_phase_tol_rad", baseline_phase)
        object.__setattr__(self, "baseline_spatial_tol_m", baseline_spatial)

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-safe tolerance-profile payload."""

        return merge_window_tolerance_profile_to_dict(self)


@dataclass(frozen=True, slots=True)
class MergeReport:
    """Audit-ready merge-window state for one sampled instant.

    Attributes:
        t: Sample timestamp in the caller's runtime units.
        phase_dispersion_rad: Maximum wrapped distance to the reference phase.
        spatial_dispersion_m: Maximum axial distance to the reference point.
        phase_margin_rad: Signed distance from phase tolerance to dispersion.
        spatial_margin_m: Signed distance from spatial tolerance to dispersion.
        phase_locked: True when phase margin is non-negative.
        spatial_locked: True when spatial margin is non-negative.
        lock_achieved: True after the required consecutive joint-lock count.
        consecutive_lock_samples: Current consecutive joint-lock count.
    """

    t: float
    phase_dispersion_rad: float
    spatial_dispersion_m: float
    phase_margin_rad: float
    spatial_margin_m: float
    phase_locked: bool
    spatial_locked: bool
    lock_achieved: bool
    consecutive_lock_samples: int

    def to_dict(self) -> dict[str, float | int | bool]:
        """Return a JSON-safe representation for audit and benchmark records."""

        return merge_window_report_to_dict(self)


def _validate_real_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _validate_tolerance(value: object, *, name: str) -> float:
    parsed = _validate_real_scalar(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _validate_positive_scalar(value: object, *, name: str) -> float:
    parsed = _validate_real_scalar(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _validate_profile_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("tolerance_profile must be a named profile string")
    name = value.strip().lower()
    if name not in MERGE_WINDOW_TOLERANCE_PROFILE_MULTIPLIERS:
        valid = ", ".join(sorted(MERGE_WINDOW_TOLERANCE_PROFILE_MULTIPLIERS))
        raise ValueError(f"tolerance_profile must be one of: {valid}")
    return name


def _validate_sample_count(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _as_float_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values)
    if array.dtype == np.dtype("O"):
        raise ValueError(f"{name} must be a finite real-valued vector")
    if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must be real-valued, not boolean or complex")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    try:
        out = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _phase_dispersion_rad(phases: FloatArray, reference_phase: float) -> float:
    wrapped = np.remainder(phases - reference_phase + np.pi, TWO_PI) - np.pi
    return float(np.max(np.abs(wrapped)))


def _spatial_dispersion_m(positions: FloatArray, reference_point: float) -> float:
    return float(np.max(np.abs(positions - reference_point)))


def resolve_merge_window_tolerance_profile(
    tolerance_profile: object,
    *,
    phase_baseline_rad: object = DEFAULT_PHASE_TOL_RAD,
    spatial_baseline_m: object = DEFAULT_SPATIAL_TOL_M,
) -> MergeWindowToleranceProfile:
    """Resolve a named PHA-C tolerance profile into numeric tolerances."""

    if isinstance(tolerance_profile, MergeWindowToleranceProfile):
        return tolerance_profile
    name = _validate_profile_name(tolerance_profile)
    phase_baseline = _validate_tolerance(
        phase_baseline_rad,
        name="phase_baseline_rad",
    )
    spatial_baseline = _validate_tolerance(
        spatial_baseline_m,
        name="spatial_baseline_m",
    )
    multiplier = MERGE_WINDOW_TOLERANCE_PROFILE_MULTIPLIERS[name]
    return MergeWindowToleranceProfile(
        name=name,
        phase_tol_rad=phase_baseline * multiplier,
        spatial_tol_m=spatial_baseline * multiplier,
        multiplier=multiplier,
        baseline_phase_tol_rad=phase_baseline,
        baseline_spatial_tol_m=spatial_baseline,
    )


def evaluate_merge_window(
    phases: ArrayLike,
    positions: ArrayLike,
    *,
    t: object = 0.0,
    reference_phase: object = 0.0,
    reference_point: object = 0.0,
    phase_tol_rad: object = DEFAULT_PHASE_TOL_RAD,
    spatial_tol_m: object = DEFAULT_SPATIAL_TOL_M,
    required_consecutive_samples: object = 3,
    prior_consecutive_lock_samples: object = 0,
    tolerance_profile: object | None = None,
) -> MergeReport:
    """Evaluate one PHA-C merge-window sample.

    Phase lock is ``max_i |wrap(theta_i - theta_ref)| <= phase_tol_rad``.
    Spatial lock is ``max_i |z_i - z_ref| <= spatial_tol_m``. The combined lock
    counter increments only when both predicates pass; otherwise it resets to
    zero. ``lock_achieved`` becomes true once the counter reaches
    ``required_consecutive_samples``.
    """

    phase_vector = _as_float_vector(phases, name="phases")
    position_vector = _as_float_vector(positions, name="positions")
    if position_vector.shape != phase_vector.shape:
        raise ValueError("positions must have the same one-dimensional shape as phases")

    timestamp = _validate_real_scalar(t, name="t")
    phase_reference = _validate_real_scalar(reference_phase, name="reference_phase")
    spatial_reference = _validate_real_scalar(reference_point, name="reference_point")
    if tolerance_profile is None:
        phase_tol = _validate_tolerance(phase_tol_rad, name="phase_tol_rad")
        spatial_tol = _validate_tolerance(spatial_tol_m, name="spatial_tol_m")
    else:
        profile = resolve_merge_window_tolerance_profile(
            tolerance_profile,
            phase_baseline_rad=phase_tol_rad,
            spatial_baseline_m=spatial_tol_m,
        )
        phase_tol = profile.phase_tol_rad
        spatial_tol = profile.spatial_tol_m
    required = _validate_sample_count(
        required_consecutive_samples,
        name="required_consecutive_samples",
        minimum=1,
    )
    prior = _validate_sample_count(
        prior_consecutive_lock_samples,
        name="prior_consecutive_lock_samples",
        minimum=0,
    )

    phase_dispersion = _phase_dispersion_rad(phase_vector, phase_reference)
    spatial_dispersion = _spatial_dispersion_m(position_vector, spatial_reference)
    phase_margin = phase_tol - phase_dispersion
    spatial_margin = spatial_tol - spatial_dispersion
    phase_locked = phase_margin >= 0.0
    spatial_locked = spatial_margin >= 0.0
    consecutive = prior + 1 if phase_locked and spatial_locked else 0
    return MergeReport(
        t=timestamp,
        phase_dispersion_rad=phase_dispersion,
        spatial_dispersion_m=spatial_dispersion,
        phase_margin_rad=phase_margin,
        spatial_margin_m=spatial_margin,
        phase_locked=bool(phase_locked),
        spatial_locked=bool(spatial_locked),
        lock_achieved=bool(consecutive >= required),
        consecutive_lock_samples=consecutive,
    )


class MergeWindowMonitor:
    """Stateful consecutive-sample gate for PHA-C merge events."""

    def __init__(
        self,
        *,
        phase_tol_rad: object = DEFAULT_PHASE_TOL_RAD,
        spatial_tol_m: object = DEFAULT_SPATIAL_TOL_M,
        required_consecutive_samples: object = 3,
        tolerance_profile: object | None = None,
    ) -> None:
        self.tolerance_profile = None
        if tolerance_profile is None:
            self.phase_tol_rad = _validate_tolerance(
                phase_tol_rad,
                name="phase_tol_rad",
            )
            self.spatial_tol_m = _validate_tolerance(
                spatial_tol_m,
                name="spatial_tol_m",
            )
        else:
            profile = resolve_merge_window_tolerance_profile(
                tolerance_profile,
                phase_baseline_rad=phase_tol_rad,
                spatial_baseline_m=spatial_tol_m,
            )
            self.tolerance_profile = profile
            self.phase_tol_rad = profile.phase_tol_rad
            self.spatial_tol_m = profile.spatial_tol_m
        self.required_consecutive_samples = _validate_sample_count(
            required_consecutive_samples,
            name="required_consecutive_samples",
            minimum=1,
        )
        self._consecutive_lock_samples = 0

    @property
    def consecutive_lock_samples(self) -> int:
        """Current consecutive joint-lock count."""

        return self._consecutive_lock_samples

    def reset(self) -> None:
        """Reset the consecutive joint-lock counter."""

        self._consecutive_lock_samples = 0

    def evaluate(
        self,
        phases: ArrayLike,
        positions: ArrayLike,
        *,
        t: object = 0.0,
        reference_phase: object = 0.0,
        reference_point: object = 0.0,
    ) -> MergeReport:
        """Evaluate one sample and update the consecutive joint-lock counter."""

        report = evaluate_merge_window(
            phases,
            positions,
            t=t,
            reference_phase=reference_phase,
            reference_point=reference_point,
            phase_tol_rad=self.phase_tol_rad,
            spatial_tol_m=self.spatial_tol_m,
            required_consecutive_samples=self.required_consecutive_samples,
            prior_consecutive_lock_samples=self._consecutive_lock_samples,
        )
        self._consecutive_lock_samples = report.consecutive_lock_samples
        return report

    def __call__(
        self,
        phases: ArrayLike,
        positions: ArrayLike,
        *,
        t: object = 0.0,
        reference_phase: object = 0.0,
        reference_point: object = 0.0,
    ) -> MergeReport:
        """Alias for :meth:`evaluate` for monitor-pipeline call sites."""

        return self.evaluate(
            phases,
            positions,
            t=t,
            reference_phase=reference_phase,
            reference_point=reference_point,
        )


def merge_window_report_to_dict(report: MergeReport) -> dict[str, float | int | bool]:
    """Convert a :class:`MergeReport` into a JSON-safe dictionary."""

    return {
        "t": float(report.t),
        "phase_dispersion_rad": float(report.phase_dispersion_rad),
        "spatial_dispersion_m": float(report.spatial_dispersion_m),
        "phase_margin_rad": float(report.phase_margin_rad),
        "spatial_margin_m": float(report.spatial_margin_m),
        "phase_locked": bool(report.phase_locked),
        "spatial_locked": bool(report.spatial_locked),
        "lock_achieved": bool(report.lock_achieved),
        "consecutive_lock_samples": int(report.consecutive_lock_samples),
    }


def merge_window_tolerance_profile_to_dict(
    profile: MergeWindowToleranceProfile,
) -> dict[str, float | str]:
    """Convert a resolved tolerance profile into a JSON-safe dictionary."""

    return {
        "name": str(profile.name),
        "phase_tol_rad": float(profile.phase_tol_rad),
        "spatial_tol_m": float(profile.spatial_tol_m),
        "multiplier": float(profile.multiplier),
        "baseline_phase_tol_rad": float(profile.baseline_phase_tol_rad),
        "baseline_spatial_tol_m": float(profile.baseline_spatial_tol_m),
    }
