# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Online digital-twin confidence scoring

"""Online digital-twin confidence scoring from model–observation divergence.

A running orchestrator and its physical (or simulated) twin both emit a phase
state and an order-parameter trajectory at every control tick. This module
turns the *disagreement* between the two streams into a single calibrated
confidence score in ``[0, 1]`` plus an operator status, using two complementary
divergences computed by the multi-language acceleration chain:

* **Phase distribution Jensen–Shannon divergence** — model and observed phase
  vectors are wrapped to ``[0, 2π)`` and binned into ``n_bins`` histograms; the
  symmetric Jensen–Shannon divergence (natural log, range ``[0, ln 2]``)
  measures how differently the two populations are distributed around the ring.
* **Order-parameter Wasserstein-1 distance** — the model and observed
  order-parameter windows ``R ∈ [0, 1]`` are compared with the closed-form
  one-dimensional Wasserstein-1 distance (mean absolute difference of the
  order-sorted samples, range ``[0, 1]``).

The raw ``(js, w1)`` pair is the compute hot path and is produced by the
Rust → Mojo → Julia → Go → NumPy fallback chain (fastest available first). The
calibration, confidence mapping, operating bands, and audit records are
deterministic NumPy/Python on top.

Calibration follows the standard online-monitoring pattern: a baseline of
nominal-operation ``(js, w1)`` samples fixes per-divergence operating means and
standard deviations together with a normal-quantile operating band. At runtime,
each new divergence is converted to a one-sided z-score against its baseline,
the two z-scores are combined into a composite deviation, and the confidence is
``exp(-z_composite / sensitivity)`` — exactly ``1.0`` while the twin tracks
inside its calibrated band, decaying smoothly as it drifts away.

The scorer is review-only: it never proposes or applies actuation. It is a
health observable consumed by the digital-twin operator evidence summary and
the observability exporters.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor._julia_runtime import require_juliacall_main

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "TwinConfidenceBaseline",
    "TwinConfidenceCalibrator",
    "TwinConfidenceScore",
    "TwinConfidenceSummary",
    "TwinDivergence",
    "phase_order_divergence",
    "score_twin_confidence",
    "summarise_twin_confidence",
    "twin_confidence_prometheus_text",
]

TWO_PI: float = 2.0 * np.pi
_JS_MAX: float = float(np.log(2.0))
_EPS: float = 1e-12
_DEFAULT_N_BINS: int = 36
_DEFAULT_SENSITIVITY: float = 3.0
_DEFAULT_WARNING_CONFIDENCE: float = 0.6
_DEFAULT_CRITICAL_CONFIDENCE: float = 0.3
_DEFAULT_BAND_Z: float = 3.0
_PARITY_TOL: float = 1e-9


# ---------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TwinDivergence:
    """Raw divergence pair between a model tick and its observed twin tick.

    Attributes
    ----------
    phase_js_divergence : float
        Jensen–Shannon divergence (natural log) between the model and observed
        phase histograms, in ``[0, ln 2]``.
    order_wasserstein : float
        One-dimensional Wasserstein-1 distance between the model and observed
        order-parameter windows, in ``[0, 1]``.
    n_bins : int
        Number of phase histogram bins used.
    backend : str
        Name of the acceleration backend that produced the pair.
    """

    phase_js_divergence: float
    order_wasserstein: float
    n_bins: int
    backend: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the divergence pair.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the divergence fields.
        """
        return {
            "phase_js_divergence": self.phase_js_divergence,
            "order_wasserstein": self.order_wasserstein,
            "n_bins": self.n_bins,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class TwinConfidenceBaseline:
    """Calibrated nominal-operation baseline for twin divergences.

    Attributes
    ----------
    phase_js_mean, phase_js_std : float
        Mean and (population) standard deviation of the nominal phase
        Jensen–Shannon divergence samples.
    order_w1_mean, order_w1_std : float
        Mean and (population) standard deviation of the nominal Wasserstein-1
        samples.
    sample_count : int
        Number of nominal samples the baseline was fitted on.
    band_z : float
        Normal-quantile multiplier defining the upper operating band
        ``mean + band_z * std`` for each divergence.
    """

    phase_js_mean: float
    phase_js_std: float
    order_w1_mean: float
    order_w1_std: float
    sample_count: int
    band_z: float

    @property
    def phase_js_upper_band(self) -> float:
        """Return the upper nominal operating band for the phase divergence.

        Returns
        -------
        float
            ``phase_js_mean + band_z * phase_js_std``.
        """
        return self.phase_js_mean + self.band_z * self.phase_js_std

    @property
    def order_w1_upper_band(self) -> float:
        """Return the upper nominal operating band for the Wasserstein distance.

        Returns
        -------
        float
            ``order_w1_mean + band_z * order_w1_std``.
        """
        return self.order_w1_mean + self.band_z * self.order_w1_std

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the baseline.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the baseline fields and bands.
        """
        return {
            "phase_js_mean": self.phase_js_mean,
            "phase_js_std": self.phase_js_std,
            "phase_js_upper_band": self.phase_js_upper_band,
            "order_w1_mean": self.order_w1_mean,
            "order_w1_std": self.order_w1_std,
            "order_w1_upper_band": self.order_w1_upper_band,
            "sample_count": self.sample_count,
            "band_z": self.band_z,
        }


@dataclass(frozen=True)
class TwinConfidenceScore:
    """Online confidence score for one twin tick against a baseline.

    Attributes
    ----------
    confidence : float
        Calibrated confidence in ``[0, 1]``; ``1.0`` while the twin tracks
        inside its nominal band, decaying as it diverges.
    status : str
        Operator status: ``"healthy"``, ``"warning"``, or ``"critical"``.
    phase_js_divergence, order_wasserstein : float
        The raw divergences scored.
    phase_js_z, order_w1_z : float
        One-sided z-scores of each divergence against its baseline.
    composite_z : float
        Euclidean combination of the two one-sided z-scores.
    phase_js_within_band, order_w1_within_band : bool
        Whether each divergence is inside its calibrated upper operating band.
    backend : str
        Acceleration backend that produced the divergences.
    score_hash : str
        Deterministic SHA-256 over the audit record (excluding the hash).
    """

    confidence: float
    status: str
    phase_js_divergence: float
    order_wasserstein: float
    phase_js_z: float
    order_w1_z: float
    composite_z: float
    phase_js_within_band: bool
    order_w1_within_band: bool
    backend: str
    score_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the confidence score.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of every score field including the
            ``score_hash``.
        """
        return {
            "confidence": self.confidence,
            "status": self.status,
            "phase_js_divergence": self.phase_js_divergence,
            "order_wasserstein": self.order_wasserstein,
            "phase_js_z": self.phase_js_z,
            "order_w1_z": self.order_w1_z,
            "composite_z": self.composite_z,
            "phase_js_within_band": self.phase_js_within_band,
            "order_w1_within_band": self.order_w1_within_band,
            "backend": self.backend,
            "score_hash": self.score_hash,
        }


# ---------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")

_BackendFn = Callable[
    [
        FloatArray,  # model_phases (N,)
        FloatArray,  # observed_phases (N,)
        FloatArray,  # model_order (W,)
        FloatArray,  # observed_order (W,)
        int,  # n
        int,  # w
        int,  # n_bins
    ],
    FloatArray,  # (js, w1)
]


def _load_rust() -> _BackendFn:
    """Load the Rust twin-confidence backend callable."""
    from spo_kernel import twin_divergence_rust

    return cast("_BackendFn", twin_divergence_rust)


def _load_mojo() -> _BackendFn:  # pragma: no cover — toolchain-gated
    """Load the Mojo twin-confidence backend callable."""
    from ..experimental.accelerators.monitor._twin_confidence_mojo import (
        _ensure_exe,
        twin_divergence_mojo,
    )

    _ensure_exe()
    return twin_divergence_mojo


def _load_julia() -> _BackendFn:  # pragma: no cover — toolchain-gated
    """Load the Julia twin-confidence backend callable."""
    require_juliacall_main()
    from ..experimental.accelerators.monitor._twin_confidence_julia import (
        twin_divergence_julia,
    )

    return twin_divergence_julia


def _load_go() -> _BackendFn:  # pragma: no cover — toolchain-gated
    """Load the Go twin-confidence backend callable."""
    from ..experimental.accelerators.monitor._twin_confidence_go import (
        _load_lib,
        twin_divergence_go,
    )

    _load_lib()
    return twin_divergence_go


_LOADERS: dict[str, Callable[[], _BackendFn]] = {
    "rust": _load_rust,
    "mojo": _load_mojo,
    "julia": _load_julia,
    "go": _load_go,
}
_BACKEND_CACHE: dict[str, _BackendFn] = {}


def _load_backend(name: str) -> _BackendFn:
    """Load and cache the named backend callable."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve the active and available backends, fastest-first."""
    _BACKEND_CACHE.clear()
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch_backend() -> tuple[str, _BackendFn | None]:
    """Return the fastest available backend callable, or ``None`` for Python."""
    ordered_backends = [ACTIVE_BACKEND, *AVAILABLE_BACKENDS]
    seen: set[str] = set()
    for backend in ordered_backends:
        if backend in seen:
            continue
        seen.add(backend)
        if backend == "python":
            return "python", None
        try:
            return backend, _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
    return "python", None


# ---------------------------------------------------------------------
# NumPy reference kernel
# ---------------------------------------------------------------------


def _phase_histogram(phases: FloatArray, n_bins: int) -> FloatArray:
    """Return a normalised histogram of phases over ``[0, 2π)``.

    Parameters
    ----------
    phases : FloatArray
        Phase samples in radians (any range; wrapped to ``[0, 2π)``).
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    FloatArray
        Probability mass per bin, summing to ``1`` (uniform when empty).
    """
    wrapped = phases - np.floor(phases / TWO_PI) * TWO_PI
    width = TWO_PI / n_bins
    idx = np.floor(wrapped / width).astype(np.int64)
    np.clip(idx, 0, n_bins - 1, out=idx)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return np.full(n_bins, 1.0 / n_bins, dtype=np.float64)
    return np.asarray(counts / total, dtype=np.float64)


def _jensen_shannon(p: FloatArray, q: FloatArray) -> float:
    """Return the Jensen–Shannon divergence (natural log) of two histograms.

    Parameters
    ----------
    p, q : FloatArray
        Probability mass functions of equal length.

    Returns
    -------
    float
        Jensen–Shannon divergence in ``[0, ln 2]``.
    """
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _kl(p: FloatArray, m: FloatArray) -> float:
    """Return the Kullback-Leibler divergence between two distributions."""
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / m[mask])))


def _wasserstein1(model_order: FloatArray, observed_order: FloatArray) -> float:
    """Return the 1-D Wasserstein-1 distance between two equal-length windows.

    Parameters
    ----------
    model_order, observed_order : FloatArray
        Order-parameter windows of equal length.

    Returns
    -------
    float
        Mean absolute difference of the order-sorted windows.
    """
    if model_order.size == 0:
        return 0.0
    sorted_model = np.sort(model_order)
    sorted_obs = np.sort(observed_order)
    return float(np.mean(np.abs(sorted_model - sorted_obs)))


def _python_kernel(
    model_phases: FloatArray,
    observed_phases: FloatArray,
    model_order: FloatArray,
    observed_order: FloatArray,
    n: int,
    w: int,
    n_bins: int,
) -> FloatArray:
    """Compute the reference ``(js, w1)`` divergence pair.

    Every compiled backend must reproduce this output to ``1e-9``.

    Parameters
    ----------
    model_phases, observed_phases : FloatArray
        Model and observed phase vectors of length ``n``.
    model_order, observed_order : FloatArray
        Model and observed order-parameter windows of length ``w``.
    n, w, n_bins : int
        Phase count, order-window length, and histogram bin count.

    Returns
    -------
    FloatArray
        Two-element array ``[phase_js_divergence, order_wasserstein]``.
    """
    del n, w
    p = _phase_histogram(model_phases, n_bins)
    q = _phase_histogram(observed_phases, n_bins)
    js = _jensen_shannon(p, q)
    w1 = _wasserstein1(model_order, observed_order)
    return np.asarray([js, w1], dtype=np.float64)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def _as_real_vector(name: str, value: object) -> FloatArray:
    """Return the value as a validated 1-D finite real vector, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _contains_complex_alias(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        parsed = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if parsed.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {parsed.shape}")
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(parsed, dtype=np.float64)


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):  # pragma: no cover - numpy always coerces
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):  # pragma: no cover - numpy always coerces
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)


def _validate_positive_int(name: str, value: object) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    value_int = int(value)
    if value_int < 1:
        raise ValueError(f"{name} must be a positive integer, got {value_int}")
    return value_int


def _validate_finite_real(name: str, value: object, *, minimum: float) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    return number


def _validate_order_window(name: str, value: object) -> FloatArray:
    """Return the validated order-parameter window, else raise."""
    parsed = _as_real_vector(name, value)
    if np.any(parsed < 0.0) or np.any(parsed > 1.0):
        raise ValueError(f"{name} order-parameter values must lie in [0, 1]")
    return parsed


def _validate_kernel_output(value: object, *, backend: str) -> tuple[float, float]:
    """Return the backend kernel output matching the reference, else raise."""
    parsed = np.asarray(value, dtype=np.float64).ravel()
    if parsed.shape != (2,):
        raise ValueError(f"backend {backend!r} output shape {parsed.shape} is not (2,)")
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"backend {backend!r} produced a non-finite divergence")
    js = float(parsed[0])
    w1 = float(parsed[1])
    if js < -_PARITY_TOL or js > _JS_MAX + _PARITY_TOL:
        raise ValueError(
            f"backend {backend!r} Jensen–Shannon divergence {js} outside [0, ln 2]"
        )
    if w1 < -_PARITY_TOL or w1 > 1.0 + _PARITY_TOL:
        raise ValueError(
            f"backend {backend!r} Wasserstein-1 distance {w1} outside [0, 1]"
        )
    return max(0.0, js), max(0.0, w1)


# ---------------------------------------------------------------------
# Public divergence entry point
# ---------------------------------------------------------------------


def phase_order_divergence(
    model_phases: FloatArray,
    observed_phases: FloatArray,
    model_order: FloatArray,
    observed_order: FloatArray,
    *,
    n_bins: int = _DEFAULT_N_BINS,
) -> TwinDivergence:
    """Compute the phase/order divergence pair for one twin tick.

    Parameters
    ----------
    model_phases, observed_phases : FloatArray
        Model and observed phase vectors (radians). Must share length ``N >= 1``.
    model_order, observed_order : FloatArray
        Model and observed order-parameter windows with values in ``[0, 1]``.
        Must share length ``W >= 1``.
    n_bins : int, optional
        Number of phase histogram bins (default ``36``, i.e. 10° per bin).

    Returns
    -------
    TwinDivergence
        The Jensen–Shannon phase divergence and Wasserstein-1 order distance,
        produced by the fastest available backend.

    Raises
    ------
    ValueError
        If shapes mismatch, lengths are empty, order values fall outside
        ``[0, 1]``, ``n_bins`` is not a positive integer, or a backend returns a
        non-physical pair.
    """
    n_bins = _validate_positive_int("n_bins", n_bins)
    model_phases64 = _as_real_vector("model_phases", model_phases)
    observed_phases64 = _as_real_vector("observed_phases", observed_phases)
    model_order64 = _validate_order_window("model_order", model_order)
    observed_order64 = _validate_order_window("observed_order", observed_order)

    n = int(model_phases64.size)
    if n == 0:
        raise ValueError("model_phases must contain at least one phase")
    if observed_phases64.size != n:
        raise ValueError(
            f"observed_phases length {observed_phases64.size} != model_phases {n}"
        )
    w = int(model_order64.size)
    if w == 0:
        raise ValueError("model_order must contain at least one sample")
    if observed_order64.size != w:
        raise ValueError(
            f"observed_order length {observed_order64.size} != model_order {w}"
        )

    backend_name, backend_fn = _dispatch_backend()
    if backend_fn is None:
        raw = _python_kernel(
            model_phases64,
            observed_phases64,
            model_order64,
            observed_order64,
            n,
            w,
            n_bins,
        )
    else:
        raw = backend_fn(
            model_phases64,
            observed_phases64,
            model_order64,
            observed_order64,
            n,
            w,
            n_bins,
        )
    js, w1 = _validate_kernel_output(raw, backend=backend_name)
    return TwinDivergence(
        phase_js_divergence=js,
        order_wasserstein=w1,
        n_bins=n_bins,
        backend=backend_name,
    )


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------


@dataclass
class TwinConfidenceCalibrator:
    """Accumulate nominal twin divergences into a calibrated baseline.

    The calibrator ingests divergence pairs gathered while the twin is known to
    track its model (commissioning, healthy replay, or a trusted window) and
    fits per-divergence means, population standard deviations, and a
    normal-quantile operating band. The resulting :class:`TwinConfidenceBaseline`
    feeds :func:`score_twin_confidence`.

    Attributes
    ----------
    band_z : float
        Normal-quantile multiplier for the upper operating band (default ``3``).
    """

    band_z: float = _DEFAULT_BAND_Z
    _phase_js: list[float] = cast("list[float]", None)
    _order_w1: list[float] = cast("list[float]", None)

    def __post_init__(self) -> None:
        self.band_z = _validate_finite_real("band_z", self.band_z, minimum=0.0)
        self._phase_js = []
        self._order_w1 = []

    @property
    def sample_count(self) -> int:
        """Return the number of nominal samples accumulated.

        Returns
        -------
        int
            Count of ingested divergence pairs.
        """
        return len(self._phase_js)

    def observe(self, divergence: TwinDivergence) -> None:
        """Add one nominal divergence pair to the calibration set.

        Parameters
        ----------
        divergence : TwinDivergence
            A divergence pair measured during trusted nominal operation.
        """
        self._phase_js.append(divergence.phase_js_divergence)
        self._order_w1.append(divergence.order_wasserstein)

    def observe_many(self, divergences: Sequence[TwinDivergence]) -> None:
        """Add several nominal divergence pairs to the calibration set.

        Parameters
        ----------
        divergences : Sequence[TwinDivergence]
            Divergence pairs measured during trusted nominal operation.
        """
        for divergence in divergences:
            self.observe(divergence)

    def baseline(self) -> TwinConfidenceBaseline:
        """Fit and return the calibrated baseline.

        Returns
        -------
        TwinConfidenceBaseline
            Per-divergence means, population standard deviations, sample count,
            and operating band multiplier.

        Raises
        ------
        ValueError
            If no nominal samples have been observed.
        """
        if not self._phase_js:
            raise ValueError("calibration requires at least one nominal sample")
        phase_js = np.asarray(self._phase_js, dtype=np.float64)
        order_w1 = np.asarray(self._order_w1, dtype=np.float64)
        return TwinConfidenceBaseline(
            phase_js_mean=float(np.mean(phase_js)),
            phase_js_std=float(np.std(phase_js)),
            order_w1_mean=float(np.mean(order_w1)),
            order_w1_std=float(np.std(order_w1)),
            sample_count=int(phase_js.size),
            band_z=self.band_z,
        )


# ---------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------


def _one_sided_z(value: float, mean: float, std: float) -> float:
    """Return the one-sided z-score of a value against a window."""
    return max(0.0, (value - mean) / max(std, _EPS))


def _confidence_status(
    confidence: float,
    *,
    warning_confidence: float,
    critical_confidence: float,
) -> str:
    """Return the confidence status label for a divergence score."""
    if confidence < critical_confidence:
        return "critical"
    if confidence < warning_confidence:
        return "warning"
    return "healthy"


def score_twin_confidence(
    divergence: TwinDivergence,
    baseline: TwinConfidenceBaseline,
    *,
    sensitivity: float = _DEFAULT_SENSITIVITY,
    warning_confidence: float = _DEFAULT_WARNING_CONFIDENCE,
    critical_confidence: float = _DEFAULT_CRITICAL_CONFIDENCE,
) -> TwinConfidenceScore:
    """Score one twin divergence against a calibrated baseline.

    Each divergence is converted to a one-sided z-score against its baseline
    mean and standard deviation; the two z-scores are combined into a composite
    Euclidean deviation, and the confidence is ``exp(-composite_z /
    sensitivity)`` — ``1.0`` while both divergences sit at or below their
    nominal means, decaying smoothly as the twin drifts.

    Parameters
    ----------
    divergence : TwinDivergence
        The divergence pair to score.
    baseline : TwinConfidenceBaseline
        The calibrated nominal baseline.
    sensitivity : float, optional
        Composite-deviation scale of the confidence decay (default ``3``):
        larger values decay more slowly. Must be ``> 0``.
    warning_confidence : float, optional
        Confidence at or above which the status is ``"healthy"`` rather than
        ``"warning"`` (default ``0.6``). In ``[0, 1]``.
    critical_confidence : float, optional
        Confidence below which the status is ``"critical"`` (default ``0.3``).
        In ``[0, 1]`` and ``<= warning_confidence``.

    Returns
    -------
    TwinConfidenceScore
        The calibrated confidence, status, z-scores, band membership, and a
        deterministic audit hash.

    Raises
    ------
    ValueError
        If ``sensitivity <= 0`` or the confidence thresholds are inconsistent.
    """
    sensitivity = _validate_finite_real("sensitivity", sensitivity, minimum=_EPS)
    warning_confidence = _validate_unit_interval(
        "warning_confidence", warning_confidence
    )
    critical_confidence = _validate_unit_interval(
        "critical_confidence", critical_confidence
    )
    if critical_confidence > warning_confidence:
        raise ValueError("critical_confidence must be <= warning_confidence")

    phase_js_z = _one_sided_z(
        divergence.phase_js_divergence,
        baseline.phase_js_mean,
        baseline.phase_js_std,
    )
    order_w1_z = _one_sided_z(
        divergence.order_wasserstein,
        baseline.order_w1_mean,
        baseline.order_w1_std,
    )
    composite_z = float(np.hypot(phase_js_z, order_w1_z))
    confidence = float(np.clip(np.exp(-composite_z / sensitivity), 0.0, 1.0))
    status = _confidence_status(
        confidence,
        warning_confidence=warning_confidence,
        critical_confidence=critical_confidence,
    )
    score = TwinConfidenceScore(
        confidence=confidence,
        status=status,
        phase_js_divergence=divergence.phase_js_divergence,
        order_wasserstein=divergence.order_wasserstein,
        phase_js_z=phase_js_z,
        order_w1_z=order_w1_z,
        composite_z=composite_z,
        phase_js_within_band=(
            divergence.phase_js_divergence <= baseline.phase_js_upper_band + _EPS
        ),
        order_w1_within_band=(
            divergence.order_wasserstein <= baseline.order_w1_upper_band + _EPS
        ),
        backend=divergence.backend,
        score_hash="",
    )
    return _with_hash(score)


def _validate_unit_interval(name: str, value: object) -> float:
    """Return ``value`` as a float in [0, 1], else raise ``ValueError``."""
    number = _validate_finite_real(name, value, minimum=0.0)
    if number > 1.0:
        raise ValueError(f"{name} must be <= 1, got {number}")
    return number


def _with_hash(score: TwinConfidenceScore) -> TwinConfidenceScore:
    """Return the record augmented with its canonical-JSON SHA-256 hash."""
    record = score.to_audit_record()
    record.pop("score_hash", None)
    serialised = json.dumps(record, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return TwinConfidenceScore(
        confidence=score.confidence,
        status=score.status,
        phase_js_divergence=score.phase_js_divergence,
        order_wasserstein=score.order_wasserstein,
        phase_js_z=score.phase_js_z,
        order_w1_z=score.order_w1_z,
        composite_z=score.composite_z,
        phase_js_within_band=score.phase_js_within_band,
        order_w1_within_band=score.order_w1_within_band,
        backend=score.backend,
        score_hash=digest,
    )


# ---------------------------------------------------------------------
# Operator-facing aggregation and Prometheus export
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TwinConfidenceSummary:
    """Operator-facing aggregate over a sequence of twin-confidence scores.

    Attributes
    ----------
    tick_count : int
        Number of scored ticks.
    healthy_count, warning_count, critical_count : int
        Per-status tick counts.
    min_confidence, mean_confidence : float
        Minimum and arithmetic-mean confidence across the scored ticks.
    latest_confidence : float
        Confidence of the most recently scored tick.
    worst_status : str
        ``"critical"`` if any tick was critical, else ``"warning"`` if any was
        warning, else ``"healthy"``.
    latest_status : str
        Status of the most recently scored tick.
    summary_hash : str
        Deterministic SHA-256 over the audit record (excluding the hash).
    """

    tick_count: int
    healthy_count: int
    warning_count: int
    critical_count: int
    min_confidence: float
    mean_confidence: float
    latest_confidence: float
    worst_status: str
    latest_status: str
    summary_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the summary.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of every summary field including
            the ``summary_hash``.
        """
        return {
            "tick_count": self.tick_count,
            "healthy_count": self.healthy_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "min_confidence": self.min_confidence,
            "mean_confidence": self.mean_confidence,
            "latest_confidence": self.latest_confidence,
            "worst_status": self.worst_status,
            "latest_status": self.latest_status,
            "summary_hash": self.summary_hash,
        }


def summarise_twin_confidence(
    scores: Sequence[TwinConfidenceScore],
) -> TwinConfidenceSummary:
    """Aggregate a sequence of twin-confidence scores into operator evidence.

    Parameters
    ----------
    scores : Sequence[TwinConfidenceScore]
        The per-tick scores in chronological order.

    Returns
    -------
    TwinConfidenceSummary
        The deterministic operator-facing aggregate.

    Raises
    ------
    ValueError
        If ``scores`` is empty.
    """
    if not scores:
        raise ValueError("summarise_twin_confidence requires at least one score")
    confidences = [score.confidence for score in scores]
    healthy = sum(1 for score in scores if score.status == "healthy")
    warning = sum(1 for score in scores if score.status == "warning")
    critical = sum(1 for score in scores if score.status == "critical")
    if critical:
        worst_status = "critical"
    elif warning:
        worst_status = "warning"
    else:
        worst_status = "healthy"
    summary = TwinConfidenceSummary(
        tick_count=len(scores),
        healthy_count=healthy,
        warning_count=warning,
        critical_count=critical,
        min_confidence=float(min(confidences)),
        mean_confidence=float(sum(confidences) / len(confidences)),
        latest_confidence=scores[-1].confidence,
        worst_status=worst_status,
        latest_status=scores[-1].status,
        summary_hash="",
    )
    return _with_summary_hash(summary)


def _with_summary_hash(summary: TwinConfidenceSummary) -> TwinConfidenceSummary:
    """Return the summary augmented with its canonical SHA-256 hash."""
    record = summary.to_audit_record()
    record.pop("summary_hash", None)
    serialised = json.dumps(record, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return TwinConfidenceSummary(
        tick_count=summary.tick_count,
        healthy_count=summary.healthy_count,
        warning_count=summary.warning_count,
        critical_count=summary.critical_count,
        min_confidence=summary.min_confidence,
        mean_confidence=summary.mean_confidence,
        latest_confidence=summary.latest_confidence,
        worst_status=summary.worst_status,
        latest_status=summary.latest_status,
        summary_hash=digest,
    )


_STATUS_LEVELS: dict[str, int] = {"healthy": 0, "warning": 1, "critical": 2}


def twin_confidence_prometheus_text(
    summary: TwinConfidenceSummary,
    *,
    prefix: str = "spo",
) -> str:
    """Render a twin-confidence summary as Prometheus exposition text.

    Parameters
    ----------
    summary : TwinConfidenceSummary
        The operator-facing aggregate to export.
    prefix : str, optional
        Metric-name prefix (default ``"spo"``).

    Returns
    -------
    str
        Prometheus exposition text with confidence gauges, per-status counters,
        and a numeric worst-status level gauge.

    Raises
    ------
    ValueError
        If ``prefix`` is not a non-empty string.
    """
    _require_non_empty(prefix, "prefix")
    lines = [
        f"# HELP {prefix}_twin_confidence_mean Mean twin confidence over scored ticks",
        f"# TYPE {prefix}_twin_confidence_mean gauge",
        f"{prefix}_twin_confidence_mean {summary.mean_confidence}",
        f"# HELP {prefix}_twin_confidence_min Minimum twin confidence over ticks",
        f"# TYPE {prefix}_twin_confidence_min gauge",
        f"{prefix}_twin_confidence_min {summary.min_confidence}",
        f"# HELP {prefix}_twin_confidence_latest Most recent twin confidence",
        f"# TYPE {prefix}_twin_confidence_latest gauge",
        f"{prefix}_twin_confidence_latest {summary.latest_confidence}",
        f"# HELP {prefix}_twin_confidence_tick_count Scored twin-confidence ticks",
        f"# TYPE {prefix}_twin_confidence_tick_count gauge",
        f"{prefix}_twin_confidence_tick_count {summary.tick_count}",
        (
            f"# HELP {prefix}_twin_confidence_status_total "
            "Twin-confidence ticks per operator status"
        ),
        f"# TYPE {prefix}_twin_confidence_status_total counter",
        (
            f'{prefix}_twin_confidence_status_total{{status="healthy"}} '
            f"{summary.healthy_count}"
        ),
        (
            f'{prefix}_twin_confidence_status_total{{status="warning"}} '
            f"{summary.warning_count}"
        ),
        (
            f'{prefix}_twin_confidence_status_total{{status="critical"}} '
            f"{summary.critical_count}"
        ),
        (
            f"# HELP {prefix}_twin_confidence_worst_status_level "
            "Worst operator status (0 healthy, 1 warning, 2 critical)"
        ),
        f"# TYPE {prefix}_twin_confidence_worst_status_level gauge",
        (
            f"{prefix}_twin_confidence_worst_status_level "
            f"{_STATUS_LEVELS[summary.worst_status]}"
        ),
    ]
    return "\n".join(lines) + "\n"


def _require_non_empty(value: str, name: str) -> None:
    """Return ``value`` if it is a non-empty string, else raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
