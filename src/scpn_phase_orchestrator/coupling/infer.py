# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Causal coupling inference

"""Infer directed coupling matrices from oscillator phase time series.

The production implementation currently exposes transfer-entropy inference.
Granger and NOTEARS are accepted as explicit method names so callers can fail
closed instead of accidentally receiving a weaker substitute.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.transfer_entropy import (
    ACTIVE_BACKEND as TRANSFER_ENTROPY_BACKEND,
)
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    AVAILABLE_BACKENDS as TRANSFER_ENTROPY_BACKENDS,
)
from scpn_phase_orchestrator.monitor.transfer_entropy import transfer_entropy_matrix

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
InferenceMethod: TypeAlias = Literal["transfer_entropy", "granger", "notears"]
NormalisationMode: TypeAlias = Literal["max", "none"]

PACKAGE_NAME = "auto-coupling-estimation"
ORIENTATION = "source_to_target"

__all__ = [
    "BoolArray",
    "CouplingInferenceConfig",
    "CouplingInferenceResult",
    "FloatArray",
    "InferenceMethod",
    "NormalisationMode",
    "auto_coupling_estimation",
    "infer_coupling_from_timeseries",
]


def _validate_optional_finite_real(value: object | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real value")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _contains_boolean_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, bool) for item in raw.ravel())


@dataclass(frozen=True, slots=True)
class CouplingInferenceConfig:
    """Configuration for data-driven coupling inference.

    ``threshold_quantile`` is applied to positive off-diagonal causal scores.
    ``threshold_absolute`` takes precedence when provided.
    """

    method: InferenceMethod = "transfer_entropy"
    n_bins: int = 8
    threshold_quantile: float | None = 0.75
    threshold_absolute: float | None = None
    normalisation: NormalisationMode = "max"
    min_timesteps: int = 4


@dataclass(frozen=True, slots=True)
class CouplingInferenceResult:
    """Causal coupling estimate and reproducibility diagnostics.

    ``knm`` uses source-to-target orientation: ``knm[i, j]`` is the inferred
    directed influence from oscillator ``i`` to oscillator ``j``. For the
    standard UPDE convention where ``K[i, j]`` means oscillator ``j`` pulls
    oscillator ``i``, use :meth:`to_upde_knm`.
    """

    knm: FloatArray
    score_matrix: FloatArray
    support_mask: BoolArray
    method: str
    score_kind: str
    n_bins: int
    threshold: float
    normalisation: str
    shape: tuple[int, int]
    package: str = PACKAGE_NAME
    orientation: str = ORIENTATION
    active_backend: str = TRANSFER_ENTROPY_BACKEND
    available_backends: tuple[str, ...] = tuple(TRANSFER_ENTROPY_BACKENDS)

    @property
    def edge_count(self) -> int:
        """Number of nonzero directed off-diagonal support edges.

        Returns
        -------
        int
            Number of nonzero directed off-diagonal support edges.
        """
        return int(np.count_nonzero(self.support_mask))

    @property
    def density(self) -> float:
        """Directed graph density over off-diagonal entries.

        Returns
        -------
        float
            Directed graph density over off-diagonal entries.
        """
        n_oscillators = self.support_mask.shape[0]
        possible = n_oscillators * (n_oscillators - 1)
        if possible == 0:
            return 0.0
        return float(self.edge_count / possible)

    def to_upde_knm(self) -> FloatArray:
        """Return matrix in UPDE target-by-source coupling convention.

        Returns
        -------
        FloatArray
            Return matrix in UPDE target-by-source coupling convention.
        """
        return np.ascontiguousarray(self.knm.T, dtype=np.float64)

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe, replay-friendly inference record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe, replay-friendly inference record.
        """
        return {
            "package": self.package,
            "method": self.method,
            "score_kind": self.score_kind,
            "orientation": self.orientation,
            "shape": list(self.shape),
            "n_bins": self.n_bins,
            "threshold": self.threshold,
            "normalisation": self.normalisation,
            "active_backend": self.active_backend,
            "available_backends": list(self.available_backends),
            "knm": self.knm.tolist(),
            "score_matrix": self.score_matrix.tolist(),
            "support_mask": self.support_mask.tolist(),
            "diagnostics": {
                "edge_count": self.edge_count,
                "density": self.density,
                "score_min": float(np.min(self.score_matrix)),
                "score_max": float(np.max(self.score_matrix)),
            },
        }


def _validate_phase_series(value: object, *, min_timesteps: int) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("phase_series must not contain boolean values")
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError("phase_series must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(
            "phase_series must be a finite 2-D array with shape "
            "(oscillators, timesteps)"
        )
    try:
        series = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "phase_series must be a finite 2-D array with shape "
            "(oscillators, timesteps)"
        ) from exc
    if series.ndim != 2:
        raise ValueError(
            "phase_series must be a finite 2-D array with shape "
            f"(oscillators, timesteps), got shape {series.shape}"
        )
    if series.shape[0] < 2:
        raise ValueError("phase_series must contain at least 2 oscillators")
    if series.shape[1] < min_timesteps:
        raise ValueError(
            f"phase_series must contain at least {min_timesteps} timesteps"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("phase_series must be a finite 2-D array")
    return np.ascontiguousarray(series, dtype=np.float64)


def _validate_config(config: CouplingInferenceConfig) -> CouplingInferenceConfig:
    if config.method not in {"transfer_entropy", "granger", "notears"}:
        raise ValueError("method must be one of: transfer_entropy, granger, notears")
    if isinstance(config.n_bins, bool) or not isinstance(config.n_bins, int):
        raise TypeError("n_bins must be an integer greater than or equal to 2")
    if config.n_bins < 2:
        raise ValueError("n_bins must be greater than or equal to 2")

    threshold_quantile = _validate_optional_finite_real(
        config.threshold_quantile, name="threshold_quantile"
    )
    if threshold_quantile is not None and not (0.0 <= threshold_quantile <= 1.0):
        raise ValueError("threshold_quantile must lie in [0, 1]")

    threshold_absolute = _validate_optional_finite_real(
        config.threshold_absolute, name="threshold_absolute"
    )
    if threshold_absolute is not None and threshold_absolute < 0.0:
        raise ValueError("threshold_absolute must be non-negative")

    if config.normalisation not in {"max", "none"}:
        raise ValueError("normalisation must be one of: max, none")
    if isinstance(config.min_timesteps, bool) or not isinstance(
        config.min_timesteps, int
    ):
        raise TypeError("min_timesteps must be an integer at least 4")
    if config.min_timesteps < 4:
        raise ValueError("min_timesteps must be at least 4")
    return config


def _threshold_scores(
    scores: FloatArray, config: CouplingInferenceConfig
) -> tuple[float, BoolArray]:
    off_diagonal = ~np.eye(scores.shape[0], dtype=bool)
    positive_scores = scores[(scores > 0.0) & off_diagonal]
    if config.threshold_absolute is not None:
        threshold = float(config.threshold_absolute)
    elif config.threshold_quantile is not None and positive_scores.size:
        threshold = float(np.quantile(positive_scores, config.threshold_quantile))
    else:
        threshold = 0.0

    support = np.asarray((scores >= threshold) & (scores > 0.0), dtype=np.bool_)
    np.fill_diagonal(support, False)
    return threshold, support


def _normalise_knm(
    scores: FloatArray, support: BoolArray, normalisation: NormalisationMode
) -> FloatArray:
    masked = np.where(support, scores, 0.0).astype(np.float64, copy=False)
    if normalisation == "none":
        return np.ascontiguousarray(masked, dtype=np.float64)
    scale = float(np.max(masked)) if np.any(masked > 0.0) else 0.0
    if scale <= 0.0:
        return np.zeros_like(masked, dtype=np.float64)
    return np.ascontiguousarray(masked / scale, dtype=np.float64)


def infer_coupling_from_timeseries(
    phase_series: object,
    *,
    config: CouplingInferenceConfig | None = None,
) -> CouplingInferenceResult:
    """Infer a directed coupling matrix from phase time-series data.

    Parameters
    ----------
    phase_series : object
        Phase time series, shape ``(T, N)``.
    config : CouplingInferenceConfig | None
        Optional inference configuration, or ``None`` for defaults.

    Returns
    -------
    CouplingInferenceResult
        The inferred directed coupling result.

    Raises
    ------
    NotImplementedError
        If the configured inference method is not implemented.
    RuntimeError
        If inference fails on the supplied series.
    """
    resolved_config = _validate_config(config or CouplingInferenceConfig())
    series = _validate_phase_series(
        phase_series, min_timesteps=resolved_config.min_timesteps
    )

    if resolved_config.method != "transfer_entropy":
        raise NotImplementedError(
            f"{resolved_config.method} coupling inference is not implemented; "
            "use method='transfer_entropy' for the current production backend"
        )

    scores = np.asarray(
        transfer_entropy_matrix(series, n_bins=resolved_config.n_bins),
        dtype=np.float64,
    )
    if scores.shape != (series.shape[0], series.shape[0]):
        raise RuntimeError(
            "transfer-entropy backend returned an unexpected matrix shape: "
            f"{scores.shape}"
        )
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("transfer-entropy backend returned non-finite scores")
    if np.any(scores < -1e-12):
        raise RuntimeError("transfer-entropy backend returned negative scores")
    if not np.allclose(np.diag(scores), 0.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("transfer-entropy backend returned non-zero self scores")
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    threshold, support = _threshold_scores(scores, resolved_config)
    knm = _normalise_knm(scores, support, resolved_config.normalisation)
    np.fill_diagonal(knm, 0.0)

    return CouplingInferenceResult(
        knm=knm,
        score_matrix=np.ascontiguousarray(scores, dtype=np.float64),
        support_mask=support,
        method=resolved_config.method,
        score_kind="transfer_entropy",
        n_bins=resolved_config.n_bins,
        threshold=threshold,
        normalisation=resolved_config.normalisation,
        shape=(int(series.shape[0]), int(series.shape[1])),
    )


def auto_coupling_estimation(
    phase_series: object,
    *,
    n_bins: int = 8,
    threshold_quantile: float | None = 0.75,
    threshold_absolute: float | None = None,
    normalisation: NormalisationMode = "max",
) -> CouplingInferenceResult:
    """Infer coupling with the packaged ``auto-coupling-estimation`` profile.

    Parameters
    ----------
    phase_series : object
        Phase time series, shape ``(T, N)``.
    n_bins : int
        Number of histogram bins.
    threshold_quantile : float | None
        Quantile threshold for edge pruning, or ``None``.
    threshold_absolute : float | None
        Absolute threshold for edge pruning, or ``None``.
    normalisation : NormalisationMode
        Edge-weight normalisation mode.

    Returns
    -------
    CouplingInferenceResult
        The inferred coupling result from the packaged auto-estimation profile.
    """
    return infer_coupling_from_timeseries(
        phase_series,
        config=CouplingInferenceConfig(
            method="transfer_entropy",
            n_bins=n_bins,
            threshold_quantile=threshold_quantile,
            threshold_absolute=threshold_absolute,
            normalisation=normalisation,
        ),
    )
