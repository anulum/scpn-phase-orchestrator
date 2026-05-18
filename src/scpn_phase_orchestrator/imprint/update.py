# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Imprint update rules

from __future__ import annotations

from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.imprint.state import ImprintState

__all__ = ["ImprintModel"]

FloatArray: TypeAlias = NDArray[np.float64]


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _finite_vector(
    value: FloatArray,
    name: str,
    *,
    length: int | None = None,
    non_negative: bool = False,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if length is not None and array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if non_negative and np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return array


def _finite_square_matrix(value: FloatArray, name: str, *, size: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_state(state: ImprintState) -> tuple[FloatArray, float]:
    if not isinstance(state, ImprintState):
        raise ValueError("imprint state must be an ImprintState")
    m_k = _finite_vector(state.m_k, "m_k", non_negative=True)
    last_update = _finite_real(state.last_update, "last_update")
    if last_update < 0.0:
        raise ValueError("last_update must be non-negative")
    if not isinstance(state.attribution, dict):
        raise ValueError("imprint attribution must be a dictionary")
    for key, value in state.attribution.items():
        if not isinstance(key, str):
            raise ValueError("imprint attribution keys must be strings")
        _finite_real(value, f"imprint attribution[{key!r}]")
    return m_k, last_update


class ImprintModel:
    """Exponential exposure accumulation with decay and saturation.

    m_k(t+dt) = m_k(t) * exp(-decay_rate * dt) + exposure * dt,
    clipped to [0, saturation].

    Modulates: K (phase coupling), alpha (lag), mu (bifurcation).
    Does NOT modulate knm_r (amplitude coupling strength) — amplitude
    coupling topology is fixed by the binding spec, not learned.
    """

    def __init__(self, decay_rate: float, saturation: float):
        decay_rate = _finite_real(decay_rate, "decay_rate")
        saturation = _finite_real(saturation, "saturation")
        if decay_rate < 0.0:
            raise ValueError(f"decay_rate must be non-negative, got {decay_rate}")
        if saturation <= 0.0:
            raise ValueError(f"saturation must be positive, got {saturation}")
        self._decay_rate = decay_rate
        self._saturation = saturation

    def update(
        self, state: ImprintState, exposure: FloatArray, dt: float
    ) -> ImprintState:
        """Decay existing imprint, add new exposure, clip to saturation."""
        m_k, last_update = _validated_state(state)
        dt = _finite_real(dt, "dt")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        exposure = _finite_vector(
            exposure, "exposure", length=m_k.shape[0], non_negative=True
        )
        decayed = m_k * np.exp(-self._decay_rate * dt)
        m_new = np.clip(decayed + exposure * dt, 0.0, self._saturation)
        return ImprintState(
            m_k=m_new,
            last_update=last_update + dt,
            attribution=state.attribution.copy(),
        )

    def modulate_coupling(self, knm: FloatArray, imprint: ImprintState) -> FloatArray:
        """Scale Knm rows by (1 + m_k)."""
        m_k, _ = _validated_state(imprint)
        knm = _finite_square_matrix(knm, "knm", size=m_k.shape[0])
        result: FloatArray = knm * (1.0 + m_k)[:, np.newaxis]
        return result

    def modulate_lag(self, alpha: FloatArray, imprint: ImprintState) -> FloatArray:
        """Shift phase lags by antisymmetric imprint offset."""
        m_k, _ = _validated_state(imprint)
        alpha = _finite_square_matrix(alpha, "alpha", size=m_k.shape[0])
        offset = m_k[:, np.newaxis] - m_k[np.newaxis, :]
        result: FloatArray = alpha + offset
        return result

    def modulate_mu(self, mu: FloatArray, imprint: ImprintState) -> FloatArray:
        """Scale bifurcation parameter: μ_k * (1 + m_k)."""
        m_k, _ = _validated_state(imprint)
        mu = _finite_vector(mu, "mu", length=m_k.shape[0])
        result: FloatArray = mu * (1.0 + m_k)
        return result
