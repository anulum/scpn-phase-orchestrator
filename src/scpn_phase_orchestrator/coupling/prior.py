# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Universal coupling prior

"""Empirical domain-agnostic prior for coupling hyperparameters.

`UniversalPrior` provides default/sample/log-probability helpers for `K_base`
and `decay_alpha`, plus a Dörfler-Bullo-style critical-coupling estimate over
the spectral module. When the optional Rust kernel is importable the
log-probability path dispatches there; otherwise the NumPy scalar fallback
preserves the same Gaussian-prior contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        prior_log_probability_rust as _rust_log_prob,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["UniversalPrior", "CouplingPrior"]

FloatArray: TypeAlias = NDArray[np.float64]

# Empirical prior from 25 domainpacks (R4-A3 cross-domain transfer study)
_K_BASE_MEAN = 0.47
_K_BASE_STD = 0.09
_DECAY_ALPHA_MEAN = 0.25
_DECAY_ALPHA_STD = 0.07
_MAX_SEED = 2**64 - 1


def _contains_boolean_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, bool) for item in raw.ravel())


def _validate_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} prior hyperparameter must be a finite real")

    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} prior hyperparameter must be finite")
    return resolved


def _validate_positive_real(value: object, *, name: str) -> float:
    resolved = _validate_finite_real(value, name=name)
    if resolved <= 0.0:
        raise ValueError(f"{name} prior hyperparameter must be positive")
    return resolved


def _validate_seed(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("seed must be an integer in the u64 range")
    seed = int(value)
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError("seed must be an integer in the u64 range")
    return seed


def _validate_frequency_vector(value: object) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("omegas must not contain boolean values")
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError("omegas must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError("omegas must be a finite 1-D frequency vector")
    try:
        omegas = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("omegas must be a finite 1-D frequency vector") from exc
    if omegas.ndim != 1:
        raise ValueError("omegas must be a finite 1-D frequency vector")
    if omegas.size == 0:
        raise ValueError("omegas must contain at least one frequency")
    if not np.all(np.isfinite(omegas)):
        raise ValueError("omegas must contain only finite values")
    return np.ascontiguousarray(omegas, dtype=np.float64)


@dataclass
class CouplingPrior:
    """Coupling configuration: base strength, decay, and K_c estimate."""

    K_base: float
    decay_alpha: float
    K_c_estimate: float


class UniversalPrior:
    """Domain-agnostic coupling prior from 25-domainpack empirical distribution.

    K_base ~ N(0.47, 0.09), decay_alpha ~ N(0.25, 0.07).
    Any new domain starts from this prior. Combined with Dörfler-Bullo K_c,
    collapses auto-tune from 5D optimization to 2D.

    Source: R4-A3 cross-domain transfer analysis (Stankovski 2017, Rev. Mod. Phys.).
    """

    def __init__(
        self,
        K_base_mean: float = _K_BASE_MEAN,
        K_base_std: float = _K_BASE_STD,
        decay_alpha_mean: float = _DECAY_ALPHA_MEAN,
        decay_alpha_std: float = _DECAY_ALPHA_STD,
    ):
        self._K_base_mean = _validate_finite_real(K_base_mean, name="K_base_mean")
        self._K_base_std = _validate_positive_real(K_base_std, name="K_base_std")
        self._decay_alpha_mean = _validate_finite_real(
            decay_alpha_mean, name="decay_alpha_mean"
        )
        self._decay_alpha_std = _validate_positive_real(
            decay_alpha_std, name="decay_alpha_std"
        )

    def sample(
        self,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> CouplingPrior:
        """Draw a random coupling configuration from the prior.

        Pass ``rng`` for an explicit generator, or ``seed`` to create a
        seeded one. If neither is given, a fresh unseeded generator is used
        (NOT reproducible across sessions).
        """
        if rng is None:
            seed = _validate_seed(seed)
            rng = np.random.default_rng(seed)
        K = max(0.01, rng.normal(self._K_base_mean, self._K_base_std))
        alpha = max(0.01, rng.normal(self._decay_alpha_mean, self._decay_alpha_std))
        return CouplingPrior(K_base=K, decay_alpha=alpha, K_c_estimate=0.0)

    def default(self) -> CouplingPrior:
        """Return the MAP (maximum a posteriori) estimate = the means."""
        return CouplingPrior(
            K_base=self._K_base_mean,
            decay_alpha=self._decay_alpha_mean,
            K_c_estimate=0.0,
        )

    def estimate_Kc(self, omegas: FloatArray, n_layers: int) -> CouplingPrior:
        """Combine prior with Dörfler-Bullo K_c for given omegas.

        K_c = max|ω_i - ω_j| / λ₂(L) where L is built from the prior's
        decay_alpha on a chain graph of n_layers.
        """
        if isinstance(n_layers, bool) or not isinstance(n_layers, Integral):
            raise TypeError(f"n_layers must be an integer, got {n_layers!r}")
        n_layers = int(n_layers)
        if n_layers <= 0:
            raise ValueError("n_layers must be a positive integer")
        from scpn_phase_orchestrator.coupling.spectral import critical_coupling

        prior = self.default()
        omega_values = _validate_frequency_vector(omegas)
        if omega_values.size != n_layers:
            raise ValueError("n_layers must match the length of omegas")
        idx = np.arange(n_layers)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        knm: FloatArray = prior.K_base * np.exp(-prior.decay_alpha * dist)
        np.fill_diagonal(knm, 0.0)
        K_c = critical_coupling(omega_values, knm)
        return CouplingPrior(
            K_base=prior.K_base,
            decay_alpha=prior.decay_alpha,
            K_c_estimate=K_c,
        )

    def log_probability(self, K_base: float, decay_alpha: float) -> float:
        """Log-probability under the Gaussian prior (unnormalized)."""
        if isinstance(K_base, bool) or not isinstance(K_base, Real):
            raise TypeError("K_base must be a finite real value")
        if isinstance(decay_alpha, bool) or not isinstance(decay_alpha, Real):
            raise TypeError("decay_alpha must be a finite real value")
        K_base = float(K_base)
        decay_alpha = float(decay_alpha)
        if not np.isfinite(K_base) or not np.isfinite(decay_alpha):
            raise ValueError("K_base and decay_alpha must be finite real values")
        if _HAS_RUST:
            log_prob = float(
                _rust_log_prob(
                    K_base,
                    decay_alpha,
                    self._K_base_mean,
                    self._K_base_std,
                    self._decay_alpha_mean,
                    self._decay_alpha_std,
                )
            )
            if not np.isfinite(log_prob):
                raise ValueError(
                    "Rust prior log-probability must be a finite real value"
                )
            return log_prob
        lp_K = -0.5 * ((K_base - self._K_base_mean) / self._K_base_std) ** 2
        lp_a = (
            -0.5 * ((decay_alpha - self._decay_alpha_mean) / self._decay_alpha_std) ** 2
        )
        return lp_K + lp_a
