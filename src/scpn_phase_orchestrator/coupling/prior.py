# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Universal coupling prior

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["UniversalPrior", "CouplingPrior"]

# Empirical prior from 25 domainpacks (R4-A3 cross-domain transfer study)
_K_BASE_MEAN = 0.47
_K_BASE_STD = 0.09
_DECAY_ALPHA_MEAN = 0.25
_DECAY_ALPHA_STD = 0.07


@dataclass
class CouplingPrior:
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
        self._K_base_mean = K_base_mean
        self._K_base_std = K_base_std
        self._decay_alpha_mean = decay_alpha_mean
        self._decay_alpha_std = decay_alpha_std

    def sample(self, rng: np.random.Generator | None = None) -> CouplingPrior:
        """Draw a random coupling configuration from the prior."""
        if rng is None:
            rng = np.random.default_rng()
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

    def estimate_Kc(self, omegas: NDArray, n_layers: int) -> CouplingPrior:
        """Combine prior with Dörfler-Bullo K_c for given omegas.

        K_c = max|ω_i - ω_j| / λ₂(L) where L is built from the prior's
        decay_alpha on a chain graph of n_layers.
        """
        from scpn_phase_orchestrator.coupling.spectral import critical_coupling

        prior = self.default()
        idx = np.arange(n_layers)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        knm = prior.K_base * np.exp(-prior.decay_alpha * dist)
        np.fill_diagonal(knm, 0.0)
        K_c = critical_coupling(omegas, knm)
        return CouplingPrior(
            K_base=prior.K_base,
            decay_alpha=prior.decay_alpha,
            K_c_estimate=K_c,
        )

    def log_probability(self, K_base: float, decay_alpha: float) -> float:
        """Log-probability under the Gaussian prior (unnormalized)."""
        lp_K = -0.5 * ((K_base - self._K_base_mean) / self._K_base_std) ** 2
        lp_a = (
            -0.5
            * ((decay_alpha - self._decay_alpha_mean) / self._decay_alpha_std) ** 2
        )
        return lp_K + lp_a
