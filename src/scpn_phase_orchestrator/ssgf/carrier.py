# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SSGF geometry carrier W(t)

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        carrier_decode_rust as _rust_decode,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["GeometryCarrier", "SSGFState"]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class SSGFState:
    z: FloatArray
    W: FloatArray
    cost: float
    grad_norm: float
    step: int


class GeometryCarrier:
    """SSGF outer cycle: z → W → microcycles → cost → grad → z.

    The latent vector z parameterizes a coupling geometry W(z) via a
    spectral decoder. The microcycle (Kuramoto integration) runs under
    W, produces a cost U_total, and the gradient ∂U/∂z drives z toward
    lower cost.

    This implements the autopoietic loop: geometry → dynamics → cost →
    gradient → geometry. The system produces its own coupling topology.
    """

    def __init__(
        self,
        n_oscillators: int,
        z_dim: int = 8,
        lr: float = 0.01,
        seed: int | None = None,
    ):
        self._n = n_oscillators
        self._z_dim = z_dim
        self._lr = lr
        self._rng = np.random.default_rng(seed)
        self._z: FloatArray = self._rng.normal(0, 0.1, z_dim)
        # Decoder: W = softplus(A @ z) reshaped to (n, n), zero diagonal
        # A is a fixed random projection (n*n × z_dim)
        self._A: FloatArray = self._rng.normal(
            0, 1.0 / np.sqrt(z_dim), (n_oscillators * n_oscillators, z_dim)
        )
        self._step = 0

    @property
    def z(self) -> FloatArray:
        return self._z.copy()

    @property
    def z_dim(self) -> int:
        return self._z_dim

    def decode(self, z: FloatArray | None = None) -> FloatArray:
        """Map z → coupling matrix W (n × n, non-negative, zero diagonal)."""
        if z is None:
            z = self._z
        if _HAS_RUST:
            zv = np.ascontiguousarray(z, dtype=np.float64)
            av = np.ascontiguousarray(self._A.ravel(), dtype=np.float64)
            w_flat: FloatArray = np.asarray(_rust_decode(zv, av, self._n))
            return w_flat.reshape(self._n, self._n)
        raw = self._A @ z
        # Softplus ensures non-negative coupling
        W = np.log1p(np.exp(raw)).reshape(self._n, self._n)
        np.fill_diagonal(W, 0.0)
        out: FloatArray = W
        return out

    def update(
        self,
        cost: float,
        cost_fn: Callable[[FloatArray], float] | None = None,
        epsilon: float = 1e-4,
    ) -> SSGFState:
        """One SSGF outer step: compute gradient of cost w.r.t. z, descend.

        If cost_fn is provided, uses finite differences on z.
        Otherwise records cost for external gradient computation.
        """
        self._step += 1
        grad: FloatArray = np.zeros(self._z_dim)

        if cost_fn is not None:
            for i in range(self._z_dim):
                z_plus = self._z.copy()
                z_plus[i] += epsilon
                z_minus = self._z.copy()
                z_minus[i] -= epsilon
                grad[i] = (
                    cost_fn(self.decode(z_plus)) - cost_fn(self.decode(z_minus))
                ) / (2 * epsilon)

        self._z -= self._lr * grad
        W = self.decode()
        return SSGFState(
            z=self._z.copy(),
            W=W,
            cost=cost,
            grad_norm=float(np.linalg.norm(grad)),
            step=self._step,
        )

    def reset(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self._z = rng.normal(0, 0.1, self._z_dim)
        self._step = 0
