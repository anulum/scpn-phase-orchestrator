# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability invariants for coupling.spectral

"""Long-run / structural invariants for ``coupling.spectral``.

The eigendecomposition isn't iterative — "stability" here means
scale-up behaviour + structural properties like monotonicity of
``λ₂`` when edges are added.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling import spectral as s_mod
from scpn_phase_orchestrator.coupling.spectral import (
    fiedler_value,
    graph_laplacian,
    spectral_eig,
)

pytestmark = pytest.mark.slow


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = s_mod.ACTIVE_BACKEND
        s_mod.ACTIVE_BACKEND = "python"
        s_mod._PRIM_CACHE = None
        try:
            return func(*args, **kwargs)
        finally:
            s_mod.ACTIVE_BACKEND = prev
            s_mod._PRIM_CACHE = None

    return wrapper


class TestScaleUp:
    @_python
    def test_handles_n_50(self):
        rng = np.random.default_rng(0)
        n = 50
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        eigvals, fiedler = spectral_eig(W)
        assert eigvals.shape == (n,)
        assert fiedler.shape == (n,)
        # Residual check: L v₂ ≈ λ₂ v₂.
        L = graph_laplacian(W)
        lam2 = eigvals[1]
        residual = L @ fiedler - lam2 * fiedler
        assert np.max(np.abs(residual)) < 1e-10


class TestMonotonicity:
    @_python
    def test_lambda2_grows_with_added_edges(self):
        """Adding a positive-weight edge to a connected graph
        cannot decrease ``λ₂`` — this is the standard graph-
        Laplacian interlacing property (Fiedler 1973)."""
        rng = np.random.default_rng(1)
        n = 6
        W = rng.uniform(0.1, 0.5, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        lam2_before = fiedler_value(W)
        # Add weight to a single edge pair.
        W2 = W.copy()
        W2[0, 1] += 2.0
        W2[1, 0] += 2.0
        lam2_after = fiedler_value(W2)
        assert lam2_after >= lam2_before - 1e-12


class TestDisconnectedDetection:
    @_python
    def test_two_components_lambda2_is_zero(self):
        n = 6
        W = np.zeros((n, n))
        # Component {0,1,2}
        W[:3, :3] = 1.0
        np.fill_diagonal(W[:3, :3], 0.0)
        # Component {3,4,5}
        W[3:, 3:] = 1.0
        np.fill_diagonal(W[3:, 3:], 0.0)
        assert fiedler_value(W) < 1e-10


class TestEigvalueSpectralBound:
    @_python
    def test_all_eigvals_leq_2_max_degree(self):
        """For any graph Laplacian, ``λ_max(L) ≤ 2 · d_max``
        (Anderson-Morley 1985). Useful sanity bound."""
        rng = np.random.default_rng(2)
        n = 8
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        d_max = float(np.max(np.sum(W, axis=1)))
        eigvals, _ = spectral_eig(W)
        assert eigvals[-1] <= 2.0 * d_max + 1e-10
