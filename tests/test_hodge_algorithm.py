# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for Hodge decomposition

"""Algorithmic properties of :func:`hodge_decomposition`.

Covered: symmetric K has zero curl; antisymmetric K has zero
gradient; gradient + curl + harmonic ≈ total coupling force;
harmonic is float-noise (~1e-15) for any K; empty input; Hypothesis
invariants.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling import hodge as h_mod
from scpn_phase_orchestrator.coupling.hodge import hodge_decomposition

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = h_mod.ACTIVE_BACKEND
        h_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            h_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestHodge:
    @_python
    def test_symmetric_k_has_zero_curl(self):
        rng = np.random.default_rng(0)
        n = 10
        k = rng.normal(0, 1, (n, n))
        k = 0.5 * (k + k.T)  # force symmetric
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)

    @_python
    def test_antisymmetric_k_has_zero_gradient(self):
        rng = np.random.default_rng(1)
        n = 10
        k = rng.normal(0, 1, (n, n))
        k = 0.5 * (k - k.T)  # force antisymmetric
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        np.testing.assert_allclose(res.gradient, 0.0, atol=1e-12)

    @_python
    def test_components_reconstruct_total(self):
        """``gradient + curl + harmonic = Σ_j K_ij cos(θ_j − θ_i)``."""
        rng = np.random.default_rng(2)
        n = 8
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        diff = phases[np.newaxis, :] - phases[:, np.newaxis]
        total = np.sum(k * np.cos(diff), axis=1)
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic,
            total,
            atol=1e-12,
        )

    @_python
    def test_harmonic_is_float_noise(self):
        """For a sym+anti decomposition of any real ``K``, the
        residual is bounded by float epsilon × matrix norm."""
        rng = np.random.default_rng(3)
        n = 12
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        norm = float(np.max(np.abs(k)))
        assert np.all(np.abs(res.harmonic) < 1e-12 * norm * n)

    @_python
    def test_empty_input(self):
        res = hodge_decomposition(np.zeros((0, 0)), np.array([]))
        assert res.gradient.size == 0
        assert res.curl.size == 0
        assert res.harmonic.size == 0


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=2, max_value=20),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_output_shape_and_finite(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        assert res.gradient.shape == (n,)
        assert res.curl.shape == (n,)
        assert res.harmonic.shape == (n,)
        assert np.all(np.isfinite(res.gradient))
        assert np.all(np.isfinite(res.curl))
        assert np.all(np.isfinite(res.harmonic))


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert h_mod.AVAILABLE_BACKENDS
        assert "python" in h_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert h_mod.AVAILABLE_BACKENDS[0] == h_mod.ACTIVE_BACKEND
