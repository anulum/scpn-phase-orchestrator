# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for envelope kernels

"""Algorithmic properties of :func:`extract_envelope` and
:func:`envelope_modulation_depth`.

Covered: constant-amplitude signal produces constant RMS; zero
window rejected; empty input safety; modulation depth on sinusoid
= 1 (in the limit of full-amplitude modulation); 2-D batched path;
Hypothesis invariants.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import envelope as e_mod
from scpn_phase_orchestrator.upde.envelope import (
    envelope_modulation_depth,
    extract_envelope,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _python(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run ``func`` with the pure-Python backend forced."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        prev = e_mod.ACTIVE_BACKEND
        e_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            e_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestExtractEnvelope:
    @_python
    def test_constant_amplitude(self) -> None:
        amps = np.full(100, 3.0)
        env = extract_envelope(amps, window=10)
        np.testing.assert_allclose(env, 3.0, atol=1e-12)

    @_python
    def test_reject_zero_window(self) -> None:
        with pytest.raises(ValueError, match="window"):
            extract_envelope(np.ones(50), window=0)

    @_python
    def test_empty_input(self) -> None:
        env = extract_envelope(np.array([]), window=5)
        assert env.size == 0

    @_python
    def test_output_shape_matches(self) -> None:
        rng = np.random.default_rng(0)
        amps = rng.normal(0, 1, 200)
        env = extract_envelope(amps, window=10)
        assert env.shape == amps.shape

    @_python
    def test_2d_batched_path(self) -> None:
        rng = np.random.default_rng(1)
        amps = rng.normal(0, 1, (100, 4))
        env = extract_envelope(amps, window=10)
        assert env.shape == amps.shape


class TestModulationDepth:
    @_python
    def test_constant_envelope_zero_depth(self) -> None:
        env = np.full(50, 2.0)
        assert envelope_modulation_depth(env) == 0.0

    @_python
    def test_empty_input_zero(self) -> None:
        assert envelope_modulation_depth(np.array([])) == 0.0

    @_python
    def test_unit_amplitude_sinusoid_full_modulation(self) -> None:
        """``|sin(t)|`` has ``min = 0, max = 1 → depth = 1``."""
        t = np.linspace(0, 2 * math.pi, 200)
        env = np.abs(np.sin(t))
        depth = envelope_modulation_depth(env)
        assert depth > 0.99

    @_python
    def test_bounded_unit_interval(self) -> None:
        rng = np.random.default_rng(3)
        env = np.abs(rng.normal(1, 0.3, 500))
        d = envelope_modulation_depth(env)
        assert 0.0 <= d <= 1.0


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=5, max_value=500),
        window=st.integers(min_value=1, max_value=20),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rms_non_negative(self, n: int, window: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        amps = rng.normal(0, 1, n)
        env = extract_envelope(amps, window=window)
        assert env.shape == amps.shape
        assert np.all(env >= 0.0 - 1e-12)


class TestDispatcherSurface:
    def test_available_non_empty(self) -> None:
        assert e_mod.AVAILABLE_BACKENDS
        assert "python" in e_mod.AVAILABLE_BACKENDS

    def test_active_is_available(self) -> None:
        assert e_mod.ACTIVE_BACKEND in e_mod.AVAILABLE_BACKENDS

    def test_available_backends_keep_canonical_fallback_order(self) -> None:
        canonical = ["rust", "mojo", "julia", "go", "python"]
        indices = [canonical.index(backend) for backend in e_mod.AVAILABLE_BACKENDS]
        assert indices == sorted(indices)
