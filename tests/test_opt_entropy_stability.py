# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability tests for OPT-entropy

"""Stability / mathematical invariants for ``monitor/opt_entropy.py``.
Marked ``pytest.mark.slow``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor.opt_entropy import (
    ordinal_pattern_sequence,
    transition_entropy,
)

pytestmark = pytest.mark.slow


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    dimension=st.integers(min_value=2, max_value=6),
)
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_transition_entropy_bounded(seed: int, dimension: int) -> None:
    """Transition entropy stays in ``[0, 1]`` across random series."""
    rng = np.random.default_rng(seed)
    for _ in range(40):
        n = int(rng.integers(30, 400))
        series = rng.standard_normal(n)
        value = transition_entropy(series, dimension, 1)
        assert 0.0 <= value <= 1.0 + 1e-12


def test_constant_series_zero_entropy() -> None:
    assert transition_entropy(np.full(256, 3.14), 3, 1) == pytest.approx(0.0, abs=1e-12)


def test_white_noise_high_entropy() -> None:
    """White noise visits ordinal transitions near-uniformly → entropy ≈ 1."""
    rng = np.random.default_rng(1)
    value = transition_entropy(rng.standard_normal(20000), 3, 1)
    assert value > 0.95


def test_periodic_signal_low_entropy() -> None:
    """A clean periodic signal cycles through few transitions → low entropy."""
    t = np.linspace(0.0, 400.0 * np.pi, 20000)
    value = transition_entropy(np.sin(t), 3, 1)
    assert value < 0.8


def test_regularisation_lowers_entropy() -> None:
    """A regularised (mostly periodic) signal has lower transition entropy
    than the disordered (pure-noise) signal it came from."""
    rng = np.random.default_rng(4)
    t = np.linspace(0.0, 200.0 * np.pi, 8000)
    base_noise = rng.standard_normal(8000)
    periodic = np.sin(t)
    disordered = transition_entropy(base_noise, 3, 1)
    regularised = transition_entropy(0.1 * base_noise + 3.0 * periodic, 3, 1)
    assert regularised < disordered


def test_amplitude_scale_invariance() -> None:
    """Ordinal patterns are invariant under positive affine rescaling."""
    rng = np.random.default_rng(9)
    series = rng.standard_normal(2000)
    base = transition_entropy(series, 3, 1)
    scaled = transition_entropy(5.0 * series + 11.0, 3, 1)
    assert base == pytest.approx(scaled, abs=1e-12)
    np.testing.assert_array_equal(
        ordinal_pattern_sequence(series, 3, 1),
        ordinal_pattern_sequence(5.0 * series + 11.0, 3, 1),
    )


def test_time_reversal_preserves_bounds() -> None:
    rng = np.random.default_rng(13)
    series = rng.standard_normal(3000)
    forward = transition_entropy(series, 4, 1)
    backward = transition_entropy(series[::-1].copy(), 4, 1)
    assert 0.0 <= forward <= 1.0
    assert 0.0 <= backward <= 1.0


@pytest.mark.parametrize("delay", [1, 2, 3, 5])
def test_delay_variation_bounded(delay: int) -> None:
    rng = np.random.default_rng(20 + delay)
    series = rng.standard_normal(4000)
    value = transition_entropy(series, 3, delay)
    assert 0.0 <= value <= 1.0
