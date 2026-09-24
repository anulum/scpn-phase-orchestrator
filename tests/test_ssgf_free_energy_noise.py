# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Langevin noise is fresh per call; NaN inputs refused

"""Langevin noise must differ between calls, and NaN must not become a weight.

With the Rust kernel and no generator, every call used seed 42 and returned
the same "noise"; a NaN temperature or step silently produced NaN; a NaN
energy or temperature made the NumPy Boltzmann path clamp to +700 and return
the largest possible weight while the Rust path returned NaN.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf.free_energy import (
    add_langevin_noise,
    boltzmann_weight,
)


def test_successive_calls_draw_fresh_noise() -> None:
    z = np.zeros(64)
    first = add_langevin_noise(z, 1.0, 0.1)
    second = add_langevin_noise(z, 1.0, 0.1)
    assert not np.array_equal(first, second)
    # the noise has the Langevin scale sqrt(2 T dt)
    assert np.std(np.concatenate([first, second])) == pytest.approx(
        math.sqrt(0.2), rel=0.35
    )


def test_seeded_generator_is_reproducible() -> None:
    z = np.zeros(8)
    a = add_langevin_noise(z, 1.0, 0.1, rng=np.random.default_rng(7))
    b = add_langevin_noise(z, 1.0, 0.1, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    ("temperature", "dt"), [(math.nan, 0.1), (1.0, math.nan), (math.inf, 0.1)]
)
def test_non_finite_langevin_parameters_are_refused(
    temperature: float, dt: float
) -> None:
    with pytest.raises(ValueError, match="must be a finite real number"):
        add_langevin_noise(np.zeros(4), temperature, dt)


@pytest.mark.parametrize(
    ("u", "t"), [(math.nan, 1.0), (1.0, math.nan), (math.inf, 1.0)]
)
def test_non_finite_boltzmann_inputs_are_refused(u: float, t: float) -> None:
    with pytest.raises(ValueError, match="must be a finite real number"):
        boltzmann_weight(u, t)


def test_finite_boltzmann_weights_are_unchanged() -> None:
    assert boltzmann_weight(0.0, 0.0) == 1.0
    assert boltzmann_weight(1.0, 0.0) == 0.0
    assert boltzmann_weight(2.0, 1.0) == pytest.approx(math.exp(-2.0))
