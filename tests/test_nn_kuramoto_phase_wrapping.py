# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable Kuramoto phase projection contracts

"""Exercise Kuramoto's real JAX direct, masked, scan and autodiff consumers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scpn_phase_orchestrator.nn.functional import (
    kuramoto_forward,
    kuramoto_forward_masked,
    kuramoto_rk4_step,
    kuramoto_rk4_step_masked,
    kuramoto_step,
    kuramoto_step_masked,
)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("method", ["euler", "rk4"])
@pytest.mark.parametrize("masked", [False, True])
def test_direct_and_scan_projection(x64: bool, method: str, masked: bool) -> None:
    """Publish the same canonical phases from actual direct and scan compilation."""
    dtype = np.float64 if x64 else np.float32
    period = dtype(2.0 * np.pi)
    interior = np.nextafter(period, dtype(0.0))
    with jax.enable_x64(x64):
        phases = jnp.array([0.0, -float(period), float(interior), 0.25])
        omega = jnp.array([-1e-15, 0.0, 0.0, 2.0])
        coupling = jnp.zeros((4, 4))
        mask = jnp.ones_like(coupling)
        if masked:
            step = (
                kuramoto_step_masked if method == "euler" else kuramoto_rk4_step_masked
            )
            result = jax.jit(step)(phases, omega, coupling, mask, 0.01)
            final, trajectory = kuramoto_forward_masked(
                phases, omega, coupling, mask, 0.01, 1, method=method
            )
        else:
            dense_step = kuramoto_step if method == "euler" else kuramoto_rk4_step
            result = jax.jit(dense_step)(phases, omega, coupling, 0.01)
            final, trajectory = kuramoto_forward(
                phases, omega, coupling, 0.01, 1, method=method
            )
        array = np.asarray(result)
        np.testing.assert_array_equal(final, result)
        np.testing.assert_array_equal(trajectory[0], result)
    assert array.dtype == dtype
    np.testing.assert_array_equal(array[:2], np.zeros(2))
    assert not np.any(np.signbit(array[:2]))
    assert array[2] == interior
    assert array[3] == pytest.approx(0.27, rel=0.0, abs=3e-8 if not x64 else 2e-16)
    assert np.all((array >= 0.0) & (array < period))


@pytest.mark.parametrize("method", ["euler", "rk4"])
@pytest.mark.parametrize("masked", [False, True])
def test_jit_jvp_and_gradient_away_from_cut(method: str, masked: bool) -> None:
    """Retain the analytical phase/frequency sensitivity through real JAX transforms."""
    with jax.enable_x64(True):
        phases = jnp.array([0.3, 0.6])
        omega = jnp.array([0.2, -0.1])
        zero = jnp.zeros((2, 2))
        mask = jnp.ones_like(zero)

        def evolve(frequency: jax.Array) -> jax.Array:
            """Return a public Kuramoto step for the differentiated frequency input."""
            if masked:
                step = (
                    kuramoto_step_masked
                    if method == "euler"
                    else kuramoto_rk4_step_masked
                )
                return step(phases, frequency, zero, mask, 0.01)
            dense_step = kuramoto_step if method == "euler" else kuramoto_rk4_step
            return dense_step(phases, frequency, zero, 0.01)

        _, tangent = jax.jvp(jax.jit(evolve), (omega,), (jnp.ones_like(omega),))
        gradient = jax.grad(lambda value: jnp.sum(evolve(value)))(omega)
        np.testing.assert_allclose(tangent, np.full(2, 0.01), rtol=0.0, atol=1e-16)
        np.testing.assert_allclose(gradient, np.full(2, 0.01), rtol=0.0, atol=1e-16)
