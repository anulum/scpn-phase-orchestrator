# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — nn integrator method contract

"""Prove every nn forward integrator refuses an unknown method name.

Each forward function chose ``rk4`` only for the exact string ``"rk4"`` and
explicit Euler for anything else, so ``"RK4"`` or an unsupported ``"rk45"``
silently changed the integrator.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")

from scpn_phase_orchestrator.nn.functional import (  # noqa: E402
    kuramoto_forward,
    kuramoto_forward_masked,
    simplicial_forward,
    stuart_landau_forward,
    winfree_forward,
)
from scpn_phase_orchestrator.nn.theta_neuron import theta_neuron_forward  # noqa: E402

N = 3
PHASES = jnp.array([0.0, 1.0, 2.0])
OMEGAS = jnp.array([0.1, 0.2, 0.3])
K = jnp.full((N, N), 0.1) - jnp.eye(N) * 0.1
# Built with the phases' dtype at import: a later test in the session may turn
# on jax_enable_x64, and a mask created at call time would then be float64
# while the phases stay float32, which scan rejects as a carry type change.
MASK = jnp.ones((N, N), dtype=PHASES.dtype)

RUNNERS: dict[str, Callable[[str], object]] = {
    "kuramoto": lambda m: kuramoto_forward(PHASES, OMEGAS, K, 0.01, 2, method=m),
    "kuramoto_masked": lambda m: kuramoto_forward_masked(
        PHASES, OMEGAS, K, MASK, 0.01, 2, method=m
    ),
    "winfree": lambda m: winfree_forward(PHASES, OMEGAS, 0.1, 0.01, 2, method=m),
    "simplicial": lambda m: simplicial_forward(PHASES, OMEGAS, K, 0.01, 2, method=m),
    "stuart_landau": lambda m: stuart_landau_forward(
        PHASES,
        jnp.ones(N),
        OMEGAS,
        jnp.ones(N),
        K,
        K,
        0.01,
        2,
        method=m,
    ),
    "theta_neuron": lambda m: theta_neuron_forward(
        PHASES, OMEGAS, K, 0.01, 2, method=m
    ),
}


@pytest.mark.parametrize("name", sorted(RUNNERS))
@pytest.mark.parametrize("method", ["RK4", "rk45", "midpoint", ""])
def test_unknown_method_is_refused(name: str, method: str) -> None:
    with pytest.raises(ValueError, match="method must be 'rk4' or 'euler'"):
        RUNNERS[name](method)


@pytest.mark.parametrize("name", sorted(RUNNERS))
@pytest.mark.parametrize("method", ["rk4", "euler"])
def test_supported_methods_run(name: str, method: str) -> None:
    RUNNERS[name](method)
