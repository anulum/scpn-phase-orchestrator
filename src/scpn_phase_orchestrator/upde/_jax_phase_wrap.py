# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Canonical differentiable JAX phase projection

"""Project JAX UPDE phases without crossing the host-array boundary."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def wrap_phases(phases: jax.Array) -> jax.Array:
    """Return half-open torus phases in the input floating-point precision.

    Parameters
    ----------
    phases : jax.Array
        Phase state in radians, including arrays traced by JIT or autodiff.

    Returns
    -------
    jax.Array
        Wrapped values with positive zero for rounded endpoints and signed
        zeros. Interior values and their gradients are unchanged. Nonfinite
        values remain nonfinite; this projection does not validate dynamics.
    """
    period = jnp.asarray(2.0 * math.pi, dtype=phases.dtype)
    wrapped = jnp.remainder(phases, period)
    return jnp.where(
        (wrapped >= period) | (wrapped == 0.0), jnp.zeros_like(wrapped), wrapped
    )
