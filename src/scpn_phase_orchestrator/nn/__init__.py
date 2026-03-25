# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable neural network module

"""Differentiable phase dynamics for neural network integration.

Functional API (jax only):
    kuramoto_step, kuramoto_rk4_step, kuramoto_forward,
    simplicial_step, simplicial_rk4_step, simplicial_forward,
    stuart_landau_step, stuart_landau_rk4_step, stuart_landau_forward,
    order_parameter, plv

Layer API (jax + equinox):
    KuramotoLayer — phase-only, learnable K and omegas
    StuartLandauLayer — phase + amplitude, learnable K, K_r, omegas, mu

Requires: jax>=0.4 for functional API, equinox>=0.11 for layer API.
"""

from __future__ import annotations

from .functional import (
    kuramoto_forward,
    kuramoto_rk4_step,
    kuramoto_step,
    order_parameter,
    plv,
    simplicial_forward,
    simplicial_rk4_step,
    simplicial_step,
    stuart_landau_forward,
    stuart_landau_rk4_step,
    stuart_landau_step,
)

__all__ = [
    "kuramoto_step",
    "kuramoto_rk4_step",
    "kuramoto_forward",
    "simplicial_step",
    "simplicial_rk4_step",
    "simplicial_forward",
    "stuart_landau_step",
    "stuart_landau_rk4_step",
    "stuart_landau_forward",
    "order_parameter",
    "plv",
    "KuramotoLayer",
    "StuartLandauLayer",
]

import contextlib

with contextlib.suppress(ImportError):
    from .kuramoto_layer import KuramotoLayer
    from .stuart_landau_layer import StuartLandauLayer
