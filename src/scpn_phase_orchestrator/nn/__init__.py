# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable neural network module

"""Differentiable Kuramoto dynamics for neural network integration.

Functional API (jax only):
    kuramoto_step, kuramoto_rk4_step, kuramoto_forward,
    order_parameter, plv

Layer API (jax + equinox):
    KuramotoLayer — equinox.Module with learnable K and omegas

Requires: jax>=0.4 for functional API, equinox>=0.11 for layer API.
"""

from __future__ import annotations

from .functional import (
    kuramoto_forward,
    kuramoto_rk4_step,
    kuramoto_step,
    order_parameter,
    plv,
)

__all__ = [
    "kuramoto_step",
    "kuramoto_rk4_step",
    "kuramoto_forward",
    "order_parameter",
    "plv",
    "KuramotoLayer",
]

try:
    from .kuramoto_layer import KuramotoLayer
except ImportError:
    pass
