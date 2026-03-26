# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable neural network module

"""Differentiable phase dynamics for neural network integration.

Requires: ``pip install scpn-phase-orchestrator[nn]`` (installs jax + equinox).

Functional API (jax only):
    kuramoto_step, kuramoto_rk4_step, kuramoto_forward,
    simplicial_step, simplicial_rk4_step, simplicial_forward,
    stuart_landau_step, stuart_landau_rk4_step, stuart_landau_forward,
    order_parameter, plv

Layer API (jax + equinox):
    KuramotoLayer — phase-only, learnable K and omegas
    StuartLandauLayer — phase + amplitude, learnable K, K_r, omegas, mu

All imports are lazy: ``import scpn_phase_orchestrator.nn`` succeeds without
JAX installed.  Symbols are resolved on first attribute access.
"""

from __future__ import annotations

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
    "coupling_laplacian",
    "saf_order_parameter",
    "saf_loss",
    "KuramotoLayer",
    "StuartLandauLayer",
    "UDEKuramotoLayer",
    "CouplingResidual",
    "infer_coupling",
    "inverse_loss",
    "coupling_correlation",
    "oim_step",
    "oim_forward",
    "extract_coloring",
    "coloring_violations",
    "coloring_energy",
    "bold_from_neural",
    "bold_signal",
    "reservoir_drive",
    "reservoir_features",
    "reservoir_predict",
    "ridge_readout",
]

_FUNCTIONAL = {
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
    "coupling_laplacian",
    "saf_order_parameter",
    "saf_loss",
}
_BOLD = {"bold_from_neural", "bold_signal"}
_INVERSE = {"infer_coupling", "inverse_loss", "coupling_correlation"}
_OIM = {
    "oim_step",
    "oim_forward",
    "extract_coloring",
    "coloring_violations",
    "coloring_energy",
}
_RESERVOIR = {
    "reservoir_drive",
    "reservoir_features",
    "reservoir_predict",
    "ridge_readout",
}
_LAYERS = {"KuramotoLayer", "StuartLandauLayer"}
_UDE = {"UDEKuramotoLayer", "CouplingResidual"}


def __getattr__(name: str) -> object:  # noqa: ANN204
    if name in _FUNCTIONAL:
        from . import functional

        return getattr(functional, name)
    if name in _BOLD:
        from . import bold

        return getattr(bold, name)
    if name in _INVERSE:
        from . import inverse

        return getattr(inverse, name)
    if name in _OIM:
        from . import oim

        return getattr(oim, name)
    if name in _RESERVOIR:
        from . import reservoir

        return getattr(reservoir, name)
    if name in _LAYERS:
        if name == "KuramotoLayer":
            from .kuramoto_layer import KuramotoLayer

            return KuramotoLayer
        from .stuart_landau_layer import StuartLandauLayer

        return StuartLandauLayer
    if name in _UDE:
        from .ude import CouplingResidual, UDEKuramotoLayer

        return UDEKuramotoLayer if name == "UDEKuramotoLayer" else CouplingResidual
    msg = f"module 'scpn_phase_orchestrator.nn' has no attribute {name!r}"
    raise AttributeError(msg)
