# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST

__all__ = ["CouplingState", "CouplingBuilder"]


@dataclass
class CouplingState:
    knm: NDArray
    alpha: NDArray
    active_template: str


class CouplingBuilder:
    """Builds Knm coupling matrices with exponential distance decay."""

    def build(
        self, n_layers: int, base_strength: float, decay_alpha: float
    ) -> CouplingState:
        """K_ij = base_strength * exp(-decay_alpha * |i - j|), zero diagonal."""
        if _HAS_RUST:
            from spo_kernel import PyCouplingBuilder

            d = PyCouplingBuilder().build(n_layers, base_strength, decay_alpha)
            n = d["n"]
            knm = np.asarray(d["knm"], dtype=np.float64).reshape(n, n)
            alpha = np.asarray(d["alpha"], dtype=np.float64).reshape(n, n)
            return CouplingState(knm=knm, alpha=alpha, active_template="default")
        idx = np.arange(n_layers)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        knm = base_strength * np.exp(-decay_alpha * dist)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n_layers, n_layers), dtype=np.float64)
        return CouplingState(knm=knm, alpha=alpha, active_template="default")

    def switch_template(
        self, state: CouplingState, template_name: str, templates: dict[str, NDArray]
    ) -> CouplingState:
        if template_name not in templates:
            raise KeyError(f"Template {template_name!r} not found")
        return CouplingState(
            knm=templates[template_name].copy(),
            alpha=state.alpha.copy(),
            active_template=template_name,
        )
