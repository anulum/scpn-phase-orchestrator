# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling matrix builder

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST

__all__ = ["CouplingState", "CouplingBuilder"]


@dataclass(frozen=True)
class CouplingState:
    knm: NDArray
    alpha: NDArray
    active_template: str
    knm_r: NDArray | None = None


class CouplingBuilder:
    """Builds Knm coupling matrices with exponential distance decay."""

    def build(
        self, n_layers: int, base_strength: float, decay_alpha: float
    ) -> CouplingState:
        """K_ij = base_strength * exp(-decay_alpha * |i - j|), zero diagonal."""
        if _HAS_RUST:  # pragma: no cover
            from spo_kernel import PyCouplingBuilder

            d = PyCouplingBuilder().build(n_layers, base_strength, decay_alpha)
            n = d["n"]
            rust_knm = np.asarray(d["knm"], dtype=np.float64).reshape(n, n)
            rust_alpha = np.asarray(d["alpha"], dtype=np.float64).reshape(n, n)
            return CouplingState(
                knm=rust_knm, alpha=rust_alpha, active_template="default"
            )
        idx = np.arange(n_layers)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        knm = base_strength * np.exp(-decay_alpha * dist)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n_layers, n_layers), dtype=np.float64)
        return CouplingState(knm=knm, alpha=alpha, active_template="default")

    def build_with_amplitude(
        self,
        n_layers: int,
        base_strength: float,
        decay_alpha: float,
        amp_strength: float,
        amp_decay: float,
    ) -> CouplingState:
        """Build phase + amplitude coupling matrices together."""
        phase = self.build(n_layers, base_strength, decay_alpha)
        idx = np.arange(n_layers)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        knm_r = amp_strength * np.exp(-amp_decay * dist)
        np.fill_diagonal(knm_r, 0.0)
        return CouplingState(
            knm=phase.knm,
            alpha=phase.alpha,
            active_template=phase.active_template,
            knm_r=knm_r,
        )

    def switch_template(
        self,
        state: CouplingState,
        template_name: str,
        templates: dict[str, NDArray],
    ) -> CouplingState:
        if template_name not in templates:
            raise KeyError(f"Template {template_name!r} not found")
        return CouplingState(
            knm=templates[template_name].copy(),
            alpha=state.alpha.copy(),
            active_template=template_name,
            knm_r=state.knm_r,
        )
