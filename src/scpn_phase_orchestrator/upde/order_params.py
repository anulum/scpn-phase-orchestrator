# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order parameter computation

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST
from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["compute_order_parameter", "compute_plv", "compute_layer_coherence"]


def compute_order_parameter(phases: NDArray) -> tuple[float, float]:
    """Kuramoto global order parameter (R, psi_mean).

    R = |mean(exp(i * theta))|, psi_mean = arg(mean(exp(i * theta))).
    """
    if phases.size == 0:
        return (0.0, 0.0)
    if _HAS_RUST:  # pragma: no cover
        from spo_kernel import order_parameter as _rust_order_param

        r, psi = _rust_order_param(np.ascontiguousarray(phases.ravel()))
        return float(r), float(psi)
    z = np.mean(np.exp(1j * phases))
    return float(np.abs(z)), float(np.angle(z) % TWO_PI)


def compute_plv(phases_a: NDArray, phases_b: NDArray) -> float:
    """Phase-locking value between two phase time series.

    PLV = |mean(exp(i * (phi_a - phi_b)))| over samples.

    Raises:
        ValueError: If arrays have different lengths.
    """
    if phases_a.size == 0 or phases_b.size == 0:
        return 0.0
    if phases_a.size != phases_b.size:
        raise ValueError(
            f"PLV requires equal-length arrays, got {phases_a.size} vs {phases_b.size}"
        )
    if _HAS_RUST:  # pragma: no cover
        from spo_kernel import plv as _rust_plv

        return float(
            _rust_plv(
                np.ascontiguousarray(phases_a.ravel()),
                np.ascontiguousarray(phases_b.ravel()),
            )
        )
    return float(np.abs(np.mean(np.exp(1j * (phases_a - phases_b)))))


def compute_layer_coherence(phases: NDArray, layer_mask: NDArray) -> float:
    """Order parameter R for the subset of oscillators selected by layer_mask.

    Args:
        phases: 1-D array of all oscillator phases.
        layer_mask: boolean mask selecting oscillators in the layer.
    """
    sub = phases[layer_mask]
    if sub.size == 0:
        return 0.0
    z = np.mean(np.exp(1j * sub))
    return float(np.abs(z))
