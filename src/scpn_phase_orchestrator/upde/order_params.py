# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import order_parameter as _rust_order_param
    from spo_kernel import plv as _rust_plv

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

TWO_PI = 2.0 * np.pi


def compute_order_parameter(phases: NDArray) -> tuple[float, float]:
    """Kuramoto global order parameter (R, psi_mean).

    R = |mean(exp(i * theta))|, psi_mean = arg(mean(exp(i * theta))).
    """
    if _HAS_RUST:
        return _rust_order_param(phases.ravel().tolist())
    z = np.mean(np.exp(1j * phases))
    return float(np.abs(z)), float(np.angle(z) % TWO_PI)


def compute_plv(phases_a: NDArray, phases_b: NDArray) -> float:
    """Phase-locking value between two phase time series.

    PLV = |mean(exp(i * (phi_a - phi_b)))| over samples.
    """
    if _HAS_RUST:
        return _rust_plv(phases_a.ravel().tolist(), phases_b.ravel().tolist())
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
