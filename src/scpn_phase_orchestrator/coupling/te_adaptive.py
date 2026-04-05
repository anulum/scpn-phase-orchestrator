# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy directed adaptive coupling

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.transfer_entropy import (
    transfer_entropy_matrix,
)

try:
    from spo_kernel import (  # type: ignore[import-untyped]
        te_adapt_coupling_rust as _rust_te_adapt,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["te_adapt_coupling"]


def te_adapt_coupling(
    knm: NDArray,
    phase_history: NDArray,
    lr: float = 0.01,
    decay: float = 0.0,
    n_bins: int = 8,
) -> NDArray:
    """Adapt coupling matrix using transfer entropy as learning signal.

    K_ij(t+1) = (1-decay) * K_ij(t) + lr * TE(i→j)

    Strengthens coupling along causal information flow channels.
    Weakens where there is no causal influence.

    Lizier 2012, "Local Information Transfer as a Spatiotemporal Filter
    for Complex Systems," Physical Review E 77(2):026110.

    Args:
        knm: current (n, n) coupling matrix.
        phase_history: (n, T) recent phase trajectories.
        lr: learning rate for TE-based update.
        decay: coupling decay rate per update (0 = no decay).
        n_bins: histogram bins for TE estimation.
    """
    te = transfer_entropy_matrix(phase_history, n_bins=n_bins)
    if _HAS_RUST:
        n = knm.shape[0]
        k_flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        t_flat = np.ascontiguousarray(te.ravel(), dtype=np.float64)
        result_flat = _rust_te_adapt(k_flat, t_flat, n, lr, decay)
        return result_flat.reshape(n, n)
    knm_new = (1.0 - decay) * knm + lr * te
    np.fill_diagonal(knm_new, 0.0)
    result: NDArray = np.maximum(knm_new, 0.0)
    return result
