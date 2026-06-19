# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy directed adaptive coupling

"""Transfer-entropy-guided coupling adaptation for offline matrix updates.

``te_adapt_coupling`` derives a directed transfer-entropy matrix from phase
history and combines it with the current coupling matrix under learning-rate
and decay parameters. The Python fallback clamps the returned coupling to
non-negative values and clears self-coupling; the optional Rust path preserves
the same dense ``N x N`` output contract. The helper returns a new matrix and
does not mutate live solver state or apply actuation.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.transfer_entropy import (
    transfer_entropy_matrix,
)

try:
    from spo_kernel import (
        te_adapt_coupling_rust as _rust_te_adapt,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["te_adapt_coupling"]
FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, bool) for item in raw.ravel())


def _as_finite_real_array(value: object, *, name: str) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be finite and real-valued")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and real-valued") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_knm(value: object) -> FloatArray:
    knm = _as_finite_real_array(value, name="knm")
    if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
        raise ValueError("knm must be a finite square coupling matrix")
    if np.any(knm < 0.0):
        raise ValueError("knm must be non-negative")
    if not np.allclose(np.diag(knm), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("knm diagonal must be zero")
    return knm


def _validate_phase_history(value: object, *, n: int) -> FloatArray:
    history = _as_finite_real_array(value, name="phase_history")
    if history.ndim != 2:
        raise ValueError("phase_history must be a finite 2-D phase matrix")
    if history.shape[0] != n:
        raise ValueError("phase_history oscillator count must match knm")
    return history


def _validate_non_negative_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real")
    return parsed


def _validate_decay(value: object) -> float:
    parsed = _validate_non_negative_real(value, name="decay")
    if parsed > 1.0:
        raise ValueError("decay must be in [0, 1]")
    return parsed


def _validate_n_bins(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("n_bins must be an integer >= 2")
    parsed = int(value)
    if parsed < 2:
        raise ValueError("n_bins must be an integer >= 2")
    return parsed


def _validate_transfer_entropy_scores(value: object, *, n: int) -> FloatArray:
    scores = _as_finite_real_array(value, name="transfer_entropy")
    if scores.shape != (n, n):
        raise RuntimeError("transfer-entropy backend returned wrong shape")
    if np.any(scores < -1e-12):
        raise RuntimeError("transfer-entropy backend returned negative scores")
    if not np.allclose(np.diag(scores), 0.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("transfer-entropy backend returned non-zero self scores")
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)
    return scores


def _validate_adapted_coupling(value: object, *, n: int) -> FloatArray:
    result = _as_finite_real_array(value, name="adapted coupling")
    if result.shape != (n, n):
        raise RuntimeError("TE adaptive backend returned wrong shape")
    if np.any(result < -1e-12):
        raise RuntimeError("TE adaptive backend returned negative coupling")
    if not np.allclose(np.diag(result), 0.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("TE adaptive backend returned non-zero self coupling")
    result = np.maximum(result, 0.0)
    np.fill_diagonal(result, 0.0)
    return result


def te_adapt_coupling(
    knm: FloatArray,
    phase_history: FloatArray,
    lr: float = 0.01,
    decay: float = 0.0,
    n_bins: int = 8,
) -> FloatArray:
    """Adapt coupling matrix using transfer entropy as learning signal.

    K_ij(t+1) = (1-decay) * K_ij(t) + lr * TE(i→j)

    Strengthens coupling along causal information flow channels.
    Weakens where there is no causal influence.

    Lizier 2012, "Local Information Transfer as a Spatiotemporal Filter
    for Complex Systems," Physical Review E 77(2):026110.

    Parameters
    ----------
    knm : FloatArray
        current (n, n) coupling matrix.
    phase_history : FloatArray
        (n, T) recent phase trajectories.
    lr : float
        learning rate for TE-based update.
    decay : float
        coupling decay rate per update (0 = no decay).
    n_bins : int
        histogram bins for TE estimation.

    Returns
    -------
    FloatArray
        FloatArray The coupling matrix adapted by the transfer-entropy learning signal.

    Raises
    ------
    RuntimeError
        If the transfer-entropy backend fails.
    """
    knm = _validate_knm(knm)
    n = knm.shape[0]
    phase_history = _validate_phase_history(phase_history, n=n)
    lr = _validate_non_negative_real(lr, name="lr")
    decay = _validate_decay(decay)
    n_bins = _validate_n_bins(n_bins)
    te = _validate_transfer_entropy_scores(
        transfer_entropy_matrix(phase_history, n_bins=n_bins),
        n=n,
    )
    if _HAS_RUST:
        k_flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        t_flat = np.ascontiguousarray(te.ravel(), dtype=np.float64)
        result_flat = np.asarray(
            _rust_te_adapt(k_flat, t_flat, n, lr, decay),
            dtype=np.float64,
        )
        if result_flat.size != n * n:
            raise RuntimeError("TE adaptive backend returned wrong shape")
        return _validate_adapted_coupling(result_flat.reshape(n, n), n=n)
    knm_new = (1.0 - decay) * knm + lr * te
    np.fill_diagonal(knm_new, 0.0)
    result: FloatArray = np.maximum(knm_new, 0.0)
    return result
