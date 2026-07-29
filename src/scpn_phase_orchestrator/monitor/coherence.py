# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coherence monitor

"""Coherence partition monitoring utilities for layer-bound phase states.

This module computes in-group and out-group Kuramoto-style locking metrics,
including R_good/R_bad summaries and PLV-based lock detection. Configuration
objects validate layer indices, threshold intervals, CLA terms, and denominator
semantics before analysis; invalid layer references fail early instead of being
silently clipped into a different biological partition.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.metrics import LayerState, LockSignature, UPDEState

__all__ = ["CoherenceMonitor"]

FloatArray = NDArray[np.float64]


def _validate_layer_indices(values: object, *, name: str) -> list[int]:
    """Return the validated layer indices, else raise ``ValueError``."""
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list of layer indices")
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(
                f"{name} must contain non-negative integer indices, got {value!r}"
            )
        indices.append(int(value))
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicates")
    return indices


def _validate_plv_threshold(value: object) -> float:
    """Return the PLV threshold as a validated value in [0, 1], else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"threshold must be finite real in [0, 1], got {value!r}")
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be finite real in [0, 1], got {value!r}")
    return threshold


def _validate_cross_layer_alignment(value: object, *, n_layers: int) -> FloatArray:
    """Return the validated cross-layer alignment input, else raise."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cross_layer_alignment must be a finite real matrix") from exc
    if raw.dtype.kind == "O":
        if any(
            isinstance(item, (bool, np.bool_, complex, str, bytes, np.str_, np.bytes_))
            or not isinstance(item, Real)
            for item in raw.flat
        ):
            raise ValueError("cross_layer_alignment must be a finite real matrix")
    elif raw.dtype.kind not in "iuf":
        raise ValueError("cross_layer_alignment must be a finite real matrix")
    cla = raw.astype(np.float64, copy=True)

    expected_shape = (n_layers, n_layers)
    if cla.shape != expected_shape:
        raise ValueError(
            f"cross_layer_alignment shape {cla.shape} does not match {expected_shape}"
        )
    if not np.all(np.isfinite(cla)):
        raise ValueError("cross_layer_alignment must contain only finite values")
    if np.any(cla < 0.0) or np.any(cla > 1.0):
        raise ValueError("cross_layer_alignment must contain values in [0, 1]")
    if not np.allclose(cla, cla.T, rtol=0.0, atol=1e-12):
        raise ValueError("cross_layer_alignment must be symmetric")
    return np.ascontiguousarray(cla, dtype=np.float64)


def _validate_upde_state(value: object) -> UPDEState:
    """Return a structurally valid diagnostic state."""
    if not isinstance(value, UPDEState):
        raise TypeError(f"upde_state must be UPDEState, got {value!r}")
    if not isinstance(value.layers, list) or any(
        not isinstance(layer, LayerState) for layer in value.layers
    ):
        raise ValueError("upde_state layers must be a list of LayerState values")
    return value


def _validate_lock_signature(value: object, *, source: int, target: int) -> float:
    """Return a validated fallback PLV with exact source-target provenance."""
    if not isinstance(value, LockSignature):
        raise ValueError(f"lock signature {source}_{target} must be LockSignature")
    if (
        isinstance(value.source_layer, bool)
        or not isinstance(value.source_layer, Integral)
        or isinstance(value.target_layer, bool)
        or not isinstance(value.target_layer, Integral)
        or int(value.source_layer) != source
        or int(value.target_layer) != target
    ):
        raise ValueError(f"lock signature {source}_{target} has invalid provenance")
    try:
        plv = _validate_plv_threshold(value.plv)
    except ValueError as exc:
        raise ValueError(f"lock signature {source}_{target} has invalid PLV") from exc
    if isinstance(value.mean_lag, bool) or not isinstance(value.mean_lag, Real):
        raise ValueError(f"lock signature {source}_{target} has invalid mean lag")
    mean_lag = float(value.mean_lag)
    if not np.isfinite(mean_lag):
        raise ValueError(f"lock signature {source}_{target} has invalid mean lag")
    return plv


def _validate_order_parameter(value: object, *, layer_index: int) -> float:
    """Return the order parameter as validated values in [0, 1], else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"layer {layer_index} R must be finite real in [0, 1]")
    r_value = float(value)
    if not np.isfinite(r_value) or r_value < 0.0 or r_value > 1.0:
        raise ValueError(f"layer {layer_index} R must be finite real in [0, 1]")
    return r_value


class CoherenceMonitor:
    """Track coherence partitioned into good vs bad layer subsets."""

    def __init__(self, good_layers: list[int], bad_layers: list[int]):
        self._good = _validate_layer_indices(good_layers, name="good_layers")
        self._bad = _validate_layer_indices(bad_layers, name="bad_layers")
        if set(self._good) & set(self._bad):
            raise ValueError("good_layers and bad_layers must be disjoint")

    def compute_r_good(self, upde_state: UPDEState) -> float:
        """Mean order parameter R across good (synchronise) layers.

        Parameters
        ----------
        upde_state : UPDEState
            The UPDE state to evaluate.

        Returns
        -------
        float
            The mean order parameter ``R`` over the maintain (good) layers.
        """
        return float(self._mean_r(upde_state, self._good, name="good_layers"))

    def compute_r_bad(self, upde_state: UPDEState) -> float:
        """Mean order parameter R across bad (desynchronise) layers.

        Parameters
        ----------
        upde_state : UPDEState
            The UPDE state to evaluate.

        Returns
        -------
        float
            The mean order parameter ``R`` over the suppress (bad) layers.
        """
        return float(self._mean_r(upde_state, self._bad, name="bad_layers"))

    # PLV lock threshold: Lachaux et al. 1999; see docs/ASSUMPTIONS.md § Quality Gating
    def detect_phase_lock(
        self, upde_state: UPDEState, threshold: float = 0.9
    ) -> list[tuple[int, int]]:
        """Return pairs of layer indices whose PLV exceeds threshold.

        Uses cross_layer_alignment matrix as the primary PLV source
        (matches Rust implementation). Falls back to lock_signatures
        if CLA entry is below threshold but a signature overrides it.

        Parameters
        ----------
        upde_state : UPDEState
            The UPDE state to evaluate.
        threshold : float
            Decision threshold.

        Returns
        -------
        list[tuple[int, int]]
            The layer-index pairs whose PLV exceeds the threshold.

        Raises
        ------
        TypeError
            If ``upde_state`` is not a diagnostic state.
        ValueError
            If state structure, alignment evidence, threshold, or a consulted
            fallback lock signature violates its public contract.
        """
        upde_state = _validate_upde_state(upde_state)
        threshold = _validate_plv_threshold(threshold)
        n = len(upde_state.layers)
        cla = _validate_cross_layer_alignment(
            upde_state.cross_layer_alignment, n_layers=n
        )
        locked = []
        for i in range(n):
            for j in range(i + 1, n):
                # Primary: use CLA matrix (always populated from phase data)
                if cla[i, j] >= threshold:
                    locked.append((i, j))
                    continue
                # Fallback: explicit lock_signatures (manually set)
                key = f"{i}_{j}"
                signatures = upde_state.layers[i].lock_signatures
                if not isinstance(signatures, dict):
                    raise ValueError(f"layer {i} lock signatures must be a dictionary")
                sig = signatures.get(key)
                if (
                    sig is not None
                    and _validate_lock_signature(sig, source=i, target=j) >= threshold
                ):
                    locked.append((i, j))
        return locked

    def _mean_r(self, upde_state: UPDEState, indices: list[int], *, name: str) -> float:
        """Return the mean order parameter (R) over the values."""
        upde_state = _validate_upde_state(upde_state)
        n_layers = len(upde_state.layers)
        invalid = [index for index in indices if index >= n_layers]
        if invalid:
            raise ValueError(
                f"{name} references layer indices outside state with "
                f"{n_layers} layers: {invalid!r}"
            )
        vals = [
            _validate_order_parameter(upde_state.layers[i].R, layer_index=i)
            for i in indices
        ]
        if not vals:
            return 0.0
        return float(np.mean(vals))
