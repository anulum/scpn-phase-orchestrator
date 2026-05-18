# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coherence monitor

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["CoherenceMonitor"]


def _validate_layer_indices(values: list[int], *, name: str) -> list[int]:
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(
                f"{name} must contain non-negative integer indices, got {value!r}"
            )
        indices.append(int(value))
    return indices


def _validate_plv_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"threshold must be finite real in [0, 1], got {value!r}")
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be finite real in [0, 1], got {value!r}")
    return threshold


def _validate_cross_layer_alignment(value: object, *, n_layers: int) -> np.ndarray:
    try:
        cla = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "cross_layer_alignment must be convertible to a finite float matrix"
        ) from exc

    expected_shape = (n_layers, n_layers)
    if cla.shape != expected_shape:
        raise ValueError(
            f"cross_layer_alignment shape {cla.shape} does not match {expected_shape}"
        )
    if not np.all(np.isfinite(cla)):
        raise ValueError("cross_layer_alignment must contain only finite values")
    return np.ascontiguousarray(cla, dtype=np.float64)


class CoherenceMonitor:
    """Track coherence partitioned into good vs bad layer subsets."""

    def __init__(self, good_layers: list[int], bad_layers: list[int]):
        self._good = _validate_layer_indices(good_layers, name="good_layers")
        self._bad = _validate_layer_indices(bad_layers, name="bad_layers")

    def compute_r_good(self, upde_state: UPDEState) -> float:
        """Mean order parameter R across good (synchronise) layers."""
        return float(self._mean_r(upde_state, self._good, name="good_layers"))

    def compute_r_bad(self, upde_state: UPDEState) -> float:
        """Mean order parameter R across bad (desynchronise) layers."""
        return float(self._mean_r(upde_state, self._bad, name="bad_layers"))

    # PLV lock threshold: Lachaux et al. 1999; see docs/ASSUMPTIONS.md § Quality Gating
    def detect_phase_lock(
        self, upde_state: UPDEState, threshold: float = 0.9
    ) -> list[tuple[int, int]]:
        """Return pairs of layer indices whose PLV exceeds threshold.

        Uses cross_layer_alignment matrix as the primary PLV source
        (matches Rust implementation). Falls back to lock_signatures
        if CLA entry is below threshold but a signature overrides it.
        """
        threshold = _validate_plv_threshold(threshold)
        n = len(upde_state.layers)
        cla = _validate_cross_layer_alignment(
            upde_state.cross_layer_alignment, n_layers=n
        )
        locked = []
        for i in range(n):
            for j in range(i + 1, n):
                # Primary: use CLA matrix (always populated from phase data)
                if i < cla.shape[0] and j < cla.shape[1] and cla[i, j] >= threshold:
                    locked.append((i, j))
                    continue
                # Fallback: explicit lock_signatures (manually set)
                key = f"{i}_{j}"
                sig = upde_state.layers[i].lock_signatures.get(key)
                if sig is not None and sig.plv >= threshold:
                    locked.append((i, j))
        return locked

    def _mean_r(self, upde_state: UPDEState, indices: list[int], *, name: str) -> float:
        n_layers = len(upde_state.layers)
        invalid = [index for index in indices if index >= n_layers]
        if invalid:
            raise ValueError(
                f"{name} references layer indices outside state with "
                f"{n_layers} layers: {invalid!r}"
            )
        vals = [upde_state.layers[i].R for i in indices]
        if not vals:
            return 0.0
        return float(np.mean(vals))
