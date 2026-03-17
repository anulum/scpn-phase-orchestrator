# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coherence monitor

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["CoherenceMonitor"]


class CoherenceMonitor:
    """Track coherence partitioned into good vs bad layer subsets."""

    def __init__(self, good_layers: list[int], bad_layers: list[int]):
        self._good = good_layers
        self._bad = bad_layers

    def compute_r_good(self, upde_state: UPDEState) -> float:
        return float(self._mean_r(upde_state, self._good))

    def compute_r_bad(self, upde_state: UPDEState) -> float:
        return float(self._mean_r(upde_state, self._bad))

    # PLV lock threshold: Lachaux et al. 1999; see docs/ASSUMPTIONS.md § Quality Gating
    def detect_phase_lock(
        self, upde_state: UPDEState, threshold: float = 0.9
    ) -> list[tuple[int, int]]:
        """Return pairs of layer indices whose PLV exceeds threshold.

        Uses cross_layer_alignment matrix as the primary PLV source
        (matches Rust implementation). Falls back to lock_signatures
        if CLA entry is below threshold but a signature overrides it.
        """
        n = len(upde_state.layers)
        cla = upde_state.cross_layer_alignment
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

    def _mean_r(self, upde_state: UPDEState, indices: list[int]) -> float:
        vals = [upde_state.layers[i].R for i in indices if i < len(upde_state.layers)]
        if not vals:
            return 0.0
        return float(np.mean(vals))
