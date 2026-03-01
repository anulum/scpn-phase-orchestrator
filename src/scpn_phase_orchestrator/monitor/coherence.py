# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.metrics import UPDEState

try:
    from spo_kernel import PyCoherenceMonitor as _RustCoherenceMonitor  # noqa: F401

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class CoherenceMonitor:
    """Track coherence partitioned into good vs bad layer subsets."""

    def __init__(self, good_layers: list[int], bad_layers: list[int]):
        self._good = good_layers
        self._bad = bad_layers

    def compute_r_good(self, upde_state: UPDEState) -> float:
        return self._mean_r(upde_state, self._good)

    def compute_r_bad(self, upde_state: UPDEState) -> float:
        return self._mean_r(upde_state, self._bad)

    def detect_phase_lock(
        self, upde_state: UPDEState, threshold=0.9
    ) -> list[tuple[int, int]]:
        """Return pairs of layer indices whose PLV exceeds threshold."""
        n = len(upde_state.layers)
        locked = []
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{i}_{j}"
                sig = upde_state.layers[i].lock_signatures.get(key)
                if sig is not None and sig.plv >= threshold:
                    locked.append((i, j))
        return locked

    def _mean_r(self, upde_state, indices):
        vals = [upde_state.layers[i].R for i in indices if i < len(upde_state.layers)]
        if not vals:
            return 0.0
        return float(np.mean(vals))
