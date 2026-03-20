# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topological Consciousness Boundary Observable
#
# Measures H1 persistent homology of delay-embedded phase dynamics.
# p_h1 > tau_h1 (default 0.72) → consciousness gate open.
#
# Requires ripser for full Vietoris-Rips persistence. Falls back to
# phase-locking value (PLV) based approximation without ripser.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["TCBOObserver", "TCBOState"]

_TAU_H1 = 0.72  # from SCPN-CODEBASE optimizations/tcbo/observer.py

try:
    from ripser import ripser as _ripser

    _HAS_RIPSER = True
except ImportError:  # pragma: no cover
    _HAS_RIPSER = False


@dataclass
class TCBOState:
    p_h1: float
    is_conscious: bool
    s_h1: float
    method: str


class TCBOObserver:
    """Topological Consciousness Boundary Observable.

    Delay-embeds multichannel phase signals, computes H1 persistent
    homology via Vietoris-Rips filtration, squashes max lifetime to
    [0,1] via logistic, gates consciousness at p_h1 > tau_h1.

    The threshold tau_h1 = 0.72 operates in the metastable R~0.4-0.8
    regime — consistent with phenomenological evidence that consciousness
    lives at intermediate synchronization, not maximum sync (R>0.95).
    """

    def __init__(
        self,
        tau_h1: float = _TAU_H1,
        embed_dim: int = 3,
        embed_delay: int = 1,
        window_size: int = 50,
        beta: float = 8.0,
    ):
        self._tau_h1 = tau_h1
        self._embed_dim = embed_dim
        self._embed_delay = embed_delay
        self._window_size = window_size
        self._beta = beta
        self._history: list[NDArray] = []

    @property
    def tau_h1(self) -> float:
        return self._tau_h1

    def observe(self, phases: NDArray) -> TCBOState:
        """Add phase snapshot, compute TCBO if enough history."""
        self._history.append(phases.copy())
        max_len = self._window_size + self._embed_dim * self._embed_delay
        if len(self._history) > max_len:
            self._history = self._history[-max_len:]

        min_len = self._embed_dim * self._embed_delay + 2
        if len(self._history) < min_len:
            return TCBOState(
                p_h1=0.0, is_conscious=False, s_h1=0.0,
                method="insufficient_data",
            )

        if _HAS_RIPSER:
            return self._observe_ripser()
        return self._observe_plv()

    def _observe_ripser(self) -> TCBOState:
        cloud = self._delay_embed()
        result = _ripser(cloud, maxdim=1)
        h1 = result["dgms"][1]
        if len(h1) == 0:
            return TCBOState(p_h1=0.0, is_conscious=False, s_h1=0.0, method="ripser")
        lifetimes = h1[:, 1] - h1[:, 0]
        finite_mask = np.isfinite(lifetimes)
        if not np.any(finite_mask):
            return TCBOState(p_h1=0.0, is_conscious=False, s_h1=0.0, method="ripser")
        s_h1 = float(np.max(lifetimes[finite_mask]))
        p_h1 = 1.0 / (1.0 + np.exp(-self._beta * s_h1))
        return TCBOState(
            p_h1=p_h1,
            is_conscious=p_h1 > self._tau_h1,
            s_h1=s_h1,
            method="ripser",
        )

    def _observe_plv(self) -> TCBOState:
        """PLV-based approximation when ripser is not available.

        Uses mean pairwise PLV as a proxy for topological integration.
        This is NOT equivalent to H1 persistence — it's an approximation
        that captures the same qualitative behavior (high PLV ↔ high p_h1)
        without the topological guarantees.
        """
        recent = np.array(self._history[-self._window_size:])
        n_osc = recent.shape[1]
        if n_osc < 2:
            return TCBOState(
                p_h1=0.0, is_conscious=False, s_h1=0.0,
                method="plv_approx",
            )

        # Mean PLV across all pairs
        plv_sum = 0.0
        count = 0
        for i in range(n_osc):
            for j in range(i + 1, n_osc):
                diff = recent[:, i] - recent[:, j]
                plv = float(np.abs(np.mean(np.exp(1j * diff))))
                plv_sum += plv
                count += 1
        mean_plv = plv_sum / count if count > 0 else 0.0

        s_h1 = mean_plv  # proxy
        p_h1 = 1.0 / (1.0 + np.exp(-self._beta * (s_h1 - 0.3)))
        return TCBOState(
            p_h1=p_h1,
            is_conscious=p_h1 > self._tau_h1,
            s_h1=s_h1,
            method="plv_approx",
        )

    def _delay_embed(self) -> NDArray:
        """Delay-embed phase history into d-dimensional point cloud."""
        recent = np.array(self._history[-self._window_size:])
        T, n_ch = recent.shape
        d = self._embed_dim
        tau = self._embed_delay
        T_out = T - (d - 1) * tau
        if T_out < 2:
            return recent[:2, :]
        embedded = np.zeros((T_out, d * n_ch))
        for k in range(d):
            start = k * tau
            embedded[:, k * n_ch : (k + 1) * n_ch] = recent[start : start + T_out]
        return embedded

    def reset(self) -> None:
        self._history.clear()
