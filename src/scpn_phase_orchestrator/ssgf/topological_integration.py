# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topological Integration Observable
#
# Measures H1 persistent homology of delay-embedded phase dynamics.
# p_h1 > tau_h1 (default 0.72) → topological-integration gate open.
#
# Requires ripser for full Vietoris-Rips persistence. Falls back to
# phase-locking value (PLV) based approximation without ripser.

"""Topological Integration Observable over phase-history windows.

The observer accumulates copied phase snapshots, delay-embeds recent history,
and computes H1 persistence through ``ripser`` when available. Without ripser
it falls back to a PLV approximation that preserves the same public state shape
but not the same topological guarantee. The scalar ``p_h1`` is a
dynamical-structure measure — the persistence of first-homology loops in the
phase point cloud — and nothing more: it does not assert phenomenology, perform
actuation, or make safety-critical decisions by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["TopologicalIntegrationObserver", "TopologicalIntegrationState"]

FloatArray: TypeAlias = NDArray[np.float64]

_TAU_H1 = 0.72

try:
    from ripser import ripser as _ripser

    _HAS_RIPSER = True
except ImportError:  # pragma: no cover
    _HAS_RIPSER = False


@dataclass
class TopologicalIntegrationState:
    """One observation: p_h1 score, the integration gate, and the method."""

    p_h1: float
    is_integrated: bool
    s_h1: float
    method: str


class TopologicalIntegrationObserver:
    """Topological Integration Observable.

    Delay-embeds multichannel phase signals, computes H1 persistent homology via
    a Vietoris-Rips filtration, squashes the maximum loop lifetime to ``[0, 1]``
    with a logistic, and opens an integration gate at ``p_h1 > tau_h1``.

    The default threshold ``tau_h1 = 0.72`` is set for the metastable
    ``R ~ 0.4-0.8`` regime, where persistent first-homology loops are most
    pronounced — full synchrony (``R > 0.95``) collapses the point cloud and
    incoherence (``R < 0.2``) fills it uniformly, and neither produces a dominant
    persistent 1-cycle. This is a topological-structure measure, not a claim
    about consciousness or phenomenology.
    """

    def __init__(
        self,
        tau_h1: float = _TAU_H1,
        embed_dim: int = 3,
        embed_delay: int = 1,
        window_size: int = 50,
        beta: float = 8.0,
    ):
        if (
            any(isinstance(v, bool) or not isinstance(v, Real) for v in (tau_h1, beta))
            or not isfinite(float(tau_h1))
            or not isfinite(float(beta))
        ):
            raise TypeError("tau_h1 and beta must be finite real values")
        if not 0.0 <= float(tau_h1) <= 1.0:
            raise ValueError(f"tau_h1 must be within [0, 1], got {tau_h1!r}")
        if float(beta) <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta!r}")
        for name, value in (
            ("embed_dim", embed_dim),
            ("embed_delay", embed_delay),
            ("window_size", window_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self._tau_h1 = tau_h1
        self._embed_dim = embed_dim
        self._embed_delay = embed_delay
        self._window_size = window_size
        self._beta = beta
        self._history: list[FloatArray] = []

    @property
    def tau_h1(self) -> float:
        """Integration gate threshold on p_h1.

        Returns
        -------
        float
            Integration gate threshold on p_h1.
        """
        return self._tau_h1

    def observe(self, phases: FloatArray) -> TopologicalIntegrationState:
        """Add a phase snapshot and compute the observable if enough history.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.

        Returns
        -------
        TopologicalIntegrationState
            The p_h1 score, integration gate, loop lifetime, and method.

        Raises
        ------
        TypeError
            If an argument has the wrong type.
        ValueError
            If the inputs are invalid or inconsistent.
        """
        if not isinstance(phases, np.ndarray):
            raise TypeError(f"phases must be a numpy.ndarray, got {phases!r}")
        if phases.ndim != 1:
            raise ValueError(f"phases must be a 1D vector, got shape {phases.shape!r}")
        if phases.size == 0:
            raise ValueError("phases must be non-empty")
        if np.issubdtype(phases.dtype, np.bool_):
            raise ValueError("phases must not use boolean dtype")
        if not np.isfinite(phases).all():
            raise ValueError("phases must contain only finite values")
        self._history.append(phases.copy())
        max_len = self._window_size + self._embed_dim * self._embed_delay
        if len(self._history) > max_len:
            self._history = self._history[-max_len:]

        min_len = self._embed_dim * self._embed_delay + 2
        if len(self._history) < min_len:
            return TopologicalIntegrationState(
                p_h1=0.0,
                is_integrated=False,
                s_h1=0.0,
                method="insufficient_data",
            )

        if _HAS_RIPSER:
            return self._observe_ripser()
        return self._observe_plv()  # pragma: no cover

    def _observe_ripser(self) -> TopologicalIntegrationState:
        """Return the Ripser persistence observation, if available."""
        cloud = self._delay_embed()
        result = _ripser(cloud, maxdim=1)
        h1 = result["dgms"][1]
        if len(h1) == 0:
            return TopologicalIntegrationState(
                p_h1=0.0, is_integrated=False, s_h1=0.0, method="ripser"
            )
        lifetimes = h1[:, 1] - h1[:, 0]
        finite_mask = np.isfinite(lifetimes)
        if not np.any(finite_mask):
            return TopologicalIntegrationState(
                p_h1=0.0, is_integrated=False, s_h1=0.0, method="ripser"
            )
        s_h1 = float(np.max(lifetimes[finite_mask]))
        p_h1 = 1.0 / (1.0 + np.exp(-self._beta * s_h1))
        return TopologicalIntegrationState(
            p_h1=p_h1,
            is_integrated=p_h1 > self._tau_h1,
            s_h1=s_h1,
            method="ripser",
        )

    def _observe_plv(self) -> TopologicalIntegrationState:  # pragma: no cover
        """PLV-based approximation when ripser is not available.

        Uses mean pairwise PLV as a proxy for topological integration.
        This is NOT equivalent to H1 persistence — it's an approximation
        that captures the same qualitative behavior (high PLV ↔ high p_h1)
        without the topological guarantees.
        """
        recent: FloatArray = np.array(self._history[-self._window_size :])
        n_osc = recent.shape[1]
        if n_osc < 2:
            return TopologicalIntegrationState(
                p_h1=0.0,
                is_integrated=False,
                s_h1=0.0,
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
        return TopologicalIntegrationState(
            p_h1=p_h1,
            is_integrated=p_h1 > self._tau_h1,
            s_h1=s_h1,
            method="plv_approx",
        )

    def _delay_embed(self) -> FloatArray:
        """Delay-embed phase history into d-dimensional point cloud."""
        recent: FloatArray = np.array(self._history[-self._window_size :])
        T, n_ch = recent.shape
        d = self._embed_dim
        tau = self._embed_delay
        T_out = T - (d - 1) * tau
        if T_out < 2:
            return recent[:2, :]
        embedded: FloatArray = np.zeros((T_out, d * n_ch))
        for k in range(d):
            start = k * tau
            embedded[:, k * n_ch : (k + 1) * n_ch] = recent[start : start + T_out]
        return embedded

    def reset(self) -> None:
        """Clear stored phase history."""
        self._history.clear()
