# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Symbolic oscillator

"""Symbolic-channel phase extraction from discrete state sequences.

`SymbolicExtractor` maps integer state indices onto ring or graph-walk phases
for semiotic and finite-state systems. It rejects invalid state counts,
non-integer signals, boolean arrays, complex arrays, and invalid sample rates
so symbolic phases remain explicit and deterministic.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

__all__ = ["SymbolicExtractor"]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _validate_n_states(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("n_states must be an integer >= 2")
    n_states = int(value)
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states}")
    return n_states


def _validate_node_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("node_id must be a non-empty string")
    return value


def _validate_signal(value: object) -> IntArray:
    signal = np.asarray(value)
    dtype = signal.dtype
    if (
        np.issubdtype(dtype, np.bool_)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.issubdtype(dtype, np.integer)
    ):
        raise ValueError("signal must be integer")
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    return signal.astype(np.int64, copy=False)


def _validate_sample_rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("sample_rate must be finite and positive")
    sample_rate = float(value)
    if not isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be finite and positive")
    return sample_rate


class SymbolicExtractor(PhaseExtractor):
    """Phase extraction from discrete symbolic state sequences.

    Maps discrete state indices to phases on the unit circle via
    theta = 2*pi*s/N (ring-phase) or via graph-walk position.
    """

    def __init__(self, n_states: int, node_id: str = "sym", mode: str = "ring"):
        """
        Args:
            n_states: total number of discrete states N
            node_id: identifier for generated PhaseState objects
            mode: "ring" for ring-phase, "graph" for graph-walk phase
        """
        n_states = _validate_n_states(n_states)
        if mode not in ("ring", "graph"):
            raise ValueError(f"mode must be 'ring' or 'graph', got {mode!r}")
        self._n_states = n_states
        self._node_id = _validate_node_id(node_id)
        self._mode = mode

    def extract(self, signal: FloatArray, sample_rate: float) -> list[PhaseState]:
        """Map discrete state indices to phases on the unit circle."""
        indices = _validate_signal(signal)
        sample_rate = _validate_sample_rate(sample_rate)
        if self._mode == "ring":
            thetas = TWO_PI * indices / self._n_states
        else:
            # Graph-walk: cumulative phase from state transitions
            # Each step adds phase proportional to the transition distance
            if len(indices) < 2:
                thetas = TWO_PI * indices.astype(np.float64) / self._n_states
            else:
                steps = np.abs(np.diff(indices)).astype(np.float64)
                cumulative = np.concatenate([[0.0], np.cumsum(steps)])
                total = cumulative[-1] if cumulative[-1] > 0 else 1.0
                thetas = TWO_PI * cumulative / total

        thetas = thetas % TWO_PI
        dt = 1.0 / sample_rate
        omegas = np.zeros_like(thetas)
        if len(thetas) > 1:
            dtheta = np.diff(thetas)
            # Unwrap jumps larger than pi
            dtheta = (dtheta + np.pi) % TWO_PI - np.pi
            omegas[1:] = dtheta / dt

        states = []
        for i in range(len(thetas)):
            states.append(
                PhaseState(
                    theta=float(thetas[i]),
                    omega=float(omegas[i]),
                    amplitude=1.0,
                    quality=self._transition_quality(indices, i),
                    channel="S",
                    node_id=self._node_id,
                )
            )
        return states

    def quality_score(self, phase_states: list[PhaseState]) -> float:
        """Mean transition quality across phase states."""
        if not phase_states:
            return 0.0
        return float(np.mean([ps.quality for ps in phase_states]))

    def _transition_quality(self, indices: IntArray, i: int) -> float:
        """Quality based on transition regularity: penalise repeated or large jumps."""
        if i == 0 or len(indices) < 2:
            return 0.5
        step = abs(int(indices[i]) - int(indices[i - 1]))
        if step == 0:
            return 0.2  # stalled
        if step == 1:
            return 1.0  # ideal single-step transition
        # Penalise large jumps proportionally
        return max(0.1, 1.0 - (step - 1) / self._n_states)
