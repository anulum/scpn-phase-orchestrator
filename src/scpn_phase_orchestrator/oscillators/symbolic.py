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

from scpn_phase_orchestrator._array_types import require_real_values
from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

__all__ = ["SymbolicExtractor", "SYMBOLIC_INITIAL_TRANSITION_QUALITY_BASELINE"]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
UnsignedIntArray: TypeAlias = NDArray[np.uint64]
SymbolicArray: TypeAlias = IntArray | UnsignedIntArray
_NATIVE_STATE_COUNT_MAX = int(np.iinfo(np.uintp).max)
SYMBOLIC_INITIAL_TRANSITION_QUALITY_BASELINE = 0.5

try:
    from spo_kernel import (
        graph_walk_phases_rust as _rust_graph_walk_phases,
    )
    from spo_kernel import (
        ring_phases_rust as _rust_ring_phases,
    )
    from spo_kernel import (
        transition_qualities_rust as _rust_transition_qualities,
    )

    _HAS_RUST_SYMBOLIC = True
except (ImportError, ModuleNotFoundError):
    _HAS_RUST_SYMBOLIC = False


def _validate_n_states(value: object) -> int:
    """Return the state count as a validated positive integer, else raise."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("n_states must be an integer >= 2")
    n_states = int(value)
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states}")
    return n_states


def _validate_node_id(value: object) -> str:
    """Return the validated node id, else raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("node_id must be a non-empty string")
    return value


def _validate_signal(value: object) -> SymbolicArray:
    """Normalise integer widths and alignment without changing signedness or labels."""
    signal = np.asarray(value)
    if signal.dtype.kind not in "iu":
        raise ValueError("signal must be integer")
    require_real_values(value, name="signal")
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    indices = (
        signal.astype(np.uint64, copy=False)
        if signal.dtype.kind == "u"
        else signal.astype(np.int64, copy=False)
    )
    if not indices.flags.aligned:
        indices = indices.copy()
    return indices


def _validate_sample_rate(value: object) -> float:
    """Return the sample rate as a validated positive value, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("sample_rate must be finite and positive")
    sample_rate = float(value)
    if not isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be finite and positive")
    return sample_rate


def _validate_initial_transition_quality(value: object) -> float:
    """Return the validated initial transition-quality value, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("initial_transition_quality must be a finite float in [0, 1]")
    quality = float(value)
    if not isfinite(quality) or quality < 0.0 or quality > 1.0:
        raise ValueError("initial_transition_quality must be a finite float in [0, 1]")
    return quality


class SymbolicExtractor(PhaseExtractor):
    """Phase extraction from discrete symbolic state sequences.

    Maps discrete state indices to phases on the unit circle via
    theta = 2*pi*s/N (ring-phase) or via graph-walk position.
    """

    def __init__(
        self,
        n_states: int,
        node_id: str = "sym",
        mode: str = "ring",
        *,
        initial_transition_quality: float = (
            SYMBOLIC_INITIAL_TRANSITION_QUALITY_BASELINE
        ),
    ):
        """Configure the symbolic oscillator over ``n_states`` discrete states.

        Parameters
        ----------
        n_states : int
            Integer vocabulary size N >= 2, with no public upper bound.
            Counts beyond the native target's usize capacity use Python.
        node_id : str
            identifier for generated PhaseState objects.
        mode : str
            "ring" for ring-phase, "graph" for graph-walk phase.
        initial_transition_quality : float
            Finite quality in [0, 1] assigned to the first sample.

        Raises
        ------
        ValueError
            If the vocabulary, node identifier, mode or initial quality is invalid.
        """
        n_states = _validate_n_states(n_states)
        if mode not in ("ring", "graph"):
            raise ValueError(f"mode must be 'ring' or 'graph', got {mode!r}")
        self._n_states = n_states
        self._node_id = _validate_node_id(node_id)
        self._mode = mode
        self._initial_transition_quality = _validate_initial_transition_quality(
            initial_transition_quality
        )

    def extract(
        self, signal: FloatArray | SymbolicArray, sample_rate: float
    ) -> list[PhaseState]:
        """Map discrete state indices to phases on the unit circle.

        Parameters
        ----------
        signal : FloatArray | SymbolicArray
            Signed or unsigned integer state indices, shape ``(T,)``; labels
            retain their values across integer-width normalisation. Text,
            boolean, floating, object and temporal values are rejected before
            mapping. Strided and read-only views are accepted; unaligned input
            is copied before native access.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        list[PhaseState]
            One state per observation, with float64 phases and frequencies,
            transition qualities, unit amplitude and the configured node id.
            Graph distances accumulate as exact integers before float conversion;
            the resulting phases remain subject to float64 rounding.

        Raises
        ------
        ValueError
            If the signal is not a one-dimensional integer array or the sampling
            rate is not finite and positive.
        """
        indices = _validate_signal(signal)
        sample_rate = _validate_sample_rate(sample_rate)
        use_native = _HAS_RUST_SYMBOLIC and self._n_states <= _NATIVE_STATE_COUNT_MAX
        if self._mode == "ring" or len(indices) < 2:
            if use_native:
                thetas = np.asarray(
                    _rust_ring_phases(indices, self._n_states),
                    dtype=np.float64,
                )
            else:
                thetas = np.fromiter(
                    (
                        TWO_PI * ((int(index) % self._n_states) / self._n_states)
                        for index in indices
                    ),
                    dtype=np.float64,
                    count=len(indices),
                )
        else:
            if use_native:
                thetas = np.asarray(
                    _rust_graph_walk_phases(indices, self._n_states),
                    dtype=np.float64,
                )
            else:
                positions = [0]
                for previous, current in zip(indices[:-1], indices[1:], strict=True):
                    positions.append(positions[-1] + abs(int(current) - int(previous)))
                cumulative = np.asarray(positions, dtype=np.float64)
                total = float(max(positions[-1], 1))
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
        rust_qualities: FloatArray | None = None
        # The kernel scores the linear index distance, which is the graph-walk
        # step. On a ring the wrap from N-1 to 0 is a single step (omega above
        # already treats it so), so ring mode scores the circular distance here.
        if use_native and self._mode == "graph":
            rust_qualities = np.asarray(
                _rust_transition_qualities(
                    indices,
                    self._n_states,
                    self._initial_transition_quality,
                ),
                dtype=np.float64,
            )
        for i in range(len(thetas)):
            states.append(
                PhaseState(
                    theta=float(thetas[i]),
                    omega=float(omegas[i]),
                    amplitude=1.0,
                    quality=(
                        float(rust_qualities[i])
                        if rust_qualities is not None
                        else self._transition_quality(indices, i)
                    ),
                    channel="S",
                    node_id=self._node_id,
                )
            )
        return states

    def quality_score(self, phase_states: list[PhaseState]) -> float:
        """Mean transition quality across phase states.

        Parameters
        ----------
        phase_states : list[PhaseState]
            Extracted per-oscillator phase states.

        Returns
        -------
        float
            Mean transition quality across phase states.
        """
        if not phase_states:
            return 0.0
        return float(np.mean([ps.quality for ps in phase_states]))

    def _transition_quality(self, indices: SymbolicArray, i: int) -> float:
        """Quality based on transition regularity: penalise repeated or large jumps.

        The jump is the linear index distance in graph mode and the circular
        distance ``min(d % N, N - d % N)`` in ring mode. Signed or unbounded
        state labels are aliases of their residues; full cycles are stalls.
        """
        if i == 0 or len(indices) < 2:
            return self._initial_transition_quality
        step = abs(int(indices[i]) - int(indices[i - 1]))
        if self._mode == "ring":
            step %= self._n_states
            step = min(step, self._n_states - step)
        if step == 0:
            return 0.2  # stalled
        if step == 1:
            return 1.0  # ideal single-step transition
        # Penalise large jumps proportionally
        return max(0.1, 1.0 - (step - 1) / self._n_states)
