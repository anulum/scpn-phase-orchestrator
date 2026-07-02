# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Generalised k-body hypergraph coupling

"""Hypergraph Kuramoto with arbitrary k-body interactions beyond pairwise.

Exposes a 5-backend fallback chain.

Extends the standard Kuramoto model with k-body coupling terms for
any k ≥ 2. The standard model (k=2) and simplicial model (k=3) are
special cases.

For a k-hyperedge {i₁, …, iₖ}, the coupling on oscillator ``iₘ`` is

    σₖ · sin( Σ_{j≠m} θ_{iⱼ} − (k−1)·θ_{iₘ} )

which generalises

    k = 2:  sin(θ_j − θ_i)            — standard Kuramoto
    k = 3:  sin(θ_j + θ_k − 2·θ_i)    — simplicial / triadic

The engine also accepts a dense pairwise coupling matrix
``pairwise_knm`` (optional) and an external-drive field ``(ζ, ψ)``.

Numerics
--------
The pairwise-derivative loop uses the Rust kernel's
``sin(θ_j − θ_i) = sin(θ_j)·cos(θ_i) − cos(θ_j)·sin(θ_i)``
expansion in the ``alpha == 0`` fast path so that floating-point
rounding matches Rust (``spo-engine/src/hypergraph.rs``) bit-for-bit.
Alpha ≠ 0 falls back to the direct ``sin(θ_j − θ_i − α)`` form in
all five backends.

References
----------
Tanaka & Aoyagi 2011, Phys. Rev. Lett. 106:224101.
Skardal & Arenas 2019, Comm. Phys. 2:22.
Bick, Gross, Harrington & Schaub 2023, Nat. Rev. Physics 5:307-317.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _hypergraph_validation,
)
from scpn_phase_orchestrator.upde._julia_runtime import require_juliacall_main

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "Hyperedge",
    "HypergraphEngine",
]


@dataclass
class Hyperedge:
    """A k-body interaction among oscillators.

    Attributes
    ----------
        nodes: Tuple of oscillator indices in this hyperedge.
        strength: Coupling strength σₖ for this hyperedge.
    """

    nodes: tuple[int, ...]
    strength: float = 1.0

    @property
    def order(self) -> int:
        """Return the number of oscillators participating in the hyperedge.

        Returns
        -------
        int
            Return the number of oscillators participating in the hyperedge.
        """
        return len(self.nodes)


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., FloatArray]:
    """Load the Rust hypergraph backend callable."""
    from spo_kernel import hypergraph_run_rust

    def _rust(
        phases: FloatArray,
        omegas: FloatArray,
        n: int,
        edge_nodes: IntArray,
        edge_offsets: IntArray,
        edge_strengths: FloatArray,
        knm_flat: FloatArray,
        alpha_flat: FloatArray,
        zeta: float,
        psi: float,
        dt: float,
        n_steps: int,
    ) -> FloatArray:
        """Call the Rust hypergraph Kuramoto step kernel."""
        return _validate_backend_output(
            hypergraph_run_rust(
                np.ascontiguousarray(phases, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                int(n),
                np.ascontiguousarray(edge_nodes, dtype=np.int64),
                np.ascontiguousarray(edge_offsets, dtype=np.int64),
                np.ascontiguousarray(edge_strengths, dtype=np.float64),
                np.ascontiguousarray(knm_flat, dtype=np.float64),
                np.ascontiguousarray(alpha_flat, dtype=np.float64),
                float(zeta),
                float(psi),
                float(dt),
                int(n_steps),
            ),
            n=int(n),
        )

    return _rust


def _load_mojo_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Mojo hypergraph backend callable."""
    from ..experimental.accelerators.upde._hypergraph_mojo import (
        _ensure_exe,
        hypergraph_run_mojo,
    )

    _ensure_exe()
    return hypergraph_run_mojo


def _load_julia_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Julia hypergraph backend callable."""
    require_juliacall_main()

    from ..experimental.accelerators.upde._hypergraph_julia import (
        hypergraph_run_julia,
    )

    return hypergraph_run_julia


def _load_go_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Go hypergraph backend callable."""
    from ..experimental.accelerators.upde._hypergraph_go import (
        _load_lib,
        hypergraph_run_go,
    )

    _load_lib()
    return hypergraph_run_go


_LOADERS: dict[str, Callable[[], Callable[..., FloatArray]]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, Callable[..., FloatArray]] = {}


def _load_backend(name: str) -> Callable[..., FloatArray]:
    """Load and cache the named backend callable."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve the active and available backends, fastest-first."""
    _BACKEND_CACHE.clear()
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch() -> Callable[..., FloatArray] | None:
    """Return the fastest available backend callable, or ``None`` for Python."""
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    deduped: list[str] = []
    for backend in ordered_backends:
        if backend in deduped:
            continue
        deduped.append(backend)
    for backend in deduped:
        if backend == "python":
            return None
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
    return None


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1 as a non-boolean integer, got {value!r}")
    return int(value)


def _validate_positive_float(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    return coerced


def _validate_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be finite real, got {value!r}")
    return coerced


def _validate_state_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    """Return the state as a validated finite array, else raise."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_optional_state_array(
    value: object | None,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    """Return the optional state as a validated array, or ``None``."""
    if value is None:
        return np.empty(0, dtype=np.float64)
    return _validate_state_array(value, name=name, shape=shape).ravel()


def _validate_backend_output(value: object, *, n: int) -> FloatArray:
    """Return a validated hypergraph backend phase vector, else raise."""
    return _hypergraph_validation.validate_hypergraph_output(value, n=n)


def _validate_hyperedge(edge: Hyperedge, *, n_oscillators: int) -> Hyperedge:
    """Return a validated hyperedge of distinct oscillator indices, else raise."""
    nodes = tuple(edge.nodes)
    if len(nodes) < 2:
        raise ValueError("hyperedge nodes must contain at least two oscillators")
    if len(set(nodes)) != len(nodes):
        raise ValueError("hyperedge nodes must be unique")
    for node in nodes:
        if isinstance(node, bool) or not isinstance(node, Integral):
            raise ValueError(f"hyperedge node must be an integer, got {node!r}")
        if int(node) < 0 or int(node) >= n_oscillators:
            raise ValueError(f"hyperedge node {node!r} outside [0, {n_oscillators})")
    if isinstance(edge.strength, bool) or not isinstance(edge.strength, Real):
        raise ValueError(
            f"hyperedge strength must be finite real, got {edge.strength!r}"
        )
    strength = float(edge.strength)
    if not np.isfinite(strength):
        raise ValueError(
            f"hyperedge strength must be finite real, got {edge.strength!r}"
        )
    return Hyperedge(nodes=tuple(int(node) for node in nodes), strength=strength)


def _python_run(
    phases: FloatArray,
    omegas: FloatArray,
    n: int,
    edge_nodes: IntArray,
    edge_offsets: IntArray,
    edge_strengths: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Python reference aligned to the Rust kernel.

    Uses the ``sin(θ_j − θ_i) = s_j·c_i − c_j·s_i`` expansion for
    the ``alpha == 0`` fast path (bit-identical to Rust) and the
    direct ``sin(diff)`` form for nonzero alpha.
    """
    p = np.asarray(phases, dtype=np.float64).copy()
    om = np.asarray(omegas, dtype=np.float64)
    has_pairwise = knm_flat.size == n * n
    has_alpha = alpha_flat.size == n * n
    alpha_zero = (not has_alpha) or bool(np.all(alpha_flat == 0.0))

    n_edges = int(edge_offsets.size)
    for _ in range(n_steps):
        deriv = om.copy()
        if has_pairwise:
            knm = knm_flat.reshape(n, n)
            s = np.sin(p)
            c = np.cos(p)
            if alpha_zero:
                # sin(θ_j − θ_i) = s_j · c_i − c_j · s_i
                sin_diff = (
                    s[np.newaxis, :] * c[:, np.newaxis]
                    - c[np.newaxis, :] * s[:, np.newaxis]
                )
                deriv += np.sum(knm * sin_diff, axis=1)
            else:
                alpha = alpha_flat.reshape(n, n)
                diff = p[np.newaxis, :] - p[:, np.newaxis] - alpha
                deriv += np.sum(knm * np.sin(diff), axis=1)
        if zeta != 0.0:
            deriv += zeta * np.sin(psi - p)
        for e in range(n_edges):
            start = int(edge_offsets[e])
            stop = int(edge_offsets[e + 1]) if e + 1 < n_edges else int(edge_nodes.size)
            nodes = edge_nodes[start:stop]
            k = stop - start
            phase_sum = float(np.sum(p[nodes]))
            sigma = float(edge_strengths[e])
            for m_idx in nodes:
                m = int(m_idx)
                deriv[m] += sigma * np.sin(phase_sum - k * p[m])
        p = (p + dt * deriv) % TWO_PI
    return p


class HypergraphEngine:
    """Kuramoto engine with arbitrary k-body hypergraph coupling.

    Supports mixed-order interactions: some edges can be pairwise,
    some 3-body, some 4-body, etc. Each ``Hyperedge`` specifies
    which oscillators participate and the coupling strength.
    """

    def __init__(
        self,
        n_oscillators: int,
        dt: float,
        hyperedges: list[Hyperedge] | None = None,
    ) -> None:
        """Initialise a validated hypergraph Kuramoto engine.

        Parameters
        ----------
        n_oscillators : int
            Positive number of oscillators in the simulated system.
        dt : float
            Positive finite explicit-Euler step size.
        hyperedges : list[Hyperedge] | None
            Optional initial hyperedge definitions to validate and store.
        """
        self._n = _validate_positive_int(n_oscillators, name="n_oscillators")
        self._dt = _validate_positive_float(dt, name="dt")
        self._hyperedges = [
            _validate_hyperedge(edge, n_oscillators=self._n)
            for edge in (hyperedges or [])
        ]

    def add_edge(self, nodes: tuple[int, ...], strength: float = 1.0) -> None:
        """Validate and append one explicit k-body hyperedge.

        Parameters
        ----------
        nodes : tuple[int, ...]
            Indices of the oscillators participating in the hyperedge.
        strength : float
            Coupling strength assigned to the hyperedge(s).
        """
        edge = _validate_hyperedge(
            Hyperedge(nodes=nodes, strength=strength),
            n_oscillators=self._n,
        )
        self._hyperedges.append(edge)

    def add_all_to_all(self, order: int, strength: float = 1.0) -> None:
        """Add all C(N, order) hyperedges of given order.

        Parameters
        ----------
        order : int
            Interaction order (number of oscillators per hyperedge).
        strength : float
            Coupling strength assigned to the hyperedge(s).

        Raises
        ------
        ValueError
            If ``order`` is outside ``2..N``.
        """
        from itertools import combinations

        order = _validate_positive_int(order, name="order")
        if order > self._n:
            raise ValueError(f"order must be <= n_oscillators, got {order!r}")
        validated_edge = _validate_hyperedge(
            Hyperedge(nodes=tuple(range(order)), strength=strength),
            n_oscillators=self._n,
        )
        for combo in combinations(range(self._n), order):
            self._hyperedges.append(
                Hyperedge(nodes=combo, strength=validated_edge.strength)
            )

    @property
    def n_edges(self) -> int:
        """Return the number of configured hyperedges.

        Returns
        -------
        int
            Return the number of configured hyperedges.
        """
        return len(self._hyperedges)

    def _encode_edges(
        self,
    ) -> tuple[IntArray, IntArray, FloatArray]:
        """Encode the hyperedges into the flat backend representation."""
        edge_nodes_list: list[int] = []
        edge_offsets_list: list[int] = []
        edge_strengths_list: list[float] = []
        for edge in self._hyperedges:
            edge_offsets_list.append(len(edge_nodes_list))
            edge_nodes_list.extend(edge.nodes)
            edge_strengths_list.append(edge.strength)
        return (
            np.array(edge_nodes_list, dtype=np.int64),
            np.array(edge_offsets_list, dtype=np.int64),
            np.array(edge_strengths_list, dtype=np.float64),
        )

    def step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        pairwise_knm: FloatArray | None = None,
        alpha: FloatArray | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
    ) -> FloatArray:
        """One explicit-Euler step.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        pairwise_knm : FloatArray | None
            Optional pairwise coupling matrix ``(N, N)``, or ``None``.
        alpha : FloatArray | None
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
        zeta : float
            External drive strength ``ζ``.
        psi : float
            External drive reference phase ``Ψ`` in radians.

        Returns
        -------
        FloatArray
            The phases after one explicit-Euler hypergraph step.
        """
        return self.run(
            phases,
            omegas,
            n_steps=1,
            pairwise_knm=pairwise_knm,
            alpha=alpha,
            zeta=zeta,
            psi=psi,
        )

    def run(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        n_steps: int,
        pairwise_knm: FloatArray | None = None,
        alpha: FloatArray | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
    ) -> FloatArray:
        """Integrate ``n_steps`` Euler steps via the fastest backend; return phases.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        n_steps : int
            Number of integration steps to run.
        pairwise_knm : FloatArray | None
            Optional pairwise coupling matrix ``(N, N)``, or ``None``.
        alpha : FloatArray | None
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
        zeta : float
            External drive strength ``ζ``.
        psi : float
            External drive reference phase ``Ψ`` in radians.

        Returns
        -------
        FloatArray
            The final phases after ``n_steps`` hypergraph steps.
        """
        n_steps = _validate_positive_int(n_steps, name="n_steps")
        phases64 = _validate_state_array(phases, name="phases", shape=(self._n,))
        omegas64 = _validate_state_array(omegas, name="omegas", shape=(self._n,))
        knm_flat = _validate_optional_state_array(
            pairwise_knm,
            name="pairwise_knm",
            shape=(self._n, self._n),
        )
        alpha_flat = _validate_optional_state_array(
            alpha,
            name="alpha",
            shape=(self._n, self._n),
        )
        zeta = _validate_finite_float(zeta, name="zeta")
        psi = _validate_finite_float(psi, name="psi")
        en, eo, es = self._encode_edges()
        backend_fn = _dispatch()
        if backend_fn is not None:
            return _validate_backend_output(
                backend_fn(
                    phases64,
                    omegas64,
                    self._n,
                    en,
                    eo,
                    es,
                    knm_flat,
                    alpha_flat,
                    float(zeta),
                    float(psi),
                    float(self._dt),
                    int(n_steps),
                ),
                n=self._n,
            )
        return _validate_backend_output(
            _python_run(
                phases64,
                omegas64,
                self._n,
                en,
                eo,
                es,
                knm_flat,
                alpha_flat,
                float(zeta),
                float(psi),
                float(self._dt),
                int(n_steps),
            ),
            n=self._n,
        )

    def order_parameter(self, phases: FloatArray) -> float:
        """Compute the standard Kuramoto R = |<exp(iθ)>|.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.

        Returns
        -------
        float
            The Kuramoto order parameter ``R``.
        """
        phases64 = _validate_state_array(
            phases,
            name="phases",
            shape=(self._n,),
        )
        return float(np.abs(np.mean(np.exp(1j * phases64))))
