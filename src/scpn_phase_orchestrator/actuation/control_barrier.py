# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Verified neural Control Barrier Function safety filter

"""Verified neural Control Barrier Function (CBF) safety filter.

A heuristic policy gate clamps actions to bounds; it cannot *prove* that the
admitted action keeps the system inside a safe set. A Control Barrier Function
can. A barrier ``h(x)`` defines the safe set ``S = {x : h(x) ≥ 0}``; the
discrete-time CBF condition

    h(x_{k+1}) ≥ (1 − γ) · h(x_k),   γ ∈ (0, 1],

keeps ``h`` non-negative once it starts non-negative, so ``S`` is forward
invariant. The control enters the one-step plant model ``x_{k+1} ≈ x_k + f + g·u``
(``f`` the uncontrolled drift, ``g`` the per-knob control sensitivity), so the
filter admits the action closest to the supervisor's proposal that still
satisfies the first-order CBF condition

    ∇h(x)·(f + g·u) ≥ −γ · h(x),

an analytic projection of the nominal control onto that half-space, then a clip
to the actuator bounds. This is strictly stronger than a bounds clamp: it is a
state-dependent constraint derived from the barrier, not a fixed box.

The barrier is a **neural** ReLU network (pure NumPy — no training-framework
dependency), and the filter is **verified**: :func:`verify_forward_invariance`
certifies, soundly, that on the boundary shell ``{x : 0 ≤ h(x) ≤ shell}`` an
admissible control always restores the CBF condition. The certificate is built
by interval bound propagation (IBP, Gowal et al. 2018) over a partition of the
state box: IBP over-approximates ``h`` on every cell, so a passing certificate
can never be a false guarantee — at worst the sound over-approximation refuses
to certify a barrier that is in fact valid.

References
----------
* Ames, Coogan, Egerstedt, Notomista, Sreenath & Tabuada 2019, *ECC* —
  control barrier functions: theory and applications.
* Agrawal & Sreenath 2017, *RSS* — discrete-time control barrier functions.
* Gowal et al. 2018, arXiv:1810.12715 — interval bound propagation for
  verified neural-network bounds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "BarrierCertificate",
    "ControlBarrierFilter",
    "NeuralBarrier",
]


def _as_float_vector(value: object, *, name: str) -> FloatArray:
    """Return ``value`` as a contiguous finite float vector, else raise."""
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real float array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar


@dataclass(frozen=True)
class NeuralBarrier:
    """A ReLU feed-forward neural control barrier function ``h(x)``.

    The network maps a state vector to a scalar barrier value; the safe set is
    ``{x : h(x) ≥ 0}``. Hidden layers use ReLU activations and the output layer
    is linear (so ``h`` can take any sign). Weights are supplied at construction
    (trained or designed elsewhere); this class evaluates, differentiates, and
    soundly bounds the network.

    Attributes
    ----------
    weights : tuple[FloatArray, ...]
        Per-layer weight matrices, each shape ``(out, in)``.
    biases : tuple[FloatArray, ...]
        Per-layer bias vectors; ``biases[i]`` has length ``weights[i].shape[0]``.
    """

    weights: tuple[FloatArray, ...]
    biases: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("barrier must have at least one layer")
        if len(self.weights) != len(self.biases):
            raise ValueError("weights and biases must have the same layer count")
        validated_w: list[FloatArray] = []
        validated_b: list[FloatArray] = []
        prev_out: int | None = None
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            w = np.asarray(weight, dtype=np.float64)
            b = np.asarray(bias, dtype=np.float64)
            if w.ndim != 2:
                raise ValueError(f"layer {index} weight must be 2-D, got {w.shape}")
            if b.ndim != 1 or b.shape[0] != w.shape[0]:
                raise ValueError(
                    f"layer {index} bias shape {b.shape} must match output "
                    f"dimension {w.shape[0]}"
                )
            if not (np.all(np.isfinite(w)) and np.all(np.isfinite(b))):
                raise ValueError(f"layer {index} weights/biases must be finite")
            if prev_out is not None and w.shape[1] != prev_out:
                raise ValueError(
                    f"layer {index} expects input {w.shape[1]} but previous layer "
                    f"emits {prev_out}"
                )
            prev_out = w.shape[0]
            validated_w.append(np.ascontiguousarray(w))
            validated_b.append(np.ascontiguousarray(b))
        if prev_out != 1:
            raise ValueError("barrier output layer must emit a single scalar")
        object.__setattr__(self, "weights", tuple(validated_w))
        object.__setattr__(self, "biases", tuple(validated_b))

    @property
    def input_dim(self) -> int:
        """Dimension of the state vector the barrier consumes."""
        return int(self.weights[0].shape[1])

    def value(self, state: FloatArray) -> float:
        """Return the barrier value ``h(state)`` (safe when ``≥ 0``).

        Parameters
        ----------
        state : FloatArray
            A state vector of length :attr:`input_dim`.

        Returns
        -------
        float
            The scalar barrier value.
        """
        activation = self._validate_state(state)
        for layer, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            activation = weight @ activation + bias
            if layer < len(self.weights) - 1:
                activation = np.maximum(activation, 0.0)
        return float(activation[0])

    def gradient(self, state: FloatArray) -> FloatArray:
        """Return ``∂h/∂state`` at ``state`` by reverse-mode differentiation.

        Parameters
        ----------
        state : FloatArray
            A state vector of length :attr:`input_dim`.

        Returns
        -------
        FloatArray
            The gradient vector, shape ``(input_dim,)``.
        """
        activation = self._validate_state(state)
        pre_masks: list[FloatArray] = []
        for layer, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            activation = weight @ activation + bias
            if layer < len(self.weights) - 1:
                mask = (activation > 0.0).astype(np.float64)
                pre_masks.append(mask)
                activation = activation * mask
        grad = np.ones(1, dtype=np.float64)
        for layer in range(len(self.weights) - 1, -1, -1):
            grad = self.weights[layer].T @ grad
            if layer > 0:
                grad = grad * pre_masks[layer - 1]
        return np.ascontiguousarray(grad, dtype=np.float64)

    def interval_bounds(
        self, lower: FloatArray, upper: FloatArray
    ) -> tuple[float, float]:
        """Return sound ``[min, max]`` bounds of ``h`` over a state box (IBP).

        Interval bound propagation pushes the input interval ``[lower, upper]``
        through each affine layer (in centre/radius form) and through ReLU,
        yielding an over-approximation: the true range of ``h`` over the box is
        contained in the returned interval.

        Parameters
        ----------
        lower, upper : FloatArray
            Per-dimension lower and upper bounds of the state box.

        Returns
        -------
        tuple[float, float]
            A sound ``(h_min, h_max)`` enclosure over the box.

        Raises
        ------
        ValueError
            If the box is malformed or ``upper < lower`` in any dimension.
        """
        lo = self._validate_state(lower)
        hi = self._validate_state(upper)
        if np.any(hi < lo):
            raise ValueError("interval upper bound must be >= lower bound")
        for layer, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            centre = (lo + hi) / 2.0
            radius = (hi - lo) / 2.0
            out_centre = weight @ centre + bias
            out_radius = np.abs(weight) @ radius
            lo = out_centre - out_radius
            hi = out_centre + out_radius
            if layer < len(self.weights) - 1:
                lo = np.maximum(lo, 0.0)
                hi = np.maximum(hi, 0.0)
        return float(lo[0]), float(hi[0])

    def _validate_state(self, state: object) -> FloatArray:
        """Return the validated system state, else raise."""
        vector = _as_float_vector(state, name="state")
        if vector.shape[0] != self.input_dim:
            raise ValueError(
                f"state dimension {vector.shape[0]} does not match barrier input "
                f"{self.input_dim}"
            )
        return vector


@dataclass(frozen=True)
class BarrierCertificate:
    """Sound forward-invariance verdict for a CBF filter over a state box.

    Attributes
    ----------
    verified : bool
        Whether every boundary-shell cell admits a control restoring the CBF
        condition (a sound guarantee — never a false positive).
    cells_checked : int
        Number of partition cells inspected.
    boundary_cells : int
        Number of cells on the safety boundary (where ``h`` may reach 0).
    worst_margin : float
        Smallest ``best_h_next − (1 − γ)·h_upper`` over boundary cells; ``≥ 0``
        iff verified. ``inf`` when no boundary cell exists in the box.
    boundary_shell : float
        The boundary-shell half-width used (cells with ``h_min ≤ shell``).
    gamma : float
        The CBF decrease rate used.
    filter_digest : str
        SHA-256 digest of the exact :class:`ControlBarrierFilter` configuration
        the certificate verifies.
    verification_digest : str
        SHA-256 digest of the filter digest plus the state/drift boxes and
        verification parameters used to produce the certificate.
    """

    verified: bool
    cells_checked: int
    boundary_cells: int
    worst_margin: float
    boundary_shell: float
    gamma: float
    filter_digest: str = ""
    verification_digest: str = ""

    def to_dict(self) -> dict[str, bool | int | float | str]:
        """Return a JSON-serialisable mapping of the certificate.

        Returns
        -------
        dict[str, bool | int | float | str]
            The verdict, cell counts, worst margin, shell width, gamma, and
            digests binding the certificate to its verified filter/envelope.
        """
        return {
            "verified": self.verified,
            "cells_checked": self.cells_checked,
            "boundary_cells": self.boundary_cells,
            "worst_margin": self.worst_margin,
            "boundary_shell": self.boundary_shell,
            "gamma": self.gamma,
            "filter_digest": self.filter_digest,
            "verification_digest": self.verification_digest,
        }


@dataclass(frozen=True)
class ControlBarrierFilter:
    """A CBF-QP safety filter over a single scalar control knob.

    Attributes
    ----------
    barrier : NeuralBarrier
        The neural control barrier function.
    gamma : float
        Discrete CBF decrease rate ``γ ∈ (0, 1]``; the barrier may fall by at
        most a factor ``γ`` of its value per step.
    control_lo, control_hi : float
        Actuator bounds on the scalar control ``u``.
    control_effect : FloatArray
        The per-unit control sensitivity ``g = ∂x/∂u``, shape ``(input_dim,)``.
    """

    barrier: NeuralBarrier
    gamma: float
    control_lo: float
    control_hi: float
    control_effect: FloatArray

    def __post_init__(self) -> None:
        gamma = _as_finite_float(self.gamma, name="gamma")
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError(f"gamma must lie in (0, 1], got {gamma}")
        lo = _as_finite_float(self.control_lo, name="control_lo")
        hi = _as_finite_float(self.control_hi, name="control_hi")
        if lo > hi:
            raise ValueError(f"control_lo {lo} must be <= control_hi {hi}")
        effect = _as_float_vector(self.control_effect, name="control_effect")
        if effect.shape[0] != self.barrier.input_dim:
            raise ValueError(
                f"control_effect dimension {effect.shape[0]} does not match barrier "
                f"input {self.barrier.input_dim}"
            )
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "control_lo", lo)
        object.__setattr__(self, "control_hi", hi)
        object.__setattr__(self, "control_effect", effect)

    @property
    def filter_digest(self) -> str:
        """Return a stable SHA-256 digest of the filter configuration.

        The digest binds a runtime CBF gate to the exact barrier weights,
        biases, CBF parameters, control bounds, and control-effect vector that a
        certificate was generated against.
        """
        return _sha256_json(
            {
                "schema": "scpn_phase_orchestrator.control_barrier_filter.v1",
                "barrier_weights": [_array_payload(w) for w in self.barrier.weights],
                "barrier_biases": [_array_payload(b) for b in self.barrier.biases],
                "gamma": self.gamma,
                "control_lo": self.control_lo,
                "control_hi": self.control_hi,
                "control_effect": _array_payload(self.control_effect),
            }
        )

    def validate_certificate(self, certificate: BarrierCertificate) -> None:
        """Raise ``ValueError`` unless ``certificate`` verifies this filter.

        Parameters
        ----------
        certificate : BarrierCertificate
            Forward-invariance certificate generated by
            :meth:`verify_forward_invariance`.

        Raises
        ------
        ValueError
            If the certificate failed, lacks a binding digest, or was generated
            for a different filter configuration.
        """
        if not certificate.verified:
            raise ValueError("barrier_certificate must be verified")
        if certificate.filter_digest == "":
            raise ValueError("barrier_certificate must carry a filter_digest")
        if certificate.filter_digest != self.filter_digest:
            raise ValueError("barrier_certificate does not match barrier_filter")
        if certificate.gamma != self.gamma:
            raise ValueError("barrier_certificate gamma does not match barrier_filter")

    def filter(
        self,
        nominal_control: float,
        state: FloatArray,
        drift: FloatArray,
    ) -> tuple[float, bool]:
        """Return the safe control nearest the nominal one, and whether it changed.

        Parameters
        ----------
        nominal_control : float
            The supervisor's proposed scalar control ``u_nom``.
        state : FloatArray
            The current state vector.
        drift : FloatArray
            The uncontrolled one-step state change ``f`` (same shape as state).

        Returns
        -------
        tuple[float, bool]
            ``(safe_control, intervened)`` — the admitted control clipped to the
            actuator bounds, and whether it differs from the (bound-clipped)
            nominal control.

        Raises
        ------
        ValueError
            If inputs are malformed.
        """
        u_nom = _as_finite_float(nominal_control, name="nominal_control")
        state_vec = self.barrier._validate_state(state)
        drift_vec = _as_float_vector(drift, name="drift")
        if drift_vec.shape[0] != self.barrier.input_dim:
            raise ValueError("drift dimension must match barrier input")

        grad = self.barrier.gradient(state_vec)
        h = self.barrier.value(state_vec)
        # First-order discrete CBF: grad·(drift + g·u) >= -gamma·h
        #   => (grad·g)·u >= -gamma·h - grad·drift
        lie_g = float(grad @ self.control_effect)
        rhs = -self.gamma * h - float(grad @ drift_vec)

        clipped_nom = min(self.control_hi, max(self.control_lo, u_nom))
        feasible = self._project(clipped_nom, lie_g, rhs)
        intervened = not np.isclose(feasible, clipped_nom, rtol=0.0, atol=1e-12)
        return feasible, intervened

    def _project(self, u_nom: float, lie_g: float, rhs: float) -> float:
        """Project ``u_nom`` onto ``lie_g·u >= rhs`` ∩ ``[lo, hi]`` (nearest point)."""
        # Already satisfied by the nominal: keep it.
        if lie_g * u_nom >= rhs or abs(lie_g) < 1e-15:
            if lie_g * u_nom >= rhs:
                return u_nom
            # No control authority and constraint violated: best effort is the
            # bound that maximises lie_g·u; nothing restores feasibility.
            return self.control_hi if lie_g >= 0.0 else self.control_lo
        boundary = rhs / lie_g
        if lie_g > 0.0:
            # Need u >= boundary; nearest feasible is max(u_nom, boundary).
            return min(self.control_hi, max(boundary, u_nom))
        # lie_g < 0: need u <= boundary; nearest feasible is min(u_nom, boundary).
        return max(self.control_lo, min(boundary, u_nom))

    def verify_forward_invariance(
        self,
        state_lo: FloatArray,
        state_hi: FloatArray,
        drift_lo: FloatArray,
        drift_hi: FloatArray,
        *,
        cells_per_axis: int = 16,
        boundary_shell: float = 0.25,
    ) -> BarrierCertificate:
        """Soundly certify forward invariance of the safe set over a state box.

        The box is partitioned into ``cells_per_axis`` cells per dimension. For
        each cell, IBP bounds ``h`` over the cell; a cell is on the boundary when
        its lower bound is ``≤ shell`` and its range straddles or approaches 0.
        For boundary cells, the next-state box ``x + f + g·u`` is formed by
        interval arithmetic (worst-case drift ``[drift_lo, drift_hi]``) for each
        control endpoint, IBP bounds ``h`` on it, and the best endpoint must keep
        ``h_next_lower ≥ (1 − γ)·h_upper``. Because IBP over-approximates, a
        ``verified`` certificate is sound.

        Parameters
        ----------
        state_lo, state_hi : FloatArray
            The state box to certify.
        drift_lo, drift_hi : FloatArray
            Worst-case interval enclosure of the uncontrolled drift ``f``.
        cells_per_axis : int
            Partition resolution per dimension (``>= 1``).
        boundary_shell : float
            Boundary-shell half-width; only cells reaching ``h ≤ shell`` are
            checked for the CBF condition.

        Returns
        -------
        BarrierCertificate
            The sound forward-invariance verdict.

        Raises
        ------
        ValueError
            If inputs are malformed.
        """
        lo = self.barrier._validate_state(state_lo)
        hi = self.barrier._validate_state(state_hi)
        d_lo = _as_float_vector(drift_lo, name="drift_lo")
        d_hi = _as_float_vector(drift_hi, name="drift_hi")
        if d_lo.shape[0] != self.barrier.input_dim or d_hi.shape[0] != lo.shape[0]:
            raise ValueError("drift bounds must match barrier input dimension")
        if np.any(hi < lo):
            raise ValueError("state_hi must be >= state_lo")
        if np.any(d_hi < d_lo):
            raise ValueError("drift_hi must be >= drift_lo")
        if isinstance(cells_per_axis, bool) or not isinstance(cells_per_axis, Integral):
            raise ValueError("cells_per_axis must be a positive integer")
        if cells_per_axis < 1:
            raise ValueError("cells_per_axis must be >= 1")
        boundary_shell_value = _as_finite_float(boundary_shell, name="boundary_shell")
        if boundary_shell_value < 0.0:
            raise ValueError("boundary_shell must be non-negative")

        dim = lo.shape[0]
        edges = [np.linspace(lo[i], hi[i], cells_per_axis + 1) for i in range(dim)]
        worst_margin = float("inf")
        cells_checked = 0
        boundary_cells = 0
        verified = True

        for cell_index in np.ndindex(*([cells_per_axis] * dim)):
            cell_lo = np.array([edges[i][cell_index[i]] for i in range(dim)])
            cell_hi = np.array([edges[i][cell_index[i] + 1] for i in range(dim)])
            cells_checked += 1
            h_min, h_max = self.barrier.interval_bounds(cell_lo, cell_hi)
            # Inside the safe set and away from the boundary: nothing to enforce.
            if h_min > boundary_shell_value:
                continue
            # Entirely outside the safe set: not part of S, skip.
            if h_max < 0.0:
                continue
            boundary_cells += 1
            cell_margin = self._cell_margin(cell_lo, cell_hi, d_lo, d_hi, h_max)
            worst_margin = min(worst_margin, cell_margin)
            if cell_margin < 0.0:
                verified = False

        filter_digest = self.filter_digest
        verification_digest = _sha256_json(
            {
                "schema": "scpn_phase_orchestrator.control_barrier_verification.v1",
                "filter_digest": filter_digest,
                "state_lo": _array_payload(lo),
                "state_hi": _array_payload(hi),
                "drift_lo": _array_payload(d_lo),
                "drift_hi": _array_payload(d_hi),
                "cells_per_axis": int(cells_per_axis),
                "boundary_shell": boundary_shell_value,
            }
        )
        return BarrierCertificate(
            verified=verified,
            cells_checked=cells_checked,
            boundary_cells=boundary_cells,
            worst_margin=worst_margin,
            boundary_shell=boundary_shell_value,
            gamma=self.gamma,
            filter_digest=filter_digest,
            verification_digest=verification_digest,
        )

    def _cell_margin(
        self,
        cell_lo: FloatArray,
        cell_hi: FloatArray,
        drift_lo: FloatArray,
        drift_hi: FloatArray,
        h_upper: float,
    ) -> float:
        """Best achievable ``h_next_lower − (1 − γ)·h_upper`` over control endpoints."""
        target = (1.0 - self.gamma) * h_upper
        best_lower = -float("inf")
        for control in (self.control_lo, self.control_hi):
            shift = drift_lo + self.control_effect * control
            shift_hi = drift_hi + self.control_effect * control
            next_lo = cell_lo + np.minimum(shift, shift_hi)
            next_hi = cell_hi + np.maximum(shift, shift_hi)
            h_next_lower, _ = self.barrier.interval_bounds(next_lo, next_hi)
            best_lower = max(best_lower, h_next_lower)
        return best_lower - target


def _array_payload(array: FloatArray) -> list[object]:
    """Return ``array`` as a nested JSON-safe Python list."""
    return cast(list[object], np.asarray(array, dtype=np.float64).tolist())


def _sha256_json(payload: dict[str, object]) -> str:
    """Return SHA-256 over canonical compact JSON."""
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()
