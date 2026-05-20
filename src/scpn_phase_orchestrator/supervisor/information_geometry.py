# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Information-geometry control primitive

"""Deterministic information-geometry control proposals for geometry-aware control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import acos
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "FloatArray",
    "InformationGeometryControlProposal",
    "InformationGeometryState",
    "propose_information_geometry_control",
]

_BACKEND: str = "numpy_jax_compatible_information_geometry"
_DEFAULT_KNOB: str = "K"
_DEFAULT_SCOPE: str = "global"
_BOUNDARY: str = "information_geometry_control_not_live_actuation"
_EPS: float = 1e-12


@dataclass(frozen=True)
class InformationGeometryState:
    """Internal geometry state on simplex coordinates.

    Attributes are designed to be deterministic and JSON-safe after conversion.
    """

    simplex_coordinates: FloatArray
    target_coordinates: FloatArray
    metric_tensor: FloatArray
    tangent_vector: FloatArray
    curvature_proxy: float
    geodesic_length: float


@dataclass(frozen=True)
class InformationGeometryControlProposal:
    """Review-only control proposal derived from information-geometry metrics."""

    action_proposals: tuple[ControlAction, ...]
    fisher_rao_distance: float
    wasserstein_distance: float
    natural_gradient_norm: float
    curvature_proxy: float
    backend: str
    claim_boundary: str
    non_actuating: bool
    execution_disabled: bool
    proposal_hash: str
    state: InformationGeometryState

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit payload for the proposal."""
        return {
            "action_proposals": [
                {
                    "knob": action.knob,
                    "scope": action.scope,
                    "value": action.value,
                    "ttl_s": action.ttl_s,
                    "justification": action.justification,
                }
                for action in self.action_proposals
            ],
            "fisher_rao_distance": self.fisher_rao_distance,
            "wasserstein_distance": self.wasserstein_distance,
            "natural_gradient_norm": self.natural_gradient_norm,
            "curvature_proxy": self.curvature_proxy,
            "backend": self.backend,
            "claim_boundary": self.claim_boundary,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "proposal_hash": self.proposal_hash,
            "state": {
                "simplex_coordinates": self.state.simplex_coordinates.tolist(),
                "target_coordinates": self.state.target_coordinates.tolist(),
                "metric_tensor": self.state.metric_tensor.tolist(),
                "tangent_vector": self.state.tangent_vector.tolist(),
                "curvature_proxy": self.state.curvature_proxy,
                "geodesic_length": self.state.geodesic_length,
            },
        }


def propose_information_geometry_control(
    current_distribution: FloatArray | list[float] | tuple[float, ...],
    target_distribution: FloatArray | list[float] | tuple[float, ...],
    coupling_gradient: FloatArray | list[float] | tuple[float, ...] | None = None,
    *,
    max_step: float,
    knob: str = _DEFAULT_KNOB,
    scope: str = _DEFAULT_SCOPE,
) -> InformationGeometryControlProposal:
    """Compute a finite, deterministic information-geometry control proposal.

    Parameters are validated eagerly and no mutation of caller arrays is performed.
    """
    simplex = _normalise_simplex(current_distribution, "current_distribution")
    target = _normalise_simplex(target_distribution, "target_distribution")
    if simplex.shape != target.shape:
        raise ValueError("current_distribution and target_distribution must match")

    max_step_value = _as_finite_real(max_step, "max_step", allow_non_positive=False)
    knob = _as_non_empty_str(knob, "knob")
    scope = _as_non_empty_str(scope, "scope")

    fisher_rao_distance = _fisher_rao_distance(simplex, target)
    wasserstein_distance = _wasserstein_distance(simplex, target)
    metric_tensor = _fisher_information_metric(simplex)

    if coupling_gradient is None:
        objective_gradient = target - simplex
    else:
        objective_gradient = _validate_gradient(
            coupling_gradient,
            simplex.shape,
            "coupling_gradient",
        )

    natural_gradient = _natural_gradient_direction(
        objective_gradient, simplex, max_step_value
    )
    geodesic_length = float(fisher_rao_distance)
    curvature_proxy = _curvature_proxy(metric_tensor)

    action_value = float(
        np.clip(np.sum(natural_gradient), -max_step_value, max_step_value)
    )
    proposals = (
        ControlAction(
            knob=knob,
            scope=scope,
            value=action_value,
            ttl_s=float(max_step_value),
            justification="information-geometry review proposal",
        ),
    )

    state = InformationGeometryState(
        simplex_coordinates=simplex,
        target_coordinates=target,
        metric_tensor=metric_tensor,
        tangent_vector=natural_gradient,
        curvature_proxy=curvature_proxy,
        geodesic_length=geodesic_length,
    )

    proposal = InformationGeometryControlProposal(
        action_proposals=proposals,
        fisher_rao_distance=fisher_rao_distance,
        wasserstein_distance=wasserstein_distance,
        natural_gradient_norm=float(np.linalg.norm(natural_gradient)),
        curvature_proxy=curvature_proxy,
        backend=_BACKEND,
        claim_boundary=_BOUNDARY,
        non_actuating=True,
        execution_disabled=True,
        proposal_hash="",
        state=state,
    )
    proposal_hash = _compute_hash(proposal.to_audit_record())

    return InformationGeometryControlProposal(
        action_proposals=proposals,
        fisher_rao_distance=fisher_rao_distance,
        wasserstein_distance=wasserstein_distance,
        natural_gradient_norm=float(np.linalg.norm(natural_gradient)),
        curvature_proxy=curvature_proxy,
        backend=_BACKEND,
        claim_boundary=_BOUNDARY,
        non_actuating=True,
        execution_disabled=True,
        proposal_hash=proposal_hash,
        state=state,
    )


def _normalise_simplex(
    values: FloatArray | list[float] | tuple[float, ...], name: str
) -> FloatArray:
    array = _as_float_array(values, name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one element")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    mass = float(np.sum(array))
    if mass <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    normalised = array / mass
    return normalised.astype(np.float64, copy=True)


def _validate_gradient(
    values: FloatArray | list[float] | tuple[float, ...],
    expected_shape: tuple[int, ...],
    name: str,
) -> FloatArray:
    array = _as_float_array(values, name)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must match distribution shape")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values only")
    return array.astype(np.float64, copy=True)


def _as_float_array(
    values: FloatArray | list[float] | tuple[float, ...], name: str
) -> FloatArray:
    if isinstance(values, bool):
        raise ValueError(f"{name} must contain numeric values")
    array = np.asarray(values, dtype=np.float64)
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be an array-like of floats")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must contain numeric float-like values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return np.array(array, dtype=np.float64, copy=True)


def _as_finite_real(value: object, name: str, *, allow_non_positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if not allow_non_positive and number <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return number


def _as_non_empty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _fisher_rao_distance(simplex: FloatArray, target: FloatArray) -> float:
    overlap = float(np.sum(np.sqrt(simplex) * np.sqrt(target)))
    overlap = max(0.0, min(1.0, overlap))
    return 2.0 * float(acos(overlap))


def _wasserstein_distance(simplex: FloatArray, target: FloatArray) -> float:
    return float(np.sum(np.abs(np.cumsum(simplex) - np.cumsum(target))))


def _fisher_information_metric(simplex: FloatArray) -> FloatArray:
    metric = 1.0 / np.maximum(simplex, _EPS)
    return np.diag(metric.astype(np.float64, copy=False))


def _natural_gradient_direction(
    gradient: FloatArray,
    simplex: FloatArray,
    max_step: float,
) -> FloatArray:
    # Fisher metric inverse for the diagonal proxy is diag(simplex), hence natural
    # gradient is element-wise product with simplex coordinates.
    direction = gradient * simplex
    norm = float(np.linalg.norm(direction))
    if norm <= max_step or norm == 0.0:
        return direction
    scale = max_step / norm
    return direction * scale


def _curvature_proxy(metric_tensor: FloatArray) -> float:
    metric_diag = np.diag(metric_tensor)
    variance = float(np.var(metric_diag))
    magnitude = float(np.mean(metric_diag))
    return float(np.clip(np.sqrt(variance) / (1.0 + magnitude), 0.0, 1.0))


def _compute_hash(record: dict[str, object]) -> str:
    payload = dict(record)
    payload.pop("proposal_hash", None)
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()
