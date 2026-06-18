# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Multiverse counterfactual rollouts

"""Deterministic counterfactual branch rollouts over branch topologies.

The implementation runs vectorised NumPy or optional JAX trajectories for
multiple branch interventions in one pass and keeps a strict non-actuation
boundary. It is an upstream-safe simulation surface for research, policy
gating, and audit review.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.actuation.mapper import ControlAction

__all__ = [
    "MultiverseBranchRecord",
    "MultiverseBranchSpec",
    "MultiverseCounterfactualManifest",
    "simulate_multiverse_counterfactual_branches",
]

FloatArray: TypeAlias = NDArray[np.float64]

_NUMPY_BACKEND_NAME = "numpy_vectorized"
_JAX_BACKEND_NAME = "jax_vectorized"
_SUPPORTED_BACKENDS = frozenset({"jax", "numpy"})
_SUPPORTED_KNOBS = {"K", "alpha", "zeta", "Psi", "psi"}


@dataclass(frozen=True)
class MultiverseBranchSpec:
    """Declarative counterfactual branch intervention specification."""

    branch_id: str
    actions: tuple[ControlAction, ...]
    topology_mask: FloatArray | None = None

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe branch specification record."""
        return {
            "branch_id": self.branch_id,
            "actions": [
                {
                    "knob": action.knob,
                    "scope": action.scope,
                    "value": float(action.value),
                    "ttl_s": float(action.ttl_s),
                    "justification": action.justification,
                }
                for action in self.actions
            ],
            "topology_mask": None
            if self.topology_mask is None
            else self.topology_mask.tolist(),
        }


@dataclass(frozen=True)
class MultiverseBranchRecord:
    """Audit record for one counterfactual branch rollout."""

    branch_id: str
    branch_hash: str
    action_count: int
    action_labels: tuple[str, ...]
    topology_edge_count: int
    topology_scale: float
    final_R: float
    mean_R: float
    min_R: float
    max_R: float
    final_psi: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe branch rollout record."""
        return {
            "branch_id": self.branch_id,
            "branch_hash": self.branch_hash,
            "action_count": self.action_count,
            "action_labels": list(self.action_labels),
            "topology_edge_count": self.topology_edge_count,
            "topology_scale": self.topology_scale,
            "final_R": self.final_R,
            "mean_R": self.mean_R,
            "min_R": self.min_R,
            "max_R": self.max_R,
            "final_psi": self.final_psi,
        }


@dataclass(frozen=True)
class MultiverseCounterfactualManifest:
    """Audit manifest for a full multiverse counterfactual rollout."""

    schema_name: str
    schema_version: str
    branch_records: tuple[MultiverseBranchRecord, ...]
    branch_count: int
    horizon: int
    backend: str
    non_actuating: bool
    execution_disabled: bool
    claim_boundary: str
    manifest_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe multiverse rollout manifest."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "branch_count": self.branch_count,
            "horizon": self.horizon,
            "backend": self.backend,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "claim_boundary": self.claim_boundary,
            "branch_records": [
                record.to_audit_record() for record in self.branch_records
            ],
            "manifest_hash": self.manifest_hash,
        }


def _coerce_float_array(name: str, value: object) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError(f"{name} must be a real ndarray")
    if _contains_complex_alias(raw):
        raise ValueError(f"{name} must be a real ndarray")
    try:
        data = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real ndarray") from exc
    return data


def _require_finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be finite")
    finite = float(value)
    if not np.isfinite(finite):
        raise ValueError(f"{field} must be finite")
    return finite


def _contains_boolean_alias(value: object) -> bool:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _contains_complex_alias(value: object) -> bool:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)


def _require_positive_real(value: object, field: str) -> float:
    value_f = _require_finite_real(value, field)
    if value_f <= 0.0:
        raise ValueError(f"{field} must be positive")
    return value_f


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a positive integer")
    if int(value) < 1 or int(value) != value:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _require_non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a non-negative integer")
    if int(value) < 0 or int(value) != value:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _normalise_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise ValueError("backend must be one of: jax, numpy")
    backend_name = backend.strip().lower()
    if backend_name not in _SUPPORTED_BACKENDS:
        raise ValueError("backend must be one of: jax, numpy")
    return backend_name


def _load_jax_numpy() -> Any:
    try:
        import jax

        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]  # jax.config.update is untyped in jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("backend='jax' requested but JAX is not available") from exc
    return jnp


def _require_shape(name: str, array: FloatArray, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ValueError(f"{name}.shape={array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


def _require_zero_diagonal(name: str, array: FloatArray) -> None:
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.allclose(np.diag(array), 0.0, rtol=0.0, atol=0.0):
        raise ValueError(f"{name} diagonal must be zero")


def _stable_hash(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(blob.encode("utf-8")).hexdigest()


def _normalise_branch_specs(
    branch_specs: tuple[MultiverseBranchSpec, ...],
    branch_action_sets: tuple[Sequence[ControlAction], ...] | None,
    topology_masks: tuple[FloatArray, ...] | None,
    n_osc: int,
) -> tuple[MultiverseBranchSpec, ...]:
    if branch_specs:
        normalised = tuple(
            MultiverseBranchSpec(
                branch_id=spec.branch_id,
                actions=tuple(spec.actions),
                topology_mask=(
                    None
                    if spec.topology_mask is None
                    else _coerce_float_array(
                        f"branch[{spec.branch_id}].topology_mask", spec.topology_mask
                    )
                ),
            )
            for spec in branch_specs
        )
        if topology_masks is not None and len(topology_masks) != len(normalised):
            raise ValueError("topology_masks must align with provided branch_specs")
    elif branch_action_sets:
        if topology_masks is not None and len(topology_masks) != len(
            branch_action_sets
        ):
            raise ValueError("topology_masks must align with branch_action_sets")
        normalised = tuple(
            MultiverseBranchSpec(
                branch_id=f"branch_{index:03d}",
                actions=tuple(actions),
                topology_mask=(
                    None
                    if topology_masks is None
                    else _coerce_float_array(
                        f"branch[{index:03d}].topology_mask", topology_masks[index]
                    )
                ),
            )
            for index, actions in enumerate(branch_action_sets)
        )
    else:
        raise ValueError("at least one branch specification is required")

    if not normalised:
        raise ValueError("at least one branch specification is required")

    result: list[MultiverseBranchSpec] = []
    for index, spec in enumerate(normalised):
        if not isinstance(spec.branch_id, str) or not spec.branch_id.strip():
            raise ValueError(f"branch[{index}].branch_id must be a non-empty string")
        if spec.actions is None:
            raise ValueError(f"branch[{index}].actions must be provided")
        actions = tuple(spec.actions)
        for action in actions:
            if not isinstance(action, ControlAction):
                raise ValueError(
                    f"branch[{index}].actions must contain ControlAction instances"
                )
        if spec.topology_mask is not None:
            if spec.topology_mask.shape != (n_osc, n_osc):
                raise ValueError(
                    f"branch[{spec.branch_id}].topology_mask.shape="
                    f"{spec.topology_mask.shape}, expected ({n_osc}, {n_osc})"
                )
            if not np.all(np.isfinite(spec.topology_mask)):
                raise ValueError(
                    f"branch[{spec.branch_id}].topology_mask must contain finite values"
                )
            _require_zero_diagonal(
                f"branch[{spec.branch_id}].topology_mask", spec.topology_mask
            )
        result.append(
            MultiverseBranchSpec(
                branch_id=spec.branch_id,
                actions=actions,
                topology_mask=spec.topology_mask,
            )
        )
    return tuple(result)


def _apply_matrix_delta(
    matrix: FloatArray, scope: str, value: float, branch_id: str
) -> None:
    if scope == "global":
        matrix += value
        return
    if scope.startswith("oscillator_") or scope.startswith("layer_"):
        suffix = scope.removeprefix("oscillator_")
        if suffix == scope:
            suffix = scope.removeprefix("layer_")
        if not suffix.isdigit():
            raise ValueError(
                f"branch[{branch_id}] scope must be 'global', 'oscillator_<index>', "
                f"or 'layer_<index>'"
            )
        idx = int(suffix)
        if idx >= matrix.shape[0]:
            raise ValueError(f"branch[{branch_id}] scope index out of range")
        matrix[idx, :] += value
        matrix[:, idx] += value
        return
    raise ValueError(
        f"branch[{branch_id}] unsupported scope {scope!r} for matrix action"
    )


def _validate_action(branch_id: str, action: ControlAction) -> tuple[str, str, float]:
    if not isinstance(action.knob, str) or action.knob not in _SUPPORTED_KNOBS:
        raise ValueError(f"branch[{branch_id}] unsupported knob {action.knob!r}")
    if not isinstance(action.scope, str) or not action.scope.strip():
        raise ValueError(f"branch[{branch_id}] action scope must be a non-empty string")
    value = _require_finite_real(
        action.value, f"branch[{branch_id}].{action.knob}.value"
    )
    return action.knob, action.scope, value


def _off_diagonal_metrics(matrix: FloatArray) -> tuple[int, float]:
    off_diag = np.ones_like(matrix, dtype=bool)
    np.fill_diagonal(off_diag, False)
    return int(np.count_nonzero(matrix[off_diag] != 0.0)), float(
        np.abs(matrix[off_diag]).sum()
    )


def _apply_branch_actions(
    branch_id: str,
    baseline_k: FloatArray,
    baseline_alpha: FloatArray,
    baseline_zeta: float,
    baseline_psi: float,
    actions: tuple[ControlAction, ...],
    topology_mask: FloatArray | None,
) -> tuple[FloatArray, FloatArray, float, float, tuple[str, ...], int, float]:
    knm = np.array(baseline_k, copy=True, dtype=np.float64)
    alpha = np.array(baseline_alpha, copy=True, dtype=np.float64)
    zeta = float(baseline_zeta)
    psi = float(baseline_psi)

    labels: list[str] = []
    for action in actions:
        knob, scope, value = _validate_action(branch_id, action)
        labels.append(f"{knob}:{scope}:{value:g}")
        if knob == "K":
            _apply_matrix_delta(knm, scope, value, branch_id)
        elif knob == "alpha":
            _apply_matrix_delta(alpha, scope, value, branch_id)
        elif knob == "zeta":
            zeta += value
        else:
            psi = (psi + value) % TWO_PI

    np.fill_diagonal(knm, 0.0)
    np.fill_diagonal(alpha, 0.0)
    if topology_mask is not None:
        knm *= topology_mask
        alpha *= topology_mask
        edge_count, topology_scale = _off_diagonal_metrics(topology_mask)
    else:
        edge_count, topology_scale = _off_diagonal_metrics(knm)

    return (
        knm,
        alpha,
        float(zeta),
        float(psi),
        tuple(labels),
        edge_count,
        topology_scale,
    )


def _branch_hash(
    spec: MultiverseBranchSpec,
    action_count: int,
    topology_edge_count: int,
    topology_scale: float,
) -> str:
    payload: dict[str, Any] = spec.to_audit_record()
    payload["action_count"] = int(action_count)
    payload["topology_edge_count"] = int(topology_edge_count)
    payload["topology_scale"] = float(topology_scale)
    return _stable_hash(payload)


def _derivative(
    theta: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: FloatArray,
    psi: FloatArray,
) -> FloatArray:
    phase_diff = theta[:, None, :] - theta[:, :, None]
    coupling = np.sum(knm * np.sin(phase_diff - alpha), axis=2)
    coupling += omegas[None, :]
    active = zeta != 0.0
    if np.any(active):
        coupling[active] += zeta[active, None] * np.sin(
            psi[active, None] - theta[active]
        )
    return np.asarray(coupling, dtype=np.float64)


def _euler_step(
    theta: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: FloatArray,
    psi: FloatArray,
    dt: float,
) -> FloatArray:
    dtheta = _derivative(theta, omegas, knm, alpha, zeta, psi)
    result: FloatArray = np.asarray((theta + dt * dtheta) % TWO_PI, dtype=np.float64)
    return result


def _rk4_step(
    theta: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: FloatArray,
    psi: FloatArray,
    dt: float,
) -> FloatArray:
    k1 = _derivative(theta, omegas, knm, alpha, zeta, psi)
    k2 = _derivative(theta + 0.5 * dt * k1, omegas, knm, alpha, zeta, psi)
    k3 = _derivative(theta + 0.5 * dt * k2, omegas, knm, alpha, zeta, psi)
    k4 = _derivative(theta + dt * k3, omegas, knm, alpha, zeta, psi)
    result: FloatArray = np.asarray(
        (theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)) % TWO_PI,
        dtype=np.float64,
    )
    return result


def _order_parameters(theta: FloatArray) -> tuple[FloatArray, FloatArray]:
    z = np.exp(1j * theta)
    mean = np.mean(z, axis=1)
    R: FloatArray = np.asarray(np.abs(mean), dtype=np.float64)
    psi: FloatArray = np.asarray(np.angle(mean) % TWO_PI, dtype=np.float64)
    return R, psi


def _rollout_numpy(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: FloatArray,
    psi: FloatArray,
    *,
    horizon: int,
    dt: float,
    method: str,
) -> tuple[FloatArray, FloatArray]:
    branch_count, n_osc, _ = knm.shape
    theta = np.broadcast_to(phases[None, :], (branch_count, n_osc)).copy()
    R_traj = np.empty((horizon + 1, branch_count), dtype=np.float64)
    psi_traj = np.empty((horizon + 1, branch_count), dtype=np.float64)
    R_traj[0], psi_traj[0] = _order_parameters(theta)

    for step in range(horizon):
        if method == "euler":
            theta = _euler_step(
                theta=theta,
                omegas=omegas,
                knm=knm,
                alpha=alpha,
                zeta=zeta,
                psi=psi,
                dt=dt,
            )
        else:
            theta = _rk4_step(
                theta=theta,
                omegas=omegas,
                knm=knm,
                alpha=alpha,
                zeta=zeta,
                psi=psi,
                dt=dt,
            )
        R_traj[step + 1], psi_traj[step + 1] = _order_parameters(theta)
    return R_traj, psi_traj


def _rollout_jax(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: FloatArray,
    psi: FloatArray,
    *,
    horizon: int,
    dt: float,
    method: str,
) -> tuple[FloatArray, FloatArray]:
    jnp = _load_jax_numpy()
    branch_count, n_osc, _ = knm.shape
    theta = jnp.broadcast_to(
        jnp.asarray(phases, dtype=jnp.float64)[None, :],
        (branch_count, n_osc),
    )
    omegas_j = jnp.asarray(omegas, dtype=jnp.float64)
    knm_j = jnp.asarray(knm, dtype=jnp.float64)
    alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
    zeta_j = jnp.asarray(zeta, dtype=jnp.float64)
    psi_j = jnp.asarray(psi, dtype=jnp.float64)
    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    two_pi = jnp.asarray(TWO_PI, dtype=jnp.float64)

    def derivative(theta_j: Any) -> Any:
        phase_diff = theta_j[:, None, :] - theta_j[:, :, None]
        coupling = jnp.sum(knm_j * jnp.sin(phase_diff - alpha_j), axis=2)
        coupling = coupling + omegas_j[None, :]
        return coupling + zeta_j[:, None] * jnp.sin(psi_j[:, None] - theta_j)

    def euler_step(theta_j: Any) -> Any:
        return jnp.mod(theta_j + dt_j * derivative(theta_j), two_pi)

    def rk4_step(theta_j: Any) -> Any:
        k1 = derivative(theta_j)
        k2 = derivative(jnp.mod(theta_j + 0.5 * dt_j * k1, two_pi))
        k3 = derivative(jnp.mod(theta_j + 0.5 * dt_j * k2, two_pi))
        k4 = derivative(jnp.mod(theta_j + dt_j * k3, two_pi))
        return jnp.mod(
            theta_j + (dt_j / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            two_pi,
        )

    def order_parameters(theta_j: Any) -> tuple[Any, Any]:
        z = jnp.exp(1j * theta_j)
        mean = jnp.mean(z, axis=1)
        return jnp.abs(mean), jnp.mod(jnp.angle(mean), two_pi)

    R_values: list[Any] = []
    psi_values: list[Any] = []
    R0, psi0 = order_parameters(theta)
    R_values.append(R0)
    psi_values.append(psi0)
    for _ in range(horizon):
        theta = euler_step(theta) if method == "euler" else rk4_step(theta)
        R_step, psi_step = order_parameters(theta)
        R_values.append(R_step)
        psi_values.append(psi_step)

    return (
        np.asarray(jnp.stack(R_values), dtype=np.float64),
        np.asarray(jnp.stack(psi_values), dtype=np.float64),
    )


def simulate_multiverse_counterfactual_branches(
    phases: NDArray[np.float64],
    omegas: NDArray[np.float64],
    baseline_k: NDArray[np.float64],
    baseline_alpha: NDArray[np.float64],
    branch_specs: tuple[MultiverseBranchSpec, ...] = (),
    *,
    branch_action_sets: tuple[Sequence[ControlAction], ...] | None = None,
    topology_masks: tuple[FloatArray, ...] | None = None,
    baseline_zeta: float = 0.0,
    baseline_psi: float = 0.0,
    horizon: int = 20,
    dt: float = 0.01,
    method: str = "rk4",
    backend: str = "numpy",
) -> MultiverseCounterfactualManifest:
    """Run deterministic branch counterfactual rollouts without actuation."""
    phases_arr = _coerce_float_array("phases", phases)
    omegas_arr = _coerce_float_array("omegas", omegas)
    baseline_k_arr = _coerce_float_array("baseline_k", baseline_k)
    baseline_alpha_arr = _coerce_float_array("baseline_alpha", baseline_alpha)

    if phases_arr.ndim != 1 or omegas_arr.ndim != 1:
        raise ValueError("phases and omegas must be 1-D arrays")

    n_osc = len(phases_arr)
    _require_shape("phases", phases_arr, (n_osc,))
    _require_shape("omegas", omegas_arr, (n_osc,))
    _require_shape("baseline_k", baseline_k_arr, (n_osc, n_osc))
    _require_shape("baseline_alpha", baseline_alpha_arr, (n_osc, n_osc))
    _require_zero_diagonal("baseline_k", baseline_k_arr)
    _require_zero_diagonal("baseline_alpha", baseline_alpha_arr)

    horizon_i = _require_positive_int(horizon, "horizon")
    dt_f = _require_positive_real(dt, "dt")
    baseline_zeta_f = _require_finite_real(baseline_zeta, "baseline_zeta")
    baseline_psi_f = _require_finite_real(baseline_psi, "baseline_psi")
    backend_name = _normalise_backend(backend)
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'")

    normalised_specs = _normalise_branch_specs(
        branch_specs=tuple(branch_specs),
        branch_action_sets=(
            tuple(tuple(action_set) for action_set in branch_action_sets)
            if branch_action_sets is not None
            else None
        ),
        topology_masks=(None if topology_masks is None else tuple(topology_masks)),
        n_osc=n_osc,
    )
    branch_count = len(normalised_specs)

    if branch_count < 1:
        raise ValueError("at least one branch is required")

    knm_cube = np.empty((branch_count, n_osc, n_osc), dtype=np.float64)
    alpha_cube = np.empty((branch_count, n_osc, n_osc), dtype=np.float64)
    zeta_vec = np.empty(branch_count, dtype=np.float64)
    psi_vec = np.empty(branch_count, dtype=np.float64)
    action_labels: list[tuple[str, ...]] = []
    topology_edge_count: list[int] = []
    topology_scale: list[float] = []
    hashes: list[str] = []

    for index, spec in enumerate(normalised_specs):
        spec_knm, spec_alpha, spec_zeta, spec_psi, labels, edge_count, topo_scale = (
            _apply_branch_actions(
                branch_id=spec.branch_id,
                baseline_k=baseline_k_arr,
                baseline_alpha=baseline_alpha_arr,
                baseline_zeta=baseline_zeta_f,
                baseline_psi=baseline_psi_f,
                actions=spec.actions,
                topology_mask=spec.topology_mask,
            )
        )
        knm_cube[index] = spec_knm
        alpha_cube[index] = spec_alpha
        zeta_vec[index] = spec_zeta
        psi_vec[index] = spec_psi
        action_labels.append(labels)
        topology_edge_count.append(edge_count)
        topology_scale.append(topo_scale)
        hashes.append(
            _branch_hash(
                spec=spec,
                action_count=len(labels),
                topology_edge_count=edge_count,
                topology_scale=topo_scale,
            )
        )

    if backend_name == "jax":
        R_traj, psi_traj = _rollout_jax(
            phases=phases_arr,
            omegas=omegas_arr,
            knm=knm_cube,
            alpha=alpha_cube,
            zeta=zeta_vec,
            psi=psi_vec,
            horizon=horizon_i,
            dt=dt_f,
            method=method,
        )
        audit_backend = _JAX_BACKEND_NAME
    else:
        R_traj, psi_traj = _rollout_numpy(
            phases=phases_arr,
            omegas=omegas_arr,
            knm=knm_cube,
            alpha=alpha_cube,
            zeta=zeta_vec,
            psi=psi_vec,
            horizon=horizon_i,
            dt=dt_f,
            method=method,
        )
        audit_backend = _NUMPY_BACKEND_NAME

    if not np.all(np.isfinite(R_traj)) or not np.all(np.isfinite(psi_traj)):
        raise ValueError("rollout produced non-finite values")

    records = tuple(
        MultiverseBranchRecord(
            branch_id=spec.branch_id,
            branch_hash=hashes[index],
            action_count=_require_non_negative_int(len(spec.actions), "action_count"),
            action_labels=action_labels[index],
            topology_edge_count=topology_edge_count[index],
            topology_scale=topology_scale[index],
            final_R=float(R_traj[-1, index]),
            mean_R=float(np.mean(R_traj[:, index])),
            min_R=float(np.min(R_traj[:, index])),
            max_R=float(np.max(R_traj[:, index])),
            final_psi=float(psi_traj[-1, index]),
        )
        for index, spec in enumerate(normalised_specs)
    )

    manifest_payload: dict[str, Any] = {
        "schema_name": "multiverse_counterfactual_rollout",
        "schema_version": "0.1.0",
        "branch_count": branch_count,
        "horizon": horizon_i,
        "backend": audit_backend,
        "non_actuating": True,
        "execution_disabled": True,
        "claim_boundary": "counterfactual_branch_rollout_not_live_actuation",
        "branch_records": [record.to_audit_record() for record in records],
        "manifest_hash": "",
    }
    manifest_hash = _stable_hash(manifest_payload)

    return MultiverseCounterfactualManifest(
        schema_name="multiverse_counterfactual_rollout",
        schema_version="0.1.0",
        branch_records=records,
        branch_count=branch_count,
        horizon=horizon_i,
        backend=audit_backend,
        non_actuating=True,
        execution_disabled=True,
        claim_boundary="counterfactual_branch_rollout_not_live_actuation",
        manifest_hash=manifest_hash,
    )
