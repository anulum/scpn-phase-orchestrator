# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Causal counterfactual supervisor rollouts

"""Causal graph learning and counterfactual supervisor rollout diagnostics.

The module estimates directed influence from traces and compares baseline UPDE
trajectories against parameter-intervention rollouts derived from proposed
control actions. Inputs are validated for finite dimensions before simulation;
action application mutates local copies of coupling and phase-lag matrices only.
Outputs are audit-ready records and attribution summaries, not live actuation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor._causal_baselines import (
    _causal_baseline_family,
    _coerce_float_array,
    _lagged_linear_effect,
    _validate_causal_trace,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

__all__ = [
    "CausalAttribution",
    "CausalGraphEstimate",
    "CausalInfluenceEdge",
    "CausalInterventionEngine",
    "CounterfactualRollout",
    "InterventionParameters",
    "build_temporal_causal_hypergraph_experiment",
    "learn_causal_graph",
]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class CausalAttribution:
    """Attribution summary derived from a paired counterfactual rollout.

    ``trajectory_consistency`` reports the fraction of the rollout horizon over
    which the per-step order-parameter delta holds the attributed effect's sign
    (or, for a ``neutral`` verdict, stays within the neutral band ``|delta| <=
    threshold``). It is a **deterministic** property of the single paired
    trajectory — how steadily the intervention pushes ``R`` in the attributed
    direction — not a statistical or frequentist confidence: the rollout draws no
    samples, so there is no sampling distribution and no p-value to report. A
    value near ``1.0`` means the effect is steady across the whole horizon; a
    lower value means the sign only settles late or oscillates. Effect magnitude
    lives in ``score``/``delta_R_final``/``delta_R_mean``, kept separate so a
    steady-but-small effect is not confused with a large-but-transient one.
    """

    effect: str
    trajectory_consistency: float
    score: float
    delta_R_final: float
    delta_R_mean: float
    threshold: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable attribution payload.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable attribution payload.
        """
        return {
            "effect": self.effect,
            "trajectory_consistency": self.trajectory_consistency,
            "score": self.score,
            "delta_R_final": self.delta_R_final,
            "delta_R_mean": self.delta_R_mean,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class CausalInfluenceEdge:
    """Signed directed influence estimate between live causal graph nodes."""

    source: str
    target: str
    weight: float
    confidence: float
    lag: int
    evidence: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable causal-edge payload.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable causal-edge payload.
        """
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "confidence": self.confidence,
            "lag": self.lag,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CausalGraphEstimate:
    """Audit-ready directed causal graph learned from traces and interventions."""

    nodes: tuple[str, ...]
    edges: tuple[CausalInfluenceEdge, ...]
    lag: int
    min_abs_weight: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable causal graph estimate.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable causal graph estimate.
        """
        return {
            "nodes": list(self.nodes),
            "edges": [edge.to_audit_record() for edge in self.edges],
            "lag": self.lag,
            "min_abs_weight": self.min_abs_weight,
        }


@dataclass(frozen=True)
class InterventionParameters:
    """UPDE parameters after applying a supervisor intervention."""

    knm: FloatArray
    alpha: FloatArray
    zeta: float
    psi: float


@dataclass(frozen=True)
class CounterfactualRollout:
    """Paired baseline/intervention rollout summary for audit logging."""

    baseline_R: list[float]
    intervention_R: list[float]
    baseline_psi: list[float]
    intervention_psi: list[float]
    delta_R_final: float
    delta_R_mean: float
    delta_psi_final: float
    actions: tuple[ControlAction, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable counterfactual audit payload.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable counterfactual audit payload.
        """
        return {
            "baseline_R": self.baseline_R,
            "intervention_R": self.intervention_R,
            "baseline_psi": self.baseline_psi,
            "intervention_psi": self.intervention_psi,
            "delta_R_final": self.delta_R_final,
            "delta_R_mean": self.delta_R_mean,
            "delta_psi_final": self.delta_psi_final,
            "actions": [
                {
                    "knob": action.knob,
                    "scope": action.scope,
                    "value": action.value,
                    "ttl_s": action.ttl_s,
                    "justification": action.justification,
                }
                for action in self.actions
            ],
        }

    def attribute(self, threshold: float = 1e-3) -> CausalAttribution:
        """Summarise whether the intervention caused a measurable R change.

        The verdict is decided by the trajectory-averaged effect ``score``; its
        steadiness across the horizon is reported separately as
        ``trajectory_consistency`` (see :class:`CausalAttribution`), an honest
        deterministic measure rather than a statistical confidence.

        Parameters
        ----------
        threshold : float
            Decision threshold.

        Returns
        -------
        CausalAttribution
            The causal attribution of the intervention.

        Raises
        ------
        ValueError
            If ``threshold`` is invalid.
        """
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold must be finite and non-negative")
        score = 0.5 * (self.delta_R_final + self.delta_R_mean)
        magnitude = abs(score)
        if magnitude <= threshold:
            effect = "neutral"
        elif score > 0.0:
            effect = "stabilising"
        else:
            effect = "destabilising"
        return CausalAttribution(
            effect=effect,
            trajectory_consistency=self._trajectory_consistency(
                score, effect, threshold
            ),
            score=float(score),
            delta_R_final=self.delta_R_final,
            delta_R_mean=self.delta_R_mean,
            threshold=threshold,
        )

    def _trajectory_consistency(
        self, score: float, effect: str, threshold: float
    ) -> float:
        """Return the fraction of the horizon consistent with the verdict.

        For a directional verdict this is the fraction of paired steps whose
        order-parameter delta shares the attributed sign; for a ``neutral``
        verdict it is the fraction of steps whose delta stays within the neutral
        band ``|delta| <= threshold``. Deterministic; an empty rollout yields
        ``0.0`` (no horizon offers evidence either way).
        """
        baseline = np.asarray(self.baseline_R, dtype=np.float64)
        intervention = np.asarray(self.intervention_R, dtype=np.float64)
        span = min(baseline.size, intervention.size)
        if span == 0:
            return 0.0
        deltas = intervention[:span] - baseline[:span]
        if effect == "neutral":
            agree = np.abs(deltas) <= threshold
        else:
            agree = np.sign(deltas) == (1.0 if score > 0.0 else -1.0)
        return float(np.count_nonzero(agree) / span)


class CausalInterventionEngine:
    """Counterfactual UPDE rollouts for supervisor actions.

    The engine answers the first causal supervision question: from the same
    state, what would the order-parameter trajectory look like with and
    without the proposed intervention?

    Parameters
    ----------
    n_oscillators : int
        Number of oscillators in the network.
    dt : float
        Integration timestep.
    horizon : int
        Number of rollout steps.
    method : str
        UPDE integration method.
    layer_membership : Mapping[str, Sequence[int]] or None
        Optional named layers with their member oscillator indices, enabling
        ``do(K, layer_<name>)`` interventions. ``"layer_<name>"`` (default) or
        ``"layer_<name>.within"`` perturbs the within-layer coupling sub-block;
        ``"layer_<name>.incident"`` perturbs every coupling incident to a layer
        member (the set generalisation of ``oscillator_``). Without it, any
        layer-scoped action is rejected.
    """

    def __init__(
        self,
        n_oscillators: int,
        dt: float,
        horizon: int = 20,
        method: str = "rk4",
        *,
        layer_membership: Mapping[str, Sequence[int]] | None = None,
    ):
        self._n = _require_positive_int(
            n_oscillators,
            type_message="n_oscillators must be an integer",
            range_message="n_oscillators must be >= 1",
        )
        self._dt = _require_positive_real(
            dt,
            message="dt must be finite and > 0",
        )
        self._horizon = _require_positive_int(
            horizon,
            type_message="horizon must be a positive integer",
            range_message="horizon must be >= 1",
        )
        self._method = method
        self._layer_membership = _validate_layer_membership(layer_membership, self._n)

    def evaluate_actions(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
        actions: list[ControlAction] | tuple[ControlAction, ...],
    ) -> CounterfactualRollout:
        """Compare no-action and intervened trajectories.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        alpha : FloatArray
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
        zeta : float
            External drive strength ``ζ``.
        psi : float
            External drive reference phase ``Ψ`` in radians.
        actions : list[ControlAction] | tuple[ControlAction, ...]
            The control actions to apply or assess.

        Returns
        -------
        CounterfactualRollout
            The counterfactual rollout comparing no-action and intervened trajectories.
        """
        phases, omegas, knm, alpha, zeta, psi = self._validate_inputs(
            phases,
            omegas,
            knm,
            alpha,
            zeta,
            psi,
        )
        action_tuple = tuple(actions)
        intervened = self.apply_actions(knm, alpha, zeta, psi, action_tuple)

        baseline_R, baseline_psi = self._rollout(phases, omegas, knm, alpha, zeta, psi)
        intervention_R, intervention_psi = self._rollout(
            phases,
            omegas,
            intervened.knm,
            intervened.alpha,
            intervened.zeta,
            intervened.psi,
        )

        baseline_arr = np.asarray(baseline_R, dtype=np.float64)
        intervention_arr = np.asarray(intervention_R, dtype=np.float64)
        return CounterfactualRollout(
            baseline_R=baseline_R,
            intervention_R=intervention_R,
            baseline_psi=baseline_psi,
            intervention_psi=intervention_psi,
            delta_R_final=float(intervention_arr[-1] - baseline_arr[-1]),
            delta_R_mean=float(np.mean(intervention_arr - baseline_arr)),
            delta_psi_final=_signed_phase_delta(
                intervention_psi[-1],
                baseline_psi[-1],
            ),
            actions=action_tuple,
        )

    def apply_actions(
        self,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
        actions: tuple[ControlAction, ...],
    ) -> InterventionParameters:
        """Apply supported supervisor actions to UPDE parameters.

        Parameters
        ----------
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        alpha : FloatArray
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
        zeta : float
            External drive strength ``ζ``.
        psi : float
            External drive reference phase ``Ψ`` in radians.
        actions : tuple[ControlAction, ...]
            The control actions to apply or assess.

        Returns
        -------
        InterventionParameters
            The UPDE parameters after applying the actions.

        Raises
        ------
        ValueError
            If an action is unsupported.
        """
        next_knm = np.array(_coerce_float_array("knm", knm), copy=True)
        next_alpha = np.array(_coerce_float_array("alpha", alpha), copy=True)
        if next_knm.shape != (self._n, self._n):
            raise ValueError(
                f"knm.shape={next_knm.shape}, expected {(self._n, self._n)}"
            )
        if next_alpha.shape != (self._n, self._n):
            raise ValueError(
                f"alpha.shape={next_alpha.shape}, expected {(self._n, self._n)}"
            )
        next_zeta = _require_finite_real(zeta, name="zeta")
        next_psi = _require_finite_real(psi, name="psi")

        for action in actions:
            action_value = _require_finite_real(action.value, name="action.value")
            if action.knob == "K":
                _apply_matrix_delta(
                    next_knm, action.scope, action_value, self._layer_membership
                )
            elif action.knob == "alpha":
                _apply_matrix_delta(
                    next_alpha, action.scope, action_value, self._layer_membership
                )
            elif action.knob == "zeta":
                next_zeta += action_value
            elif action.knob in {"Psi", "psi"}:
                next_psi = (next_psi + action_value) % TWO_PI
            else:
                msg = f"unsupported causal intervention knob {action.knob!r}"
                raise ValueError(msg)

        np.fill_diagonal(next_knm, 0.0)
        np.fill_diagonal(next_alpha, 0.0)
        return InterventionParameters(
            knm=next_knm,
            alpha=next_alpha,
            zeta=next_zeta,
            psi=next_psi,
        )

    def _rollout(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
    ) -> tuple[list[float], list[float]]:
        """Run the causal-intervention rollout."""
        engine = UPDEEngine(self._n, self._dt, method=self._method)
        theta = np.array(phases, dtype=np.float64, copy=True)
        r0, psi0 = compute_order_parameter(theta)
        r_values = [r0]
        psi_values = [psi0]
        for _ in range(self._horizon):
            theta = engine.step(theta, omegas, knm, zeta, psi, alpha)
            r_value, psi_value = compute_order_parameter(theta)
            r_values.append(r_value)
            psi_values.append(psi_value)
        return r_values, psi_values

    def _validate_inputs(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, float, float]:
        """Validate and normalise the causal-intervention inputs, else raise."""
        n = self._n
        checks = (
            ("phases", phases, (n,)),
            ("omegas", omegas, (n,)),
            ("knm", knm, (n, n)),
            ("alpha", alpha, (n, n)),
        )
        validated: list[FloatArray] = []
        for name, arr, expected in checks:
            data = _coerce_float_array(name, arr)
            if data.shape != expected:
                raise ValueError(f"{name}.shape={data.shape}, expected {expected}")
            if not np.all(np.isfinite(data)):
                raise ValueError(f"{name} contains NaN/Inf")
            validated.append(data)
        return (
            validated[0],
            validated[1],
            validated[2],
            validated[3],
            _require_finite_real(zeta, name="zeta"),
            _require_finite_real(psi, name="psi"),
        )


def learn_causal_graph(
    trace: dict[str, list[float]],
    rollouts: list[CounterfactualRollout] | tuple[CounterfactualRollout, ...] = (),
    *,
    lag: int = 1,
    min_abs_weight: float = 1e-6,
) -> CausalGraphEstimate:
    """Estimate a signed live causal graph from traces and interventions.

    Trace edges use lagged linear influence from ``source[t]`` to
    ``target[t + lag] - target[t]``. Intervention edges summarise paired
    counterfactual rollouts as explicit ``do(knob:scope) -> R`` effects. The
    estimate is intentionally lightweight and audit-first; it is not a formal
    do-calculus proof.

    Parameters
    ----------
    trace : dict[str, list[float]]
        Signal trace keyed by variable name, each a sequence of floats.
    rollouts : list[CounterfactualRollout] | tuple[CounterfactualRollout, ...]
        Counterfactual rollouts used to estimate causal edges.
    lag : int
        Lag in samples for the causal estimate.
    min_abs_weight : float
        Minimum absolute edge weight retained in the graph.

    Returns
    -------
    CausalGraphEstimate
        The estimated signed causal graph.

    Raises
    ------
    ValueError
        If trace or intervention values are non-finite, or the influence
        arithmetic cannot be represented as finite values.
    """
    trace_arrays = _validate_causal_trace(trace, lag, min_abs_weight)
    nodes = tuple(trace_arrays)
    edges: list[CausalInfluenceEdge] = []
    for source in nodes:
        source_values = trace_arrays[source]
        source_window = source_values[:-lag]
        for target in nodes:
            if source == target:
                continue
            target_values = trace_arrays[target]
            target_delta = target_values[lag:] - target_values[:-lag]
            weight, confidence = _lagged_linear_effect(source_window, target_delta)
            if abs(weight) >= min_abs_weight:
                edges.append(
                    CausalInfluenceEdge(
                        source=source,
                        target=target,
                        weight=weight,
                        confidence=confidence,
                        lag=lag,
                        evidence="lagged_trace",
                    )
                )

    intervention_nodes: list[str] = []
    for rollout in rollouts:
        if not all(
            np.isfinite(value)
            for value in (rollout.delta_R_mean, rollout.delta_R_final)
        ):
            raise ValueError("counterfactual causal effects must be finite")
        for action in rollout.actions:
            if not np.isfinite(action.value):
                raise ValueError("counterfactual causal action values must be finite")
            source = f"do({action.knob}:{action.scope})"
            intervention_nodes.append(source)
            effect_scale = action.value if action.value != 0.0 else 1.0
            weight = float(rollout.delta_R_mean / effect_scale)
            if not np.isfinite(weight):
                raise ValueError("counterfactual causal influence must be finite")
            if abs(weight) < min_abs_weight:
                continue
            threshold = max(min_abs_weight, 1e-12)
            effects = (abs(rollout.delta_R_mean), abs(rollout.delta_R_final))
            confidence = (
                1.0
                if max(effects) >= threshold
                else min(1.0, sum(value / threshold for value in effects))
            )
            edges.append(
                CausalInfluenceEdge(
                    source=source,
                    target="R",
                    weight=weight,
                    confidence=confidence,
                    lag=0,
                    evidence="counterfactual_rollout",
                )
            )

    all_nodes = tuple(dict.fromkeys((*nodes, *intervention_nodes, "R")))
    edges.sort(key=lambda edge: (edge.source, edge.target, edge.evidence))
    return CausalGraphEstimate(
        nodes=all_nodes,
        edges=tuple(edges),
        lag=lag,
        min_abs_weight=min_abs_weight,
    )


def build_temporal_causal_hypergraph_experiment(
    trace: dict[str, list[float]],
    candidate_hyperedges: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    lag: int = 1,
    min_abs_weight: float = 1e-6,
    required_baseline_margin: float = 0.0,
) -> dict[str, object]:
    """Build a research-only temporal-causal hypergraph experiment manifest.

    The manifest compares candidate time-symmetric hyperedges against a
    deterministic family of conventional causal baselines: lagged-linear graph
    edges, lagged Pearson correlation, lagged-delta correlation,
    Granger-style residual improvement, and target persistence. It never
    permits production claims, hot-patching, or actuation; baseline failure
    keeps all candidates blocked as research evidence only.

    Parameters
    ----------
    trace : dict[str, list[float]]
        Signal trace keyed by variable name, each a sequence of floats.
    candidate_hyperedges : list[dict[str, object]] | tuple[dict[str, object], ...]
        Candidate causal hyperedges to test.
    lag : int
        Lag in samples for the causal estimate.
    min_abs_weight : float
        Minimum absolute edge weight retained in the graph.
    required_baseline_margin : float
        Minimum baseline margin a hyperedge must beat.

    Returns
    -------
    dict[str, object]
        The temporal-causal hypergraph experiment manifest.

    Raises
    ------
    ValueError
        If the trace or candidate hyperedges are invalid.
    """
    if not np.isfinite(required_baseline_margin) or required_baseline_margin < 0.0:
        raise ValueError("required_baseline_margin must be finite and non-negative")
    baseline = learn_causal_graph(trace, lag=lag, min_abs_weight=min_abs_weight)
    candidates = _validated_temporal_hyperedges(candidate_hyperedges)
    baseline_family = _causal_baseline_family(
        trace,
        lag=lag,
        min_abs_weight=min_abs_weight,
        graph=baseline,
    )
    strongest_baseline = max(
        baseline_family,
        key=lambda record: (
            float(record["score"]),
            str(record["name"]),
        ),
    )
    baseline_score = float(strongest_baseline["score"])
    accepted: list[dict[str, object]] = []
    evaluated: list[dict[str, object]] = []
    for candidate in candidates:
        score = candidate["score"]
        if not isinstance(score, int | float) or isinstance(score, bool):
            raise ValueError("candidate score must be finite")
        advantage = float(score) - baseline_score
        record = {
            **candidate,
            "baseline_score": baseline_score,
            "baseline_margin": advantage,
            "accepted": advantage > required_baseline_margin,
        }
        evaluated.append(record)
        if record["accepted"]:
            accepted.append(record)

    blocked_reasons: list[str] = []
    baseline_beaten = bool(accepted)
    if not baseline_beaten:
        blocked_reasons.append("conventional_causal_baseline_not_beaten")
    manifest: dict[str, object] = {
        "schema": "scpn_temporal_causal_hypergraph_experiment_v1",
        "research_only": True,
        "production_claim_permitted": False,
        "hot_patch_permitted": False,
        "actuation_permitted": False,
        "baseline_beaten": baseline_beaten,
        "blocked_reasons": blocked_reasons,
        "required_baseline_margin": float(required_baseline_margin),
        "baseline": {
            "edge_count": len(baseline.edges),
            "node_count": len(baseline.nodes),
            "score": baseline_score,
            "strongest_baseline": strongest_baseline["name"],
            "baseline_family": baseline_family,
            "lag": baseline.lag,
            "min_abs_weight": baseline.min_abs_weight,
            "edges": [edge.to_audit_record() for edge in baseline.edges],
        },
        "candidate_hyperedge_count": len(evaluated),
        "accepted_hyperedge_count": len(accepted),
        "evaluated_hyperedges": evaluated,
        "accepted_hyperedges": accepted,
    }
    manifest["experiment_sha256"] = _stable_json_hash(manifest)
    return manifest


def _validate_layer_membership(
    layer_membership: Mapping[str, Sequence[int]] | None,
    n_oscillators: int,
) -> dict[str, tuple[int, ...]]:
    """Return a validated ``layer name -> oscillator indices`` mapping.

    Parameters
    ----------
    layer_membership : Mapping[str, Sequence[int]] or None
        Named layers with their member oscillator indices, or ``None`` for a
        network with no declared layers.
    n_oscillators : int
        The oscillator count the indices must fall within.

    Returns
    -------
    dict[str, tuple[int, ...]]
        The validated mapping (empty when ``layer_membership`` is ``None``).

    Raises
    ------
    ValueError
        If a layer name is not a non-empty string, a membership list is empty,
        or an index is not an integer in ``[0, n_oscillators)``.
    """
    if layer_membership is None:
        return {}
    validated: dict[str, tuple[int, ...]] = {}
    for name, members in layer_membership.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("layer name must be a non-empty string")
        indices: list[int] = []
        for member in members:
            if isinstance(member, bool) or not isinstance(member, Integral):
                raise ValueError(f"layer {name!r} membership indices must be integers")
            index = int(member)
            if not 0 <= index < n_oscillators:
                raise ValueError(
                    f"layer {name!r} membership index {index} out of range "
                    f"[0, {n_oscillators})"
                )
            indices.append(index)
        if not indices:
            raise ValueError(f"layer {name!r} must have at least one member")
        validated[name] = tuple(indices)
    return validated


def _apply_matrix_delta(
    matrix: FloatArray,
    scope: str,
    value: float,
    layer_membership: Mapping[str, tuple[int, ...]],
) -> None:
    """Apply an intervention delta to the coupling matrix in place per scope."""
    if not isinstance(scope, str):
        raise ValueError(f"unsupported causal intervention scope {scope!r}")
    if scope == "global":
        matrix += value
        return
    if scope.startswith("oscillator_"):
        suffix = scope.removeprefix("oscillator_")
        if not suffix.isdecimal():
            raise ValueError("oscillator-scoped causal intervention requires an index")
        idx = int(suffix)
        if idx >= matrix.shape[0]:
            raise ValueError("oscillator-scoped causal intervention index out of range")
        matrix[idx, :] += value
        matrix[:, idx] += value
        return
    if scope.startswith("layer_"):
        _apply_layer_delta(matrix, scope, value, layer_membership)
        return
    raise ValueError(f"unsupported causal intervention scope {scope!r}")


def _apply_layer_delta(
    matrix: FloatArray,
    scope: str,
    value: float,
    layer_membership: Mapping[str, tuple[int, ...]],
) -> None:
    """Apply a layer-scoped intervention delta in place.

    The scope ``"layer_<name>"`` (or ``"layer_<name>.within"``) adds *value* to
    the within-layer coupling sub-block — the couplings between members of the
    layer. The scope ``"layer_<name>.incident"`` instead adds *value* once to
    every coupling incident to any layer member (the set generalisation of the
    ``oscillator_`` scope, to which a single-member incident intervention
    reduces). The audit record keeps the full scope string, so which semantics
    applied is always recoverable.

    Raises
    ------
    ValueError
        If the mode is unknown, the layer name is empty, or the name is not a
        declared layer.
    """
    remainder = scope.removeprefix("layer_")
    name, _, mode = remainder.partition(".")
    mode = mode or "within"
    if mode not in {"within", "incident"}:
        raise ValueError(
            f"layer-scoped causal intervention mode must be 'within' or "
            f"'incident', got {mode!r}"
        )
    if not name:
        raise ValueError("layer-scoped causal intervention requires a layer name")
    if name not in layer_membership:
        raise ValueError(
            f"layer-scoped causal intervention requires layer membership for {name!r}"
        )
    members = list(layer_membership[name])
    if mode == "within":
        matrix[np.ix_(members, members)] += value
        return
    mask = np.zeros(matrix.shape, dtype=bool)
    mask[members, :] = True
    mask[:, members] = True
    matrix[mask] += value


def _signed_phase_delta(a: float, b: float) -> float:
    """Return the signed wrapped phase delta."""
    return float((a - b + np.pi) % TWO_PI - np.pi)


def _validated_temporal_hyperedges(
    candidate_hyperedges: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    """Return the validated temporal hyperedges, else raise."""
    if not isinstance(candidate_hyperedges, list | tuple) or not candidate_hyperedges:
        raise ValueError("candidate_hyperedges must be a non-empty sequence")
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(candidate_hyperedges):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate_hyperedges[{index}] must be a mapping")
        sources = candidate.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"candidate_hyperedges[{index}].sources must be a list")
        if not all(isinstance(source, str) and source for source in sources):
            raise ValueError("candidate_hyperedge sources must be non-empty strings")
        target = candidate.get("target")
        if not isinstance(target, str) or not target:
            raise ValueError(f"candidate_hyperedges[{index}].target is required")
        offsets = candidate.get("time_offsets")
        if not isinstance(offsets, list) or not offsets:
            raise ValueError(
                f"candidate_hyperedges[{index}].time_offsets must be a list"
            )
        if not all(
            isinstance(offset, int) and not isinstance(offset, bool)
            for offset in offsets
        ):
            raise ValueError("candidate_hyperedge time_offsets must be integers")
        score = _require_finite_real(candidate.get("score", 0.0), name="score")
        records.append(
            {
                "sources": list(sources),
                "target": target,
                "time_offsets": list(offsets),
                "score": float(score),
                "evidence": str(candidate.get("evidence", "temporal_hypergraph")),
            }
        )
    records.sort(
        key=lambda item: (item["target"], item["sources"], item["time_offsets"])
    )
    return records


def _stable_json_hash(payload: object) -> str:
    """Return the canonical-JSON SHA-256 hash of the value."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _require_positive_int(
    value: object,
    *,
    type_message: str,
    range_message: str,
) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(type_message)
    coerced = int(value)
    if coerced < 1:
        raise ValueError(range_message)
    return coerced


def _require_positive_real(value: object, *, message: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    coerced = float(value)
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(message)
    return coerced


def _require_finite_real(value: object, *, name: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be finite")
    return coerced
