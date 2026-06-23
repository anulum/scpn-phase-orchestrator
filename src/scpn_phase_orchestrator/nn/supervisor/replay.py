# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor replay and manifest builders

"""Replay proposal, baseline report, and experiment manifest builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from ..functional import order_parameter
from ._shared import (
    _audit_record_from_object,
    _is_finite_number,
    _json_object,
    _non_negative_int,
    masked_order_parameter,
)
from ._types import (
    SupervisorBaselineReport,
    SupervisorCorpusReplayProposals,
    SupervisorExperimentManifest,
    SupervisorReplayProposal,
)
from .policy import supervisor_action_bound_penalty
from .projection import project_supervisor_action_for_audit

if TYPE_CHECKING:
    from ._types import (
        KuramotoSupervisorScenario,
        SupervisorAction,
        SupervisorScenarioCorpus,
    )
    from .policy import DifferentiableSupervisorPolicy


def build_supervisor_replay_proposal(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
    *,
    scenario_metadata: dict[str, Any] | None = None,
    previous_action: SupervisorAction | None = None,
    ttl_s: float = 5.0,
    max_ttl_s: float = 5.0,
    rate_limit_fraction: float = 1.0,
    include_layer_actions: bool = True,
    regime_churn_score: float | None = None,
    max_regime_churn: float | None = None,
) -> SupervisorReplayProposal:
    """Build a deterministic replay-only proposal from a neural supervisor.

    The proposal is an audit artefact only. It does not return
    ``ControlAction`` objects and carries ``actuation_permitted=False`` so that
    downstream replay/autotune surfaces can review it without enabling live
    actuation.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    scenario : KuramotoSupervisorScenario
        The Kuramoto supervisor scenario.
    scenario_metadata : dict[str, Any] | None
        Scenario metadata, or ``None``.
    previous_action : SupervisorAction | None
        The previous supervisor action, or ``None``.
    ttl_s : float
        Action time-to-live in seconds.
    max_ttl_s : float
        Maximum action time-to-live in seconds.
    rate_limit_fraction : float
        Maximum fractional change per step.
    include_layer_actions : bool
        Whether to include per-layer actions.
    regime_churn_score : float | None
        The regime-churn score, or ``None``.
    max_regime_churn : float | None
        Maximum allowed regime churn, or ``None``.

    Returns
    -------
    SupervisorReplayProposal
        The replay-only supervisor proposal.
    """
    metadata = _json_object(scenario_metadata, "scenario_metadata")
    action = policy(scenario)
    projection = project_supervisor_action_for_audit(
        action,
        policy.config,
        previous_action=previous_action,
        ttl_s=ttl_s,
        max_ttl_s=max_ttl_s,
        rate_limit_fraction=rate_limit_fraction,
        include_layer_actions=include_layer_actions,
        regime_churn_score=regime_churn_score,
        max_regime_churn=max_regime_churn,
    )
    metrics = {
        "current_R_global": float(order_parameter(scenario.phases)),
        "current_R_good": float(
            masked_order_parameter(scenario.phases, scenario.good_mask)
        ),
        "current_R_bad": float(
            masked_order_parameter(scenario.phases, scenario.bad_mask)
        ),
        "value_estimate": float(action.value_estimate),
        "bound_penalty": float(supervisor_action_bound_penalty(action, policy.config)),
    }
    scenario_summary = {
        "n_oscillators": int(scenario.phases.shape[0]),
        "dt": float(scenario.dt),
        "inner_steps": int(scenario.inner_steps),
        "horizon": int(scenario.horizon),
    }
    return SupervisorReplayProposal(
        action=action,
        projection=projection,
        scenario_summary=scenario_summary,
        scenario_metadata=metadata,
        metrics=metrics,
        actuation_permitted=False,
    )


def build_supervisor_corpus_replay_proposals(
    policy: DifferentiableSupervisorPolicy,
    corpus: SupervisorScenarioCorpus,
    *,
    previous_action: SupervisorAction | None = None,
    ttl_s: float = 5.0,
    max_ttl_s: float = 5.0,
    rate_limit_fraction: float = 1.0,
    include_layer_actions: bool = True,
    regime_churn_scores: tuple[float, ...] | None = None,
    max_regime_churn: float | None = None,
) -> SupervisorCorpusReplayProposals:
    """Build deterministic replay-only proposals for every corpus scenario.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    corpus : SupervisorScenarioCorpus
        The validated supervisor scenario corpus.
    previous_action : SupervisorAction | None
        The previous supervisor action, or ``None``.
    ttl_s : float
        Action time-to-live in seconds.
    max_ttl_s : float
        Maximum action time-to-live in seconds.
    rate_limit_fraction : float
        Maximum fractional change per step.
    include_layer_actions : bool
        Whether to include per-layer actions.
    regime_churn_scores : tuple[float, ...] | None
        Per-scenario regime-churn scores, or ``None``.
    max_regime_churn : float | None
        Maximum allowed regime churn, or ``None``.

    Returns
    -------
    SupervisorCorpusReplayProposals
        The replay-only proposals for every corpus scenario.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not corpus.scenarios:
        raise ValueError("scenario corpus requires at least one scenario")
    if len(corpus.metadata) != len(corpus.scenarios):
        raise ValueError("scenario corpus metadata must match scenario count")
    if regime_churn_scores is not None and len(regime_churn_scores) != len(
        corpus.scenarios
    ):
        raise ValueError("regime_churn_scores must match scenario count")

    proposals = []
    for index, (scenario, metadata) in enumerate(
        zip(corpus.scenarios, corpus.metadata, strict=True)
    ):
        regime_churn_score = (
            None if regime_churn_scores is None else regime_churn_scores[index]
        )
        proposal_metadata = _json_object(
            {**metadata, "corpus_index": index},
            f"corpus metadata {index}",
        )
        proposals.append(
            build_supervisor_replay_proposal(
                policy,
                scenario,
                scenario_metadata=proposal_metadata,
                previous_action=previous_action,
                ttl_s=ttl_s,
                max_ttl_s=max_ttl_s,
                rate_limit_fraction=rate_limit_fraction,
                include_layer_actions=include_layer_actions,
                regime_churn_score=regime_churn_score,
                max_regime_churn=max_regime_churn,
            )
        )
    return SupervisorCorpusReplayProposals(
        proposals=tuple(proposals),
        actuation_permitted=False,
    )


def build_supervisor_baseline_report(
    comparisons: Iterable[Any],
    *,
    report_label: str = "supervisor_baseline_report",
) -> SupervisorBaselineReport:
    """Aggregate already-generated supervisor comparison records for review.

    Parameters
    ----------
    comparisons : Iterable[Any]
        The supervisor comparison records.
    report_label : str
        Label for the baseline report.

    Returns
    -------
    SupervisorBaselineReport
        The aggregated baseline report.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not report_label:
        raise ValueError("report_label must not be empty")
    records = tuple(
        _comparison_record_from_object(comparison, f"comparison {index}")
        for index, comparison in enumerate(comparisons)
    )
    if not records:
        raise ValueError("baseline report requires at least one comparison")
    return SupervisorBaselineReport(
        comparisons=records,
        summary=_baseline_report_summary(records),
        report_label=report_label,
        actuation_permitted=False,
    )


def build_supervisor_experiment_manifest(
    baseline_report: SupervisorBaselineReport,
    *,
    command: str,
    git_sha: str,
    dependency_lock: Mapping[str, Any],
    device_info: Mapping[str, Any],
    seed_list: Iterable[int],
    config_json_path: str | None = None,
    metrics_jsonl_path: str | None = None,
    summary_table_path: str | None = None,
    checkpoint_manifest_path: str | None = None,
    plot_manifest_path: str | None = None,
) -> SupervisorExperimentManifest:
    """Build a reproducibility manifest for a supervisor baseline report.

    Parameters
    ----------
    baseline_report : SupervisorBaselineReport
        The aggregated baseline report.
    command : str
        The command line recorded with the run.
    git_sha : str
        Git commit SHA recorded with the run.
    dependency_lock : Mapping[str, Any]
        Dependency lock mapping recorded with the run.
    device_info : Mapping[str, Any]
        Device information recorded with the run.
    seed_list : Iterable[int]
        Seeds for the experiment runs.
    config_json_path : str | None
        Filesystem path to the config json, or ``None``.
    metrics_jsonl_path : str | None
        Filesystem path to the metrics jsonl, or ``None``.
    summary_table_path : str | None
        Filesystem path to the summary table, or ``None``.
    checkpoint_manifest_path : str | None
        Filesystem path to the checkpoint manifest, or ``None``.
    plot_manifest_path : str | None
        Filesystem path to the plot manifest, or ``None``.

    Returns
    -------
    SupervisorExperimentManifest
        The reproducibility manifest.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not command:
        raise ValueError("command must not be empty")
    if not git_sha:
        raise ValueError("git_sha must not be empty")
    seeds = tuple(
        _non_negative_int(seed, f"seed_list[{index}]")
        for index, seed in enumerate(seed_list)
    )
    if not seeds:
        raise ValueError("seed_list must contain at least one seed")
    lock_record = _json_object(dict(dependency_lock), "dependency_lock")
    if not lock_record:
        raise ValueError("dependency_lock must not be empty")
    device_record = _json_object(dict(device_info), "device_info")
    if not device_record:
        raise ValueError("device_info must not be empty")
    report_record = _json_object(
        baseline_report.to_audit_record(),
        "baseline_report",
    )
    if report_record.get("actuation_permitted") is not False:
        raise ValueError("baseline_report must be non-actuating")
    artifacts = _json_object(
        {
            "config_json_path": config_json_path,
            "metrics_jsonl_path": metrics_jsonl_path,
            "summary_table_path": summary_table_path,
            "checkpoint_manifest_path": checkpoint_manifest_path,
            "plot_manifest_path": plot_manifest_path,
        },
        "artifacts",
    )
    return SupervisorExperimentManifest(
        baseline_report=report_record,
        command=command,
        git_sha=git_sha,
        dependency_lock=lock_record,
        device_info=device_record,
        seed_list=seeds,
        artifacts=artifacts,
        actuation_permitted=False,
    )


def _comparison_record_from_object(value: Any, field: str) -> dict[str, Any]:
    """Return a validated comparison record from an object."""
    record = _audit_record_from_object(value, field)
    if record.get("actuation_permitted") is not False:
        raise ValueError(f"{field} must be non-actuating")
    proposal_type = record.get("proposal_type")
    if not isinstance(proposal_type, str):
        raise ValueError(f"{field} must include proposal_type")
    if not (
        proposal_type.startswith("differentiable_supervisor_")
        and (
            proposal_type.endswith("_baseline") or proposal_type.endswith("_comparison")
        )
    ):
        raise ValueError(f"{field} is not a supervisor comparison record")
    return record


def _baseline_report_summary(
    records: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Return a summary of the baseline replay report."""
    delta_rewards = []
    baseline_count = 0
    replay_count = 0
    proposal_types: list[str] = []
    for record in records:
        proposal_type = str(record["proposal_type"])
        proposal_types.append(proposal_type)
        if proposal_type.endswith("_baseline"):
            baseline_count += 1
        if proposal_type == "differentiable_supervisor_replay_comparison":
            replay_count += 1
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            delta_reward = metrics.get("delta_reward")
            if _is_finite_number(delta_reward):
                delta_rewards.append(float(delta_reward))
    summary: dict[str, Any] = {
        "comparison_count": len(records),
        "baseline_comparison_count": baseline_count,
        "replay_comparison_count": replay_count,
        "proposal_types": proposal_types,
    }
    if delta_rewards:
        summary["best_delta_reward"] = max(delta_rewards)
        summary["worst_delta_reward"] = min(delta_rewards)
        summary["mean_delta_reward"] = sum(delta_rewards) / len(delta_rewards)
    return summary
