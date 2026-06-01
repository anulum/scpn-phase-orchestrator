# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Autopoietic lineage sandbox

"""Review-only child-policy lineage manifests for replay sandboxes."""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any, TypedDict, cast

import numpy as np

__all__ = [
    "build_autopoietic_lineage_replay_corpus",
    "build_autopoietic_lineage_sandbox",
    "build_intergenerational_policy_inheritance",
]


class _ReplaySummary(TypedDict):
    replay_count: int
    mean_reward: float
    min_reward: float
    mean_safety_margin: float
    min_safety_margin: float
    violation_count: int


def build_autopoietic_lineage_replay_corpus() -> tuple[dict[str, object], ...]:
    """Return a deterministic multi-domain replay corpus for lineage review.

    The corpus is intentionally offline and compact. It gives the lineage
    sandbox domain-diverse replay evidence without loading partner data,
    contacting services, or enabling any live merge path.
    """

    return (
        {
            "replay_id": "power_grid_frequency_recovery",
            "domain": "power_grid",
            "scenario": "frequency_recovery_after_load_step",
            "reward": 0.82,
            "safety_margin": 0.24,
            "violations": [],
        },
        {
            "replay_id": "cardiac_rhythm_pacing_recovery",
            "domain": "cardiac_rhythm",
            "scenario": "ventricular_pacing_recovery",
            "reward": 0.78,
            "safety_margin": 0.19,
            "violations": [],
        },
        {
            "replay_id": "traffic_flow_platoon_recovery",
            "domain": "traffic_flow",
            "scenario": "corridor_platoon_recovery",
            "reward": 0.75,
            "safety_margin": 0.16,
            "violations": [],
        },
        {
            "replay_id": "cyber_industrial_recontainment",
            "domain": "cyber_industrial",
            "scenario": "lateral_movement_recontainment",
            "reward": 0.73,
            "safety_margin": 0.14,
            "violations": [],
        },
    )


def build_autopoietic_lineage_sandbox(
    parent_policy: Mapping[str, object],
    audit_replays: Sequence[Mapping[str, object]],
    *,
    child_budget: int = 3,
    mutation_step: float = 0.02,
    minimum_replay_reward: float = 0.0,
    minimum_safety_margin: float = 0.0,
) -> dict[str, object]:
    """Build a deterministic offline child-policy lineage review manifest.

    The sandbox mutates a numeric parent-policy mapping into a bounded set of
    child candidates, evaluates each candidate only against supplied replay
    summaries, and emits reviewable policy diffs. It never permits live merge,
    hot patching, or actuation.
    """

    if not isinstance(child_budget, int) or isinstance(child_budget, bool):
        raise ValueError("child_budget must be a positive integer")
    if child_budget <= 0:
        raise ValueError("child_budget must be a positive integer")
    step = _finite_non_negative(mutation_step, "mutation_step")
    if step <= 0.0:
        raise ValueError("mutation_step must be positive")
    min_reward = _finite_non_negative(minimum_replay_reward, "minimum_replay_reward")
    min_margin = _finite_non_negative(minimum_safety_margin, "minimum_safety_margin")
    parent = _validated_policy(parent_policy)
    replays = _validated_replays(audit_replays)

    replay_summary = _replay_summary(replays)
    replay_corpus = _replay_corpus_rows(replays)
    replay_domains = tuple(sorted({str(row["domain"]) for row in replay_corpus}))
    child_candidates = [
        _child_candidate(
            parent,
            index=index,
            mutation_step=step,
            replay_summary=replay_summary,
            minimum_replay_reward=min_reward,
            minimum_safety_margin=min_margin,
        )
        for index in range(child_budget)
    ]
    accepted_child_count = sum(
        candidate["status"] == "accepted_for_review" for candidate in child_candidates
    )
    rejected_child_count = child_budget - accepted_child_count
    manifest: dict[str, object] = {
        "schema": "scpn_autopoietic_lineage_sandbox_v1",
        "parent_policy_sha256": _stable_hash(parent),
        "parent_policy_genome": parent,
        "replay_corpus_sha256": _stable_hash(replay_corpus),
        "child_budget": child_budget,
        "child_candidate_count": len(child_candidates),
        "accepted_child_count": accepted_child_count,
        "rejected_child_count": rejected_child_count,
        "minimum_replay_reward": min_reward,
        "minimum_safety_margin": min_margin,
        "mutation_step": step,
        "review_required": True,
        "execution_disabled": True,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "actuation_permitted": False,
        "replay_corpus": replay_corpus,
        "replay_corpus_count": len(replay_corpus),
        "replay_domain_count": len(replay_domains),
        "replay_domains": replay_domains,
        "replay_summary": replay_summary,
        "child_candidates": child_candidates,
    }
    manifest["lineage_sha256"] = _stable_hash(manifest)
    return manifest


def build_intergenerational_policy_inheritance(
    lineage_manifest: Mapping[str, object],
    child_candidate: Mapping[str, object],
    *,
    signer_id: str,
    signing_key: str,
    objective_weights: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build signed review metadata for inherited child-policy genomes.

    The resulting manifest materialises the inherited policy genome from a
    reviewed child diff, records replay-fitness components, and signs metadata
    for operator review. It does not permit direct hot patches or actuation.
    """

    lineage = _validated_lineage_manifest(lineage_manifest)
    child = _validated_review_child(child_candidate)
    signer = _non_empty_string(signer_id, "signer_id")
    key = _non_empty_string(signing_key, "signing_key")
    weights = _validated_objective_weights(objective_weights)
    parent_genome = _parent_genome_from_lineage(lineage)
    inherited_genome = dict(parent_genome)
    for diff in _policy_diff_items(child):
        inherited_genome[str(diff["knob"])] = _finite_number(
            diff["child_value"], "policy_diff.child_value"
        )
    fitness = _multi_objective_fitness(child, weights)
    signed_payload: dict[str, object] = {
        "lineage_sha256": str(lineage["lineage_sha256"]),
        "child_sha256": str(child["child_sha256"]),
        "inherited_policy_genome": inherited_genome,
        "multi_objective_replay_fitness": fitness,
    }
    metadata = {
        "signer_id": signer,
        "signature_algorithm": "hmac-sha256",
        "signature_sha256": _signature(signed_payload, signer=signer, key=key),
    }
    manifest: dict[str, object] = {
        "schema": "scpn_intergenerational_policy_inheritance_v1",
        "lineage_sha256": lineage["lineage_sha256"],
        "parent_policy_sha256": lineage["parent_policy_sha256"],
        "child_sha256": child["child_sha256"],
        "inherited_policy_genome": inherited_genome,
        "policy_diff": child["policy_diff"],
        "multi_objective_replay_fitness": fitness,
        "signed_metadata": metadata,
        "hot_patch_review_required": True,
        "direct_hot_patch_permitted": False,
        "merge_strategy": "reviewed_hot_patch_only",
        "actuation_permitted": False,
    }
    manifest["inheritance_sha256"] = _stable_hash(manifest)
    return manifest


def _validated_policy(policy: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(policy, Mapping) or not policy:
        raise ValueError("parent_policy must be a non-empty mapping")
    validated: dict[str, float] = {}
    for key, value in sorted(policy.items()):
        if not isinstance(key, str) or not key:
            raise ValueError("parent_policy keys must be non-empty strings")
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError("parent_policy values must be numeric")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("parent_policy values must be finite")
        validated[key] = numeric
    return validated


def _validated_replays(
    audit_replays: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(audit_replays, Sequence) or isinstance(audit_replays, str):
        raise ValueError("audit_replays must be a sequence of mappings")
    if not audit_replays:
        raise ValueError("audit_replays must contain at least one replay summary")
    replays: list[dict[str, object]] = []
    for index, replay in enumerate(audit_replays):
        if not isinstance(replay, Mapping):
            raise ValueError(f"audit_replays[{index}] must be a mapping")
        copied = dict(replay)
        if "reward" not in copied:
            raise ValueError(f"audit_replays[{index}].reward is required")
        _finite_number(copied["reward"], f"audit_replays[{index}].reward")
        if "safety_margin" in copied:
            _finite_number(
                copied["safety_margin"],
                f"audit_replays[{index}].safety_margin",
            )
        if "replay_id" in copied:
            _non_empty_string(copied["replay_id"], f"audit_replays[{index}].replay_id")
        if "domain" in copied:
            _non_empty_string(copied["domain"], f"audit_replays[{index}].domain")
        if "scenario" in copied:
            _non_empty_string(copied["scenario"], f"audit_replays[{index}].scenario")
        violations = copied.get("violations", [])
        if not isinstance(violations, list) or not all(
            isinstance(item, str) for item in violations
        ):
            raise ValueError(f"audit_replays[{index}].violations must be string list")
        replays.append(copied)
    return replays


def _replay_summary(replays: Sequence[Mapping[str, object]]) -> _ReplaySummary:
    rewards = [_finite_number(replay["reward"], "reward") for replay in replays]
    margins = [
        _finite_number(replay.get("safety_margin", 0.0), "safety_margin")
        for replay in replays
    ]
    violation_count = 0
    for replay in replays:
        violations = replay.get("violations", [])
        if not isinstance(violations, list):
            raise ValueError("violations must be a list")
        violation_count += len(violations)
    return {
        "replay_count": len(replays),
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "mean_safety_margin": float(np.mean(margins)),
        "min_safety_margin": float(np.min(margins)),
        "violation_count": int(violation_count),
    }


def _replay_corpus_rows(
    replays: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for index, replay in enumerate(replays):
        violations = replay.get("violations", [])
        if not isinstance(violations, list):
            raise ValueError("violations must be a list")
        rows.append(
            {
                "replay_id": str(replay.get("replay_id", f"replay_{index + 1:03d}")),
                "domain": str(replay.get("domain", "unspecified")),
                "scenario": str(replay.get("scenario", "unspecified")),
                "reward": _finite_number(replay["reward"], "reward"),
                "safety_margin": _finite_number(
                    replay.get("safety_margin", 0.0),
                    "safety_margin",
                ),
                "violation_count": len(violations),
            }
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                str(row["domain"]),
                str(row["scenario"]),
                str(row["replay_id"]),
            ),
        )
    )


def _child_candidate(
    parent: Mapping[str, float],
    *,
    index: int,
    mutation_step: float,
    replay_summary: _ReplaySummary,
    minimum_replay_reward: float,
    minimum_safety_margin: float,
) -> dict[str, object]:
    knob = sorted(parent)[index % len(parent)]
    direction = 1.0 if index % 2 == 0 else -1.0
    mutation = mutation_step * float(index + 1) * direction
    child_value = float(parent[knob] + mutation)
    policy_diff = [
        {
            "knob": knob,
            "parent_value": float(parent[knob]),
            "child_value": child_value,
            "delta": mutation,
        }
    ]
    blocked_reasons = _blocked_reasons(
        replay_summary,
        minimum_replay_reward=minimum_replay_reward,
        minimum_safety_margin=minimum_safety_margin,
    )
    status = "rejected" if blocked_reasons else "accepted_for_review"
    record = {
        "child_id": f"child_{index + 1:03d}",
        "status": status,
        "policy_diff": policy_diff,
        "replay_reward": replay_summary["mean_reward"],
        "minimum_replay_reward": minimum_replay_reward,
        "safety_margin": replay_summary["min_safety_margin"],
        "minimum_safety_margin": minimum_safety_margin,
        "blocked_reasons": blocked_reasons,
        "review_required": True,
        "execution_disabled": True,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "actuation_permitted": False,
    }
    record["child_sha256"] = _stable_hash(record)
    return record


def _blocked_reasons(
    replay_summary: _ReplaySummary,
    *,
    minimum_replay_reward: float,
    minimum_safety_margin: float,
) -> list[str]:
    reasons: list[str] = []
    if replay_summary["mean_reward"] < minimum_replay_reward:
        reasons.append("replay_reward_below_minimum")
    if replay_summary["min_safety_margin"] < minimum_safety_margin:
        reasons.append("safety_margin_below_minimum")
    if replay_summary["violation_count"] > 0:
        reasons.append("replay_violations_present")
    return reasons


def _policy_diff_items(child: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = child.get("policy_diff")
    if not isinstance(raw, list):
        raise ValueError("policy_diff must be a list")
    if not all(isinstance(item, Mapping) for item in raw):
        raise ValueError("policy_diff entries must be mappings")
    return cast(list[Mapping[str, object]], raw)


def _finite_number(value: object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _finite_non_negative(value: object, field: str) -> float:
    numeric = _finite_number(value, field)
    if numeric < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return numeric


def _stable_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _validated_lineage_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(manifest, Mapping):
        raise ValueError("lineage_manifest must be a mapping")
    data = dict(manifest)
    if data.get("schema") != "scpn_autopoietic_lineage_sandbox_v1":
        raise ValueError("lineage_manifest schema is unsupported")
    for key in ("lineage_sha256", "parent_policy_sha256", "child_candidates"):
        if key not in data:
            raise ValueError(f"lineage_manifest.{key} is required")
    if data.get("live_merge_permitted") is not False:
        raise ValueError("lineage_manifest must disable live merge")
    if data.get("execution_disabled") is not True:
        raise ValueError("lineage_manifest must disable execution")
    if data.get("hot_patch_permitted") is not False:
        raise ValueError("lineage_manifest must disable hot patching")
    if data.get("actuation_permitted") is not False:
        raise ValueError("lineage_manifest must disable actuation")
    candidates = data["child_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("lineage_manifest.child_candidates must be a non-empty list")
    return data


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _validated_review_child(candidate: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(candidate, Mapping):
        raise ValueError("child_candidate must be a mapping")
    child = dict(candidate)
    if child.get("status") != "accepted_for_review":
        raise ValueError("child_candidate must have status accepted_for_review")
    if child.get("review_required") is not True:
        raise ValueError("child_candidate must require review")
    if child.get("execution_disabled") is not True:
        raise ValueError("child_candidate must disable execution")
    if child.get("live_merge_permitted") is not False:
        raise ValueError("child_candidate must disable live merge")
    if child.get("hot_patch_permitted") is not False:
        raise ValueError("child_candidate must disable hot patching")
    if child.get("actuation_permitted") is not False:
        raise ValueError("child_candidate must disable actuation")
    if not isinstance(child.get("child_sha256"), str):
        raise ValueError("child_candidate.child_sha256 is required")
    diffs = child.get("policy_diff")
    if not isinstance(diffs, list) or not diffs:
        raise ValueError("child_candidate.policy_diff must be a non-empty list")
    for index, diff in enumerate(diffs):
        if not isinstance(diff, Mapping):
            raise ValueError(f"child_candidate.policy_diff[{index}] must be a mapping")
        _non_empty_string(diff.get("knob"), f"policy_diff[{index}].knob")
        _finite_number(diff.get("parent_value"), f"policy_diff[{index}].parent_value")
        _finite_number(diff.get("child_value"), f"policy_diff[{index}].child_value")
        _finite_number(diff.get("delta"), f"policy_diff[{index}].delta")
    return child


def _parent_genome_from_lineage(manifest: Mapping[str, object]) -> dict[str, float]:
    if "parent_policy_genome" in manifest:
        return _validated_policy(
            _require_mapping(manifest["parent_policy_genome"], "parent_policy_genome")
        )
    genome: dict[str, float] = {}
    candidates = manifest["child_candidates"]
    if not isinstance(candidates, list):
        raise ValueError("lineage_manifest.child_candidates must be a list")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("lineage child candidate must be a mapping")
        diffs = candidate.get("policy_diff")
        if not isinstance(diffs, list):
            raise ValueError("lineage child candidate policy_diff must be a list")
        for diff in diffs:
            if not isinstance(diff, Mapping):
                raise ValueError("lineage policy diff must be a mapping")
            knob = _non_empty_string(diff.get("knob"), "policy_diff.knob")
            genome[knob] = _finite_number(diff.get("parent_value"), "parent_value")
    if not genome:
        raise ValueError("lineage manifest does not contain parent genome evidence")
    return dict(sorted(genome.items()))


def _validated_objective_weights(
    objective_weights: Mapping[str, object] | None,
) -> dict[str, float]:
    if objective_weights is None:
        return {"reward": 0.5, "safety": 0.4, "simplicity": 0.1}
    if not isinstance(objective_weights, Mapping):
        raise ValueError("objective_weights must be a mapping")
    weights = {
        key: _finite_non_negative(value, f"objective_weights.{key}")
        for key, value in objective_weights.items()
        if isinstance(key, str) and key
    }
    required = {"reward", "safety", "simplicity"}
    if set(weights) != required:
        raise ValueError("objective_weights must contain reward, safety, simplicity")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("objective_weights total must be positive")
    return {key: weights[key] / total for key in sorted(weights)}


def _multi_objective_fitness(
    child: Mapping[str, object],
    weights: Mapping[str, float],
) -> dict[str, object]:
    reward = _finite_number(child.get("replay_reward"), "child.replay_reward")
    safety = _finite_number(child.get("safety_margin"), "child.safety_margin")
    diff_count = len(_policy_diff_items(child))
    simplicity = 1.0 / float(1 + diff_count)
    score = (
        reward * weights["reward"]
        + safety * weights["safety"]
        + simplicity * weights["simplicity"]
    )
    return {
        "reward_component": reward,
        "safety_component": safety,
        "simplicity_component": simplicity,
        "objective_weights": dict(weights),
        "fitness_score": float(score),
    }


def _signature(payload: Mapping[str, object], *, signer: str, key: str) -> str:
    body = json.dumps(
        {"payload": payload, "signer_id": signer},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hmac.new(key.encode("utf-8"), body.encode("utf-8"), sha256).hexdigest()


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value
