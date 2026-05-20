# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Autopoietic lineage sandbox

"""Review-only child-policy lineage manifests for replay sandboxes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

import numpy as np

__all__ = ["build_autopoietic_lineage_sandbox"]


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
        "replay_corpus_sha256": _stable_hash(replays),
        "child_budget": child_budget,
        "child_candidate_count": len(child_candidates),
        "accepted_child_count": accepted_child_count,
        "rejected_child_count": rejected_child_count,
        "minimum_replay_reward": min_reward,
        "minimum_safety_margin": min_margin,
        "mutation_step": step,
        "review_required": True,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "actuation_permitted": False,
        "replay_summary": replay_summary,
        "child_candidates": child_candidates,
    }
    manifest["lineage_sha256"] = _stable_hash(manifest)
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
        violations = copied.get("violations", [])
        if not isinstance(violations, list) or not all(
            isinstance(item, str) for item in violations
        ):
            raise ValueError(f"audit_replays[{index}].violations must be string list")
        replays.append(copied)
    return replays


def _replay_summary(replays: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rewards = [_finite_number(replay["reward"], "reward") for replay in replays]
    margins = [
        _finite_number(replay.get("safety_margin", 0.0), "safety_margin")
        for replay in replays
    ]
    violation_count = sum(len(replay.get("violations", [])) for replay in replays)
    return {
        "replay_count": len(replays),
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "mean_safety_margin": float(np.mean(margins)),
        "min_safety_margin": float(np.min(margins)),
        "violation_count": int(violation_count),
    }


def _child_candidate(
    parent: Mapping[str, float],
    *,
    index: int,
    mutation_step: float,
    replay_summary: Mapping[str, object],
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
        "live_merge_permitted": False,
        "actuation_permitted": False,
    }
    record["child_sha256"] = _stable_hash(record)
    return record


def _blocked_reasons(
    replay_summary: Mapping[str, object],
    *,
    minimum_replay_reward: float,
    minimum_safety_margin: float,
) -> list[str]:
    reasons: list[str] = []
    if float(replay_summary["mean_reward"]) < minimum_replay_reward:
        reasons.append("replay_reward_below_minimum")
    if float(replay_summary["min_safety_margin"]) < minimum_safety_margin:
        reasons.append("safety_margin_below_minimum")
    if int(replay_summary["violation_count"]) > 0:
        reasons.append("replay_violations_present")
    return reasons


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
