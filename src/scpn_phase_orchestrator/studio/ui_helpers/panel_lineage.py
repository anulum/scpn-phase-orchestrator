# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio lineage review panels

"""Autopoietic and intergenerational lineage review panel builders."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from numbers import Real
from typing import cast

import numpy as np

_AUTOPOIETIC_LINEAGE_SCHEMA = "scpn_autopoietic_lineage_sandbox_v1"


_AUTOPOIETIC_LINEAGE_BOUNDARY = "autopoietic_lineage_sandbox_review_not_live_merge"


_INTERGENERATIONAL_HISTORY_SCHEMA = (
    "scpn_intergenerational_policy_inheritance_history_v1"
)


_INTERGENERATIONAL_HISTORY_BOUNDARY = (
    "intergenerational_inheritance_review_not_direct_hot_patch"
)


def build_autopoietic_lineage_studio_panel(
    manifests: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a passive Studio panel for autopoietic lineage sandbox review.

    Parameters
    ----------
    manifests : Sequence[Mapping[str, object]]
        The manifest records.

    Returns
    -------
    dict[str, object]
        A passive Studio panel for autopoietic lineage sandbox review.
    """
    normalised_manifests = _normalise_autopoietic_lineage_manifests(manifests)
    replay_corpus_rows = tuple(
        row
        for manifest in normalised_manifests
        for row in cast(tuple[dict[str, object], ...], manifest["replay_corpus"])
    )
    replay_domains = tuple(sorted({str(row["domain"]) for row in replay_corpus_rows}))
    child_rows = tuple(
        child
        for manifest in normalised_manifests
        for child in cast(tuple[dict[str, object], ...], manifest["child_candidates"])
    )
    accepted_child_rows = tuple(
        child for child in child_rows if child["status"] == "accepted_for_review"
    )
    rejected_child_rows = tuple(
        child for child in child_rows if child["status"] == "rejected"
    )

    return {
        "panel_kind": "studio_autopoietic_lineage_panel",
        "supervisor": "autopoietic_lineage_sandbox",
        "manifest_count": len(normalised_manifests),
        "claim_boundary": _AUTOPOIETIC_LINEAGE_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "operator_review_required": True,
        "hot_patch_permitted": False,
        "live_merge_permitted": False,
        "actuation_permitted": False,
        "lineage_manifests": normalised_manifests,
        "replay_corpus_rows": replay_corpus_rows,
        "replay_domains": replay_domains,
        "replay_domain_count": len(replay_domains),
        "child_candidate_total": len(child_rows),
        "accepted_child_total": len(accepted_child_rows),
        "rejected_child_total": len(rejected_child_rows),
        "accepted_child_rows": accepted_child_rows,
        "rejected_child_rows": rejected_child_rows,
        "operator_summary": (
            "autopoietic lineage review: "
            f"{len(normalised_manifests)} manifest(s), "
            f"{len(replay_domains)} replay domain(s), "
            f"{len(accepted_child_rows)} accepted child candidate(s)"
        ),
        "operator_action": (
            "render as offline lineage sandbox evidence only; compare replay "
            "domains, policy diffs, and blocked reasons before a separately "
            "reviewed inheritance workflow"
        ),
    }


def build_intergenerational_inheritance_studio_panel(
    histories: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a passive Studio panel for inheritance-history review.

    Parameters
    ----------
    histories : Sequence[Mapping[str, object]]
        Inheritance-history records.

    Returns
    -------
    dict[str, object]
        A passive Studio panel for inheritance-history review.
    """
    normalised_histories = _normalise_intergenerational_inheritance_histories(histories)
    child_rows = tuple(
        child
        for history in normalised_histories
        for child in cast(tuple[dict[str, object], ...], history["child_rows"])
    )
    replay_domains = tuple(
        sorted(
            {
                str(domain)
                for history in normalised_histories
                for domain in cast(tuple[str, ...], history["replay_domains"])
            }
        )
    )
    fitness_scores = tuple(
        float(cast(float, row["fitness_score"])) for row in child_rows
    )

    return {
        "panel_kind": "studio_intergenerational_inheritance_panel",
        "supervisor": "intergenerational_policy_inheritance",
        "history_count": len(normalised_histories),
        "claim_boundary": _INTERGENERATIONAL_HISTORY_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "operator_review_required": True,
        "direct_hot_patch_permitted": False,
        "hot_patch_permitted": False,
        "live_merge_permitted": False,
        "actuation_permitted": False,
        "histories": normalised_histories,
        "inheritance_child_rows": child_rows,
        "history_record_total": len(child_rows),
        "signed_metadata_total": sum(
            int(cast(int, history["signed_metadata_count"]))
            for history in normalised_histories
        ),
        "replay_domains": replay_domains,
        "replay_domain_count": len(replay_domains),
        "fitness_range": {
            "minimum": min(fitness_scores),
            "maximum": max(fitness_scores),
        },
        "operator_summary": (
            "intergenerational inheritance review: "
            f"{len(normalised_histories)} history package(s), "
            f"{len(child_rows)} signed child record(s), "
            f"{len(replay_domains)} replay domain(s)"
        ),
        "operator_action": (
            "render as signed inheritance-history evidence only; require "
            "separate operator approval before any reviewed hot-patch workflow"
        ),
    }


def _normalise_autopoietic_lineage_manifests(
    manifests: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate schema-tagged autopoietic lineage manifests for the panel."""
    rows = _autopoietic_lineage_mapping_sequence(manifests, "lineage manifests")
    normalised: list[dict[str, object]] = []
    seen_lineages: set[str] = set()
    for index, manifest in enumerate(rows):
        name = f"lineage manifest {index}"
        if manifest.get("schema") != _AUTOPOIETIC_LINEAGE_SCHEMA:
            raise ValueError(f"{name} has unsupported schema")
        lineage_sha256 = _autopoietic_lineage_sha(
            manifest.get("lineage_sha256"), f"{name} lineage_sha256"
        )
        if lineage_sha256 in seen_lineages:
            raise ValueError(f"{name} duplicates lineage_sha256")
        seen_lineages.add(lineage_sha256)
        _autopoietic_lineage_bool(manifest, "review_required", True, name)
        _autopoietic_lineage_bool(manifest, "execution_disabled", True, name)
        _autopoietic_lineage_bool(manifest, "live_merge_permitted", False, name)
        _autopoietic_lineage_bool(manifest, "hot_patch_permitted", False, name)
        _autopoietic_lineage_bool(manifest, "actuation_permitted", False, name)

        replay_corpus = _normalise_autopoietic_replay_corpus(
            manifest.get("replay_corpus"), name
        )
        replay_domains = tuple(sorted({str(row["domain"]) for row in replay_corpus}))
        children = _normalise_autopoietic_lineage_children(
            manifest.get("child_candidates"), name
        )
        accepted_count = sum(
            1 for child in children if child["status"] == "accepted_for_review"
        )
        rejected_count = sum(1 for child in children if child["status"] == "rejected")

        child_candidate_count = _autopoietic_lineage_int(
            manifest.get("child_candidate_count"), f"{name} child_candidate_count"
        )
        accepted_child_count = _autopoietic_lineage_int(
            manifest.get("accepted_child_count"), f"{name} accepted_child_count"
        )
        rejected_child_count = _autopoietic_lineage_int(
            manifest.get("rejected_child_count"), f"{name} rejected_child_count"
        )
        replay_corpus_count = _autopoietic_lineage_int(
            manifest.get("replay_corpus_count"), f"{name} replay_corpus_count"
        )
        replay_domain_count = _autopoietic_lineage_int(
            manifest.get("replay_domain_count"), f"{name} replay_domain_count"
        )
        if child_candidate_count != len(children):
            raise ValueError(f"{name} child_candidate_count does not match rows")
        if accepted_child_count != accepted_count:
            raise ValueError(f"{name} accepted_child_count does not match rows")
        if rejected_child_count != rejected_count:
            raise ValueError(f"{name} rejected_child_count does not match rows")
        if replay_corpus_count != len(replay_corpus):
            raise ValueError(f"{name} replay_corpus_count does not match rows")
        if replay_domain_count != len(replay_domains):
            raise ValueError(f"{name} replay_domain_count does not match rows")

        normalised.append(
            {
                "schema": _AUTOPOIETIC_LINEAGE_SCHEMA,
                "lineage_sha256": lineage_sha256,
                "parent_policy_sha256": _autopoietic_lineage_sha(
                    manifest.get("parent_policy_sha256"),
                    f"{name} parent_policy_sha256",
                ),
                "replay_corpus_sha256": _autopoietic_lineage_sha(
                    manifest.get("replay_corpus_sha256"),
                    f"{name} replay_corpus_sha256",
                ),
                "review_required": True,
                "execution_disabled": True,
                "live_merge_permitted": False,
                "hot_patch_permitted": False,
                "actuation_permitted": False,
                "child_candidate_count": child_candidate_count,
                "accepted_child_count": accepted_child_count,
                "rejected_child_count": rejected_child_count,
                "child_candidates": children,
                "replay_corpus_count": replay_corpus_count,
                "replay_domain_count": replay_domain_count,
                "replay_domains": replay_domains,
                "replay_corpus": replay_corpus,
                "lineage_summary": _autopoietic_lineage_text(
                    manifest.get("lineage_summary"), f"{name} lineage_summary"
                )
                if "lineage_summary" in manifest
                else "",
            }
        )
    return tuple(normalised)


def _normalise_intergenerational_inheritance_histories(
    histories: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate schema-tagged intergenerational inheritance histories."""
    rows = _autopoietic_lineage_mapping_sequence(histories, "inheritance histories")
    normalised: list[dict[str, object]] = []
    seen_histories: set[str] = set()
    for index, history in enumerate(rows):
        name = f"inheritance history {index}"
        if history.get("schema") != _INTERGENERATIONAL_HISTORY_SCHEMA:
            raise ValueError(f"{name} has unsupported schema")
        history_sha256 = _autopoietic_lineage_sha(
            history.get("history_sha256"), f"{name} history_sha256"
        )
        if history_sha256 in seen_histories:
            raise ValueError(f"{name} duplicates history_sha256")
        seen_histories.add(history_sha256)
        _intergenerational_history_hash_check(history, name, history_sha256)
        _autopoietic_lineage_bool(history, "hot_patch_review_required", True, name)
        _autopoietic_lineage_bool(history, "direct_hot_patch_permitted", False, name)
        _autopoietic_lineage_bool(history, "actuation_permitted", False, name)
        _autopoietic_lineage_bool(history, "operator_review_required", True, name)
        if history.get("merge_strategy") != "reviewed_hot_patch_only":
            raise ValueError(f"{name} merge_strategy is unsupported")

        child_rows = _normalise_intergenerational_child_rows(
            history.get("child_rows"), name
        )
        history_record_count = _autopoietic_lineage_int(
            history.get("history_record_count"), f"{name} history_record_count"
        )
        signed_metadata_count = _autopoietic_lineage_int(
            history.get("signed_metadata_count"), f"{name} signed_metadata_count"
        )
        replay_domain_count = _autopoietic_lineage_int(
            history.get("replay_domain_count"), f"{name} replay_domain_count"
        )
        replay_domains = _autopoietic_lineage_text_tuple(
            history.get("replay_domains"), f"{name} replay_domains"
        )
        if history_record_count != len(child_rows):
            raise ValueError(f"{name} history_record_count does not match rows")
        if signed_metadata_count != len(child_rows):
            raise ValueError(f"{name} signed_metadata_count does not match rows")
        if replay_domain_count != len(set(replay_domains)):
            raise ValueError(f"{name} replay_domain_count does not match rows")
        fitness_values = tuple(
            float(cast(float, row["fitness_score"])) for row in child_rows
        )
        minimum_fitness = _autopoietic_lineage_float(
            history.get("minimum_fitness_score"), f"{name} minimum_fitness_score"
        )
        maximum_fitness = _autopoietic_lineage_float(
            history.get("maximum_fitness_score"), f"{name} maximum_fitness_score"
        )
        mean_fitness = _autopoietic_lineage_float(
            history.get("mean_fitness_score"), f"{name} mean_fitness_score"
        )
        if not np.isclose(minimum_fitness, min(fitness_values)):
            raise ValueError(f"{name} minimum_fitness_score does not match rows")
        if not np.isclose(maximum_fitness, max(fitness_values)):
            raise ValueError(f"{name} maximum_fitness_score does not match rows")
        if not np.isclose(mean_fitness, float(np.mean(fitness_values))):
            raise ValueError(f"{name} mean_fitness_score does not match rows")

        normalised.append(
            {
                "schema": _INTERGENERATIONAL_HISTORY_SCHEMA,
                "history_sha256": history_sha256,
                "lineage_sha256": _autopoietic_lineage_sha(
                    history.get("lineage_sha256"), f"{name} lineage_sha256"
                ),
                "parent_policy_sha256": _autopoietic_lineage_sha(
                    history.get("parent_policy_sha256"),
                    f"{name} parent_policy_sha256",
                ),
                "history_record_count": history_record_count,
                "signed_metadata_count": signed_metadata_count,
                "replay_domain_count": replay_domain_count,
                "replay_domains": tuple(sorted(replay_domains)),
                "child_rows": child_rows,
                "minimum_fitness_score": minimum_fitness,
                "maximum_fitness_score": maximum_fitness,
                "mean_fitness_score": mean_fitness,
                "hot_patch_review_required": True,
                "direct_hot_patch_permitted": False,
                "merge_strategy": "reviewed_hot_patch_only",
                "actuation_permitted": False,
                "operator_review_required": True,
            }
        )
    return tuple(normalised)


def _normalise_intergenerational_child_rows(
    value: object,
    history_name: str,
) -> tuple[dict[str, object], ...]:
    """Validate the child rows recorded for one inheritance history."""
    rows = _autopoietic_lineage_mapping_sequence(value, f"{history_name} child_rows")
    if not rows:
        raise ValueError(f"{history_name} must contain child_rows")
    normalised: list[dict[str, object]] = []
    seen_inheritances: set[str] = set()
    for index, row in enumerate(rows):
        name = f"{history_name} child row {index}"
        inheritance_sha256 = _autopoietic_lineage_sha(
            row.get("inheritance_sha256"), f"{name} inheritance_sha256"
        )
        if inheritance_sha256 in seen_inheritances:
            raise ValueError(f"{name} duplicates inheritance_sha256")
        seen_inheritances.add(inheritance_sha256)
        generation_index = _autopoietic_lineage_int(
            row.get("generation_index"), f"{name} generation_index"
        )
        if generation_index != index:
            raise ValueError(f"{name} generation_index must be contiguous")
        _autopoietic_lineage_bool(row, "hot_patch_review_required", True, name)
        _autopoietic_lineage_bool(row, "direct_hot_patch_permitted", False, name)
        _autopoietic_lineage_bool(row, "actuation_permitted", False, name)
        if row.get("merge_strategy") != "reviewed_hot_patch_only":
            raise ValueError(f"{name} merge_strategy is unsupported")
        normalised.append(
            {
                "generation_index": generation_index,
                "inheritance_sha256": inheritance_sha256,
                "lineage_sha256": _autopoietic_lineage_sha(
                    row.get("lineage_sha256"), f"{name} lineage_sha256"
                ),
                "child_sha256": _autopoietic_lineage_sha(
                    row.get("child_sha256"), f"{name} child_sha256"
                ),
                "signer_id": _autopoietic_lineage_text(
                    row.get("signer_id"), f"{name} signer_id"
                ),
                "signature_sha256": _autopoietic_lineage_sha(
                    row.get("signature_sha256"), f"{name} signature_sha256"
                ),
                "policy_gene_count": _autopoietic_lineage_int(
                    row.get("policy_gene_count"), f"{name} policy_gene_count"
                ),
                "policy_diff_count": _autopoietic_lineage_int(
                    row.get("policy_diff_count"), f"{name} policy_diff_count"
                ),
                "fitness_score": _autopoietic_lineage_float(
                    row.get("fitness_score"), f"{name} fitness_score"
                ),
                "reward_component": _autopoietic_lineage_float(
                    row.get("reward_component"), f"{name} reward_component"
                ),
                "safety_component": _autopoietic_lineage_float(
                    row.get("safety_component"), f"{name} safety_component"
                ),
                "simplicity_component": _autopoietic_lineage_float(
                    row.get("simplicity_component"), f"{name} simplicity_component"
                ),
                "merge_strategy": "reviewed_hot_patch_only",
                "hot_patch_review_required": True,
                "direct_hot_patch_permitted": False,
                "actuation_permitted": False,
            }
        )
    return tuple(normalised)


def _intergenerational_history_hash_check(
    history: Mapping[str, object],
    name: str,
    history_sha256: str,
) -> None:
    """Assert the history's recorded hash matches its expected SHA-256 digest."""
    body = dict(history)
    body.pop("history_sha256", None)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    expected = sha256(canonical.encode("utf-8")).hexdigest()
    if expected != history_sha256:
        raise ValueError(f"{name} history_sha256 does not match content")


def _normalise_autopoietic_lineage_children(
    value: object,
    manifest_name: str,
) -> tuple[dict[str, object], ...]:
    """Validate the child candidates recorded in one lineage manifest."""
    children = _autopoietic_lineage_mapping_sequence(
        value, f"{manifest_name} child_candidates"
    )
    if not children:
        raise ValueError(f"{manifest_name} must contain child candidates")
    seen_children: set[str] = set()
    normalised: list[dict[str, object]] = []
    for index, child in enumerate(children):
        name = f"{manifest_name} child {index}"
        child_sha256 = _autopoietic_lineage_sha(
            child.get("child_sha256"), f"{name} child_sha256"
        )
        if child_sha256 in seen_children:
            raise ValueError(f"{name} duplicates child_sha256")
        seen_children.add(child_sha256)
        _autopoietic_lineage_bool(child, "review_required", True, name)
        _autopoietic_lineage_bool(child, "execution_disabled", True, name)
        _autopoietic_lineage_bool(child, "live_merge_permitted", False, name)
        _autopoietic_lineage_bool(child, "hot_patch_permitted", False, name)
        _autopoietic_lineage_bool(child, "actuation_permitted", False, name)
        status = _autopoietic_lineage_text(child.get("status"), f"{name} status")
        if status not in {"accepted_for_review", "rejected"}:
            raise ValueError(f"{name} has unsupported status")
        blocked_reasons = _autopoietic_lineage_text_tuple(
            child.get("blocked_reasons", ()), f"{name} blocked_reasons"
        )
        if status == "accepted_for_review" and blocked_reasons:
            raise ValueError(f"{name} accepted child cannot have blocked reasons")
        if status == "rejected" and not blocked_reasons:
            raise ValueError(f"{name} rejected child must explain blocked reasons")
        normalised.append(
            {
                "child_id": _autopoietic_lineage_text(
                    child.get("child_id"), f"{name} child_id"
                ),
                "child_sha256": child_sha256,
                "status": status,
                "review_required": True,
                "execution_disabled": True,
                "live_merge_permitted": False,
                "hot_patch_permitted": False,
                "actuation_permitted": False,
                "blocked_reasons": blocked_reasons,
                "policy_diff": _normalise_autopoietic_policy_diff(
                    child.get("policy_diff"), name
                ),
            }
        )
    return tuple(normalised)


def _normalise_autopoietic_policy_diff(
    value: object,
    child_name: str,
) -> tuple[dict[str, object], ...]:
    """Validate the policy-diff rows recorded for one lineage child."""
    diff_rows = _autopoietic_lineage_mapping_sequence(
        value, f"{child_name} policy_diff"
    )
    if not diff_rows:
        raise ValueError(f"{child_name} must contain policy_diff rows")
    normalised: list[dict[str, object]] = []
    for index, row in enumerate(diff_rows):
        name = f"{child_name} policy_diff {index}"
        normalised.append(
            {
                "knob": _autopoietic_lineage_text(row.get("knob"), f"{name} knob"),
                "parent_value": _autopoietic_lineage_float(
                    row.get("parent_value"), f"{name} parent_value"
                ),
                "child_value": _autopoietic_lineage_float(
                    row.get("child_value"), f"{name} child_value"
                ),
                "delta": _autopoietic_lineage_float(row.get("delta"), f"{name} delta"),
            }
        )
    return tuple(normalised)


def _normalise_autopoietic_replay_corpus(
    value: object,
    manifest_name: str,
) -> tuple[dict[str, object], ...]:
    """Validate the replay-corpus rows recorded in one lineage manifest."""
    rows = _autopoietic_lineage_mapping_sequence(
        value, f"{manifest_name} replay_corpus"
    )
    if not rows:
        raise ValueError(f"{manifest_name} must contain replay corpus rows")
    normalised: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        name = f"{manifest_name} replay {index}"
        normalised.append(
            {
                "replay_id": _autopoietic_lineage_text(
                    row.get("replay_id"), f"{name} replay_id"
                ),
                "domain": _autopoietic_lineage_text(
                    row.get("domain"), f"{name} domain"
                ),
                "scenario": _autopoietic_lineage_text(
                    row.get("scenario"), f"{name} scenario"
                ),
                "reward": _autopoietic_lineage_float(
                    row.get("reward"), f"{name} reward"
                ),
                "safety_margin": _autopoietic_lineage_float(
                    row.get("safety_margin"), f"{name} safety_margin"
                ),
                "violation_count": _autopoietic_lineage_int(
                    row.get("violation_count"), f"{name} violation_count"
                ),
            }
        )
    return tuple(normalised)


def _autopoietic_lineage_mapping_sequence(
    value: object,
    name: str,
) -> tuple[Mapping[str, object], ...]:
    """Return ``value`` as a tuple of mappings, else raise ``ValueError``."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    rows: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} item {index} must be a mapping")
        rows.append(item)
    return tuple(rows)


def _autopoietic_lineage_text(value: object, name: str) -> str:
    """Return ``value`` if it is a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _autopoietic_lineage_text_tuple(value: object, name: str) -> tuple[str, ...]:
    """Return an empty tuple for a null value, else a tuple of non-empty strings."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    return tuple(_autopoietic_lineage_text(item, f"{name} item") for item in value)


def _autopoietic_lineage_sha(value: object, name: str) -> str:
    """Return ``value`` if it is a lowercase 64-char SHA-256 digest, else raise."""
    text = _autopoietic_lineage_text(value, name)
    if (
        len(text) != 64
        or text.lower() != text
        or any(char not in "0123456789abcdef" for char in text)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _autopoietic_lineage_bool(
    mapping: Mapping[str, object],
    key: str,
    expected: bool,
    name: str,
) -> None:
    """Assert ``mapping[key]`` is exactly ``expected``, else raise ``ValueError``."""
    if mapping.get(key) is not expected:
        raise ValueError(f"{name} {key} must be {expected}")


def _autopoietic_lineage_float(value: object, name: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _autopoietic_lineage_int(value: object, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
