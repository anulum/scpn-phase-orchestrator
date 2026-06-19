# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for autopoietic lineage sandbox

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.supervisor.lineage import (
    _stable_hash,
    build_autopoietic_lineage_replay_corpus,
    build_autopoietic_lineage_sandbox,
    build_intergenerational_policy_inheritance,
    build_intergenerational_policy_inheritance_history,
)

_PARENT_POLICY = {"K": 0.42, "alpha": 0.18, "zeta": 0.09}
_REPLAYS = [{"replay_id": "nominal", "reward": 0.82, "safety_margin": 0.24}]


def _valid_lineage() -> dict[str, object]:
    return build_autopoietic_lineage_sandbox(
        _PARENT_POLICY,
        _REPLAYS,
        child_budget=2,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )


def _accepted_child(lineage: dict[str, object]) -> dict[str, object]:
    candidates = lineage["child_candidates"]
    assert isinstance(candidates, list)
    return next(c for c in candidates if c["status"] == "accepted_for_review")


def _inherit(
    lineage: dict[str, object], child: dict[str, object], **kwargs: object
) -> dict[str, object]:
    return build_intergenerational_policy_inheritance(
        lineage,
        child,
        signer_id="operator-review-key",
        signing_key="local-test-key",
        **kwargs,
    )


def _resigned(inheritance: dict[str, object], **changes: object) -> dict[str, object]:
    """Return an inheritance manifest with ``changes`` and a matching hash.

    The validator recomputes the body hash, so corrupting a post-hash field
    requires re-stamping ``inheritance_sha256`` to reach the targeted check.
    """
    mutated = {**inheritance, **changes}
    body = {k: v for k, v in mutated.items() if k != "inheritance_sha256"}
    mutated["inheritance_sha256"] = _stable_hash(body)
    return mutated


def test_autopoietic_lineage_sandbox_is_deterministic_and_review_only() -> None:
    parent_policy = {
        "K": 0.42,
        "alpha": 0.18,
        "zeta": 0.09,
    }
    audit_replays = build_autopoietic_lineage_replay_corpus()

    first = build_autopoietic_lineage_sandbox(
        parent_policy,
        audit_replays,
        child_budget=3,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    second = build_autopoietic_lineage_sandbox(
        parent_policy,
        audit_replays,
        child_budget=3,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )

    assert first == second
    assert first["schema"] == "scpn_autopoietic_lineage_sandbox_v1"
    assert first["child_budget"] == 3
    assert first["child_candidate_count"] == 3
    assert first["accepted_child_count"] >= 1
    assert first["review_required"] is True
    assert first["execution_disabled"] is True
    assert first["live_merge_permitted"] is False
    assert first["hot_patch_permitted"] is False
    assert first["actuation_permitted"] is False
    assert first["replay_corpus_count"] == len(audit_replays)
    assert first["replay_domain_count"] == 4
    assert first["replay_domains"] == (
        "cardiac_rhythm",
        "cyber_industrial",
        "power_grid",
        "traffic_flow",
    )
    assert len(str(first["lineage_sha256"])) == 64
    assert len(str(first["replay_corpus_sha256"])) == 64
    assert all(
        candidate["execution_disabled"] is True
        and candidate["hot_patch_permitted"] is False
        for candidate in first["child_candidates"]
    )
    assert all(
        candidate["policy_diff"]
        for candidate in first["child_candidates"]
        if candidate["status"] == "accepted_for_review"
    )


def test_autopoietic_lineage_replay_corpus_is_curated_and_deterministic() -> None:
    first = build_autopoietic_lineage_replay_corpus()
    second = build_autopoietic_lineage_replay_corpus()

    assert first == second
    assert len(first) == 4
    assert {row["domain"] for row in first} == {
        "cardiac_rhythm",
        "cyber_industrial",
        "power_grid",
        "traffic_flow",
    }
    assert all(row["reward"] >= 0.7 for row in first)
    assert all(row["safety_margin"] >= 0.1 for row in first)
    assert all(row["violations"] == [] for row in first)


def test_autopoietic_lineage_sandbox_rejects_unsafe_child_candidates() -> None:
    manifest = build_autopoietic_lineage_sandbox(
        {"K": 0.8, "alpha": 0.4},
        [
            {
                "replay_id": "unsafe_replay",
                "reward": 0.3,
                "safety_margin": 0.02,
                "violations": ["stl_margin_breach"],
            }
        ],
        child_budget=2,
        mutation_step=0.1,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )

    assert manifest["accepted_child_count"] == 0
    assert manifest["rejected_child_count"] == 2
    assert {candidate["status"] for candidate in manifest["child_candidates"]} == {
        "rejected"
    }
    assert all(
        "replay_reward_below_minimum" in candidate["blocked_reasons"]
        for candidate in manifest["child_candidates"]
    )
    assert all(
        "safety_margin_below_minimum" in candidate["blocked_reasons"]
        for candidate in manifest["child_candidates"]
    )
    assert all(
        "replay_violations_present" in candidate["blocked_reasons"]
        for candidate in manifest["child_candidates"]
    )


def test_autopoietic_lineage_sandbox_rejects_invalid_resource_bounds() -> None:
    with pytest.raises(ValueError, match="child_budget"):
        build_autopoietic_lineage_sandbox({"K": 0.1}, [], child_budget=0)

    with pytest.raises(ValueError, match="audit_replays"):
        build_autopoietic_lineage_sandbox({"K": 0.1}, [], child_budget=1)

    with pytest.raises(ValueError, match="numeric"):
        build_autopoietic_lineage_sandbox({"K": "high"}, [{"reward": 1.0}])

    with pytest.raises(ValueError, match="mutation_step"):
        build_autopoietic_lineage_sandbox(
            {"K": 0.1},
            [{"reward": 1.0}],
            mutation_step=0.0,
        )


def test_intergenerational_policy_inheritance_is_signed_and_review_only() -> None:
    lineage = build_autopoietic_lineage_sandbox(
        {"K": 0.42, "alpha": 0.18, "zeta": 0.09},
        [{"replay_id": "nominal", "reward": 0.82, "safety_margin": 0.24}],
        child_budget=2,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    child = lineage["child_candidates"][0]

    first = build_intergenerational_policy_inheritance(
        lineage,
        child,
        signer_id="operator-review-key",
        signing_key="local-test-key",
        objective_weights={"reward": 0.6, "safety": 0.3, "simplicity": 0.1},
    )
    second = build_intergenerational_policy_inheritance(
        lineage,
        child,
        signer_id="operator-review-key",
        signing_key="local-test-key",
        objective_weights={"reward": 0.6, "safety": 0.3, "simplicity": 0.1},
    )

    assert first == second
    assert first["schema"] == "scpn_intergenerational_policy_inheritance_v1"
    assert first["signed_metadata"]["signer_id"] == "operator-review-key"
    assert len(str(first["signed_metadata"]["signature_sha256"])) == 64
    assert len(str(first["inheritance_sha256"])) == 64
    assert first["inherited_policy_genome"]["K"] == pytest.approx(0.44)
    assert first["multi_objective_replay_fitness"]["fitness_score"] > 0.0
    assert first["hot_patch_review_required"] is True
    assert first["direct_hot_patch_permitted"] is False
    assert first["actuation_permitted"] is False
    assert first["merge_strategy"] == "reviewed_hot_patch_only"


def test_intergenerational_policy_inheritance_history_preserves_review_records() -> (
    None
):
    lineage = build_autopoietic_lineage_sandbox(
        {"K": 0.42, "alpha": 0.18, "zeta": 0.09},
        build_autopoietic_lineage_replay_corpus(),
        child_budget=2,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    inheritances = [
        build_intergenerational_policy_inheritance(
            lineage,
            child,
            signer_id="operator-review-key",
            signing_key="local-test-key",
            objective_weights={"reward": 0.6, "safety": 0.3, "simplicity": 0.1},
        )
        for child in lineage["child_candidates"]
        if child["status"] == "accepted_for_review"
    ]

    first = build_intergenerational_policy_inheritance_history(lineage, inheritances)
    second = build_intergenerational_policy_inheritance_history(lineage, inheritances)

    assert first == second
    assert first["schema"] == "scpn_intergenerational_policy_inheritance_history_v1"
    assert first["history_record_count"] == len(inheritances)
    assert first["signed_metadata_count"] == len(inheritances)
    assert first["replay_domain_count"] == 4
    assert first["direct_hot_patch_permitted"] is False
    assert first["actuation_permitted"] is False
    assert first["operator_review_required"] is True
    assert len(str(first["history_sha256"])) == 64
    assert [row["generation_index"] for row in first["child_rows"]] == [0, 1]
    assert all(row["fitness_score"] > 0.0 for row in first["child_rows"])


def test_intergenerational_policy_inheritance_rejects_unreviewed_children() -> None:
    lineage = build_autopoietic_lineage_sandbox(
        {"K": 0.42, "alpha": 0.18},
        [
            {
                "replay_id": "unsafe",
                "reward": 0.2,
                "safety_margin": 0.01,
                "violations": ["stl_margin_breach"],
            }
        ],
        child_budget=1,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )

    with pytest.raises(ValueError, match="accepted_for_review"):
        build_intergenerational_policy_inheritance(
            lineage,
            lineage["child_candidates"][0],
            signer_id="operator-review-key",
            signing_key="local-test-key",
        )


def test_intergenerational_policy_inheritance_rejects_bad_signature_inputs() -> None:
    lineage = build_autopoietic_lineage_sandbox(
        {"K": 0.42},
        [{"reward": 0.82, "safety_margin": 0.24}],
        child_budget=1,
        mutation_step=0.02,
    )

    with pytest.raises(ValueError, match="signer_id"):
        build_intergenerational_policy_inheritance(
            lineage,
            lineage["child_candidates"][0],
            signer_id="",
            signing_key="local-test-key",
        )

    with pytest.raises(ValueError, match="signing_key"):
        build_intergenerational_policy_inheritance(
            lineage,
            lineage["child_candidates"][0],
            signer_id="operator-review-key",
            signing_key="",
        )


@pytest.mark.parametrize(
    ("changes", "drop", "match"),
    [
        ({"schema": "wrong"}, None, "schema is unsupported"),
        (None, "lineage_sha256", "lineage_sha256 is required"),
        ({"live_merge_permitted": True}, None, "must disable live merge"),
        ({"execution_disabled": False}, None, "must disable execution"),
        ({"hot_patch_permitted": True}, None, "must disable hot patching"),
        ({"actuation_permitted": True}, None, "must disable actuation"),
        ({"child_candidates": []}, None, "child_candidates must be a non-empty list"),
        ({"child_candidates": "x"}, None, "child_candidates must be a non-empty list"),
    ],
)
def test_inheritance_rejects_corrupt_lineage_manifest(changes, drop, match) -> None:
    lineage = _valid_lineage()
    child = _accepted_child(lineage)
    corrupt = {k: v for k, v in lineage.items() if k != drop}
    if changes:
        corrupt.update(changes)
    with pytest.raises(ValueError, match=match):
        _inherit(corrupt, child)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"status": "rejected"}, "accepted_for_review"),
        ({"review_required": False}, "must require review"),
        ({"execution_disabled": False}, "must disable execution"),
        ({"live_merge_permitted": True}, "must disable live merge"),
        ({"hot_patch_permitted": True}, "must disable hot patching"),
        ({"actuation_permitted": True}, "must disable actuation"),
        ({"child_sha256": 123}, "child_sha256 is required"),
        ({"policy_diff": []}, "policy_diff must be a non-empty list"),
        ({"policy_diff": ["x"]}, r"policy_diff\[0\] must be a mapping"),
    ],
)
def test_inheritance_rejects_corrupt_child_candidate(changes, match) -> None:
    lineage = _valid_lineage()
    child = {**_accepted_child(lineage), **changes}
    with pytest.raises(ValueError, match=match):
        _inherit(lineage, child)


@pytest.mark.parametrize(
    ("diff_changes", "match"),
    [
        ({"knob": ""}, r"policy_diff\[0\].knob"),
        ({"parent_value": "x"}, r"parent_value must be numeric"),
        ({"child_value": float("nan")}, r"child_value must be finite"),
    ],
)
def test_inheritance_rejects_corrupt_policy_diff_entry(diff_changes, match) -> None:
    lineage = _valid_lineage()
    child = _accepted_child(lineage)
    first_diff = {**child["policy_diff"][0], **diff_changes}
    corrupt_child = {**child, "policy_diff": [first_diff]}
    with pytest.raises(ValueError, match=match):
        _inherit(lineage, corrupt_child)


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ("not-a-mapping", "objective_weights must be a mapping"),
        ({"reward": 0.6}, "must contain reward, safety, simplicity"),
        ({"reward": -0.1, "safety": 0.4, "simplicity": 0.1}, "must be non-negative"),
    ],
)
def test_inheritance_rejects_corrupt_objective_weights(weights, match) -> None:
    lineage = _valid_lineage()
    child = _accepted_child(lineage)
    with pytest.raises(ValueError, match=match):
        _inherit(lineage, child, objective_weights=weights)


def test_inheritance_reconstructs_parent_genome_without_explicit_field() -> None:
    """The parent genome is rebuilt from child policy diffs when absent."""
    lineage = _valid_lineage()
    child = _accepted_child(lineage)
    without_genome = {k: v for k, v in lineage.items() if k != "parent_policy_genome"}
    result = _inherit(without_genome, child)
    assert result["schema"] == "scpn_intergenerational_policy_inheritance_v1"
    assert result["inherited_policy_genome"]


def test_parent_genome_fallback_requires_diff_evidence() -> None:
    """A lineage with no parent genome and empty diffs is rejected."""
    lineage = _valid_lineage()
    child = _accepted_child(lineage)
    corrupt = {k: v for k, v in lineage.items() if k != "parent_policy_genome"}
    corrupt["child_candidates"] = [{"policy_diff": []}]
    with pytest.raises(ValueError, match="does not contain parent genome evidence"):
        _inherit(corrupt, child)


def _valid_history_inputs() -> tuple[dict[str, object], dict[str, object]]:
    lineage = _valid_lineage()
    child = _accepted_child(lineage)
    return lineage, _inherit(lineage, child)


@pytest.mark.parametrize(
    ("changes", "rehash", "match"),
    [
        ({"schema": "wrong"}, False, "schema is unsupported"),
        ({"lineage_sha256": "0" * 64}, False, "lineage_sha256 does not match"),
        ({"parent_policy_sha256": "0" * 64}, False, "parent_policy_sha256 does not"),
        ({"actuation_permitted": True}, False, "hash does not match content"),
        ({"hot_patch_review_required": False}, True, "must require hot patch review"),
        (
            {"direct_hot_patch_permitted": True},
            True,
            "must disable direct hot patching",
        ),
        ({"actuation_permitted": True}, True, "must disable actuation"),
        ({"merge_strategy": "auto"}, True, "merge strategy is unsupported"),
        ({"policy_diff": []}, True, "policy_diff must be non-empty"),
        ({"policy_diff": "x"}, True, "policy_diff must be a list"),
        ({"policy_diff": ["x"]}, True, "policy_diff entries must be mappings"),
    ],
)
def test_history_rejects_corrupt_inheritance_manifest(changes, rehash, match) -> None:
    lineage, inheritance = _valid_history_inputs()
    corrupt = (
        _resigned(inheritance, **changes) if rehash else {**inheritance, **changes}
    )
    with pytest.raises(ValueError, match=match):
        build_intergenerational_policy_inheritance_history(lineage, [corrupt])


def test_history_rejects_corrupt_signed_metadata() -> None:
    lineage, inheritance = _valid_history_inputs()
    bad_algorithm = _resigned(
        inheritance,
        signed_metadata={
            **inheritance["signed_metadata"],
            "signature_algorithm": "rsa",
        },
    )
    with pytest.raises(ValueError, match="signature algorithm is unsupported"):
        build_intergenerational_policy_inheritance_history(lineage, [bad_algorithm])

    bad_signature = _resigned(
        inheritance,
        signed_metadata={
            **inheritance["signed_metadata"],
            "signature_sha256": "NOTHEX",
        },
    )
    with pytest.raises(ValueError, match="must be lowercase SHA-256"):
        build_intergenerational_policy_inheritance_history(lineage, [bad_signature])


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("not-a-mapping", "lineage_manifest must be a mapping"),
    ],
)
def test_inheritance_rejects_non_mapping_lineage(value, match) -> None:
    with pytest.raises(ValueError, match=match):
        _inherit(value, {})


def test_inheritance_rejects_non_mapping_child() -> None:
    lineage = _valid_lineage()
    with pytest.raises(ValueError, match="child_candidate must be a mapping"):
        _inherit(lineage, "not-a-mapping")


def test_inheritance_rejects_non_mapping_parent_genome() -> None:
    lineage = _valid_lineage()
    child = _accepted_child(lineage)
    corrupt = {**lineage, "parent_policy_genome": "not-a-mapping"}
    with pytest.raises(ValueError, match="parent_policy_genome must be a mapping"):
        _inherit(corrupt, child)


@pytest.mark.parametrize(
    ("candidates", "match"),
    [
        (["not-a-mapping"], "lineage child candidate must be a mapping"),
        ([{"policy_diff": "x"}], "policy_diff must be a list"),
        ([{"policy_diff": ["x"]}], "lineage policy diff must be a mapping"),
    ],
)
def test_parent_genome_fallback_rejects_corrupt_candidates(candidates, match) -> None:
    lineage = _valid_lineage()
    child = _accepted_child(lineage)
    corrupt = {k: v for k, v in lineage.items() if k != "parent_policy_genome"}
    corrupt["child_candidates"] = candidates
    with pytest.raises(ValueError, match=match):
        _inherit(corrupt, child)


def test_history_rejects_non_mapping_inheritance() -> None:
    lineage = _valid_lineage()
    with pytest.raises(ValueError, match="inheritance_manifest must be a mapping"):
        build_intergenerational_policy_inheritance_history(lineage, ["not-a-mapping"])


@pytest.mark.parametrize(
    ("parent_policy", "replays", "match"),
    [
        ({}, _REPLAYS, "parent_policy must be a non-empty mapping"),
        ({"": 0.5}, _REPLAYS, "parent_policy keys must be non-empty strings"),
        ({"K": float("inf")}, _REPLAYS, "parent_policy values must be finite"),
        (_PARENT_POLICY, "not-a-sequence", "audit_replays must be a sequence"),
        (_PARENT_POLICY, [123], r"audit_replays\[0\] must be a mapping"),
        (_PARENT_POLICY, [{}], r"audit_replays\[0\].reward is required"),
        (
            _PARENT_POLICY,
            [{"reward": 0.8, "violations": [1]}],
            r"audit_replays\[0\].violations must be string list",
        ),
    ],
)
def test_sandbox_rejects_malformed_inputs(parent_policy, replays, match) -> None:
    with pytest.raises(ValueError, match=match):
        build_autopoietic_lineage_sandbox(parent_policy, replays)
