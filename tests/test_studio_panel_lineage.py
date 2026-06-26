# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio lineage panel contract tests

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from math import inf
from typing import Any, cast

import pytest

from scpn_phase_orchestrator import studio
from scpn_phase_orchestrator.supervisor import (
    build_autopoietic_lineage_replay_corpus,
    build_autopoietic_lineage_sandbox,
    build_intergenerational_policy_inheritance,
    build_intergenerational_policy_inheritance_history,
)


def _lineage() -> dict[str, object]:
    return build_autopoietic_lineage_sandbox(
        {"K": 0.42, "alpha": 0.18},
        build_autopoietic_lineage_replay_corpus(),
        child_budget=2,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )


def _history() -> dict[str, object]:
    lineage = _lineage()
    inheritances = [
        build_intergenerational_policy_inheritance(
            lineage,
            child,
            signer_id="studio-review-key",
            signing_key="studio-local-signing-key",
        )
        for child in cast(Sequence[Mapping[str, object]], lineage["child_candidates"])
        if child["status"] == "accepted_for_review"
    ]
    return build_intergenerational_policy_inheritance_history(lineage, inheritances)


def _with_path(
    record: Mapping[str, object],
    path: Sequence[str | int],
    value: object,
) -> dict[str, object]:
    updated = deepcopy(dict(record))
    cursor: Any = updated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return updated


def _with_history_hash(history: Mapping[str, object]) -> dict[str, object]:
    updated = deepcopy(dict(history))
    updated.pop("history_sha256", None)
    updated["history_sha256"] = sha256(
        json.dumps(updated, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return updated


def _bad_history(path: Sequence[str | int], value: object) -> dict[str, object]:
    return _with_history_hash(_with_path(_history(), path, value))


def test_lineage_panel_accepts_null_blocked_reasons_as_empty_review_evidence() -> None:
    lineage = _with_path(_lineage(), ("child_candidates", 0, "blocked_reasons"), None)

    panel = studio.build_autopoietic_lineage_studio_panel([lineage])

    assert panel["accepted_child_rows"][0]["blocked_reasons"] == ()


def test_lineage_panels_accept_real_supervisor_review_payloads() -> None:
    lineage = _lineage()
    history = _history()

    lineage_panel = studio.build_autopoietic_lineage_studio_panel([lineage])
    inheritance_panel = studio.build_intergenerational_inheritance_studio_panel(
        [history]
    )

    assert lineage_panel["panel_kind"] == "studio_autopoietic_lineage_panel"
    assert lineage_panel["manifest_count"] == 1
    assert lineage_panel["accepted_child_total"] == 2
    assert lineage_panel["rejected_child_total"] == 0
    assert lineage_panel["replay_domain_count"] == 4
    assert lineage_panel["actuation_permitted"] is False
    assert "control_actions" not in lineage_panel
    assert inheritance_panel["panel_kind"] == (
        "studio_intergenerational_inheritance_panel"
    )
    assert inheritance_panel["history_record_total"] == 2
    assert inheritance_panel["signed_metadata_total"] == 2
    assert inheritance_panel["fitness_range"] == {
        "minimum": 0.491,
        "maximum": 0.491,
    }
    assert inheritance_panel["direct_hot_patch_permitted"] is False
    assert "actions_to_apply" not in inheritance_panel


@pytest.mark.parametrize(
    ("manifests", "message"),
    [
        ("manifest", "lineage manifests must be a sequence"),
        ((object(),), "lineage manifests item 0 must be a mapping"),
        ((_with_path(_lineage(), ("schema",), "unsupported"),), "unsupported schema"),
        (
            (_with_path(_lineage(), ("parent_policy_sha256",), "A" * 64),),
            "parent_policy_sha256",
        ),
        ((_with_path(_lineage(), ("review_required",), False),), "review_required"),
        (
            (
                _lineage(),
                _with_path(_lineage(), ("parent_policy_sha256",), "b" * 64),
            ),
            "duplicates lineage_sha256",
        ),
        (
            (_with_path(_lineage(), ("child_candidate_count",), 1),),
            "child_candidate_count",
        ),
        (
            (_with_path(_lineage(), ("accepted_child_count",), 1),),
            "accepted_child_count",
        ),
        (
            (_with_path(_lineage(), ("rejected_child_count",), 1),),
            "rejected_child_count",
        ),
        ((_with_path(_lineage(), ("replay_corpus_count",), 3),), "replay_corpus_count"),
        ((_with_path(_lineage(), ("replay_domain_count",), 3),), "replay_domain_count"),
        ((_with_path(_lineage(), ("child_candidates",), ()),), "child candidates"),
        (
            (
                _with_path(
                    _lineage(),
                    ("child_candidates", 1, "child_sha256"),
                    cast(
                        Sequence[Mapping[str, object]], _lineage()["child_candidates"]
                    )[0]["child_sha256"],
                ),
            ),
            "duplicates child_sha256",
        ),
        (
            (
                _with_path(
                    _lineage(), ("child_candidates", 0, "status"), "pending_review"
                ),
            ),
            "unsupported status",
        ),
        (
            (
                _with_path(
                    _lineage(),
                    ("child_candidates", 0, "blocked_reasons"),
                    ("should_not_block",),
                ),
            ),
            "accepted child cannot have blocked reasons",
        ),
        (
            (
                {
                    **_with_path(
                        _lineage(), ("child_candidates", 0, "status"), "rejected"
                    ),
                    "accepted_child_count": 1,
                    "rejected_child_count": 1,
                },
            ),
            "rejected child must explain blocked reasons",
        ),
        (
            (_with_path(_lineage(), ("child_candidates", 0, "policy_diff"), ()),),
            "policy_diff rows",
        ),
        ((_with_path(_lineage(), ("replay_corpus",), ()),), "replay corpus rows"),
        (
            (_with_path(_lineage(), ("replay_corpus", 0, "domain"), ""),),
            "domain must be a non-empty string",
        ),
        (
            (
                _with_path(
                    _lineage(), ("child_candidates", 0, "blocked_reasons"), "bad"
                ),
            ),
            "blocked_reasons must be a sequence of strings",
        ),
        (
            (_with_path(_lineage(), ("replay_corpus", 0, "reward"), "high"),),
            "reward must be a finite real number",
        ),
        (
            (_with_path(_lineage(), ("replay_corpus", 0, "reward"), inf),),
            "reward must be finite",
        ),
        (
            (_with_path(_lineage(), ("replay_corpus", 0, "violation_count"), -1),),
            "violation_count must be a non-negative integer",
        ),
    ],
)
def test_autopoietic_lineage_panel_rejects_malformed_manifests(
    manifests: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        studio.build_autopoietic_lineage_studio_panel(
            cast("Sequence[Mapping[str, object]]", manifests)
        )


@pytest.mark.parametrize(
    ("histories", "message"),
    [
        (
            (_with_history_hash(_with_path(_history(), ("schema",), "unsupported")),),
            "unsupported schema",
        ),
        (
            (_history(), _with_path(_history(), ("lineage_sha256",), "b" * 64)),
            "duplicates history_sha256",
        ),
        ((_with_path(_history(), ("history_sha256",), "0" * 64),), "history_sha256"),
        ((_bad_history(("merge_strategy",), "direct_hot_patch"),), "merge_strategy"),
        ((_bad_history(("history_record_count",), 1),), "history_record_count"),
        ((_bad_history(("signed_metadata_count",), 1),), "signed_metadata_count"),
        ((_bad_history(("replay_domain_count",), 3),), "replay_domain_count"),
        ((_bad_history(("minimum_fitness_score",), 0.1),), "minimum_fitness_score"),
        ((_bad_history(("maximum_fitness_score",), 0.9),), "maximum_fitness_score"),
        ((_bad_history(("mean_fitness_score",), 0.1),), "mean_fitness_score"),
        ((_bad_history(("child_rows",), ()),), "child_rows"),
        (
            (
                _bad_history(
                    ("child_rows", 1, "inheritance_sha256"),
                    cast(Sequence[Mapping[str, object]], _history()["child_rows"])[0][
                        "inheritance_sha256"
                    ],
                ),
            ),
            "duplicates inheritance_sha256",
        ),
        ((_bad_history(("child_rows", 1, "generation_index"), 3),), "generation_index"),
        (
            (_bad_history(("child_rows", 0, "merge_strategy"), "direct_hot_patch"),),
            "merge_strategy",
        ),
    ],
)
def test_intergenerational_inheritance_panel_rejects_malformed_histories(
    histories: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        studio.build_intergenerational_inheritance_studio_panel(
            cast("Sequence[Mapping[str, object]]", histories)
        )
