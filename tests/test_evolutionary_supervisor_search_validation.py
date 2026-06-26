# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Evolutionary supervisor validation contracts

"""Fail-closed validation contracts for offline evolutionary supervisor search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pytest

from scpn_phase_orchestrator.supervisor.evolutionary_search import (
    run_offline_evolutionary_supervisor_search,
)


def _parent_policy() -> Mapping[str, object]:
    return {"K": 0.42, "alpha": 0.18}


def _audit_replays() -> Sequence[Mapping[str, object]]:
    return [
        {
            "replay_id": "nominal",
            "reward": 0.86,
            "safety_margin": 0.11,
            "violations": [],
        }
    ]


def _trace() -> Mapping[str, Sequence[object]]:
    return {"R": [0.96, 0.97, 0.98]}


def test_rejects_non_mapping_parent_policy() -> None:
    with pytest.raises(ValueError, match="parent_policy must be a mapping"):
        run_offline_evolutionary_supervisor_search(
            cast(Mapping[str, object], object()),
            _audit_replays(),
            stl_spec="always (R >= 0.8)",
            trace=_trace(),
        )


def test_rejects_empty_or_non_string_parent_policy_keys() -> None:
    with pytest.raises(ValueError, match="parent_policy keys"):
        run_offline_evolutionary_supervisor_search(
            cast(Mapping[str, object], {"": 0.1}),
            _audit_replays(),
            stl_spec="always (R >= 0.8)",
            trace=_trace(),
        )

    with pytest.raises(ValueError, match="parent_policy keys"):
        run_offline_evolutionary_supervisor_search(
            cast(Mapping[str, object], {1: 0.1}),
            _audit_replays(),
            stl_spec="always (R >= 0.8)",
            trace=_trace(),
        )


def test_rejects_non_finite_parent_policy_value() -> None:
    with pytest.raises(ValueError, match=r"parent_policy\[K\] must be finite"):
        run_offline_evolutionary_supervisor_search(
            {"K": float("nan")},
            _audit_replays(),
            stl_spec="always (R >= 0.8)",
            trace=_trace(),
        )


@pytest.mark.parametrize(
    ("audit_replays", "match"),
    [
        (
            cast(Sequence[Mapping[str, object]], "not-replays"),
            "audit_replays must be a sequence of mappings",
        ),
        (
            cast(Sequence[Mapping[str, object]], [object()]),
            r"audit_replays\[0\] must be a mapping",
        ),
        (
            [{"reward": 0.5, "safety_margin": 0.1}],
            r"audit_replays\[0\]\.violations must be a sequence of strings",
        ),
        (
            [{"reward": 0.5, "safety_margin": 0.1, "violations": "bad"}],
            r"audit_replays\[0\]\.violations must be a sequence of strings",
        ),
        (
            [{"reward": 0.5, "safety_margin": 0.1, "violations": [1]}],
            r"audit_replays\[0\]\.violations must be a sequence of strings",
        ),
    ],
)
def test_rejects_malformed_replay_evidence(
    audit_replays: Sequence[Mapping[str, object]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_offline_evolutionary_supervisor_search(
            _parent_policy(),
            audit_replays,
            stl_spec="always (R >= 0.8)",
            trace=_trace(),
        )


def test_minimum_reward_and_safety_thresholds_block_candidates() -> None:
    report = run_offline_evolutionary_supervisor_search(
        _parent_policy(),
        _audit_replays(),
        stl_spec="always (R >= 0.8)",
        trace=_trace(),
        generation_count=1,
        population_size=2,
        mutation_step=0.01,
        minimum_replay_reward=0.9,
        minimum_safety_margin=0.2,
    )

    assert report.accepted_count == 0
    assert all(
        {
            "replay_reward_below_minimum",
            "safety_margin_below_minimum",
        }.issubset(candidate.blocked_reasons)
        for candidate in report.candidates
    )


@pytest.mark.parametrize(
    ("trace", "match"),
    [
        (
            cast(Mapping[str, Sequence[object]], object()),
            "trace must be a mapping of signal names to sequences",
        ),
        ({}, "trace must contain at least one signal"),
        (cast(Mapping[str, Sequence[object]], {"": [0.9]}), "trace signal names"),
        ({"R": "not-a-sequence"}, "trace values must be a sequence"),
        (
            cast(Mapping[str, Sequence[object]], {"R": object()}),
            "trace values must be a sequence",
        ),
        ({"R": []}, "trace signals must be non-empty"),
        ({"R": [0.9], "S": [0.9, 0.8]}, "all signals in trace must have equal length"),
        ({"R": [float("nan")]}, r"trace\[R\] must be finite"),
    ],
)
def test_rejects_malformed_stl_trace(
    trace: Mapping[str, Sequence[object]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_offline_evolutionary_supervisor_search(
            _parent_policy(),
            _audit_replays(),
            stl_spec="always (R >= 0.8)",
            trace=trace,
        )


def test_rejects_invalid_config_scalar_values() -> None:
    with pytest.raises(ValueError, match="mutation_step must be positive"):
        run_offline_evolutionary_supervisor_search(
            _parent_policy(),
            _audit_replays(),
            stl_spec="always (R >= 0.8)",
            trace=_trace(),
            mutation_step=0.0,
        )

    with pytest.raises(ValueError, match="generation_count must be a positive integer"):
        run_offline_evolutionary_supervisor_search(
            _parent_policy(),
            _audit_replays(),
            stl_spec="always (R >= 0.8)",
            trace=_trace(),
            generation_count=cast(int, True),
        )
