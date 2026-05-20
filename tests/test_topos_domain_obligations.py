# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topos obligation example tests

from __future__ import annotations

import json

from scpn_phase_orchestrator.binding.topos_examples import (
    BOUNDARY_TAG,
    build_topos_domain_obligation_examples,
)

EXPECTED_DOMAINS = ("power_grid", "cardiac_rhythm", "cyber_industrial")
REQUIRED_KEYS = {
    "domain",
    "symbolic_prompt",
    "binding_object_count",
    "policy_object_count",
    "obligation_names",
    "passed",
    "non_actuating",
    "proof_boundary",
    "example_hash",
}


def _is_json_safe_record(record: dict[str, object]) -> bool:
    try:
        json.dumps(record, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _domain_parts(domain: str) -> tuple[str, ...]:
    return tuple(part for part in domain.split("_") if part)


def test_build_topos_domain_obligation_examples_minimum_count_and_keys():
    examples = build_topos_domain_obligation_examples()

    assert isinstance(examples, tuple)
    assert len(examples) >= 3
    for example in examples:
        assert REQUIRED_KEYS.issubset(example.keys())


def test_examples_are_deterministic_hashes_and_payloads():
    first = build_topos_domain_obligation_examples()
    second = build_topos_domain_obligation_examples()

    assert first == second
    hashes_first = [example["example_hash"] for example in first]
    hashes_second = [example["example_hash"] for example in second]
    assert hashes_first == hashes_second
    assert len(set(hashes_first)) == len(hashes_first)


def test_examples_are_json_safe_records():
    examples = build_topos_domain_obligation_examples()

    for example in examples:
        assert _is_json_safe_record(example)


def test_examples_include_non_actuating_and_proof_boundary():
    examples = build_topos_domain_obligation_examples()

    for example in examples:
        assert example["non_actuating"] is True
        assert example["proof_boundary"] == BOUNDARY_TAG
        assert example["passed"] is True


def test_counts_and_domain_specific_obligation_names():
    examples = build_topos_domain_obligation_examples()

    for example in examples:
        assert isinstance(example["binding_object_count"], int)
        assert isinstance(example["policy_object_count"], int)
        assert example["binding_object_count"] > 0
        assert example["policy_object_count"] > 0

        obligations = example["obligation_names"]
        assert isinstance(obligations, list)
        assert obligations
        for obligation in obligations:
            assert isinstance(obligation, str)
            assert obligation.strip()

        domain_parts = _domain_parts(example["domain"])
        assert domain_parts
        assert any(
            any(part in obligation for obligation in obligations)
            for part in domain_parts
        ), f"obligation names are not domain specific: {example['domain']}"


def test_expected_domains_are_present():
    examples = build_topos_domain_obligation_examples()
    example_domains = {example["domain"] for example in examples}
    for expected in EXPECTED_DOMAINS:
        assert expected in example_domains
