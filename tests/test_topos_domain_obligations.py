# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topos obligation example tests

from __future__ import annotations

import dataclasses
import json
import time

import pytest

from scpn_phase_orchestrator.binding.semantic import compile_symbolic_binding
from scpn_phase_orchestrator.binding.topos_examples import (
    BOUNDARY_TAG,
    ToposDomainObligation,
    ToposProofObligation,
    build_topos_domain_obligation_examples,
)
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyAction,
    PolicyCondition,
    PolicyRule,
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


def test_non_finite_policy_numbers_are_rejected_from_example_hash():
    artifacts = compile_symbolic_binding(
        "A 1 layer power grid symbolic control prompt",
        name="topos_non_finite_policy_payload",
        oscillators_per_layer=1,
        dry_run_steps=1,
        retrieval_root=None,
        docs_root=None,
    )
    example = ToposDomainObligation(
        domain="power_grid",
        symbolic_prompt="A 1 layer power grid symbolic control prompt",
        binding_spec=artifacts.binding_spec,
        policy_rules=(
            PolicyRule(
                name="non_finite_action",
                regimes=["CRITICAL"],
                condition=PolicyCondition(
                    metric="R",
                    layer=0,
                    op="<",
                    threshold=0.5,
                ),
                actions=[
                    PolicyAction(
                        knob="K",
                        scope="global",
                        value=float("nan"),
                        ttl_s=1.0,
                    )
                ],
            ),
        ),
        obligations=(
            ToposProofObligation(
                name="power_grid_finite_policy_payload",
                description="Policy payloads must remain finite for stable hashes.",
            ),
        ),
        binding_object_count=1,
        policy_object_count=1,
        non_actuating=True,
        proof_boundary=BOUNDARY_TAG,
        passed=True,
    )

    with pytest.raises(ValueError, match="finite JSON"):
        example.to_audit_record()


def _valid_domain_obligation() -> ToposDomainObligation:
    """Build one fully valid example so error tests can mutate single fields."""
    artifacts = compile_symbolic_binding(
        "A 1 layer power grid symbolic control prompt",
        name="topos_domain_obligation_valid_base",
        oscillators_per_layer=1,
        dry_run_steps=1,
        retrieval_root=None,
        docs_root=None,
    )
    return ToposDomainObligation(
        domain="power_grid",
        symbolic_prompt="A 1 layer power grid symbolic control prompt",
        binding_spec=artifacts.binding_spec,
        policy_rules=(
            PolicyRule(
                name="grid_coherence_recovery",
                regimes=["CRITICAL"],
                condition=PolicyCondition(metric="R", layer=0, op="<", threshold=0.5),
                actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
            ),
        ),
        obligations=(
            ToposProofObligation(
                name="power_grid_coherence_guard",
                description="Maintain categorical coherence under load shifts.",
            ),
        ),
        binding_object_count=1,
        policy_object_count=1,
        non_actuating=True,
        proof_boundary=BOUNDARY_TAG,
        passed=True,
    )


def test_valid_domain_obligation_serialises_cleanly():
    record = _valid_domain_obligation().to_audit_record()

    assert REQUIRED_KEYS.issubset(record.keys())
    assert record["domain"] == "power_grid"
    assert record["non_actuating"] is True
    assert record["proof_boundary"] == BOUNDARY_TAG
    assert isinstance(record["example_hash"], str) and record["example_hash"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"domain": ""}, "example domain must be a non-empty string"),
        ({"domain": "   "}, "example domain must be a non-empty string"),
        ({"symbolic_prompt": 123}, "symbolic prompt must be a non-empty string"),
        ({"symbolic_prompt": "  "}, "symbolic prompt must be a non-empty string"),
        ({"binding_spec": None}, "binding spec must be a BindingSpec"),
        ({"policy_rules": ()}, "policy rules must be a non-empty tuple"),
        ({"obligations": ()}, "obligations must be a non-empty tuple"),
        (
            {"obligations": (1,)},
            "obligations must contain only ToposProofObligation values",
        ),
        (
            {"obligations": (ToposProofObligation(name="", description="blank name"),)},
            "obligation names must be non-empty",
        ),
        ({"proof_boundary": "wrong_boundary"}, "unexpected proof boundary"),
        (
            {"binding_object_count": 0},
            "binding_object_count must be a positive integer",
        ),
        (
            {"binding_object_count": 1.5},
            "binding_object_count must be a positive integer",
        ),
        ({"policy_object_count": 0}, "policy_object_count must be a positive integer"),
        (
            {"policy_object_count": 1.5},
            "policy_object_count must be a positive integer",
        ),
        ({"non_actuating": False}, "non_actuating must be True"),
        ({"passed": False}, "all obligations must be passed"),
    ],
)
def test_to_audit_record_rejects_invalid_fields(overrides, match):
    candidate = dataclasses.replace(_valid_domain_obligation(), **overrides)
    with pytest.raises(ValueError, match=match):
        candidate.to_audit_record()


def test_build_examples_rejects_non_dict_audit_record(monkeypatch):
    monkeypatch.setattr(
        ToposDomainObligation, "to_audit_record", lambda self: "not-a-dict"
    )
    with pytest.raises(ValueError, match="example manifest must be a dict"):
        build_topos_domain_obligation_examples()


def test_build_examples_rejects_blank_example_hash(monkeypatch):
    monkeypatch.setattr(
        ToposDomainObligation, "to_audit_record", lambda self: {"example_hash": ""}
    )
    with pytest.raises(ValueError, match="stable example_hash"):
        build_topos_domain_obligation_examples()


def test_build_domain_example_rejects_empty_policy_rules():
    from scpn_phase_orchestrator.binding.topos_examples import _build_domain_example

    with pytest.raises(ValueError, match="policy rules must be non-empty"):
        _build_domain_example(
            domain="power_grid",
            symbolic_prompt="A 1 layer power grid symbolic control prompt",
            compilation_name="topos_empty_policy_rules",
            oscillators_per_layer=1,
            dry_run_steps=1,
            policy_rules=(),
            obligations=(("coherence_guard", "Maintain coherence."),),
        )


def test_build_domain_example_raises_on_compilation_failure(monkeypatch):
    from scpn_phase_orchestrator.binding import topos_examples

    class _FailingArtifacts:
        validation_errors = ["layer count mismatch"]

    monkeypatch.setattr(
        topos_examples,
        "compile_symbolic_binding",
        lambda *args, **kwargs: _FailingArtifacts(),
    )

    with pytest.raises(ValueError, match="compilation for 'power_grid' failed"):
        topos_examples._build_domain_example(
            domain="power_grid",
            symbolic_prompt="A 1 layer power grid symbolic control prompt",
            compilation_name="topos_compilation_failure",
            oscillators_per_layer=1,
            dry_run_steps=1,
            policy_rules=(
                PolicyRule(
                    name="grid_coherence_recovery",
                    regimes=["CRITICAL"],
                    condition=PolicyCondition(
                        metric="R", layer=0, op="<", threshold=0.5
                    ),
                    actions=[
                        PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)
                    ],
                ),
            ),
            obligations=(("coherence_guard", "Maintain coherence."),),
        )


def test_build_examples_performance_budget():
    build_topos_domain_obligation_examples()  # warm import/compile caches

    def _build_once() -> tuple[float, int]:
        start = time.perf_counter()
        examples = build_topos_domain_obligation_examples()
        return time.perf_counter() - start, len(examples)

    # Take the best (minimum) build time over several runs. The fastest run
    # reflects achievable throughput and is robust to transient host load that
    # inflates a single measurement; a real algorithmic regression degrades
    # every run.
    timings = [_build_once() for _ in range(5)]
    elapsed = min(t for t, _ in timings)
    count = timings[0][1]

    assert count >= 3
    # Measured mean ~29 ms on the i5-11600K workstation (non-isolated functional
    # budget, not a published benchmark). 0.5 s ceiling guards against
    # algorithmic regression while tolerating a loaded host.
    assert elapsed < 0.5, f"example build regressed: {elapsed * 1000:.1f} ms"
