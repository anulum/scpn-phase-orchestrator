# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio report-panel seal tests

"""The Topos, evolutionary and lineage panels recompute report seals.

Each report these panels render carries a SHA-256 of its own canonical JSON:
the Topos validation reports and the evolutionary search report over the
record without the seal field, the policy-DSL report over the record with it
blank, the lineage sandbox manifest over the record without it. The panels
checked only the digest format, so an edited report rendered under its
original seal.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import cast

import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.binding.semantic import compile_symbolic_binding
from scpn_phase_orchestrator.binding.topos_semantic import (
    validate_symbolic_binding_functor,
)
from scpn_phase_orchestrator.supervisor import (
    build_autopoietic_lineage_replay_corpus,
    build_autopoietic_lineage_sandbox,
)
from scpn_phase_orchestrator.supervisor.evolutionary_policy_dsl import (
    run_offline_evolutionary_policy_dsl_search,
)
from scpn_phase_orchestrator.supervisor.evolutionary_search import (
    run_offline_evolutionary_supervisor_search,
)
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyAction,
    PolicyCondition,
    PolicyRule,
)
from scpn_phase_orchestrator.supervisor.topos_policy import (
    validate_policy_composition_category,
)
from tests.sealing import seal

Record = dict[str, object]


def _copy(record: object) -> Record:
    """Return a mutable deep copy of a producer audit record."""
    return cast("Record", deepcopy(record))


def _symbolic_report() -> Record:
    """Return a production symbolic-binding functor report."""
    artifacts = compile_symbolic_binding(
        "1-layer Studio Topos review binding with deterministic evidence morphisms",
        name="studio_topos_seal_contract",
        oscillators_per_layer=1,
        dry_run_steps=1,
    )
    return _copy(validate_symbolic_binding_functor(artifacts).to_audit_record())


def _policy_report() -> Record:
    """Return a production policy-composition category report."""
    rule = PolicyRule(
        name="studio_topos_seal_guard",
        regimes=["DEGRADED"],
        condition=PolicyCondition(metric="R", layer=0, op="<", threshold=0.6),
        actions=[PolicyAction(knob="K", scope="global", value=0.05, ttl_s=3.0)],
    )
    return _copy(validate_policy_composition_category([rule]).to_audit_record())


def _search_report() -> Record:
    """Return a production offline evolutionary search report."""
    replays = [
        {
            "replay_id": "nominal",
            "reward": 0.92,
            "safety_margin": 0.08,
            "violations": [],
        },
        {
            "replay_id": "disturbance",
            "reward": 0.84,
            "safety_margin": 0.06,
            "violations": [],
        },
    ]
    report = run_offline_evolutionary_supervisor_search(
        {"K": 0.42, "alpha": 0.18, "zeta": 0.09},
        replays,
        stl_spec="always (R >= 0.82)",
        trace={"R": [0.91, 0.90, 0.89, 0.88]},
        generation_count=1,
        population_size=4,
        mutation_step=0.04,
        minimum_replay_reward=0.70,
        minimum_safety_margin=0.04,
    )
    return _copy(report.to_audit_record())


def _dsl_report() -> Record:
    """Return a production offline policy-DSL evolution report."""
    report = run_offline_evolutionary_policy_dsl_search(
        "rule grid_guard: if R < 0.90 and K > 0.10 then set K += 0.03\n"
        "rule recovery_guard: if R >= 0.20 then set K -= 0.02",
        generation_count=1,
        population_size=3,
        mutation_step=0.01,
    )
    return _copy(report.to_audit_record())


def _lineage_manifest() -> Record:
    """Return a production autopoietic lineage sandbox manifest."""
    manifest = build_autopoietic_lineage_sandbox(
        {"K": 0.42, "alpha": 0.18, "zeta": 0.09},
        build_autopoietic_lineage_replay_corpus(),
        child_budget=3,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    return _copy(manifest)


def _render_topos_symbolic(report: Record) -> dict[str, object]:
    return studio.build_topos_semantic_binding_studio_panel(
        [report], [_policy_report()]
    )


def _render_topos_policy(report: Record) -> dict[str, object]:
    return studio.build_topos_semantic_binding_studio_panel(
        [_symbolic_report()], [report]
    )


def _render_search(report: Record) -> dict[str, object]:
    return studio.build_evolutionary_supervisor_policy_search_studio_panel([report])


def _render_dsl(report: Record) -> dict[str, object]:
    return studio.build_evolutionary_supervisor_policy_search_studio_panel(
        [_search_report()], dsl_reports=[report]
    )


def _render_lineage(manifest: Record) -> dict[str, object]:
    return studio.build_autopoietic_lineage_studio_panel([manifest])


# (build, render, seal field, edited field, edited value, blank-field seal)
CASES = [
    pytest.param(
        _symbolic_report,
        _render_topos_symbolic,
        "report_hash",
        "passed",
        False,
        False,
        id="topos-symbolic",
    ),
    pytest.param(
        _policy_report,
        _render_topos_policy,
        "report_hash",
        "passed",
        False,
        False,
        id="topos-policy",
    ),
    pytest.param(
        _search_report,
        _render_search,
        "report_hash",
        "stl_spec",
        "always (R >= 0.10)",
        False,
        id="evolutionary-search",
    ),
    pytest.param(
        _dsl_report,
        _render_dsl,
        "report_hash",
        "mutation_step",
        0.5,
        True,
        id="evolutionary-dsl",
    ),
    pytest.param(
        _lineage_manifest,
        _render_lineage,
        "lineage_sha256",
        "mutation_step",
        0.5,
        False,
        id="lineage-manifest",
    ),
]

Build = Callable[[], Record]
Render = Callable[[Record], dict[str, object]]


@pytest.mark.parametrize(
    ("build", "render", "seal_field", "field", "value", "blanked"), CASES
)
def test_edited_report_is_refused(
    build: Build,
    render: Render,
    seal_field: str,
    field: str,
    value: object,
    blanked: bool,
) -> None:
    """A report edited after sealing no longer renders."""
    report = build()
    assert report[field] != value
    report[field] = value

    with pytest.raises(ValueError, match=f"{seal_field} does not match the record"):
        render(report)


@pytest.mark.parametrize(
    ("build", "render", "seal_field", "field", "value", "blanked"), CASES
)
def test_resealed_report_renders(
    build: Build,
    render: Render,
    seal_field: str,
    field: str,
    value: object,
    blanked: bool,
) -> None:
    """The same edit, resealed with the producer's rule, renders."""
    report = build()
    report[field] = value

    panel = render(seal(report, seal_field, blanked=blanked))

    assert panel["actuation_permitted"] is False
