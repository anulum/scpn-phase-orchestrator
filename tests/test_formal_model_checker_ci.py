# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal checker CI smoke tool tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.exceptions import PolicyError
from tools.formal_model_checker_ci import (
    build_domainpack_formal_packages,
    build_smoke_package,
    run_ci_smoke,
)


def test_formal_checker_ci_package_binds_spin_and_smt_commands() -> None:
    package = build_smoke_package()
    record = package.to_audit_record()

    assert record["artifact_types"] == {
        "barrier_smt": "smt2",
        "protocol_spin": "promela",
    }
    assert record["checker_commands"] == [
        {
            "property_name": "spin_token_bound",
            "checker": "spin",
            "artifact_name": "protocol_spin",
            "command": ["spin", "-run", "protocol_spin.pml"],
            "execution_permitted": False,
        },
        {
            "property_name": "smt_unit_interval_feasible",
            "checker": "smt",
            "artifact_name": "barrier_smt",
            "command": ["z3", "barrier_smt.smt2"],
            "execution_permitted": False,
        },
    ]


def test_formal_checker_ci_domainpack_packages_bind_safety_invariants() -> None:
    bundles = build_domainpack_formal_packages()
    records = [bundle.to_audit_record() for bundle in bundles]

    assert [record["domainpack"] for record in records] == [
        "cardiac_rhythm",
        "chemical_reactor",
        "power_grid",
        "pll_clock",
    ]
    assert all(record["artifact_count"] == 2 for record in records)
    assert all(
        record["invariant_summary"].endswith("hard-bound SMT feasibility.")
        for record in records
    )
    assert {
        command["checker"]
        for record in records
        for command in record["package"]["checker_commands"]
    } == {"spin", "smt"}
    assert all(
        command["execution_permitted"] is False
        for record in records
        for command in record["package"]["checker_commands"]
    )
    assert all(
        set(record["package"]["artifact_types"].values()) == {"promela", "smt2"}
        for record in records
    )


def test_formal_checker_ci_execution_is_rejected_outside_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    with pytest.raises(PolicyError, match="CI-only"):
        run_ci_smoke(execute=True)
