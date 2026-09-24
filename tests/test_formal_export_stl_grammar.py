# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL export grammar parity tests

"""Prove the PRISM STL exporter reads the builtin monitor's grammar.

The exporter once kept its own copy of the predicate grammar, so a formula the
monitor evaluated could be refused by the exporter. These tests drive both
through their public entry points on the same formulas.
"""

from __future__ import annotations

import re

import pytest

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.monitor.stl import (
    PHASE_FIELD_SPECIFICATIONS,
    STLMonitor,
    synthesise_stl_monitoring_automaton,
)
from scpn_phase_orchestrator.supervisor.formal_export import export_stl_specs_prism
from scpn_phase_orchestrator.supervisor.policy_rules import PolicySTLSpec

_FORMULAS = (
    "always (R >= .5)",
    "always (R >= 5.)",
    "always (R >= 5e-1)",
    "eventually (x <= 1E-20)",
    "always (R > 0.3 and K <= 10)",
    *(spec.spec for spec in PHASE_FIELD_SPECIFICATIONS),
)


def _exported_thresholds(model: str) -> list[float]:
    """Return every threshold in the model's ``..._satisfied`` label expression."""
    satisfied = re.search(r'label "\w+_satisfied" = (.*);', model)
    assert satisfied is not None
    return [
        float(value)
        for value in re.findall(r"(?:>=|>|<=|<|=)\s*([-+0-9.eE]+)", satisfied.group(1))
    ]


@pytest.mark.parametrize("formula", _FORMULAS)
def test_exporter_accepts_every_formula_the_monitor_evaluates(formula: str) -> None:
    """A builtin formula exports, and the export keeps its thresholds exactly."""
    signals = sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*(?:>=|>|<=|<|==)", formula)))
    trace = {signal: [0.0, 1.0] for signal in signals}
    assert STLMonitor(formula).evaluate_result(trace).backend == "builtin"

    export = export_stl_specs_prism([PolicySTLSpec(name="probe", spec=formula)])

    expected = sorted(
        float(value)
        for value in re.findall(
            r"(?:>=|>|<=|<|==)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            formula,
        )
    )
    assert sorted(set(_exported_thresholds(export.model))) == sorted(set(expected))


def test_exporter_rejects_a_non_finite_threshold_as_policy_error() -> None:
    """A threshold that overflows to infinity is a policy error, not a crash."""
    with pytest.raises(PolicyError, match="threshold must be finite"):
        export_stl_specs_prism(
            [PolicySTLSpec(name="probe", spec="always (R >= 1e999)")]
        )


def test_automaton_guard_keeps_the_full_threshold() -> None:
    """Audit guard text does not shorten a threshold to six digits."""
    automaton = synthesise_stl_monitoring_automaton(
        "always (R >= 0.30000001)", {"R": [0.5, 0.6]}
    )
    assert {transition.guard for transition in automaton.transitions} == {
        "R >= 0.30000001"
    }
