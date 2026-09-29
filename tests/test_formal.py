# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Petri PRISM numeric fidelity tests

"""Public Petri exporter contracts for guard precision and transition priority."""

from __future__ import annotations

import math
import re
import sys

import pytest

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor.formal import export_petri_net_to_prism
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Guard,
    Marking,
    PetriNet,
    Place,
    Transition,
)


@pytest.mark.parametrize("operator", [">", ">=", "<", "<=", "=="])
@pytest.mark.parametrize(
    "threshold",
    [
        0.30000000000000004,
        -1.2345678901234567,
        1.2345678901234567e-100,
        1.2345678901234567e100,
        math.ulp(0.0),
        sys.float_info.max,
        -0.0,
    ],
)
def test_export_preserves_guard_threshold_and_boundary_decisions(
    operator: str, threshold: float
) -> None:
    """Exported literals retain the runtime guard's exact binary64 boundary."""
    guard = Guard("coherence", operator, threshold)
    net = PetriNet(
        [Place("ready"), Place("done")],
        [Transition("advance", [Arc("ready")], [Arc("done")], guard)],
    )
    model = export_petri_net_to_prism(net, Marking({"ready": 1}))
    prism_operator = "=" if operator == "==" else operator
    match = re.search(rf"coherence {re.escape(prism_operator)} ([^;]+);", model)
    assert match is not None
    exported_threshold = float(match.group(1))
    assert exported_threshold.hex() == threshold.hex()
    exported_guard = Guard("coherence", operator, exported_threshold)
    for metric in (
        math.nextafter(threshold, -math.inf),
        threshold,
        math.nextafter(threshold, math.inf),
    ):
        if math.isfinite(metric):
            assert exported_guard.evaluate({"coherence": metric}) == guard.evaluate(
                {"coherence": metric}
            )
    assert "const double coherence;" in model
    assert "[advance] enabled_advance -> (done'=done+1) & (ready'=ready-1);" in model


def test_export_preserves_priority_identifiers_and_token_bounds() -> None:
    """Distinct sanitised names retain ordered commands and bounded updates."""
    net = PetriNet(
        [Place("1 ready"), Place("done")],
        [
            Transition("take-a", [Arc("1 ready")], [Arc("done", 2)]),
            Transition("take a", [], []),
        ],
    )
    model = export_petri_net_to_prism(net, Marking({"1 ready": 1}), module_name="!!!")
    assert "module module" in model
    assert "p_1_ready : [0..2] init 1;" in model
    assert "formula enabled_take_a = p_1_ready >= 1 & done <= 0;" in model
    assert "formula enabled_take_a_2 = true;" in model
    assert "[take_a_2] enabled_take_a_2 & !(enabled_take_a) -> true;" in model
    assert "[idle] !(enabled_take_a | enabled_take_a_2) -> true;" in model


@pytest.mark.parametrize("include_idle", [False, True])
def test_export_empty_net_idle_contract(include_idle: bool) -> None:
    """An empty net exports without transitions or invented metric constants."""
    model = export_petri_net_to_prism(
        PetriNet([], []), Marking(), include_idle=include_idle
    )
    assert ("[idle] true -> true;" in model) is include_idle
    assert "const double" not in model
    assert model.endswith("endmodule\n")


@pytest.mark.parametrize("bound", [0, -1])
def test_export_refuses_nonpositive_bound(bound: int) -> None:
    """Invalid explicit bounds never produce a model."""
    with pytest.raises(PolicyError, match="max_tokens must be >= 1"):
        export_petri_net_to_prism(PetriNet([], []), Marking(), max_tokens=bound)


def test_export_refuses_initial_marking_outside_bound() -> None:
    """Explicit bounds cannot truncate the supplied runtime marking."""
    with pytest.raises(PolicyError, match="exceeds max_tokens=1"):
        export_petri_net_to_prism(
            PetriNet([Place("ready")], []), Marking({"ready": 2}), max_tokens=1
        )
