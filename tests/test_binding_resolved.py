# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Resolved binding summary tests

from __future__ import annotations

from scpn_phase_orchestrator.binding.resolved import resolve_binding_summary
from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    BindingSpec,
    BoundaryDef,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ObjectivePartition,
    OscillatorFamily,
)


def _spec() -> BindingSpec:
    return BindingSpec(
        name="resolved-test",
        version="1.0.0",
        safety_tier="research",
        sample_period_s=0.02,
        control_period_s=0.1,
        layers=[
            HierarchyLayer("sensor", 0, ["p0", "p1"], omegas=[1.2, 1.4]),
            HierarchyLayer("controller", 1, ["s0"]),
        ],
        oscillator_families={
            "physical": OscillatorFamily("P", "hilbert", {"units": "bar"}),
            "symbolic": OscillatorFamily("S", "ring", {"states": ["a", "b"]}),
        },
        coupling=CouplingSpec(0.35, 0.2, {"near": "local"}),
        drivers=DriverSpec(
            physical={"psi": 0.3},
            informational={"zeta": 0.05},
            symbolic={},
        ),
        objectives=ObjectivePartition([0], [1], good_weight=2.0, bad_weight=0.5),
        boundaries=[BoundaryDef("floor", "R", 0.2, None, "soft")],
        actuators=[ActuatorMapping("global-k", "K", "global", (0.0, 3.0))],
    )


def test_resolve_binding_summary_counts_ranges_and_defaults() -> None:
    summary = resolve_binding_summary(_spec(), spec_path="/private/domain.yaml")

    assert summary["source"] == "domain.yaml"
    assert summary["counts"]["oscillators"] == 3
    assert summary["timing"]["control_interval_steps"] == 5
    assert summary["layers"][0]["range"] == (0, 2)
    assert summary["layers"][1]["range"] == (2, 3)
    assert summary["layers"][0]["omega_source"] == "explicit"
    assert summary["layers"][1]["omega_source"] == "default"
    assert summary["layers"][1]["omegas"] == [1.0]
    assert summary["defaults_applied"]["omegas"] == ["controller"]


def test_resolve_binding_summary_actuation_and_drivers() -> None:
    summary = resolve_binding_summary(_spec())

    assert summary["actuation"]["value_bounds_source"] == "actuators"
    assert summary["actuation"]["value_bounds"] == {"K": (0.0, 3.0)}
    assert summary["actuation"]["rate_limits"]["Psi"] == 0.5
    assert summary["drivers"]["zeta_initial"] == 0.05
    assert summary["drivers"]["psi_initial"] == 0.3
    assert summary["drivers"]["psi_driver"] is None
    assert summary["oscillator_families"]["physical"]["config_keys"] == ["units"]
