# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Multi-domain topology Lyapunov workflow tests

"""Workflow contract: topology adaptation must publish Lyapunov evidence.

The dynamic higher-order topology workflow spans domainpack demos and the
supervisor mutation surface.  Each reviewed domain must emit non-actuating,
auditable mutation evidence whose Lyapunov energy does not increase for the
proposed pairwise topology update.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from domainpacks.network_security.topology_adaptation_demo import (
    run_demo as run_network_security_demo,
)
from domainpacks.plasma_control.topology_adaptation_demo import (
    run_demo as run_plasma_demo,
)
from domainpacks.traffic_flow.topology_adaptation_demo import (
    run_demo as run_traffic_demo,
)

DemoRunner = Callable[[], dict[str, object]]


@pytest.mark.parametrize(
    ("expected_domain", "run_demo"),
    [
        ("plasma_control", run_plasma_demo),
        ("traffic_flow", run_traffic_demo),
        ("network_security", run_network_security_demo),
    ],
)
def test_multi_domain_topology_mutations_emit_lyapunov_evidence(
    expected_domain: str,
    run_demo: DemoRunner,
) -> None:
    payload = run_demo()

    assert payload["domainpack"] == expected_domain
    audit = payload["audit"]
    lyapunov = payload["lyapunov_validation"]

    assert isinstance(audit, dict)
    assert isinstance(lyapunov, dict)
    assert audit["pairwise_delta_norm"] > 0.0
    assert audit["added_simplices"]
    assert lyapunov["non_increasing"] is True
    assert lyapunov["delta_V"] <= 0.0
    assert lyapunov["after_V"] <= lyapunov["before_V"]
    assert isinstance(lyapunov["after_in_basin"], bool)
    assert lyapunov["after_max_phase_diff"] >= 0.0
    assert all(
        math.isfinite(float(lyapunov[key]))
        for key in ("before_V", "after_V", "delta_V", "after_max_phase_diff")
    )


def test_multi_domain_topology_workflow_domains_are_unique() -> None:
    payloads = [
        run_plasma_demo(),
        run_traffic_demo(),
        run_network_security_demo(),
    ]

    assert {payload["domainpack"] for payload in payloads} == {
        "plasma_control",
        "traffic_flow",
        "network_security",
    }
    assert all(payload.get("actuating") is not True for payload in payloads)
