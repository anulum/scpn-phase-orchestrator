# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Geometry walk domainpack tests

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec

SPEC_PATH = "domainpacks/geometry_walk/binding_spec.yaml"


def test_spec_loads():
    spec = load_binding_spec(SPEC_PATH)
    assert spec.name == "geometry_walk"


def test_spec_valid():
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    assert errors == []


def test_run_script():
    result = subprocess.run(
        [sys.executable, "domainpacks/geometry_walk/run.py"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0
    assert "R_good=" in result.stdout
    assert "coalescence" in result.stdout


def test_policy_loads():
    from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules

    rules = load_policy_rules(Path("domainpacks/geometry_walk/policy.yaml"))
    assert len(rules) > 0


class TestGeometryWalkPipelineWiring:
    """Pipeline: geometry_walk spec → engine → R."""

    def test_domainpack_spec_drives_engine(self):
        """Load geometry_walk spec → engine → R∈[0,1]."""
        import numpy as np

        from scpn_phase_orchestrator.coupling import CouplingBuilder
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        spec = load_binding_spec(SPEC_PATH)
        n = sum(len(ly.oscillator_ids) for ly in spec.layers)
        cs = CouplingBuilder().build(
            n,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
        )
        eng = UPDEEngine(n, dt=spec.sample_period_s)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        for _ in range(100):
            phases = eng.step(
                phases,
                omegas,
                cs.knm,
                0.0,
                0.0,
                cs.alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
