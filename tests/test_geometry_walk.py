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
        timeout=90,
    )
    assert result.returncode == 0
    assert "R_good=" in result.stdout
    assert "coalescence" in result.stdout


def test_policy_loads():
    from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules

    rules = load_policy_rules(Path("domainpacks/geometry_walk/policy.yaml"))
    assert len(rules) > 0
