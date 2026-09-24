# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — resilience is scored under the pack's own closed loop

"""``spo chaos`` must run the same closed loop as ``spo run``.

``run_resilience_experiment`` called ``simulate`` without the spec path, so a
domainpack's ``policy.yaml`` never loaded: for 18 of the 35 packs that ship a
policy, the loop ``spo run`` executes fires actions the chaos runs never saw
(``epidemic_sir``: 32 actions against 0 over 200 steps).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.runtime.chaos import (
    ChaosFault,
    ChaosSchedule,
    run_resilience_experiment,
)
from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.runtime.simulation import simulate

_SPEC = (
    Path(__file__).resolve().parents[1]
    / "domainpacks"
    / "epidemic_sir"
    / "binding_spec.yaml"
)
_STEPS = 200


def _policy_run_r() -> tuple[float, int]:
    result = simulate(
        load_binding_spec(_SPEC),
        steps=_STEPS,
        seed=42,
        policy_enabled=True,
        binding_spec_path=_SPEC,
    )
    return result.r_good, result.action_total


def test_policy_changes_the_closed_loop_of_this_pack() -> None:
    without = simulate(load_binding_spec(_SPEC), steps=_STEPS, seed=42)
    _, with_policy_actions = _policy_run_r()
    assert with_policy_actions > without.action_total


def test_resilience_nominal_run_is_the_spo_run_loop() -> None:
    result = run_resilience_experiment(
        load_binding_spec(_SPEC),
        ChaosSchedule((ChaosFault("coupling_drop", 20, 30, 0.5),)),
        steps=_STEPS,
        seed=42,
        binding_spec_path=_SPEC,
    )
    expected_r, _ = _policy_run_r()
    assert result.nominal_final_r == pytest.approx(expected_r, abs=1e-12)


def test_chaos_cli_loads_the_pack_policy() -> None:
    cli = CliRunner().invoke(
        main,
        [
            "chaos",
            str(_SPEC),
            "--fault",
            "coupling_drop:20:30:0.5",
            "--steps",
            str(_STEPS),
            "--seed",
            "42",
            "--json-out",
        ],
    )
    assert cli.exit_code == 0, cli.output
    expected_r, _ = _policy_run_r()
    assert json.loads(cli.output)["nominal_final_r"] == pytest.approx(
        expected_r, abs=1e-12
    )
