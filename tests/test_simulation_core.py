# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for the shared simulation core + eval API

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.api import evaluate_binding_spec
from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.runtime import cli
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.runtime.simulation import SimulationResult, simulate

# Representative specs across the engine/feature space.
# digital_twin_nchannel is research-tier and Kuramoto (no amplitude), covering
# the non-amplitude engine path.
KURAMOTO_SPEC = "domainpacks/digital_twin_nchannel/binding_spec.yaml"
SL_RESEARCH_SPEC = "domainpacks/sleep_architecture/binding_spec.yaml"  # research, SL
SL_IMPRINT_SPEC = "domainpacks/queuewaves/binding_spec.yaml"  # research, SL + imprint
GATED_SL_SPEC = "domainpacks/cardiac_rhythm/binding_spec.yaml"  # clinical, SL
ACTING_SPEC = (
    "domainpacks/chemical_reactor/binding_spec.yaml"  # production, policy acts
)


def _spec(path: str):
    return load_binding_spec(Path(path))


def _cli_run_finals(
    spec_path: str, *, steps: int, seed: int
) -> tuple[float, float, str]:
    result = CliRunner().invoke(
        cli.main, ["run", spec_path, "--steps", str(steps), "--seed", str(seed)]
    )
    assert result.exit_code == 0, result.output
    match = re.search(r"R_good=([\d.]+)\s+R_bad=([\d.]+)\s+regime=(\w+)", result.output)
    assert match is not None, result.output
    return float(match.group(1)), float(match.group(2)), match.group(3)


class TestParityWithCli:
    """The simulation core must reproduce `spo run` exactly — it is the single
    fidelity contract guarding the CLI loop and the eval API from drifting."""

    def test_cli_run_delegates_to_simulate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []
        original = cli.simulate

        def spy(spec: Any, **kwargs: Any) -> SimulationResult:
            calls.append(dict(kwargs))
            return original(spec, **kwargs)

        monkeypatch.setattr(cli, "simulate", spy)
        result = CliRunner().invoke(
            cli.main,
            ["run", KURAMOTO_SPEC, "--steps", "7", "--seed", "13"],
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0]["steps"] == 7
        assert calls[0]["seed"] == 13
        assert calls[0]["policy_enabled"] is True
        assert calls[0]["audit_logger"] is None
        assert calls[0]["binding_spec_path"] == Path(KURAMOTO_SPEC)
        assert "R_good=" in result.output

    @pytest.mark.parametrize(
        "spec_path", [KURAMOTO_SPEC, SL_RESEARCH_SPEC, SL_IMPRINT_SPEC]
    )
    def test_closed_loop_matches_cli(self, spec_path: str) -> None:
        cli_rg, cli_rb, cli_reg = _cli_run_finals(spec_path, steps=40, seed=11)
        res = simulate(
            _spec(spec_path),
            steps=40,
            seed=11,
            policy_enabled=True,
            binding_spec_path=Path(spec_path),
        )
        assert res.r_good == pytest.approx(cli_rg, abs=1e-4)
        assert res.r_bad == pytest.approx(cli_rb, abs=1e-4)
        assert res.final_regime == cli_reg


class TestSimulateContract:
    def test_returns_result_with_histories(self) -> None:
        res = simulate(_spec(KURAMOTO_SPEC), steps=15, seed=5)
        assert isinstance(res, SimulationResult)
        assert res.steps == 15
        assert len(res.r_good_history) == 15
        assert len(res.r_bad_history) == 15
        assert res.separation == pytest.approx(res.r_good - res.r_bad)

    def test_order_parameters_are_bounded(self) -> None:
        res = simulate(_spec(SL_RESEARCH_SPEC), steps=20, seed=5)
        assert 0.0 <= res.r_good <= 1.0
        assert 0.0 <= res.r_bad <= 1.0
        for r in (*res.r_good_history, *res.r_bad_history):
            assert 0.0 <= r <= 1.0

    def test_amplitude_mode_reports_amplitudes(self) -> None:
        res = simulate(_spec(SL_RESEARCH_SPEC), steps=10, seed=5)
        assert res.amplitude_mode is True
        assert res.final_amplitudes is not None
        assert res.mean_amplitude is not None

    def test_kuramoto_mode_has_no_amplitudes(self) -> None:
        res = simulate(_spec(KURAMOTO_SPEC), steps=10, seed=5)
        assert res.amplitude_mode is False
        assert res.final_amplitudes is None
        assert res.mean_amplitude is None

    def test_deterministic_same_seed(self) -> None:
        a = simulate(_spec(SL_IMPRINT_SPEC), steps=25, seed=3)
        b = simulate(_spec(SL_IMPRINT_SPEC), steps=25, seed=3)
        assert a.r_good == b.r_good
        assert a.r_bad == b.r_bad
        assert np.array_equal(a.final_phases, b.final_phases)

    def test_different_seed_differs(self) -> None:
        a = simulate(_spec(SL_RESEARCH_SPEC), steps=25, seed=1)
        b = simulate(_spec(SL_RESEARCH_SPEC), steps=25, seed=2)
        assert not np.array_equal(a.final_phases, b.final_phases)

    def test_no_oscillators_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = _spec(KURAMOTO_SPEC)
        object.__setattr__(spec, "layers", [])
        with pytest.raises(ValueError, match="no oscillators"):
            simulate(spec, steps=5)

    def test_zero_steps(self) -> None:
        res = simulate(_spec(KURAMOTO_SPEC), steps=0, seed=5)
        assert res.steps == 0
        assert res.r_good_history == ()
        assert res.r_good == 0.0


class TestFeatureBranches:
    """Cover the optional spec-feature branches of the core."""

    def test_geometry_prior_is_applied(self) -> None:
        # neuroscience_eeg declares a geometry_prior, exercising the constraint
        # build and per-step projection.
        spec = _spec("domainpacks/neuroscience_eeg/binding_spec.yaml")
        assert spec.geometry_prior is not None
        res = simulate(spec, steps=12, seed=5)
        assert np.isfinite(res.r_good)

    def test_physical_driver_branch(self) -> None:
        spec = _spec(SL_RESEARCH_SPEC)
        spec.drivers.physical["frequency"] = 1.5
        spec.drivers.physical["amplitude"] = 0.5
        res = simulate(spec, steps=10, seed=5)
        assert res.steps == 10

    def test_symbolic_driver_branch(self) -> None:
        spec = _spec(SL_RESEARCH_SPEC)
        spec.drivers.physical.clear()
        spec.drivers.informational.clear()
        spec.drivers.symbolic["sequence"] = [0.1, 0.2, 0.3]
        res = simulate(spec, steps=10, seed=5)
        assert res.steps == 10

    def test_informational_driver_branch(self) -> None:
        spec = _spec(SL_RESEARCH_SPEC)
        spec.drivers.physical.clear()
        spec.drivers.informational["cadence_hz"] = 2.0
        res = simulate(spec, steps=10, seed=5)
        assert res.steps == 10

    def test_objective_r_empty_returns_zero(self) -> None:
        from scpn_phase_orchestrator.runtime.simulation import _objective_r

        assert _objective_r(np.zeros(4), [], {}) == 0.0

    def test_petri_net_from_protocol_builds_net_and_marking(self) -> None:
        from scpn_phase_orchestrator.binding.types import (
            ProtocolNetSpec,
            ProtocolTransitionSpec,
        )
        from scpn_phase_orchestrator.runtime.simulation import petri_net_from_protocol
        from scpn_phase_orchestrator.supervisor.petri_net import Marking, PetriNet

        transition = ProtocolTransitionSpec(
            name="fire",
            inputs=[{"place": "p0", "weight": 1}],
            outputs=[{"place": "p1", "weight": 2}],
            guard="R > 0.5",
        )
        protocol = ProtocolNetSpec(
            places=["p0", "p1"],
            initial={"p0": 1, "p1": 0},
            place_regime={"p0": "nominal"},
            transitions=[transition],
        )
        net, marking = petri_net_from_protocol(protocol)
        assert isinstance(net, PetriNet)
        assert isinstance(marking, Marking)
        assert marking.tokens["p0"] == 1


class TestPolicySwitch:
    def test_closed_loop_acts_open_loop_does_not(self) -> None:
        closed = simulate(
            _spec(ACTING_SPEC),
            steps=120,
            seed=7,
            policy_enabled=True,
            binding_spec_path=Path(ACTING_SPEC),
        )
        open_loop = simulate(
            _spec(ACTING_SPEC),
            steps=120,
            seed=7,
            policy_enabled=False,
            binding_spec_path=Path(ACTING_SPEC),
        )
        assert open_loop.action_total == 0
        assert closed.action_total > 0
        # When the controller acts, the closed-loop trajectory must diverge from
        # the open-loop baseline on the same seed.
        assert not np.array_equal(closed.final_phases, open_loop.final_phases)

    def test_open_loop_is_policy_free_but_runs_drivers(self) -> None:
        res = simulate(_spec(SL_RESEARCH_SPEC), steps=30, seed=7, policy_enabled=False)
        assert res.action_total == 0
        assert res.policy_enabled is False


class TestAuditLogging:
    def test_audit_log_is_replayable(self, tmp_path: Path) -> None:
        audit = tmp_path / "run.jsonl"
        logger = AuditLogger(str(audit))
        simulate(
            _spec(SL_RESEARCH_SPEC),
            steps=8,
            seed=3,
            policy_enabled=True,
            audit_logger=logger,
            binding_spec_path=Path(SL_RESEARCH_SPEC),
        )
        logger.close()
        entries = ReplayEngine(str(audit)).load()
        # Header + one record per step.
        step_records = [e for e in entries if "step" in e]
        assert len(step_records) == 8
        integrity_ok, _ = ReplayEngine.verify_integrity(entries)
        assert integrity_ok

    @pytest.mark.parametrize(
        "spec_path", [SL_RESEARCH_SPEC, SL_IMPRINT_SPEC, KURAMOTO_SPEC]
    )
    def test_audit_passes_replay_determinism_verify(
        self, tmp_path: Path, spec_path: str
    ) -> None:
        # Regression guard: the audit must log pre-step amplitudes paired with
        # pre-step phases, so `replay --verify` reproduces every transition.
        # Logging post-step amplitudes previously broke amplitude-driven specs.
        audit = tmp_path / "run.jsonl"
        logger = AuditLogger(str(audit))
        simulate(
            _spec(spec_path),
            steps=20,
            seed=5,
            policy_enabled=True,
            audit_logger=logger,
            binding_spec_path=Path(spec_path),
        )
        logger.close()
        engine = CliRunner().invoke(cli.main, ["replay", str(audit), "--verify"])
        assert engine.exit_code == 0, engine.output
        assert "Determinism verified" in engine.output


class TestEvaluateApi:
    def test_evaluates_gated_sl_spec(self) -> None:
        # cardiac_rhythm is clinical-tier Stuart-Landau: blocked for live
        # `spo run` but allowed for non-actuating evaluation.
        res = evaluate_binding_spec(GATED_SL_SPEC, steps=20, seed=7)
        assert isinstance(res, SimulationResult)
        assert res.spec_name == "cardiac_rhythm"
        assert res.amplitude_mode is True

    def test_accepts_spec_object(self) -> None:
        res = evaluate_binding_spec(_spec(KURAMOTO_SPEC), steps=10, seed=7)
        assert res.spec_name == "digital_twin_nchannel"

    def test_accepts_path_object(self) -> None:
        res = evaluate_binding_spec(Path(GATED_SL_SPEC), steps=8, seed=7)
        assert res.steps == 8

    def test_open_vs_closed_loop(self) -> None:
        open_loop = evaluate_binding_spec(
            ACTING_SPEC, steps=120, seed=7, policy_enabled=False
        )
        closed = evaluate_binding_spec(
            ACTING_SPEC, steps=120, seed=7, policy_enabled=True
        )
        assert open_loop.action_total == 0
        assert closed.action_total > 0

    def test_to_record_is_json_serialisable(self) -> None:
        record = evaluate_binding_spec(GATED_SL_SPEC, steps=8, seed=7).to_record()
        assert json.loads(json.dumps(record)) == record
        assert record["spec_name"] == "cardiac_rhythm"

    def test_rejects_non_spec(self) -> None:
        with pytest.raises(TypeError, match="BindingSpec or path"):
            evaluate_binding_spec(42)  # type: ignore[arg-type]

    def test_rejects_invalid_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scpn_phase_orchestrator.api as api_mod

        monkeypatch.setattr(api_mod, "validate_binding_spec", lambda _spec: ["broken"])
        with pytest.raises(ValueError, match="validation failed"):
            evaluate_binding_spec(_spec(KURAMOTO_SPEC), steps=5)
