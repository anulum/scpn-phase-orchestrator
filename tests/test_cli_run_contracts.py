# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI run command contracts

"""Public CLI contracts for run/report/queuewaves/replay commands.

The assertions cover concrete exit-code and output behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine


class TestCLIRun:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_run_non_research_safety_tier(self, runner, tmp_path):
        spec = {
            "name": "tier-test",
            "version": "1.0.0",
            "safety_tier": "clinical",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "5"])
        assert result.exit_code != 0
        assert "safety_tier='clinical' is not enforced" in result.output
        assert "R_good=" not in result.output

    def test_run_no_oscillators_exits_1(self, runner, tmp_path):
        spec = {
            "name": "empty-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": []},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "5"])
        assert result.exit_code != 0
        assert "no oscillators" in result.output

    def test_run_amplitude_mode(self, runner, tmp_path):
        spec = {
            "name": "sl-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
                {"name": "L2", "index": 1, "oscillator_ids": ["o2", "o3"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": [1]},
            "boundaries": [],
            "actuators": [],
            "amplitude": {
                "mu": 1.0,
                "epsilon": 1.0,
                "amp_coupling_strength": 0.1,
                "amp_coupling_decay": 0.3,
            },
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "25"])
        assert result.exit_code == 0
        assert "mean_amplitude" in result.output

    def test_run_with_audit_log(self, runner, tmp_path):
        spec = {
            "name": "audit-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
        }
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")
        audit_path = str(tmp_path / "audit.jsonl")
        result = runner.invoke(
            main,
            ["run", str(spec_path), "--steps", "5", "--audit", audit_path],
        )
        assert result.exit_code == 0

    def test_report_empty_log_exits_1(self, runner, tmp_path):
        log = tmp_path / "empty.jsonl"
        log.write_text(json.dumps({"event": "x"}) + "\n", encoding="utf-8")
        result = runner.invoke(main, ["report", str(log)])
        assert result.exit_code != 0

    def test_report_with_actions(self, runner, tmp_path):
        entries = []
        for i in range(5):
            e = {
                "step": i,
                "regime": "DEGRADED",
                "stability": 0.4,
                "layers": [{"R": 0.4, "psi": 0.0}],
                "actions": [{"knob": "K", "scope": "global", "value": 0.05}],
            }
            entries.append(e)
        log = tmp_path / "actions.jsonl"
        log.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["report", str(log)])
        assert result.exit_code == 0


class TestCLIQueuewavesCheck:
    def test_queuewaves_check_with_anomalies(self, tmp_path):
        cfg = {
            "prometheus_url": "http://localhost:9090",
            "services": [
                {"name": "svc-a", "promql": "up", "layer": "micro"},
                {"name": "svc-b", "promql": "up", "layer": "macro"},
            ],
            "scrape_interval_s": 1.0,
            "buffer_length": 16,
        }
        path = tmp_path / "qw.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["queuewaves", "check", "--config", str(path)])
        assert result.exit_code in (0, 1)
        assert "R_good" in result.output

    def test_queuewaves_check_no_anomalies(self, tmp_path):
        """Permissive thresholds ensure no anomalies → line 622 covered."""
        cfg = {
            "prometheus_url": "http://localhost:9090",
            "services": [
                {"name": "svc-a", "promql": "up", "layer": "macro"},
            ],
            "scrape_interval_s": 1.0,
            "buffer_length": 16,
            "thresholds": {
                "r_bad_warn": 99.0,
                "r_bad_critical": 99.0,
                "plv_cascade": 99.0,
                "imprint_chronic": 99.0,
            },
        }
        path = tmp_path / "qw.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["queuewaves", "check", "--config", str(path)])
        assert result.exit_code == 0
        assert "No anomalies detected" in result.output


class TestCLIReplay:
    def test_replay_sl_chained(self, tmp_path):
        n = 2
        engine = StuartLandauEngine(n, dt=0.01)
        engine._use_rust = False
        state = np.array([0.1, 0.2, 0.8, 0.9])
        omegas = np.ones(n)
        mu = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        knm_r = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm_r, 0.0)
        alpha = np.zeros((n, n))

        entries = [
            {
                "header": True,
                "n_oscillators": n,
                "dt": 0.01,
                "seed": 0,
                "amplitude_mode": True,
            }
        ]
        for _ in range(5):
            entries.append(
                {
                    "step": len(entries) - 1,
                    "phases": state.tolist(),
                    "omegas": omegas.tolist(),
                    "knm": knm.tolist(),
                    "alpha": alpha.tolist(),
                    "mu": mu.tolist(),
                    "knm_r": knm_r.tolist(),
                    "zeta": 0.0,
                    "psi_drive": 0.0,
                }
            )
            state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

        log = tmp_path / "sl_replay.jsonl"
        log.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(main, ["replay", str(log), "--verify"])
        assert result.exit_code == 0
        assert "verified" in result.output.lower()

    def test_replay_failed_determinism(self, tmp_path):
        n = 2
        engine = StuartLandauEngine(n, dt=0.01)
        engine._use_rust = False
