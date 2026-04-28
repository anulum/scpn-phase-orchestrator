# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau CLI tests

"""Tests for Stuart-Landau integration in CLI run command."""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

_MINIMAL_AMPLITUDE_SPEC = {
    "name": "sl_test",
    "version": "0.1.0",
    "safety_tier": "research",
    "sample_period_s": 0.01,
    "control_period_s": 0.1,
    "layers": [
        {"name": "L0", "index": 0, "oscillator_ids": ["a", "b"]},
        {"name": "L1", "index": 1, "oscillator_ids": ["c", "d"]},
    ],
    "oscillator_families": {
        "fam": {"channel": "P", "extractor_type": "hilbert"},
    },
    "coupling": {"base_strength": 0.5, "decay_alpha": 0.3},
    "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
    "objectives": {"good_layers": [0], "bad_layers": [1]},
    "boundaries": [],
    "actuators": [],
    "amplitude": {
        "mu": 1.0,
        "epsilon": 0.5,
        "amp_coupling_strength": 0.3,
        "amp_coupling_decay": 0.3,
    },
}

_PHASE_ONLY_SPEC = {
    "name": "phase_test",
    "version": "0.1.0",
    "safety_tier": "research",
    "sample_period_s": 0.01,
    "control_period_s": 0.1,
    "layers": [
        {"name": "L0", "index": 0, "oscillator_ids": ["a", "b"]},
    ],
    "oscillator_families": {
        "fam": {"channel": "P", "extractor_type": "hilbert"},
    },
    "coupling": {"base_strength": 0.5, "decay_alpha": 0.3},
    "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
    "objectives": {"good_layers": [0], "bad_layers": []},
    "boundaries": [],
    "actuators": [],
}


def _write_spec(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "binding_spec.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


class TestCLIAmplitudeMode:
    def test_run_amplitude_mode(self, tmp_path: Path) -> None:
        spec_path = _write_spec(tmp_path, _MINIMAL_AMPLITUDE_SPEC)
        runner = CliRunner()
        result = runner.invoke(main, ["run", str(spec_path), "--steps", "30"])
        assert result.exit_code == 0
        assert "mean_amplitude" in result.output

    def test_run_phase_only_no_amplitude(self, tmp_path: Path) -> None:
        spec_path = _write_spec(tmp_path, _PHASE_ONLY_SPEC)
        runner = CliRunner()
        result = runner.invoke(main, ["run", str(spec_path), "--steps", "10"])
        assert result.exit_code == 0
        assert "mean_amplitude" not in result.output

    def test_run_amplitude_output_format(self, tmp_path: Path) -> None:
        spec_path = _write_spec(tmp_path, _MINIMAL_AMPLITUDE_SPEC)
        runner = CliRunner()
        result = runner.invoke(main, ["run", str(spec_path), "--steps", "10"])
        assert "R_good=" in result.output
        assert "R_bad=" in result.output
        assert "regime=" in result.output

    def test_amplitude_audit_log(self, tmp_path: Path) -> None:
        spec_path = _write_spec(tmp_path, _MINIMAL_AMPLITUDE_SPEC)
        audit_path = tmp_path / "audit.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                str(spec_path),
                "--steps",
                "10",
                "--audit",
                str(audit_path),
            ],
        )
        assert result.exit_code == 0
        assert audit_path.exists()

        replay = ReplayEngine(audit_path)
        entries = replay.load()
        header = replay.load_header(entries)
        assert header is not None
        assert header.get("amplitude_mode") is True

    def test_amplitude_audit_replay_builds_sl(self, tmp_path: Path) -> None:
        spec_path = _write_spec(tmp_path, _MINIMAL_AMPLITUDE_SPEC)
        audit_path = tmp_path / "audit.jsonl"
        runner = CliRunner()
        runner.invoke(
            main,
            [
                "run",
                str(spec_path),
                "--steps",
                "5",
                "--audit",
                str(audit_path),
            ],
        )
        replay = ReplayEngine(audit_path)
        entries = replay.load()
        header = replay.load_header(entries)
        assert header is not None
        engine = replay.build_engine(header)
        assert isinstance(engine, StuartLandauEngine)

    def test_phase_only_replay_builds_upde(self, tmp_path: Path) -> None:
        spec_path = _write_spec(tmp_path, _PHASE_ONLY_SPEC)
        audit_path = tmp_path / "audit.jsonl"
        runner = CliRunner()
        runner.invoke(
            main,
            [
                "run",
                str(spec_path),
                "--steps",
                "5",
                "--audit",
                str(audit_path),
            ],
        )
        replay = ReplayEngine(audit_path)
        entries = replay.load()
        header = replay.load_header(entries)
        assert header is not None
        engine = replay.build_engine(header)
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        assert isinstance(engine, UPDEEngine)

    def test_amplitude_with_imprint(self, tmp_path: Path) -> None:
        data = {
            **_MINIMAL_AMPLITUDE_SPEC,
            "imprint_model": {
                "decay_rate": 0.1,
                "saturation": 2.0,
                "modulates": ["K"],
            },
        }
        spec_path = _write_spec(tmp_path, data)
        runner = CliRunner()
        result = runner.invoke(main, ["run", str(spec_path), "--steps", "20"])
        assert result.exit_code == 0

    def test_validate_amplitude_spec(self, tmp_path: Path) -> None:
        spec_path = _write_spec(tmp_path, _MINIMAL_AMPLITUDE_SPEC)
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(spec_path)])
        assert result.exit_code == 0

    def test_validate_invalid_amplitude(self, tmp_path: Path) -> None:
        data = {
            **_MINIMAL_AMPLITUDE_SPEC,
            "amplitude": {"mu": 1.0, "epsilon": -0.5},
        }
        spec_path = _write_spec(tmp_path, data)
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(spec_path)])
        assert result.exit_code == 1
        assert "epsilon" in result.output


class TestCLIAmplitudeReporting:
    def test_amplitude_output_fields(self, tmp_path: Path) -> None:
        spec_path = _write_spec(tmp_path, _MINIMAL_AMPLITUDE_SPEC)
        runner = CliRunner()
        result = runner.invoke(main, ["run", str(spec_path), "--steps", "30"])
        assert result.exit_code == 0
        assert "mean_amplitude=" in result.output
        assert "R_good=" in result.output
        assert "regime=" in result.output

    def test_amplitude_with_policy_rules(self, tmp_path: Path) -> None:
        policy_data = {
            "rules": [
                {
                    "name": "amp_boost",
                    "regime": ["DEGRADED"],
                    "condition": {
                        "metric": "mean_amplitude",
                        "op": "<",
                        "threshold": 0.5,
                    },
                    "action": {
                        "knob": "K",
                        "scope": "global",
                        "value": 0.1,
                        "ttl_s": 10.0,
                    },
                }
            ]
        }
        spec = {
            **_MINIMAL_AMPLITUDE_SPEC,
            "policy": "policy.yaml",
        }
        spec_path = _write_spec(tmp_path, spec)
        policy_path = tmp_path / "policy.yaml"
        policy_path.write_text(
            yaml.dump(policy_data),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["run", str(spec_path), "--steps", "20"])
        assert result.exit_code == 0


class TestCLIRealDomainpacks:
    _PACK_DIR = Path(__file__).parent.parent / "domainpacks"

    def test_cardiac_rhythm_runs(self) -> None:
        spec = self._PACK_DIR / "cardiac_rhythm" / "binding_spec.yaml"
        runner = CliRunner()
        result = runner.invoke(main, ["run", str(spec), "--steps", "20"])
        assert result.exit_code == 0, result.output
        assert "mean_amplitude=" in result.output

    def test_neuroscience_eeg_runs(self) -> None:
        spec = self._PACK_DIR / "neuroscience_eeg" / "binding_spec.yaml"
        runner = CliRunner()
        result = runner.invoke(main, ["run", str(spec), "--steps", "20"])
        assert result.exit_code == 0, result.output
        assert "mean_amplitude=" in result.output

    def test_laser_array_runs(self) -> None:
        spec = self._PACK_DIR / "laser_array" / "binding_spec.yaml"
        runner = CliRunner()
        result = runner.invoke(main, ["run", str(spec), "--steps", "20"])
        assert result.exit_code == 0

    def test_validate_all_domainpacks(self) -> None:
        runner = CliRunner()
        for pack_dir in sorted(self._PACK_DIR.iterdir()):
            spec = pack_dir / "binding_spec.yaml"
            if not spec.exists():
                continue
            result = runner.invoke(main, ["validate", str(spec)])
            assert result.exit_code == 0, (
                f"{pack_dir.name} validation failed: {result.output}"
            )


# Pipeline wiring is proven by TestCLIRealDomainpacks above:
# cardiac_rhythm, neuroscience_eeg, laser_array domainpacks drive
# the full CLI → StuartLandauEngine pipeline end-to-end.
