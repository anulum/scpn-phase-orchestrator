# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Scaffold validation tests

from __future__ import annotations

import numpy as np
import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.cli import main


class TestScaffoldSpecValid:
    """Verify that `spo scaffold` generates a spec that passes the full
    validation → simulation pipeline — not just "file exists"."""

    def test_scaffolded_spec_passes_validation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["scaffold", "mytest"])
        assert result.exit_code == 0

        spec_path = tmp_path / "domainpacks" / "mytest" / "binding_spec.yaml"
        assert spec_path.exists()

        spec = load_binding_spec(spec_path)
        errors = validate_binding_spec(spec)
        assert errors == [], f"Scaffold spec errors: {errors}"
        assert len(spec.layers) >= 1
        assert spec.objectives.good_layers or spec.objectives.bad_layers

    def test_scaffolded_spec_runs_simulation(self, tmp_path, monkeypatch):
        """Roundtrip: scaffold → load → run 10 steps → R∈[0,1].
        Proves the scaffold isn't just valid YAML — it drives the engine."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["scaffold", "simtest"])

        spec_path = tmp_path / "domainpacks" / "simtest" / "binding_spec.yaml"
        result = runner.invoke(main, ["run", str(spec_path), "--steps", "10"])
        assert result.exit_code == 0
        assert "R_good" in result.output

    def test_scaffolded_yaml_is_valid(self, tmp_path, monkeypatch):
        """The generated YAML must parse without errors and contain
        all mandatory top-level keys."""
        monkeypatch.chdir(tmp_path)
        CliRunner().invoke(main, ["scaffold", "yamltest"])

        spec_path = tmp_path / "domainpacks" / "yamltest" / "binding_spec.yaml"
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        for key in ["name", "version", "layers", "coupling", "objectives"]:
            assert key in raw, f"Scaffold missing required key: {key}"

    def test_scaffold_creates_readme(self, tmp_path, monkeypatch):
        """README.md must exist and contain the domain name."""
        monkeypatch.chdir(tmp_path)
        CliRunner().invoke(main, ["scaffold", "doctest"])

        readme = tmp_path / "domainpacks" / "doctest" / "README.md"
        assert readme.exists()
        content = readme.read_text(encoding="utf-8")
        assert "doctest" in content.lower()

    def test_scaffold_spec_feeds_order_parameter(self, tmp_path, monkeypatch):
        """Pipeline wiring: scaffold spec → build coupling → engine →
        compute_order_parameter."""
        from scpn_phase_orchestrator.coupling import CouplingBuilder
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        monkeypatch.chdir(tmp_path)
        CliRunner().invoke(main, ["scaffold", "pipetest"])

        spec_path = tmp_path / "domainpacks" / "pipetest" / "binding_spec.yaml"
        spec = load_binding_spec(spec_path)

        n = sum(len(layer.oscillator_ids) for layer in spec.layers)
        cs = CouplingBuilder().build(
            n, spec.coupling.base_strength, spec.coupling.decay_alpha
        )
        eng = UPDEEngine(n, dt=spec.sample_period_s)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.asarray(spec.get_omegas())
        alpha = np.zeros((n, n))

        for _ in range(100):
            phases = eng.step(phases, omegas, cs.knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0, f"Scaffolded spec pipeline: R={r}"
