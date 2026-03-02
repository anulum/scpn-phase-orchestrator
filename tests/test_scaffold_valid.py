# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from click.testing import CliRunner

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.cli import main


def test_scaffold_generates_valid_spec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["scaffold", "mytest"])
    assert result.exit_code == 0

    spec_path = tmp_path / "domainpacks" / "mytest" / "binding_spec.yaml"
    assert spec_path.exists()

    spec = load_binding_spec(spec_path)
    errors = validate_binding_spec(spec)
    assert errors == [], f"scaffold spec has validation errors: {errors}"
    assert len(spec.layers) >= 1
    assert spec.objectives.good_layers or spec.objectives.bad_layers
