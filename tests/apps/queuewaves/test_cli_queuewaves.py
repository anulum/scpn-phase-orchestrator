# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves CLI tests

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.cli import main


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    yaml_text = textwrap.dedent("""\
        prometheus_url: "http://localhost:9090"
        scrape_interval_s: 1
        buffer_length: 16
        services:
          - name: svc-a
            promql: 'up'
            layer: micro
          - name: tput
            promql: 'up'
            layer: macro
        thresholds:
          r_bad_warn: 0.50
          r_bad_critical: 0.70
    """)
    f = tmp_path / "qw.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    return f


def test_check_subcommand(config_file: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["queuewaves", "check", "--config", str(config_file)])
    assert result.exit_code in (0, 1)
    assert "R_good=" in result.output
    assert "R_bad=" in result.output


def test_check_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["queuewaves", "check", "--help"])
    assert result.exit_code == 0
    assert "one-shot" in result.output.lower() or "scrape" in result.output.lower()


def test_serve_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["queuewaves", "serve", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_queuewaves_group_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["queuewaves", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.output
    assert "check" in result.output
