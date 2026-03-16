# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves config tests

from __future__ import annotations

import textwrap
from pathlib import Path

from scpn_phase_orchestrator.apps.queuewaves.config import (
    ConfigCompiler,
    QueueWavesConfig,
    load_config,
)


def test_load_config(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent("""\
        prometheus_url: "http://prom:9090"
        scrape_interval_s: 10
        buffer_length: 32
        services:
          - name: svc-a
            promql: 'rate(http[1m])'
            layer: micro
            channel: P
          - name: tput
            promql: 'rate(http[5m])'
            layer: macro
        thresholds:
          r_bad_warn: 0.45
          r_bad_critical: 0.65
        coupling:
          strength: 0.40
          decay: 0.20
        alert_sinks:
          - url: "https://example.com/hook"
            format: slack
        server:
          port: 9999
    """)
    f = tmp_path / "qw.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(f)
    assert cfg.prometheus_url == "http://prom:9090"
    assert len(cfg.services) == 2
    assert cfg.services[0].name == "svc-a"
    assert cfg.thresholds.r_bad_warn == 0.45
    assert cfg.coupling.strength == 0.40
    assert len(cfg.alert_sinks) == 1
    assert cfg.server.port == 9999


def test_config_compiler_layers(minimal_config: QueueWavesConfig) -> None:
    compiler = ConfigCompiler()
    spec = compiler.compile(minimal_config)
    assert spec.name == "queuewaves"
    assert len(spec.layers) == 2  # micro + macro
    micro = [ly for ly in spec.layers if ly.name == "micro"][0]
    macro = [ly for ly in spec.layers if ly.name == "macro"][0]
    assert micro.index == 0
    assert macro.index == 2
    assert set(micro.oscillator_ids) == {"svc-a", "svc-b"}
    assert "throughput" in macro.oscillator_ids


def test_config_compiler_objectives(minimal_config: QueueWavesConfig) -> None:
    spec = ConfigCompiler().compile(minimal_config)
    assert 0 in spec.objectives.bad_layers  # micro = bad
    assert 2 in spec.objectives.good_layers  # macro = good


def test_config_compiler_boundaries(minimal_config: QueueWavesConfig) -> None:
    spec = ConfigCompiler().compile(minimal_config)
    assert len(spec.boundaries) == 2
    names = {b.name for b in spec.boundaries}
    assert "r_bad_warn" in names
    assert "r_bad_critical" in names


def test_config_compiler_oscillator_families(minimal_config: QueueWavesConfig) -> None:
    spec = ConfigCompiler().compile(minimal_config)
    assert "svc-a" in spec.oscillator_families
    assert spec.oscillator_families["svc-a"].channel == "P"
    assert spec.oscillator_families["svc-a"].extractor_type == "hilbert"


def test_load_config_defaults(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent("""\
        prometheus_url: "http://prom:9090"
        services:
          - name: s1
            promql: 'up'
    """)
    f = tmp_path / "qw.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(f)
    assert cfg.scrape_interval_s == 15.0
    assert cfg.buffer_length == 64
    assert cfg.thresholds.r_bad_warn == 0.50
