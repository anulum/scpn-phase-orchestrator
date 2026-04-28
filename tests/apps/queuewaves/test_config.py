# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves config tests

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from scpn_phase_orchestrator.apps.queuewaves.config import (
    AlertSink,
    ConfigCompiler,
    CouplingConfig,
    QueueWavesConfig,
    SecurityConfig,
    ServerConfig,
    ServiceDef,
    ThresholdConfig,
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
        security:
          mode: production
          api_key_env: QUEUEWAVES_API_KEY
          rate_limit_per_minute: 240
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
    assert cfg.security.mode == "production"
    assert cfg.security.api_key_env == "QUEUEWAVES_API_KEY"
    assert cfg.security.rate_limit_per_minute == 240


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
    assert cfg.security.mode == "development"
    assert cfg.security.rate_limit_per_minute == 120


def test_load_config_normalises_numeric_scalars(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent("""\
        prometheus_url: "http://prom:9090"
        scrape_interval_s: "2.5"
        buffer_length: "8"
        services:
          - name: s1
            promql: up
        thresholds:
          r_bad_warn: "0.1"
          r_bad_critical: "0.2"
        server:
          port: "9091"
        security:
          rate_limit_per_minute: "30"
    """)
    f = tmp_path / "qw.yaml"
    f.write_text(yaml_text, encoding="utf-8")

    cfg = load_config(f)

    assert cfg.scrape_interval_s == 2.5
    assert cfg.buffer_length == 8
    assert cfg.thresholds.r_bad_warn == 0.1
    assert cfg.thresholds.r_bad_critical == 0.2
    assert cfg.server.port == 9091
    assert cfg.security.rate_limit_per_minute == 30
    assert isinstance(cfg.scrape_interval_s, float)
    assert isinstance(cfg.buffer_length, int)
    assert isinstance(cfg.server.port, int)
    assert isinstance(cfg.security.rate_limit_per_minute, int)


@pytest.mark.parametrize(
    ("yaml_text", "pattern"),
    [
        (
            """\
            prometheus_url: "file:///tmp/prom"
            services:
              - name: s1
                promql: up
            """,
            "prometheus_url",
        ),
        (
            """\
            prometheus_url: "http://prom:9090"
            services: []
            """,
            "services",
        ),
        (
            """\
            prometheus_url: "http://prom:9090"
            services:
              - name: ""
                promql: up
            """,
            "service.name",
        ),
        (
            """\
            prometheus_url: "http://prom:9090"
            services:
              - name: s1
                promql: up
                layer: unknown
            """,
            "service.layer",
        ),
        (
            """\
            prometheus_url: "http://prom:9090"
            services:
              - name: s1
                promql: up
                channel: X
            """,
            "service.channel",
        ),
        (
            """\
            prometheus_url: "http://prom:9090"
            services:
              - name: s1
                promql: up
            scrape_interval_s: 0
            """,
            "scrape_interval_s",
        ),
        (
            """\
            prometheus_url: "http://prom:9090"
            services:
              - name: s1
                promql: up
            buffer_length: 3
            """,
            "buffer_length",
        ),
        (
            """\
            prometheus_url: "http://prom:9090"
            services:
              - name: s1
                promql: up
            security:
              mode: exposed
            """,
            "security.mode",
        ),
    ],
)
def test_load_config_rejects_invalid_values(
    tmp_path: Path, yaml_text: str, pattern: str
) -> None:
    path = tmp_path / "qw.yaml"
    path.write_text(textwrap.dedent(yaml_text), encoding="utf-8")

    with pytest.raises(ValueError, match=pattern):
        load_config(path)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ServiceDef(name="svc", promql="up", layer="micro", channel="S"),
        lambda: ThresholdConfig(r_bad_warn=2.0, r_bad_critical=1.0),
        lambda: CouplingConfig(strength=-0.1),
        lambda: AlertSink(url="ftp://example.com/hook"),
        lambda: AlertSink(url="https://example.com/hook", format="pager"),
        lambda: ServerConfig(host="", port=8080),
        lambda: ServerConfig(host="127.0.0.1", port=70000),
        lambda: SecurityConfig(mode="production", rate_limit_per_minute=0),
    ],
)
def test_direct_config_constructors_validate(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        factory()
