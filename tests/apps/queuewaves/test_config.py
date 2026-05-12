# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves config tests

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from dataclasses import asdict
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


def test_load_config_recursion_error_is_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scpn_phase_orchestrator.apps.queuewaves import config as config_mod

    path = tmp_path / "qw.yaml"
    path.write_text(
        "prometheus_url: http://prom:9090\nservices: []\n", encoding="utf-8"
    )

    def raise_recursion(_: str) -> object:
        raise RecursionError("nested YAML")

    monkeypatch.setattr(config_mod.yaml, "safe_load", raise_recursion)
    with pytest.raises(ValueError, match="QueueWaves config YAML parse error"):
        load_config(path)


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


def test_config_compiler_accepts_standard_and_named_channels(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent("""\
        prometheus_url: "http://prom:9090"
        services:
          - name: latency
            promql: histogram_quantile(0.99, rate(http_duration_bucket[5m]))
            layer: micro
            channel: P
          - name: release-state
            promql: deployment_state
            layer: meso
            channel: S
          - name: retry-budget
            promql: retry_budget_remaining
            layer: macro
            channel: RetryBudget
            extractor_type: event
    """)
    path = tmp_path / "qw-nchannel.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    spec = ConfigCompiler().compile(load_config(path))

    assert spec.oscillator_families["latency"].channel == "P"
    assert spec.oscillator_families["latency"].extractor_type == "hilbert"
    assert spec.oscillator_families["release-state"].channel == "S"
    assert spec.oscillator_families["release-state"].extractor_type == "ring"
    assert spec.oscillator_families["retry-budget"].channel == "RetryBudget"
    assert spec.oscillator_families["retry-budget"].extractor_type == "event"


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


def test_production_template_requires_auth_and_rate_limit() -> None:
    template = Path("domainpacks/queuewaves/queuewaves.production.yaml")

    cfg = load_config(template)

    assert cfg.security.mode == "production"
    assert cfg.security.api_key_env == "QUEUEWAVES_API_KEY"
    assert cfg.security.rate_limit_per_minute > 0
    assert cfg.server.host == "0.0.0.0"
    assert cfg.services
    assert cfg.alert_sinks


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


def test_load_config_rejects_malformed_yaml_without_leaking_secret_path(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "vault" / "secret" / "queuewaves.yaml"
    secret_path.parent.mkdir(parents=True)
    secret_path.write_text("- prometheus_url: http://prom:9090\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_config(secret_path)

    message = str(exc_info.value)
    assert message == "QueueWaves config must be a YAML mapping, got list"
    assert str(secret_path) not in message
    assert "vault" not in message
    assert "secret" not in message


def test_load_config_requires_prometheus_url_without_leaking_secret_path(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "secrets" / "queuewaves.yaml"
    secret_path.parent.mkdir()
    secret_path.write_text(
        textwrap.dedent("""\
            services:
              - name: s1
                promql: up
        """),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(secret_path)

    message = str(exc_info.value)
    assert message == "QueueWaves config missing required key 'prometheus_url'"
    assert str(secret_path) not in message
    assert "secrets" not in message


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ThresholdConfig(r_bad_warn=object()),
        lambda: CouplingConfig(decay="not-a-number"),
        lambda: QueueWavesConfig(
            prometheus_url="http://prom:9090",
            services=[ServiceDef(name="s1", promql="up", layer="micro")],
            scrape_interval_s=None,
        ),
    ],
)
def test_numeric_config_fields_reject_non_floatable_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ServerConfig(port=True),
        lambda: SecurityConfig(rate_limit_per_minute=False),
    ],
)
def test_integer_config_fields_reject_boolean_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ServerConfig(port="not-a-port"),
        lambda: SecurityConfig(rate_limit_per_minute="bursty"),
        lambda: QueueWavesConfig(
            prometheus_url="http://prom:9090",
            services=[ServiceDef(name="s1", promql="up", layer="micro")],
            buffer_length="many",
        ),
    ],
)
def test_integer_config_fields_reject_non_integral_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        factory()


def test_config_compiler_fails_closed_for_unresolved_service_extractor() -> None:
    service = ServiceDef(name="s1", promql="up", layer="micro")
    object.__setattr__(service, "extractor_type", None)
    cfg = QueueWavesConfig(
        prometheus_url="http://prom:9090",
        services=[service],
    )

    with pytest.raises(ValueError, match="service.extractor_type was not resolved"):
        ConfigCompiler().compile(cfg)


def test_compiled_binding_serialisation_is_deterministic() -> None:
    cfg = QueueWavesConfig(
        prometheus_url="http://prom:9090",
        services=[
            ServiceDef(name="macro-tput", promql="rate(done[5m])", layer="macro"),
            ServiceDef(name="micro-latency", promql="rate(latency[1m])", layer="micro"),
            ServiceDef(name="meso-errors", promql="rate(errors[5m])", layer="meso"),
        ],
        scrape_interval_s=5.0,
        coupling=CouplingConfig(strength=0.4, decay=0.2),
    )

    spec = ConfigCompiler().compile(cfg)
    serialised_once = json.dumps(asdict(spec), sort_keys=True)
    serialised_twice = json.dumps(asdict(ConfigCompiler().compile(cfg)), sort_keys=True)

    assert [layer.name for layer in spec.layers] == ["micro", "meso", "macro"]
    assert serialised_once == serialised_twice
    assert '"sample_period_s": 5.0' in serialised_once
    assert '"base_strength": 0.4' in serialised_once


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
                channel: 1bad
            """,
            "service.channel",
        ),
        (
            """\
            prometheus_url: "http://prom:9090"
            services:
              - name: s1
                promql: up
                channel: ExtraChannel
            """,
            "service.extractor_type",
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
        lambda: ServiceDef(name="svc", promql="up", layer="micro", channel="1bad"),
        lambda: ServiceDef(name="svc", promql="up", layer="micro", channel="Extra"),
        lambda: ServiceDef(
            name="svc",
            promql="up",
            layer="micro",
            channel="Extra",
            extractor_type="unknown",
        ),
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
