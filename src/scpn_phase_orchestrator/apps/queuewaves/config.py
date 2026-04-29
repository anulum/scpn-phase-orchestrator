# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves configuration

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from urllib.parse import urlparse

import yaml

from scpn_phase_orchestrator.binding.types import (
    VALID_EXTRACTORS,
    ActuatorMapping,
    BindingSpec,
    BoundaryDef,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ObjectivePartition,
    OscillatorFamily,
    is_valid_channel_id,
    resolve_extractor_type,
)

__all__ = [
    "ServiceDef",
    "ThresholdConfig",
    "AlertSink",
    "SecurityConfig",
    "QueueWavesConfig",
    "ConfigCompiler",
]

_LAYER_ORDER = {"micro": 0, "meso": 1, "macro": 2}
_STANDARD_CHANNEL_EXTRACTORS = {"P": "hilbert", "I": "event", "S": "ring"}
_VALID_ALERT_FORMATS = frozenset({"generic", "slack"})
_VALID_SECURITY_MODES = frozenset({"development", "production"})


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_finite_non_negative(value: float, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and non-negative") from exc
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return parsed


def _require_int_range(
    value: int,
    field_name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return parsed


def _require_http_url(value: str, field_name: str) -> str:
    parsed_value = _require_non_empty(value, field_name)
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{field_name} must be an http(s) URL")
    return parsed_value


@dataclass(frozen=True)
class ServiceDef:
    """A monitored service: name, PromQL query, hierarchy layer, and channel."""

    name: str
    promql: str
    layer: str  # micro, meso, macro
    channel: str = "P"
    extractor_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty(self.name, "service.name"))
        object.__setattr__(
            self, "promql", _require_non_empty(self.promql, "service.promql")
        )
        if self.layer not in _LAYER_ORDER:
            raise ValueError("service.layer must be micro, meso, or macro")
        if not is_valid_channel_id(self.channel):
            raise ValueError("service.channel must match [A-Za-z][A-Za-z0-9_-]{0,63}")
        extractor_type = self.extractor_type
        if extractor_type is None:
            extractor_type = _STANDARD_CHANNEL_EXTRACTORS.get(self.channel)
            if extractor_type is None:
                raise ValueError(
                    "service.extractor_type is required for named service channels"
                )
        else:
            extractor_type = resolve_extractor_type(extractor_type)
        if extractor_type not in VALID_EXTRACTORS:
            raise ValueError("service.extractor_type must be a valid extractor")
        object.__setattr__(self, "extractor_type", extractor_type)


@dataclass(frozen=True)
class ThresholdConfig:
    """Anomaly detection thresholds for R_bad, PLV cascade, and chronic imprint."""

    r_bad_warn: float = 0.50
    r_bad_critical: float = 0.70
    plv_cascade: float = 0.85
    imprint_chronic: float = 1.5
    cooldown_seconds: float = 300.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "r_bad_warn",
            _require_finite_non_negative(self.r_bad_warn, "thresholds.r_bad_warn"),
        )
        object.__setattr__(
            self,
            "r_bad_critical",
            _require_finite_non_negative(
                self.r_bad_critical, "thresholds.r_bad_critical"
            ),
        )
        object.__setattr__(
            self,
            "plv_cascade",
            _require_finite_non_negative(self.plv_cascade, "thresholds.plv_cascade"),
        )
        object.__setattr__(
            self,
            "imprint_chronic",
            _require_finite_non_negative(
                self.imprint_chronic, "thresholds.imprint_chronic"
            ),
        )
        object.__setattr__(
            self,
            "cooldown_seconds",
            _require_finite_non_negative(
                self.cooldown_seconds, "thresholds.cooldown_seconds"
            ),
        )
        if self.r_bad_warn > self.r_bad_critical:
            raise ValueError("thresholds.r_bad_warn must be <= r_bad_critical")


@dataclass(frozen=True)
class CouplingConfig:
    """Phase coupling strength and decay parameters."""

    strength: float = 0.50
    decay: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strength",
            _require_finite_non_negative(self.strength, "coupling.strength"),
        )
        object.__setattr__(
            self, "decay", _require_finite_non_negative(self.decay, "coupling.decay")
        )


@dataclass(frozen=True)
class AlertSink:
    """Webhook endpoint for anomaly alerts."""

    url: str
    format: str = "generic"  # "slack" or "generic"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "url", _require_http_url(self.url, "alert_sinks[].url")
        )
        if self.format not in _VALID_ALERT_FORMATS:
            raise ValueError("alert_sinks[].format must be generic or slack")


@dataclass(frozen=True)
class ServerConfig:
    """QueueWaves HTTP server bind address."""

    host: str = "127.0.0.1"
    port: int = 8080

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", _require_non_empty(self.host, "server.host"))
        object.__setattr__(
            self, "port", _require_int_range(self.port, "server.port", 0, 65535)
        )


@dataclass(frozen=True)
class SecurityConfig:
    """QueueWaves production network security policy."""

    mode: str = "development"
    api_key_env: str = "QUEUEWAVES_API_KEY"
    rate_limit_per_minute: int = 120

    def __post_init__(self) -> None:
        if self.mode not in _VALID_SECURITY_MODES:
            raise ValueError("security.mode must be development or production")
        object.__setattr__(
            self,
            "api_key_env",
            _require_non_empty(self.api_key_env, "security.api_key_env"),
        )
        object.__setattr__(
            self,
            "rate_limit_per_minute",
            _require_int_range(
                self.rate_limit_per_minute, "security.rate_limit_per_minute", 1
            ),
        )


@dataclass
class QueueWavesConfig:
    """Top-level configuration for a QueueWaves deployment."""

    prometheus_url: str
    services: list[ServiceDef]
    scrape_interval_s: float = 15.0
    buffer_length: int = 64
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    alert_sinks: list[AlertSink] = field(default_factory=list)
    server: ServerConfig = field(default_factory=ServerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __post_init__(self) -> None:
        self.prometheus_url = _require_http_url(self.prometheus_url, "prometheus_url")
        if not self.services:
            raise ValueError("services must contain at least one service")
        self.scrape_interval_s = _require_finite_non_negative(
            self.scrape_interval_s, "scrape_interval_s"
        )
        if self.scrape_interval_s <= 0.0:
            raise ValueError("scrape_interval_s must be > 0")
        self.buffer_length = _require_int_range(self.buffer_length, "buffer_length", 4)


def load_config(path: Path) -> QueueWavesConfig:
    """Load QueueWavesConfig from a YAML file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (RecursionError, yaml.YAMLError):
        raise ValueError("QueueWaves config YAML parse error") from None
    if not isinstance(raw, dict):
        msg = f"QueueWaves config must be a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)
    if "prometheus_url" not in raw:
        msg = "QueueWaves config missing required key 'prometheus_url'"
        raise ValueError(msg)
    services = [
        ServiceDef(
            name=s["name"],
            promql=s["promql"],
            layer=s.get("layer", "micro"),
            channel=s.get("channel", "P"),
            extractor_type=s.get("extractor_type"),
        )
        for s in raw.get("services", [])
    ]
    thresh_raw = raw.get("thresholds", {})
    thresholds = ThresholdConfig(
        r_bad_warn=thresh_raw.get("r_bad_warn", 0.50),
        r_bad_critical=thresh_raw.get("r_bad_critical", 0.70),
        plv_cascade=thresh_raw.get("plv_cascade", 0.85),
        imprint_chronic=thresh_raw.get("imprint_chronic", 1.5),
        cooldown_seconds=thresh_raw.get("cooldown_seconds", 300.0),
    )
    coup_raw = raw.get("coupling", {})
    coupling_cfg = CouplingConfig(
        strength=coup_raw.get("strength", 0.50),
        decay=coup_raw.get("decay", 0.25),
    )
    sinks = [
        AlertSink(url=s["url"], format=s.get("format", "generic"))
        for s in raw.get("alert_sinks", [])
    ]
    srv_raw = raw.get("server", {})
    server_cfg = ServerConfig(
        host=srv_raw.get("host", "127.0.0.1"),
        port=srv_raw.get("port", 8080),
    )
    sec_raw = raw.get("security", {})
    security_cfg = SecurityConfig(
        mode=sec_raw.get("mode", "development"),
        api_key_env=sec_raw.get("api_key_env", "QUEUEWAVES_API_KEY"),
        rate_limit_per_minute=sec_raw.get("rate_limit_per_minute", 120),
    )
    return QueueWavesConfig(
        prometheus_url=raw["prometheus_url"],
        services=services,
        scrape_interval_s=raw.get("scrape_interval_s", 15.0),
        buffer_length=raw.get("buffer_length", 64),
        thresholds=thresholds,
        coupling=coupling_cfg,
        alert_sinks=sinks,
        server=server_cfg,
        security=security_cfg,
    )


class ConfigCompiler:
    """Converts user-facing QueueWavesConfig into an SPO BindingSpec."""

    def compile(self, cfg: QueueWavesConfig) -> BindingSpec:
        """Translate QueueWavesConfig into an SPO BindingSpec."""
        layers_by_name: dict[str, list[ServiceDef]] = {}
        for svc in cfg.services:
            layers_by_name.setdefault(svc.layer, []).append(svc)

        layers: list[HierarchyLayer] = []
        osc_families: dict[str, OscillatorFamily] = {}
        good_layers: list[int] = []
        bad_layers: list[int] = []

        for layer_name, svcs in sorted(
            layers_by_name.items(), key=lambda kv: _LAYER_ORDER.get(kv[0], 99)
        ):
            idx = _LAYER_ORDER.get(layer_name, len(layers))
            osc_ids = [svc.name for svc in svcs]
            layers.append(
                HierarchyLayer(
                    name=layer_name,
                    index=idx,
                    oscillator_ids=osc_ids,
                )
            )

            for svc in svcs:
                ext = svc.extractor_type
                if ext is None:
                    raise ValueError("service.extractor_type was not resolved")
                osc_families[svc.name] = OscillatorFamily(
                    channel=svc.channel, extractor_type=ext, config={}
                )

            # micro = bad (retry storms sync), macro = good (coordinated throughput)
            if layer_name == "micro":
                bad_layers.append(idx)
            else:
                good_layers.append(idx)

        boundaries = [
            BoundaryDef(
                name="r_bad_warn",
                variable="R_bad",
                lower=None,
                upper=cfg.thresholds.r_bad_warn,
                severity="soft",
            ),
            BoundaryDef(
                name="r_bad_critical",
                variable="R_bad",
                lower=None,
                upper=cfg.thresholds.r_bad_critical,
                severity="hard",
            ),
        ]
        actuators = [
            ActuatorMapping(
                name="coupling_adj", knob="K", scope="global", limits=(-0.5, 0.5)
            ),
            ActuatorMapping(
                name="damping_adj", knob="zeta", scope="global", limits=(0.0, 0.5)
            ),
        ]

        return BindingSpec(
            name="queuewaves",
            version="0.1.0",
            safety_tier="production",
            sample_period_s=cfg.scrape_interval_s,
            control_period_s=cfg.scrape_interval_s,
            layers=layers,
            oscillator_families=osc_families,
            coupling=CouplingSpec(
                base_strength=cfg.coupling.strength,
                decay_alpha=cfg.coupling.decay,
                templates={},
            ),
            drivers=DriverSpec(physical={}, informational={}, symbolic={}),
            objectives=ObjectivePartition(
                good_layers=good_layers, bad_layers=bad_layers
            ),
            boundaries=boundaries,
            actuators=actuators,
        )
