# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves configuration

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    BindingSpec,
    BoundaryDef,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ObjectivePartition,
    OscillatorFamily,
)

__all__ = [
    "ServiceDef",
    "ThresholdConfig",
    "AlertSink",
    "QueueWavesConfig",
    "ConfigCompiler",
]

_LAYER_ORDER = {"micro": 0, "meso": 1, "macro": 2}


@dataclass(frozen=True)
class ServiceDef:
    name: str
    promql: str
    layer: str  # micro, meso, macro
    channel: str = "P"  # P(hysical) or I(nformational)


@dataclass(frozen=True)
class ThresholdConfig:
    r_bad_warn: float = 0.50
    r_bad_critical: float = 0.70
    plv_cascade: float = 0.85
    imprint_chronic: float = 1.5
    cooldown_seconds: float = 300.0


@dataclass(frozen=True)
class CouplingConfig:
    strength: float = 0.50
    decay: float = 0.25


@dataclass(frozen=True)
class AlertSink:
    url: str
    format: str = "generic"  # "slack" or "generic"


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass
class QueueWavesConfig:
    prometheus_url: str
    services: list[ServiceDef]
    scrape_interval_s: float = 15.0
    buffer_length: int = 64
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    alert_sinks: list[AlertSink] = field(default_factory=list)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config(path: Path) -> QueueWavesConfig:
    """Load QueueWavesConfig from a YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    services = [
        ServiceDef(
            name=s["name"],
            promql=s["promql"],
            layer=s.get("layer", "micro"),
            channel=s.get("channel", "P"),
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
    return QueueWavesConfig(
        prometheus_url=raw["prometheus_url"],
        services=services,
        scrape_interval_s=raw.get("scrape_interval_s", 15.0),
        buffer_length=raw.get("buffer_length", 64),
        thresholds=thresholds,
        coupling=coupling_cfg,
        alert_sinks=sinks,
        server=server_cfg,
    )


class ConfigCompiler:
    """Converts user-facing QueueWavesConfig into an SPO BindingSpec."""

    def compile(self, cfg: QueueWavesConfig) -> BindingSpec:
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
                ext = "hilbert" if svc.channel == "P" else "event"
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
