# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prometheus text exposition exporter

"""Prometheus text exposition formatter for reduced UPDE metrics.

``MetricsExporter`` validates metric prefix syntax, regime labels, latency, and
finite UPDE metric values before producing deterministic Prometheus text lines.
It performs no HTTP serving, scraping, registry mutation, or background export;
callers decide where the generated text is published.
"""

from __future__ import annotations

import re
from math import isfinite
from numbers import Real

from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["MetricsExporter"]

_PROMETHEUS_PREFIX_RE = re.compile(r"^[A-Za-z_:][A-Za-z0-9_:]*$")


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _validated_latency_ms(latency_ms: float) -> float:
    if (
        not isinstance(latency_ms, Real)
        or isinstance(latency_ms, bool)
        or not isfinite(float(latency_ms))
        or latency_ms < 0.0
    ):
        raise ValueError("latency_ms must be a finite non-negative real value")
    return float(latency_ms)


def _validated_regime(regime: object) -> str:
    if not isinstance(regime, str) or not regime:
        raise ValueError("regime must be a non-empty string")
    if "\x00" in regime:
        raise ValueError("regime must not contain NUL characters")
    return regime


def _validated_finite_metric(value: object, *, field: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


class MetricsExporter:
    """Format UPDE state, regime, and latency as Prometheus text exposition."""

    def __init__(self, prefix: str = "spo") -> None:
        if not isinstance(prefix, str) or not _PROMETHEUS_PREFIX_RE.fullmatch(prefix):
            raise ValueError(
                "prefix must be a valid Prometheus metric prefix "
                "([A-Za-z_:][A-Za-z0-9_:]*)"
            )
        self._prefix = prefix

    def exposition_lines(
        self, upde_state: UPDEState, regime: str, latency_ms: float
    ) -> list[str]:
        """Build individual metric lines in Prometheus text format."""
        p = self._prefix
        regime_label = _escape_label_value(_validated_regime(regime))
        latency_ms = _validated_latency_ms(latency_ms)
        lines: list[str] = []

        r_values = [
            _validated_finite_metric(layer.R, field=f"layer {idx} R")
            for idx, layer in enumerate(upde_state.layers)
        ]
        r_global = sum(r_values) / len(r_values) if r_values else 0.0
        stability_proxy = _validated_finite_metric(
            upde_state.stability_proxy,
            field="stability_proxy",
        )
        pac_max = _validated_finite_metric(upde_state.pac_max, field="pac_max")

        lines.append(f"# HELP {p}_r_global Global Kuramoto order parameter R")
        lines.append(f"# TYPE {p}_r_global gauge")
        lines.append(f'{p}_r_global{{regime="{regime_label}"}} {r_global:.6f}')

        lines.append(f"# HELP {p}_stability_proxy Mean R across layers")
        lines.append(f"# TYPE {p}_stability_proxy gauge")
        lines.append(
            f'{p}_stability_proxy{{regime="{regime_label}"}} {stability_proxy:.6f}'
        )

        lines.append(f"# HELP {p}_pac_max Maximum phase-amplitude coupling")
        lines.append(f"# TYPE {p}_pac_max gauge")
        lines.append(f'{p}_pac_max{{regime="{regime_label}"}} {pac_max:.6f}')

        lines.append(f"# HELP {p}_latency_ms UPDE step latency in milliseconds")
        lines.append(f"# TYPE {p}_latency_ms gauge")
        lines.append(f'{p}_latency_ms{{regime="{regime_label}"}} {latency_ms:.3f}')

        lines.append(f"# HELP {p}_layer_count Number of active UPDE layers")
        lines.append(f"# TYPE {p}_layer_count gauge")
        lines.append(f"{p}_layer_count {len(upde_state.layers)}")

        for i, r_value in enumerate(r_values):
            lines.append(
                f'{p}_layer_r{{layer="{i}",regime="{regime_label}"}} {r_value:.6f}'
            )

        return lines

    def export(self, upde_state: UPDEState, regime: str, latency_ms: float) -> str:
        """Return full Prometheus text exposition as a single string."""
        return "\n".join(self.exposition_lines(upde_state, regime, latency_ms)) + "\n"
