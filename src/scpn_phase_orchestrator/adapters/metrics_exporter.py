# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prometheus text exposition exporter

from __future__ import annotations

from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["MetricsExporter"]


class MetricsExporter:
    """Format UPDE state, regime, and latency as Prometheus text exposition."""

    def __init__(self, prefix: str = "spo") -> None:
        self._prefix = prefix

    def exposition_lines(
        self, upde_state: UPDEState, regime: str, latency_ms: float
    ) -> list[str]:
        """Build individual metric lines in Prometheus text format."""
        p = self._prefix
        lines: list[str] = []

        r_values = [layer.R for layer in upde_state.layers]
        r_global = sum(r_values) / len(r_values) if r_values else 0.0

        lines.append(f"# HELP {p}_r_global Global Kuramoto order parameter R")
        lines.append(f"# TYPE {p}_r_global gauge")
        lines.append(f'{p}_r_global{{regime="{regime}"}} {r_global:.6f}')

        lines.append(f"# HELP {p}_stability_proxy Mean R across layers")
        lines.append(f"# TYPE {p}_stability_proxy gauge")
        lines.append(
            f'{p}_stability_proxy{{regime="{regime}"}} {upde_state.stability_proxy:.6f}'
        )

        lines.append(f"# HELP {p}_pac_max Maximum phase-amplitude coupling")
        lines.append(f"# TYPE {p}_pac_max gauge")
        lines.append(f'{p}_pac_max{{regime="{regime}"}} {upde_state.pac_max:.6f}')

        lines.append(f"# HELP {p}_latency_ms UPDE step latency in milliseconds")
        lines.append(f"# TYPE {p}_latency_ms gauge")
        lines.append(f'{p}_latency_ms{{regime="{regime}"}} {latency_ms:.3f}')

        lines.append(f"# HELP {p}_layer_count Number of active UPDE layers")
        lines.append(f"# TYPE {p}_layer_count gauge")
        lines.append(f"{p}_layer_count {len(upde_state.layers)}")

        for i, layer in enumerate(upde_state.layers):
            lines.append(f'{p}_layer_r{{layer="{i}",regime="{regime}"}} {layer.R:.6f}')

        return lines

    def export(self, upde_state: UPDEState, regime: str, latency_ms: float) -> str:
        """Return full Prometheus text exposition as a single string."""
        return "\n".join(self.exposition_lines(upde_state, regime, latency_ms)) + "\n"
