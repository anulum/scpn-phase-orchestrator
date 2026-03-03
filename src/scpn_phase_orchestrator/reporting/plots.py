# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations


class CoherencePlot:
    """Audit log visualisation (matplotlib-based, planned for v0.3)."""

    def __init__(self, log_data: list[dict]):
        self._data = log_data

    def plot_r_timeline(self, output_path: str) -> None:
        raise NotImplementedError("Plotting planned for v0.3, see ROADMAP.md")

    def plot_regime_timeline(self, output_path: str) -> None:
        raise NotImplementedError("Plotting planned for v0.3, see ROADMAP.md")

    def plot_action_audit(self, output_path: str) -> None:
        raise NotImplementedError("Plotting planned for v0.3, see ROADMAP.md")
