# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations


class OTelAdapter:
    """Extract phase-relevant events from OpenTelemetry span data."""

    def __init__(self, service_name: str):
        self._service_name = service_name

    def extract_events(self, span_data: list[dict]) -> list[float]:
        raise NotImplementedError("OTel adapter planned for v0.3, see ROADMAP.md")
