# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from numpy.typing import NDArray


class PrometheusAdapter:
    """Fetch time-series metrics from a Prometheus endpoint."""

    def __init__(self, endpoint: str):
        self._endpoint = endpoint

    def fetch_metric(
        self, query: str, start: float, end: float, step: float
    ) -> NDArray:
        raise NotImplementedError(  # TODO(gh-21): implement fetch_metric
            "Prometheus adapter planned for v0.3, see ROADMAP.md"
        )
