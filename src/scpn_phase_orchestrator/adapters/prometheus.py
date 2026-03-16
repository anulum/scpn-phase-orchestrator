# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prometheus metrics adapter

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
