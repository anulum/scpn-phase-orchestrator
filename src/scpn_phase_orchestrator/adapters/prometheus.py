# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prometheus metrics adapter

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray


class PrometheusAdapter:
    """Fetch time-series metrics from a Prometheus endpoint."""

    def __init__(self, endpoint: str, timeout: float = 10.0):
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def fetch_metric(
        self, query: str, start: float, end: float, step: float
    ) -> NDArray:
        """Query Prometheus range API, return values as 1-D float array.

        Raises ConnectionError on network failure, ValueError on bad response.
        """
        url = (
            f"{self._endpoint}/api/v1/query_range"
            f"?query={query}&start={start}&end={end}&step={step}"
        )
        req = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(req, timeout=self._timeout) as resp:  # nosec B310
                body = json.loads(resp.read())
        except (URLError, OSError):
            raise ConnectionError("Prometheus query failed") from None

        if body.get("status") != "success":
            raise ValueError(f"Prometheus returned status={body.get('status')}")

        results = body.get("data", {}).get("result", [])
        if not results:
            return np.array([], dtype=np.float64)

        # First result series: extract values (timestamp, value pairs)
        values = [float(v[1]) for v in results[0].get("values", [])]
        return np.array(values, dtype=np.float64)

    def fetch_instant(self, query: str) -> float:
        """Query Prometheus instant API, return scalar value."""
        url = f"{self._endpoint}/api/v1/query?query={query}"
        req = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(req, timeout=self._timeout) as resp:  # nosec B310
                body = json.loads(resp.read())
        except (URLError, OSError):
            raise ConnectionError("Prometheus query failed") from None

        if body.get("status") != "success":
            raise ValueError(f"Prometheus returned status={body.get('status')}")

        results = body.get("data", {}).get("result", [])
        if not results:
            raise ValueError("Prometheus returned empty result set")

        return float(results[0]["value"][1])
