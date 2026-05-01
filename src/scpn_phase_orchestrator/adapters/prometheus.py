# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prometheus metrics adapter

from __future__ import annotations

import json
from math import isfinite
from typing import TypeAlias
from urllib.error import URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class PrometheusAdapter:
    """Fetch time-series metrics from a Prometheus endpoint."""

    def __init__(self, endpoint: str, timeout: float = 10.0):
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("Prometheus endpoint must be a non-empty http(s) URL")
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("Prometheus endpoint must be a non-empty http(s) URL")
        if isinstance(timeout, bool):
            raise ValueError("Prometheus timeout must be finite and positive")
        try:
            parsed_timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("Prometheus timeout must be finite and positive") from exc
        if not isfinite(parsed_timeout) or parsed_timeout <= 0.0:
            raise ValueError("Prometheus timeout must be finite and positive")
        self._endpoint = endpoint.rstrip("/")
        self._timeout = parsed_timeout

    def fetch_metric(
        self, query: str, start: float, end: float, step: float
    ) -> FloatArray:
        """Query Prometheus range API, return values as 1-D float array.

        Raises ConnectionError on network failure, ValueError on bad response.
        """
        query_text = _require_query_text(query)
        start_f = _require_finite_float(start, "start")
        end_f = _require_finite_float(end, "end")
        step_f = _require_finite_float(step, "step")
        if end_f < start_f:
            raise ValueError("Prometheus end must be >= start")
        if step_f <= 0.0:
            raise ValueError("Prometheus step must be positive")
        params = urlencode(
            {"query": query_text, "start": start_f, "end": end_f, "step": step_f}
        )
        url = f"{self._endpoint}/api/v1/query_range?{params}"
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
        result: FloatArray = np.array(values, dtype=np.float64)
        return result

    def fetch_instant(self, query: str) -> float:
        """Query Prometheus instant API, return scalar value."""
        query_text = _require_query_text(query)
        params = urlencode({"query": query_text})
        url = f"{self._endpoint}/api/v1/query?{params}"
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


def _require_query_text(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("Prometheus query must be a non-empty string")
    query_text = query.strip()
    if not query_text:
        raise ValueError("Prometheus query must be a non-empty string")
    return query_text


def _require_finite_float(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Prometheus {field_name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Prometheus {field_name} must be finite") from exc
    if not isfinite(parsed):
        raise ValueError(f"Prometheus {field_name} must be finite")
    return parsed
