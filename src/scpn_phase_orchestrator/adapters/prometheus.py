# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prometheus metrics adapter

"""Prometheus HTTP adapter for validated instant and range metric queries.

``PrometheusAdapter`` validates endpoint URLs, timeouts, query text, range
bounds, and step size before issuing standard Prometheus API requests. Network
failures are reported as ``ConnectionError`` and malformed responses as
``ValueError``. The adapter fetches metric values only; it does not run a server
or mutate orchestration state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, TypeAlias, cast
from urllib.error import URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
FloatLike: TypeAlias = str | bytes | bytearray | memoryview | int | float


def _reject_json_constant(value: str) -> None:
    """Raise if the JSON value is a forbidden constant."""
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_duplicate_json_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Raise if the JSON object has duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
        result[key] = value
    return result


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

        Parameters
        ----------
        query : str
            PromQL query string.
        start : float
            Range start time as a UNIX timestamp.
        end : float
            Range end time as a UNIX timestamp.
        step : float
            Sampling step in seconds.

        Returns
        -------
        FloatArray
            The range query values as a 1-D float array.

        Raises
        ------
        ConnectionError
            If the Prometheus server is unreachable.
        ValueError
            If the query or response is invalid.
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
            # Endpoint scheme is constrained to http/https in __init__, so
            # B310's file://-style scheme concern does not apply here.
            with urlopen(req, timeout=self._timeout) as resp:  # nosec B310
                body = _load_response_body(resp.read())
        except (URLError, OSError):
            raise ConnectionError("Prometheus query failed") from None

        if body.get("status") != "success":
            raise ValueError(f"Prometheus returned status={body.get('status')}")

        results = _response_results(body)
        if not results:
            return np.array([], dtype=np.float64)

        values = _range_values(results[0])
        result: FloatArray = np.array(values, dtype=np.float64)
        return result

    def fetch_instant(self, query: str) -> float:
        """Query Prometheus instant API, return scalar value.

        Parameters
        ----------
        query : str
            PromQL query string.

        Returns
        -------
        float
            The instant query scalar value.

        Raises
        ------
        ConnectionError
            If the Prometheus server is unreachable.
        ValueError
            If the query or response is invalid.
        """
        query_text = _require_query_text(query)
        params = urlencode({"query": query_text})
        url = f"{self._endpoint}/api/v1/query?{params}"
        req = Request(url, headers={"Accept": "application/json"})
        try:
            # Endpoint scheme is constrained to http/https in __init__, so
            # B310's file://-style scheme concern does not apply here.
            with urlopen(req, timeout=self._timeout) as resp:  # nosec B310
                body = _load_response_body(resp.read())
        except (URLError, OSError):
            raise ConnectionError("Prometheus query failed") from None

        if body.get("status") != "success":
            raise ValueError(f"Prometheus returned status={body.get('status')}")

        results = _response_results(body)
        if not results:
            raise ValueError("Prometheus returned empty result set")

        return _instant_value(results[0])


def _require_query_text(query: str) -> str:
    """Return the validated non-empty PromQL query text, else raise."""
    if not isinstance(query, str):
        raise ValueError("Prometheus query must be a non-empty string")
    query_text = query.strip()
    if not query_text:
        raise ValueError("Prometheus query must be a non-empty string")
    return query_text


def _require_finite_float(value: object, field_name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"Prometheus {field_name} must be finite")
    if not isinstance(value, (str, bytes, bytearray, memoryview, int, float)):
        raise ValueError(f"Prometheus {field_name} must be finite")
    try:
        parsed = float(cast(FloatLike, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Prometheus {field_name} must be finite") from exc
    if not isfinite(parsed):
        raise ValueError(f"Prometheus {field_name} must be finite")
    return parsed


def _load_response_body(raw: bytes) -> Mapping[str, Any]:
    """Return the parsed JSON body from a Prometheus response, else raise."""
    try:
        body = json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_key,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Prometheus returned malformed JSON") from exc
    except ValueError as exc:
        raise ValueError(
            "Prometheus JSON must contain finite JSON numbers and unique object keys"
        ) from exc
    if not isinstance(body, Mapping):
        raise ValueError("Prometheus response must be a JSON object")
    return body


def _response_results(body: Mapping[str, Any]) -> Sequence[Any]:
    """Return the result list from a Prometheus response, else raise."""
    data = body.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Prometheus response data must be an object")
    results = data.get("result")
    if not isinstance(results, list):
        raise ValueError("Prometheus response result must be a list")
    return results


def _require_sample_pair(sample: object, *, field_name: str) -> Sequence[Any]:
    """Return a validated [timestamp, value] sample pair, else raise."""
    if (
        not isinstance(sample, Sequence)
        or isinstance(sample, (bytes, str))
        or len(sample) != 2
    ):
        raise ValueError(f"Prometheus {field_name} must contain sample pairs")
    return sample


def _require_sample_timestamp(value: object) -> float:
    """Return the validated sample timestamp, else raise."""
    timestamp = _require_finite_float(value, "sample timestamp")
    if timestamp < 0.0:
        raise ValueError("Prometheus sample timestamp must be non-negative")
    return timestamp


def _range_values(series: object) -> list[float]:
    """Return the (timestamp, value) series from a range query result."""
    if not isinstance(series, Mapping):
        raise ValueError("Prometheus result series must be an object")
    samples = series.get("values")
    if not isinstance(samples, list):
        raise ValueError("Prometheus range result values must be a list")

    values: list[float] = []
    for sample in samples:
        pair = _require_sample_pair(sample, field_name="range result values")
        _require_sample_timestamp(pair[0])
        values.append(_require_finite_float(pair[1], "sample value"))
    return values


def _instant_value(series: object) -> float:
    """Return the single (timestamp, value) from an instant query result."""
    if not isinstance(series, Mapping):
        raise ValueError("Prometheus result series must be an object")
    pair = _require_sample_pair(series.get("value"), field_name="instant result value")
    _require_sample_timestamp(pair[0])
    return _require_finite_float(pair[1], "sample value")
