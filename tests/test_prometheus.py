# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prometheus adapter tests

from __future__ import annotations

import io
import json
from unittest.mock import patch

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter


def _mock_response(body: dict):
    data = json.dumps(body).encode()
    resp = io.BytesIO(data)
    resp.status = 200
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: None
    resp.read = lambda: data
    return resp


class TestFetchMetric:
    def test_returns_values(self):
        body = {
            "status": "success",
            "data": {
                "result": [
                    {"values": [[1, "0.5"], [2, "0.7"], [3, "0.9"]]}
                ]
            },
        }
        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen",
            return_value=_mock_response(body),
        ):
            result = adapter.fetch_metric("up", 0, 100, 15)
        np.testing.assert_array_almost_equal(result, [0.5, 0.7, 0.9])

    def test_empty_result(self):
        body = {"status": "success", "data": {"result": []}}
        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen",
            return_value=_mock_response(body),
        ):
            result = adapter.fetch_metric("missing", 0, 100, 15)
        assert len(result) == 0

    def test_error_status(self):
        body = {"status": "error", "errorType": "bad_data"}
        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen",
            return_value=_mock_response(body),
        ):
            with pytest.raises(ValueError, match="status="):
                adapter.fetch_metric("bad", 0, 100, 15)

    def test_connection_error(self):
        from urllib.error import URLError

        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen",
            side_effect=URLError("refused"),
        ):
            with pytest.raises(ConnectionError):
                adapter.fetch_metric("up", 0, 100, 15)


class TestFetchInstant:
    def test_returns_scalar(self):
        body = {
            "status": "success",
            "data": {"result": [{"value": [1234, "42.5"]}]},
        }
        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen",
            return_value=_mock_response(body),
        ):
            result = adapter.fetch_instant("up")
        assert result == 42.5

    def test_empty_result_raises(self):
        body = {"status": "success", "data": {"result": []}}
        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen",
            return_value=_mock_response(body),
        ):
            with pytest.raises(ValueError, match="empty"):
                adapter.fetch_instant("missing")

    def test_strips_trailing_slash(self):
        adapter = PrometheusAdapter("http://localhost:9090/")
        assert adapter._endpoint == "http://localhost:9090"
