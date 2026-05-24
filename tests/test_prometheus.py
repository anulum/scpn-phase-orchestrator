# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prometheus adapter tests

from __future__ import annotations

import io
import json
from typing import get_type_hints
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter
from tests.typing_contracts import assert_precise_ndarray_hint


def _mock_response(body: dict):
    data = json.dumps(body).encode()
    resp = io.BytesIO(data)
    resp.status = 200
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: None
    resp.read = lambda: data
    return resp


class TestPrometheusConfigValidation:
    @pytest.mark.parametrize("endpoint", ["", "localhost:9090", "file:///tmp/prom"])
    def test_endpoint_must_be_http_url(self, endpoint: str):
        with pytest.raises(ValueError, match="endpoint"):
            PrometheusAdapter(endpoint)

    @pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan"), True])
    def test_timeout_must_be_finite_and_positive(self, timeout: float):
        with pytest.raises(ValueError, match="timeout"):
            PrometheusAdapter("http://localhost:9090", timeout=timeout)

    def test_timeout_is_normalised_to_float(self):
        adapter = PrometheusAdapter("http://localhost:9090", timeout=1)
        assert adapter._timeout == 1.0


class TestFetchMetric:
    def test_public_array_contracts_are_parameterised(self):
        hint = get_type_hints(PrometheusAdapter.fetch_metric)["return"]
        assert_precise_ndarray_hint(hint)
        assert "float64" in str(hint)

    def test_returns_values(self):
        body = {
            "status": "success",
            "data": {"result": [{"values": [[1, "0.5"], [2, "0.7"], [3, "0.9"]]}]},
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
        with (
            patch(
                "scpn_phase_orchestrator.adapters.prometheus.urlopen",
                return_value=_mock_response(body),
            ),
            pytest.raises(ValueError, match="status="),
        ):
            adapter.fetch_metric("bad", 0, 100, 15)

    def test_connection_error(self):
        from urllib.error import URLError

        adapter = PrometheusAdapter("http://localhost:9090")
        with (
            patch(
                "scpn_phase_orchestrator.adapters.prometheus.urlopen",
                side_effect=URLError("refused"),
            ),
            pytest.raises(ConnectionError),
        ):
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
        with (
            patch(
                "scpn_phase_orchestrator.adapters.prometheus.urlopen",
                return_value=_mock_response(body),
            ),
            pytest.raises(ValueError, match="empty"),
        ):
            adapter.fetch_instant("missing")

    def test_strips_trailing_slash(self):
        adapter = PrometheusAdapter("http://localhost:9090/")
        assert adapter._endpoint == "http://localhost:9090"


# Pipeline wiring: PrometheusAdapter is an input adapter — tests above verify
# fetch_metric/fetch_instant that feed monitoring data into SPO.


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestPrometheusAdapter:
    def test_fetch_metric_rejects_invalid_query_and_range(self):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        adapter = PrometheusAdapter("http://localhost:9090")
        with pytest.raises(ValueError, match="non-empty string"):
            adapter.fetch_metric("  ", 0, 10, 1)
        with pytest.raises(ValueError, match="end must be >= start"):
            adapter.fetch_metric("up", 10, 0, 1)
        with pytest.raises(ValueError, match="step must be positive"):
            adapter.fetch_metric("up", 0, 10, 0)

    def test_fetch_instant_rejects_invalid_query(self):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        adapter = PrometheusAdapter("http://localhost:9090")
        with pytest.raises(ValueError, match="non-empty string"):
            adapter.fetch_instant("")

    def test_fetch_metric_url_encodes_query_text(self):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        mock_response = json.dumps(
            {
                "status": "success",
                "data": {"resultType": "matrix", "result": []},
            }
        ).encode()
        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            adapter.fetch_metric(
                'sum(rate(http_requests_total{job="api"}[5m]))',
                0,
                60,
                5,
            )

            request_arg = mock_urlopen.call_args.args[0]
            encoded = (
                "query=sum%28rate%28http_requests_total%7Bjob%3D%22api%22%7D"
                "%5B5m%5D%29%29"
            )
            assert encoded in request_arg.full_url

    def test_fetch_metric_success(self):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        mock_response = json.dumps(
            {
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [{"values": [[1, "0.5"], [2, "0.8"], [3, "0.3"]]}],
                },
            }
        ).encode()

        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = adapter.fetch_metric("up", 0, 10, 1)
            assert len(result) == 3
            np.testing.assert_allclose(result, [0.5, 0.8, 0.3])

    def test_fetch_metric_empty(self):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        mock_response = json.dumps(
            {"status": "success", "data": {"resultType": "matrix", "result": []}}
        ).encode()

        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = adapter.fetch_metric("up", 0, 10, 1)
            np.testing.assert_array_equal(result, np.array([], dtype=np.float64))
            request_arg = mock_urlopen.call_args.args[0]
            assert request_arg.full_url == (
                "http://localhost:9090/api/v1/query_range?"
                "query=up&start=0.0&end=10.0&step=1.0"
            )
            assert request_arg.headers["Accept"] == "application/json"

    def test_fetch_metric_network_error(self):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        adapter = PrometheusAdapter("http://localhost:9090")
        with (
            patch(
                "scpn_phase_orchestrator.adapters.prometheus.urlopen",
                side_effect=URLError("connection refused"),
            ),
            pytest.raises(ConnectionError),
        ):
            adapter.fetch_metric("up", 0, 10, 1)

    def test_fetch_metric_network_error_does_not_leak_endpoint_or_query(self):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        adapter = PrometheusAdapter("http://metrics.internal:9090/private")
        with (
            patch(
                "scpn_phase_orchestrator.adapters.prometheus.urlopen",
                side_effect=URLError(
                    "http://metrics.internal:9090/private/api/v1/query_range"
                    "?query=tenant_secret"
                ),
            ),
            pytest.raises(ConnectionError) as excinfo,
        ):
            adapter.fetch_metric("tenant_secret", 0, 10, 1)
        msg = str(excinfo.value)
        assert msg == "Prometheus query failed"
        assert "metrics.internal" not in msg
        assert "tenant_secret" not in msg
        assert excinfo.value.__cause__ is None

    def test_fetch_instant_network_error_does_not_leak_endpoint_or_query(self):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        adapter = PrometheusAdapter("http://metrics.internal:9090/private")
        with (
            patch(
                "scpn_phase_orchestrator.adapters.prometheus.urlopen",
                side_effect=URLError(
                    "http://metrics.internal:9090/private/api/v1/query"
                    "?query=tenant_secret"
                ),
            ),
            pytest.raises(ConnectionError) as excinfo,
        ):
            adapter.fetch_instant("tenant_secret")
        msg = str(excinfo.value)
        assert msg == "Prometheus query failed"
        assert "metrics.internal" not in msg
        assert "tenant_secret" not in msg
        assert excinfo.value.__cause__ is None

    def test_fetch_metric_bad_json(self):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"not json"
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with pytest.raises(ValueError):
                adapter.fetch_metric("up", 0, 10, 1)

    @pytest.mark.parametrize(
        "body",
        [
            [],
            {"status": "success"},
            {"status": "success", "data": []},
            {"status": "success", "data": {"result": {}}},
            {"status": "success", "data": {"result": [{}]}},
            {"status": "success", "data": {"result": [{"values": [["bad"]]}]}},
            {
                "status": "success",
                "data": {"result": [{"values": [[1, "nan"]]}]},
            },
        ],
    )
    def test_fetch_metric_rejects_malformed_response_shape(self, body: object):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        encoded = json.dumps(body).encode()
        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = encoded
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with pytest.raises(ValueError, match="Prometheus"):
                adapter.fetch_metric("up", 0, 10, 1)

    @pytest.mark.parametrize(
        "body",
        [
            {"status": "success", "data": {"result": [{}]}},
            {"status": "success", "data": {"result": [{"value": ["bad"]}]}},
            {"status": "success", "data": {"result": [{"value": [1, "inf"]}]}},
        ],
    )
    def test_fetch_instant_rejects_malformed_response_shape(self, body: object):
        from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter

        encoded = json.dumps(body).encode()
        adapter = PrometheusAdapter("http://localhost:9090")
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = encoded
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with pytest.raises(ValueError, match="Prometheus"):
                adapter.fetch_instant("up")


# Salvaged module-specific behavioural contracts from deleted sprint file.



def _mock_resp(body: dict):
    data = json.dumps(body).encode()
    resp = io.BytesIO(data)
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: None
    resp.read = lambda: data
    return resp


# ---------------------------------------------------------------------------
# Prometheus adapter: error handling
# ---------------------------------------------------------------------------


class TestPrometheusErrorHandling:
    """Verify that PrometheusAdapter raises correct exceptions
    for connection failures and API errors."""

    def test_connection_refused_raises_connection_error(self):
        adapter = PrometheusAdapter("http://localhost:9090")
        with (
            patch(
                "scpn_phase_orchestrator.adapters.prometheus.urlopen",
                side_effect=URLError("refused"),
            ),
            pytest.raises(ConnectionError),
        ):
            adapter.fetch_instant("up")

    def test_api_error_status_raises_value_error(self):
        adapter = PrometheusAdapter("http://localhost:9090")
        with (
            patch(
                "scpn_phase_orchestrator.adapters.prometheus.urlopen",
                return_value=_mock_resp({"status": "error"}),
            ),
            pytest.raises(ValueError, match="status="),
        ):
            adapter.fetch_instant("up")

    def test_successful_response_returns_data(self):
        """Valid Prometheus response must be parsed correctly."""
        adapter = PrometheusAdapter("http://localhost:9090")
        body = {
            "status": "success",
            "data": {"resultType": "vector", "result": [{"value": [1234, "0.5"]}]},
        }
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen",
            return_value=_mock_resp(body),
        ):
            result = adapter.fetch_instant("up")
        assert result is not None


# ---------------------------------------------------------------------------
# NPE: normalised phase entropy
# ---------------------------------------------------------------------------
