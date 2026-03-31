# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coverage tests for adapter modules

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestSampleBuffer:
    def test_push_and_get(self):
        from scpn_phase_orchestrator.adapters.hardware_io import SampleBuffer

        buf = SampleBuffer(capacity=10, n_channels=2)
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        buf.push(data)
        recent = buf.get_recent(3)
        assert recent.shape == (2, 3)
        np.testing.assert_array_equal(recent, data)

    def test_get_recent_empty(self):
        from scpn_phase_orchestrator.adapters.hardware_io import SampleBuffer

        buf = SampleBuffer(capacity=10, n_channels=2)
        recent = buf.get_recent(5)
        assert recent.shape == (2, 0)

    def test_wrap_around(self):
        from scpn_phase_orchestrator.adapters.hardware_io import SampleBuffer

        buf = SampleBuffer(capacity=4, n_channels=1)
        for i in range(6):
            buf.push(np.array([[float(i)]]))
        recent = buf.get_recent(4)
        assert recent.shape == (1, 4)
        np.testing.assert_array_equal(recent[0], [2, 3, 4, 5])

    def test_get_more_than_available(self):
        from scpn_phase_orchestrator.adapters.hardware_io import SampleBuffer

        buf = SampleBuffer(capacity=10, n_channels=1)
        buf.push(np.array([[1.0, 2.0]]))
        recent = buf.get_recent(5)
        assert recent.shape == (1, 2)


class TestSimulatedBoardAdapter:
    def test_start_stop(self):
        from scpn_phase_orchestrator.adapters.hardware_io import SimulatedBoardAdapter

        adapter = SimulatedBoardAdapter(n_channels=4, sample_rate=256)
        adapter.start()
        assert adapter._running
        adapter.stop()
        assert not adapter._running

    def test_properties(self):
        from scpn_phase_orchestrator.adapters.hardware_io import SimulatedBoardAdapter

        adapter = SimulatedBoardAdapter(n_channels=8, sample_rate=512)
        assert adapter.sample_rate == 512
        assert adapter.n_channels == 8

    def test_get_channel_data(self):
        from scpn_phase_orchestrator.adapters.hardware_io import SimulatedBoardAdapter

        adapter = SimulatedBoardAdapter(n_channels=4)
        adapter.start()
        data = adapter.get_channel_data(0, n_samples=100)
        assert data.shape == (100,)
        assert np.all(np.abs(data) <= 1.0)

    def test_get_all_eeg(self):
        from scpn_phase_orchestrator.adapters.hardware_io import SimulatedBoardAdapter

        adapter = SimulatedBoardAdapter(n_channels=4)
        adapter.start()
        data = adapter.get_all_eeg(n_samples=100)
        assert data.shape == (4, 100)

    def test_custom_frequencies(self):
        from scpn_phase_orchestrator.adapters.hardware_io import SimulatedBoardAdapter

        freqs = np.array([10.0, 20.0])
        adapter = SimulatedBoardAdapter(n_channels=2, frequencies=freqs)
        np.testing.assert_array_equal(adapter._freqs, freqs)


class TestPrometheusAdapter:
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
            assert len(result) == 0

    def test_fetch_metric_network_error(self):
        from urllib.error import URLError

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


class TestMetricsExporter:
    def test_export(self):
        from scpn_phase_orchestrator.adapters.metrics_exporter import MetricsExporter
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        exporter = MetricsExporter(prefix="test")
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=1.0), LayerState(R=0.7, psi=2.0)],
            cross_layer_alignment=np.eye(2),
            stability_proxy=0.6,
            regime_id=0,
        )
        text = exporter.export(state, "nominal", 1.5)
        assert "test_r_global" in text
        assert "test_latency_ms" in text
        assert "nominal" in text

    def test_exposition_lines(self):
        from scpn_phase_orchestrator.adapters.metrics_exporter import MetricsExporter
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        exporter = MetricsExporter()
        state = UPDEState(
            layers=[LayerState(R=0.4, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.4,
            regime_id=0,
        )
        lines = exporter.exposition_lines(state, "recovery", 2.0)
        assert any("recovery" in line for line in lines)
        assert any("layer_count" in line for line in lines)


class TestRedisStore:
    def test_save_load_with_mock(self):
        from scpn_phase_orchestrator.adapters.redis_store import RedisStateStore

        mock_client = MagicMock()
        store = RedisStateStore(client=mock_client, key="test:state")

        store.save_state({"R": 0.5})
        mock_client.set.assert_called_once()

        mock_client.get.return_value = b'{"R": 0.5}'
        result = store.load_state()
        assert result == {"R": 0.5}

    def test_load_missing(self):
        from scpn_phase_orchestrator.adapters.redis_store import RedisStateStore

        mock_client = MagicMock()
        mock_client.get.return_value = None
        store = RedisStateStore(client=mock_client)
        assert store.load_state() is None

    def test_delete_state(self):
        from scpn_phase_orchestrator.adapters.redis_store import RedisStateStore

        mock_client = MagicMock()
        store = RedisStateStore(client=mock_client, key="test:key")
        store.delete_state()
        mock_client.delete.assert_called_once_with("test:key")

    def test_key_property(self):
        from scpn_phase_orchestrator.adapters.redis_store import RedisStateStore

        mock_client = MagicMock()
        store = RedisStateStore(client=mock_client, key="my:key")
        assert store.key == "my:key"


class TestSNNControllerBridge:
    def test_upde_to_current(self):
        from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        bridge = SNNControllerBridge(n_neurons=100)
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=i * 0.1) for i in range(4)],
            cross_layer_alignment=np.eye(4),
            stability_proxy=0.5,
            regime_id=0,
        )
        currents = bridge.upde_state_to_input_current(state)
        assert currents.shape == (4,)
        assert np.all(np.isfinite(currents))

    def test_spike_rates_to_actions(self):
        from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge

        bridge = SNNControllerBridge()
        rates = np.array([10.0, 50.0, 100.0, 200.0])
        actions = bridge.spike_rates_to_actions(rates, layer_assignments=[0, 1, 2, 3])
        assert len(actions) > 0

    def test_lif_rate_estimate_monotonic(self):
        """Higher input current → higher firing rate (LIF model property)."""
        from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge

        bridge = SNNControllerBridge()
        currents = np.array([0.5, 1.0, 1.5, 2.0])
        rates = bridge.lif_rate_estimate(currents)
        assert rates.shape == (4,)
        assert np.all(rates >= 0.0), "Firing rates must be non-negative"
        # Monotonicity: higher current → higher or equal rate
        for i in range(len(rates) - 1):
            assert rates[i + 1] >= rates[i] - 1e-10, (
                f"LIF rate not monotonic: I={currents[i]:.1f}→{rates[i]:.1f}, "
                f"I={currents[i+1]:.1f}→{rates[i+1]:.1f}"
            )

    def test_build_numpy_network_returns_valid_object(self):
        from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge

        bridge = SNNControllerBridge()
        net = bridge.build_numpy_network(4)
        assert net is not None

    def test_snn_bridge_pipeline_wiring(self):
        """End-to-end: UPDEState → SNN currents → rates → actions.
        Verifies the full adapter pipeline, not just individual methods."""
        from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        bridge = SNNControllerBridge(n_neurons=100)
        state = UPDEState(
            layers=[LayerState(R=0.3, psi=i * 0.5) for i in range(4)],
            cross_layer_alignment=np.eye(4),
            stability_proxy=0.3,
            regime_id="degraded",
        )
        currents = bridge.upde_state_to_input_current(state)
        rates = bridge.lif_rate_estimate(currents)
        actions = bridge.spike_rates_to_actions(rates, layer_assignments=[0, 1, 2, 3])
        # Pipeline must produce finite currents, non-negative rates, and ≥0 actions
        assert np.all(np.isfinite(currents))
        assert np.all(rates >= 0.0)
        assert isinstance(actions, list)
