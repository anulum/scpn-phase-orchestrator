# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for Remanentia memory bridge

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.remanentia_bridge import (
    CoherenceMemorySnapshot,
    RemanentiaBridge,
)


class _FakeHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler returning canned JSON."""

    def do_GET(self):
        if self.path == "/health":
            self._respond({"status": "ok"})
        elif self.path == "/status":
            self._respond({"status": "ok", "entities": 42, "memories": 7})
        else:
            self._respond({}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/recall":
            self._respond(
                {
                    "results": [
                        {"text": "a", "score": 0.8},
                        {"text": "b", "score": 0.6},
                    ],
                }
            )
        elif self.path == "/consolidate":
            self._respond({"status": "ok"})
        else:
            self._respond({}, 404)

    def _respond(self, data: dict, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *_a):
        pass


@pytest.fixture(scope="module")
def bridge():
    server = HTTPServer(("127.0.0.1", 0), _FakeHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield RemanentiaBridge(remanentia_url=f"http://127.0.0.1:{port}", timeout=2.0)
    server.shutdown()


class TestRemanentiaBridge:
    def test_health_check(self, bridge: RemanentiaBridge) -> None:
        assert bridge.health_check() is True

    def test_health_check_fail(self) -> None:
        bad = RemanentiaBridge(remanentia_url="http://127.0.0.1:1", timeout=0.5)
        assert bad.health_check() is False

    def test_get_entity_count(self, bridge: RemanentiaBridge) -> None:
        assert bridge.get_entity_count() == 42

    def test_novelty_score_bounded(self, bridge: RemanentiaBridge) -> None:
        score = bridge.get_novelty_score("test query")
        assert 0.0 <= score <= 1.0

    def test_novelty_score_offline(self) -> None:
        bad = RemanentiaBridge(remanentia_url="http://127.0.0.1:1", timeout=0.5)
        assert bad.get_novelty_score("test") == 0.5

    def test_trigger_consolidation(self, bridge: RemanentiaBridge) -> None:
        assert bridge.trigger_consolidation() is True

    def test_report_coherence(self, bridge: RemanentiaBridge) -> None:
        bridge.report_coherence(R=0.85, regime="nominal")
        assert bridge._last_R == 0.85
        assert bridge._last_regime == "nominal"

    def test_novelty_to_coupling_delta(self, bridge: RemanentiaBridge) -> None:
        deltas = bridge.novelty_to_coupling_delta(["q1", "q2"], scale=0.5)
        assert deltas.shape == (2,)
        assert np.all(deltas >= 1.0)

    def test_snapshot(self, bridge: RemanentiaBridge) -> None:
        bridge.report_coherence(R=0.9, regime="sync")
        snap = bridge.snapshot()
        assert isinstance(snap, CoherenceMemorySnapshot)
        assert snap.R_global == 0.9
        assert snap.consolidation_suggested is True
        assert snap.n_entities == 42

    def test_url_scheme_validation(self) -> None:
        bad = RemanentiaBridge(remanentia_url="file:///etc/passwd")
        with pytest.raises(ValueError, match="Refusing non-HTTP"):
            bad._get("/health")

    def test_entity_count_offline(self) -> None:
        bad = RemanentiaBridge(remanentia_url="http://127.0.0.1:1", timeout=0.5)
        assert bad.get_entity_count() == 0

    def test_consolidation_offline(self) -> None:
        bad = RemanentiaBridge(remanentia_url="http://127.0.0.1:1", timeout=0.5)
        assert bad.trigger_consolidation() is False

    def test_snapshot_offline(self) -> None:
        bad = RemanentiaBridge(remanentia_url="http://127.0.0.1:1", timeout=0.5)
        bad._last_R = 0.3
        bad._last_regime = "drift"
        snap = bad.snapshot()
        assert snap.R_global == 0.3
        assert snap.n_entities == 0
        assert snap.n_memories == 0


class _EmptyRecallHandler(BaseHTTPRequestHandler):
    """Returns empty recall results."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"results": []}).encode())

    def log_message(self, *_a):
        pass


class TestNoveltyEmpty:
    def test_empty_recall_returns_one(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _EmptyRecallHandler)
        port = server.server_address[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        bridge = RemanentiaBridge(
            remanentia_url=f"http://127.0.0.1:{port}", timeout=2.0
        )
        assert bridge.get_novelty_score("unknown topic") == 1.0
        server.shutdown()


class TestRemanentiaPipelineWiring:
    """Pipeline: engine state → snapshot → Remanentia memory store."""

    def test_snapshot_from_engine_state(self):
        """UPDEEngine → R → CoherenceMemorySnapshot: proves bridge
        consumes engine output for persistent memory storage."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, np.zeros((n, n)))
        r, _ = compute_order_parameter(phases)

        snap = CoherenceMemorySnapshot(
            R_global=r,
            regime="nominal" if r > 0.6 else "degraded",
            n_entities=0,
            n_memories=0,
            novelty_score=0.5,
            consolidation_suggested=False,
        )
        assert 0.0 <= snap.R_global <= 1.0
        assert snap.regime in ("nominal", "degraded")
