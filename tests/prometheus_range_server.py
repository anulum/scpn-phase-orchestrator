# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — local HTTP server speaking the Prometheus range API

"""A real local HTTP server answering ``/api/v1/query_range`` and ``/api/v1/query``.

Justification (CEO_DIRECTIVES §8): Prometheus is an external service that CI
cannot run. The code under test performs real HTTP requests over a real
socket; only the far end is a small deterministic server returning
Prometheus-format range results, so the collector's request, parsing and
buffering paths run unmodified.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def _series(query: str, start: float, end: float, step: float) -> list[list[object]]:
    offset = sum(map(ord, query)) % 7
    points: list[list[object]] = []
    t = start
    index = 0
    while t <= end + 1e-9:
        value = 1.0 + 0.5 * math.sin(0.35 * index + offset)
        points.append([t, f"{value:.6f}"])
        index += 1
        t = start + index * step
    return points


class _Handler(BaseHTTPRequestHandler):
    empty_queries: frozenset[str] = frozenset()
    instant_calls: int = 0

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if parsed.path == "/api/v1/query":
            self._instant(params["query"])
            return
        if parsed.path != "/api/v1/query_range":
            self.send_error(404)
            return
        query = params["query"]
        result = (
            []
            if query in self.empty_queries
            else [
                {
                    "metric": {"__name__": query},
                    "values": _series(
                        query,
                        float(params["start"]),
                        float(params["end"]),
                        float(params["step"]),
                    ),
                }
            ]
        )
        body = json.dumps(
            {"status": "success", "data": {"resultType": "matrix", "result": result}}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _instant(self, query: str) -> None:
        type(self).instant_calls += 1
        result = (
            []
            if query in self.empty_queries
            else [
                {
                    "metric": {"__name__": query},
                    "value": _series(query, 0.0, 0.0, 1.0)[0][:1]
                    + [f"{1.0 + 0.1 * type(self).instant_calls:.6f}"],
                }
            ]
        )
        self._send(
            {"status": "success", "data": {"resultType": "vector", "result": result}}
        )

    def _send(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


@contextmanager
def prometheus_range_server(
    empty_queries: frozenset[str] = frozenset(),
) -> Iterator[str]:
    """Serve the range API on 127.0.0.1 and yield its base URL."""
    handler = type(
        "Handler", (_Handler,), {"empty_queries": empty_queries, "instant_calls": 0}
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
