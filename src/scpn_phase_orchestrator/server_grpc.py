# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC streaming service

"""gRPC server-streaming service for live UPDE phase data.

Install grpcio to use this module::

    pip install grpcio grpcio-tools

Without grpcio, importing this module raises ImportError.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

__all__ = ["PhaseStreamServicer", "HAS_GRPC"]

try:
    import grpc  # pragma: no cover

    HAS_GRPC = True  # pragma: no cover
except ModuleNotFoundError:
    grpc = None
    HAS_GRPC = False


class _PhaseResponse:
    """Minimal response wrapper when no protobuf definitions are compiled."""

    def __init__(self, payload: str) -> None:
        self.payload = payload


class PhaseStreamServicer:
    """gRPC servicer that streams UPDE state snapshots.

    Accepts a callable ``state_source`` that yields UPDEState-like
    objects on each call. The stream terminates when ``max_steps``
    is reached or the context is cancelled.
    """

    def __init__(
        self,
        state_source: Any,
        max_steps: int = 100,
        interval_s: float = 0.05,
    ) -> None:
        self._source = state_source
        self._max_steps = max_steps
        self._interval = interval_s

    def StreamPhases(self, request: Any, context: Any) -> Iterator[_PhaseResponse]:
        """Server-streaming RPC: yield phase snapshots as JSON payloads."""
        for step in range(self._max_steps):
            if (
                context is not None
                and hasattr(context, "is_active")
                and not context.is_active()
            ):
                return
            state = self._source()
            payload = json.dumps(_serialise_state(state, step), default=str)
            yield _PhaseResponse(payload)
            time.sleep(self._interval)


def _serialise_state(state: Any, step: int) -> dict:
    """Convert a UPDEState (or plain dict) to JSON-safe dict."""
    if hasattr(state, "__dataclass_fields__"):
        d = asdict(state)
    elif isinstance(state, dict):
        d = state
    else:
        d = {"value": str(state)}
    d["step"] = step
    d["timestamp"] = time.time()
    return d
