# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Remanentia memory bridge

"""Bidirectional bridge between SPO coherence monitoring and
Remanentia's memory system.

Direction 1 (SPO -> Remanentia):
  Agent coherence metrics feed consolidation decisions.
  High R = agents aligned = consolidate their outputs together.
  Low R = agents diverged = index separately, flag conflicts.

Direction 2 (Remanentia -> SPO):
  Memory recall novelty feeds coupling adaptation.
  Novel recall = agents exploring new ground = boost K.
  Stale recall = repetitive work = decay K.

Requires: Remanentia API running at http://localhost:8001
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, TypeAlias
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray

__all__ = ["RemanentiaBridge", "CoherenceMemorySnapshot"]
FloatArray: TypeAlias = NDArray[np.float64]
logger = logging.getLogger(__name__)


@dataclass
class CoherenceMemorySnapshot:
    """Combined coherence + memory state."""

    R_global: float
    regime: str
    n_entities: int
    n_memories: int
    novelty_score: float
    consolidation_suggested: bool

    def __post_init__(self) -> None:
        self.R_global = _validated_unit_interval(self.R_global, name="R_global")
        self.regime = _validated_label(self.regime, name="regime")
        if (
            not isinstance(self.n_entities, int)
            or isinstance(self.n_entities, bool)
            or self.n_entities < 0
        ):
            raise ValueError("n_entities must be a non-negative integer")
        if (
            not isinstance(self.n_memories, int)
            or isinstance(self.n_memories, bool)
            or self.n_memories < 0
        ):
            raise ValueError("n_memories must be a non-negative integer")
        self.novelty_score = _validated_unit_interval(
            self.novelty_score,
            name="novelty_score",
        )
        if not isinstance(self.consolidation_suggested, bool):
            raise ValueError("consolidation_suggested must be a bool")


def _validated_remanentia_url(remanentia_url: object) -> str:
    if not isinstance(remanentia_url, str) or not remanentia_url.strip():
        raise ValueError("remanentia_url must be a non-empty http(s) URL")
    url = remanentia_url.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("remanentia_url must be a non-empty http(s) URL")
    return url


def _validated_timeout(timeout: object) -> float:
    if (
        not isinstance(timeout, Real)
        or isinstance(timeout, bool)
        or not isfinite(float(timeout))
    ):
        raise ValueError("timeout must be a finite positive real value")
    value = float(timeout)
    if value <= 0.0:
        raise ValueError("timeout must be a finite positive real value")
    return value


def _validated_positive_real(value: object, *, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite positive real value")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real value")
    return result


def _validated_unit_interval(value: object, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    result = float(value)
    if not isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return result


def _validated_label(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _validated_query(query: object) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return query


def _validated_status_payload(payload: object) -> tuple[int, int]:
    if not isinstance(payload, dict):
        raise ValueError("status payload must be a mapping")
    entities = payload.get("entities")
    memories = payload.get("memories")
    if not isinstance(entities, int) or isinstance(entities, bool) or entities < 0:
        raise ValueError("status.entities must be a non-negative integer")
    if not isinstance(memories, int) or isinstance(memories, bool) or memories < 0:
        raise ValueError("status.memories must be a non-negative integer")
    return entities, memories


def _validated_recall_scores(payload: object) -> list[float]:
    if not isinstance(payload, dict):
        raise ValueError("recall payload must be a mapping")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("recall.results must be a list")
    scores: list[float] = []
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"recall.results[{index}] must be a mapping")
        score = item.get("score")
        if not isinstance(score, Real) or isinstance(score, bool):
            raise ValueError(f"recall.results[{index}].score must be a real number")
        score_f = float(score)
        if not isfinite(score_f):
            raise ValueError(f"recall.results[{index}].score must be finite")
        scores.append(score_f)
    return scores


def _validated_response_payload(payload: object, *, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} payload must be a mapping")
    return payload


def _validated_agent_phases(agent_phases: object) -> dict[str, float] | None:
    if agent_phases is None:
        return None
    if not isinstance(agent_phases, dict):
        raise ValueError("agent_phases must map agent names to finite phases")
    validated: dict[str, float] = {}
    for agent, phase in agent_phases.items():
        name = _validated_label(agent, name="agent_phases key")
        if not isinstance(phase, Real) or isinstance(phase, bool):
            raise ValueError("agent_phases values must be finite real numbers")
        parsed = float(phase)
        if not isfinite(parsed):
            raise ValueError("agent_phases values must be finite real numbers")
        validated[name] = parsed
    return validated


class RemanentiaBridge:
    """Bidirectional SPO <-> Remanentia bridge.

    Usage::

        bridge = RemanentiaBridge(remanentia_url="http://localhost:8001")

        # After each SPO step:
        bridge.report_coherence(R=0.85, regime="nominal", agent_phases={...})

        # Get memory-informed coupling adjustment:
        novelty = bridge.get_novelty_score("What coupling topology works?")
        K_boost = novelty * 0.5  # novel = explore more = stronger coupling

        # Trigger consolidation when agents are aligned:
        if R > 0.8:
            bridge.trigger_consolidation()
    """

    def __init__(
        self,
        remanentia_url: str = "http://localhost:8002",
        timeout: float = 5.0,
    ):
        self._url = _validated_remanentia_url(remanentia_url)
        self._timeout = _validated_timeout(timeout)
        self._last_R = 0.0
        self._last_regime = "unknown"
        self._last_novelty_score = 0.5
        self._last_entities = 0
        self._last_memories = 0

    @staticmethod
    def _is_transport_or_decode_error(exc: BaseException) -> bool:
        return isinstance(
            exc,
            (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                ValueError,
                json.JSONDecodeError,
            ),
        )

    def _open(self, req: urllib.request.Request) -> dict[str, Any]:
        """Execute a urllib request, enforcing http(s) scheme."""
        url = req.full_url
        if not url.startswith(("http://", "https://")):
            # Reject without echoing the offending URL — it may contain a
            # malicious scheme or caller-supplied payload.
            raise ValueError(
                "Refusing request: only http:// and https:// URLs are allowed"
            )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # nosec B310
            return _validated_response_payload(
                json.loads(resp.read()),
                name="response",
            )

    def _get(self, path: str) -> dict[str, Any]:
        """GET request to Remanentia API."""
        return self._open(urllib.request.Request(f"{self._url}{path}"))

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """POST request to Remanentia API."""
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self._url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        return self._open(req)

    def health_check(self) -> bool:
        """Check if Remanentia is running."""
        try:
            resp = _validated_response_payload(self._get("/health"), name="health")
            return resp.get("status") == "ok"
        except BaseException as exc:
            if not self._is_transport_or_decode_error(exc):
                raise
            logger.warning("remanentia.health_check_failed: %s", type(exc).__name__)
            return False

    def report_coherence(
        self,
        R: float,
        regime: str,
        agent_phases: dict[str, float] | None = None,
    ) -> None:
        """Report current SPO coherence state to Remanentia.

        Remanentia can use this to decide when to consolidate:
        high R = aligned agents = good time to merge their traces.
        """
        R = _validated_unit_interval(R, name="R")
        regime = _validated_label(regime, name="regime")
        _validated_agent_phases(agent_phases)
        self._last_R = R
        self._last_regime = regime
        # Store as a reasoning trace that Remanentia can index
        try:
            self._post(
                "/recall",
                {
                    "query": f"SPO coherence report: R={R:.3f} regime={regime}",
                    "top_k": 0,
                },
            )
        except BaseException as exc:
            if not self._is_transport_or_decode_error(exc):
                raise
            logger.warning(
                "remanentia.report_coherence_trace_failed: %s",
                type(exc).__name__,
            )

    def get_novelty_score(self, query: str) -> float:
        """Query Remanentia and estimate novelty from recall results.

        If recall returns many relevant memories -> low novelty (known ground).
        If recall returns few/none -> high novelty (unexplored territory).
        Novelty feeds SPO coupling: novel = boost K (explore together).
        """
        query = _validated_query(query)
        try:
            resp = self._post("/recall", {"query": query, "top_k": 5})
            scores = _validated_recall_scores(resp)
            if not scores:
                self._last_novelty_score = 1.0
                return 1.0  # fully novel — no relevant memories
            # Novelty = 1 - mean relevance score
            novelty = float(max(0.0, 1.0 - float(np.mean(scores))))
            self._last_novelty_score = novelty
            return novelty
        except BaseException as exc:
            if not self._is_transport_or_decode_error(exc):
                raise
            logger.warning(
                "remanentia.get_novelty_score_failed: %s; using_last=%.6f",
                type(exc).__name__,
                self._last_novelty_score,
            )
            return self._last_novelty_score

    def get_entity_count(self) -> int:
        """Get number of entities in Remanentia's knowledge graph."""
        try:
            resp = self._get("/status")
            entities, memories = _validated_status_payload(resp)
            self._last_entities = entities
            self._last_memories = memories
            return entities
        except BaseException as exc:
            if not self._is_transport_or_decode_error(exc):
                raise
            logger.warning(
                "remanentia.get_entity_count_failed: %s; using_last=%d",
                type(exc).__name__,
                self._last_entities,
            )
            return self._last_entities

    def trigger_consolidation(self, force: bool = False) -> bool:
        """Trigger memory consolidation in Remanentia.

        Best called when R is high (agents aligned, traces coherent).
        """
        if not isinstance(force, bool):
            raise ValueError("force must be a bool")
        try:
            resp = _validated_response_payload(
                self._post("/consolidate", {"force": force}),
                name="consolidate",
            )
            return resp.get("status") == "ok"
        except BaseException as exc:
            if not self._is_transport_or_decode_error(exc):
                raise
            logger.warning(
                "remanentia.trigger_consolidation_failed: %s",
                type(exc).__name__,
            )
            return False

    def novelty_to_coupling_delta(
        self,
        queries: list[str],
        scale: float = 0.5,
    ) -> FloatArray:
        """Convert per-agent novelty scores to coupling adjustment.

        Each agent's recent work is queried against Remanentia.
        Novel agents get coupling boosted (explore together).
        Redundant agents get coupling decayed (avoid repetition).

        Returns (N,) array of per-agent K multipliers.
        """
        if not isinstance(queries, list):
            raise ValueError("queries must be a list of non-empty strings")
        scale = _validated_positive_real(scale, name="scale")
        deltas = []
        for q in queries:
            novelty = self.get_novelty_score(_validated_query(q))
            deltas.append(1.0 + novelty * scale)
        return np.array(deltas, dtype=np.float64)

    def snapshot(self) -> CoherenceMemorySnapshot:
        """Combined coherence + memory state."""
        n_ent = self.get_entity_count()
        try:
            status = self._get("/status")
            entities, n_memories = _validated_status_payload(status)
            self._last_entities = entities
            self._last_memories = n_memories
        except BaseException as exc:
            if not self._is_transport_or_decode_error(exc):
                raise
            logger.warning(
                "remanentia.snapshot_status_failed: %s; using_last=%d",
                type(exc).__name__,
                self._last_memories,
            )
            n_memories = self._last_memories

        return CoherenceMemorySnapshot(
            R_global=self._last_R,
            regime=self._last_regime,
            n_entities=n_ent,
            n_memories=n_memories,
            novelty_score=self._last_novelty_score,
            consolidation_suggested=self._last_R > 0.8,
        )
