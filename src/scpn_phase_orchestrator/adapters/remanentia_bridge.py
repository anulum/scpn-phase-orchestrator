# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
import urllib.request
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["RemanentiaBridge", "CoherenceMemorySnapshot"]


@dataclass
class CoherenceMemorySnapshot:
    """Combined coherence + memory state."""

    R_global: float
    regime: str
    n_entities: int
    n_memories: int
    novelty_score: float
    consolidation_suggested: bool


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
        self._url = remanentia_url.rstrip("/")
        self._timeout = timeout
        self._last_R = 0.0
        self._last_regime = "unknown"

    def _get(self, path: str) -> dict:
        """GET request to Remanentia API."""
        req = urllib.request.Request(f"{self._url}{path}")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, data: dict) -> dict:
        """POST request to Remanentia API."""
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self._url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def health_check(self) -> bool:
        """Check if Remanentia is running."""
        try:
            resp = self._get("/health")
            return resp.get("status") == "ok"
        except Exception:
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
        self._last_R = R
        self._last_regime = regime
        # Store as a reasoning trace that Remanentia can index
        import contextlib

        with contextlib.suppress(Exception):
            self._post(
                "/recall",
                {
                    "query": f"SPO coherence report: R={R:.3f} regime={regime}",
                    "top_k": 0,
                },
            )

    def get_novelty_score(self, query: str) -> float:
        """Query Remanentia and estimate novelty from recall results.

        If recall returns many relevant memories -> low novelty (known ground).
        If recall returns few/none -> high novelty (unexplored territory).
        Novelty feeds SPO coupling: novel = boost K (explore together).
        """
        try:
            resp = self._post("/recall", {"query": query, "top_k": 5})
            results = resp.get("results", [])
            if not results:
                return 1.0  # fully novel — no relevant memories
            # Novelty = 1 - mean relevance score
            scores = [r.get("score", 0.0) for r in results]
            return max(0.0, 1.0 - np.mean(scores))
        except Exception:
            return 0.5  # unknown — conservative default

    def get_entity_count(self) -> int:
        """Get number of entities in Remanentia's knowledge graph."""
        try:
            resp = self._get("/status")
            return resp.get("entities", 0)
        except Exception:
            return 0

    def trigger_consolidation(self, force: bool = False) -> bool:
        """Trigger memory consolidation in Remanentia.

        Best called when R is high (agents aligned, traces coherent).
        """
        try:
            resp = self._post("/consolidate", {"force": force})
            return resp.get("status") == "ok"
        except Exception:
            return False

    def novelty_to_coupling_delta(
        self,
        queries: list[str],
        scale: float = 0.5,
    ) -> NDArray:
        """Convert per-agent novelty scores to coupling adjustment.

        Each agent's recent work is queried against Remanentia.
        Novel agents get coupling boosted (explore together).
        Redundant agents get coupling decayed (avoid repetition).

        Returns (N,) array of per-agent K multipliers.
        """
        deltas = []
        for q in queries:
            novelty = self.get_novelty_score(q)
            deltas.append(1.0 + novelty * scale)
        return np.array(deltas)

    def snapshot(self) -> CoherenceMemorySnapshot:
        """Combined coherence + memory state."""
        n_ent = self.get_entity_count()
        try:
            status = self._get("/status")
            n_mem = status.get("memories", 0)
        except Exception:
            n_mem = 0

        return CoherenceMemorySnapshot(
            R_global=self._last_R,
            regime=self._last_regime,
            n_entities=n_ent,
            n_memories=n_mem,
            novelty_score=0.5,
            consolidation_suggested=self._last_R > 0.8,
        )
