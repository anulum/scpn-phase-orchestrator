# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Remanentia bridge defensive contract tests

"""Defensive contract coverage for the Remanentia adapter surface."""

from __future__ import annotations

import urllib.request

import pytest

from scpn_phase_orchestrator.adapters.remanentia_bridge import (
    CoherenceMemorySnapshot,
    RemanentiaBridge,
)

ResponsePayload = dict[str, object]


def _offline_bridge() -> RemanentiaBridge:
    """Return a bridge whose network calls fail fast by default."""
    return RemanentiaBridge(remanentia_url="http://127.0.0.1:1", timeout=0.5)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n_entities", True, "n_entities must be a non-negative integer"),
        ("n_entities", -1, "n_entities must be a non-negative integer"),
        ("n_memories", True, "n_memories must be a non-negative integer"),
        ("n_memories", -1, "n_memories must be a non-negative integer"),
    ],
)
def test_snapshot_record_rejects_invalid_count_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    """Snapshot records must reject boolean aliases and negative counts."""
    payload: dict[str, object] = {
        "R_global": 0.5,
        "regime": "nominal",
        "n_entities": 1,
        "n_memories": 2,
        "novelty_score": 0.4,
        "consolidation_suggested": False,
    }
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        CoherenceMemorySnapshot(**payload)


@pytest.mark.parametrize(
    ("payload", "expected_last"),
    [
        (["not", "a", "mapping"], 8),
        ({"entities": 8, "memories": True}, 8),
        ({"entities": 8, "memories": -1}, 8),
    ],
)
def test_get_entity_count_reuses_cache_for_malformed_status_payloads(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    expected_last: int,
) -> None:
    """Status schema errors are fail-closed transport/decode failures."""
    bridge = _offline_bridge()
    bridge._last_entities = expected_last

    def get_status(_path: str) -> object:
        return payload

    monkeypatch.setattr(bridge, "_get", get_status)

    assert bridge.get_entity_count() == expected_last


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "a", "mapping"],
        {"results": "not-a-list"},
        {"results": ["not-a-mapping"]},
        {"results": [{"score": float("inf")}]},
    ],
)
def test_novelty_score_reuses_cache_for_malformed_recall_payloads(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    """Recall schema errors fall back to the last known novelty score."""
    bridge = _offline_bridge()
    bridge._last_novelty_score = 0.27

    def post_recall(_path: str, _data: ResponsePayload) -> object:
        return payload

    monkeypatch.setattr(bridge, "_post", post_recall)

    assert bridge.get_novelty_score("phase coupling evidence") == pytest.approx(0.27)


@pytest.mark.parametrize(
    "agent_phases",
    [
        "not-a-mapping",
        {"agent\none": 0.1},
        {"agent-one": True},
        {"agent-one": float("inf")},
    ],
)
def test_report_coherence_rejects_malformed_agent_phase_payloads(
    agent_phases: object,
) -> None:
    """Coherence reports validate the real per-agent phase payload contract."""
    bridge = _offline_bridge()

    with pytest.raises(ValueError, match="agent_phases"):
        bridge.report_coherence(
            R=0.5,
            regime="nominal",
            agent_phases=agent_phases,  # type: ignore[arg-type]  # negative contract
        )


def test_report_coherence_accepts_valid_agent_phase_payload() -> None:
    """Valid per-agent phase payloads must pass the report contract."""
    bridge = _offline_bridge()

    bridge.report_coherence(
        R=0.5,
        regime="nominal",
        agent_phases={"agent-one": 0.25, "agent-two": 1.5},
    )

    assert bridge._last_R == 0.5
    assert bridge._last_regime == "nominal"


def test_report_coherence_swallows_transport_failure_after_state_update() -> None:
    """Coherence reports preserve local state when the service is offline."""
    bridge = _offline_bridge()

    bridge.report_coherence(R=0.7, regime="recovering")

    assert bridge._last_R == 0.7
    assert bridge._last_regime == "recovering"


def test_health_check_rejects_mutated_non_http_url_without_leaking_url() -> None:
    """Request opening must fail closed if internal URL state is corrupted."""
    bridge = _offline_bridge()
    bridge._url = "file:///etc/passwd"

    assert bridge.health_check() is False


@pytest.mark.parametrize(
    "method_name",
    [
        "health_check",
        "report_coherence",
        "get_novelty_score",
        "get_entity_count",
        "trigger_consolidation",
        "snapshot",
    ],
)
def test_public_methods_reraise_non_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    """Programmer faults must propagate instead of being cached as outages."""
    bridge = _offline_bridge()

    def raise_programmer_fault(
        _request: urllib.request.Request,
    ) -> ResponsePayload:
        raise RuntimeError("programmer fault")

    monkeypatch.setattr(bridge, "_open", raise_programmer_fault)

    with pytest.raises(RuntimeError, match="programmer fault"):
        if method_name == "health_check":
            bridge.health_check()
        elif method_name == "report_coherence":
            bridge.report_coherence(R=0.4, regime="nominal")
        elif method_name == "get_novelty_score":
            bridge.get_novelty_score("memory novelty")
        elif method_name == "get_entity_count":
            bridge.get_entity_count()
        elif method_name == "trigger_consolidation":
            bridge.trigger_consolidation()
        else:
            bridge.snapshot()


def test_snapshot_reraises_non_transport_failure_from_second_status_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot must propagate programmer faults after cached counts refresh."""
    bridge = _offline_bridge()
    calls = 0

    def get_status(_path: str) -> ResponsePayload:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"entities": 3, "memories": 5}
        raise RuntimeError("programmer fault")

    monkeypatch.setattr(bridge, "_get", get_status)

    with pytest.raises(RuntimeError, match="programmer fault"):
        bridge.snapshot()
