# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit logger tests

from __future__ import annotations

import json

import numpy as np

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _sample_state():
    return UPDEState(
        layers=[LayerState(R=0.8, psi=0.5), LayerState(R=0.6, psi=1.0)],
        cross_layer_alignment=np.eye(2),
        stability_proxy=0.7,
        regime_id="nominal",
    )


def _sample_actions():
    return [
        ControlAction(
            knob="K", scope="global", value=0.05, ttl_s=5.0, justification="boost"
        ),
    ]


def test_log_step_writes_line(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)
    logger.log_step(0, _sample_state(), _sample_actions())
    logger.close()

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["step"] == 0
    assert record["regime"] == "nominal"
    assert len(record["layers"]) == 2
    assert len(record["actions"]) == 1


def test_log_event_writes_line(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)
    logger.log_event("regime_change", {"from": "nominal", "to": "degraded"})
    logger.close()

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "regime_change"
    assert record["from"] == "nominal"


def test_valid_json_per_line(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)
    logger.log_step(0, _sample_state(), [])
    logger.log_step(1, _sample_state(), _sample_actions())
    logger.log_event("test_event", {"x": 42})
    logger.close()

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        parsed = json.loads(line)
        assert "ts" in parsed


def test_close_flushes(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)
    logger.log_step(0, _sample_state(), [])
    logger.close()
    content = log_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0
