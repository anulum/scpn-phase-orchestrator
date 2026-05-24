# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit logger validation contracts

"""
Validation contracts for AuditLogger path, header, step, event, and finite-array
payload boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger


def test_u1_audit_logger_rejects_directory_path(tmp_path) -> None:
    with pytest.raises(Exception, match="directory"):
        AuditLogger(tmp_path)


def test_u1_audit_logger_rejects_blank_event_stream_path(tmp_path) -> None:
    with pytest.raises(Exception, match="event_stream path must be non-empty"):
        AuditLogger(tmp_path / "audit.jsonl", event_stream=" ")


def test_u1_audit_logger_rejects_non_path_type() -> None:
    with pytest.raises(Exception, match="str or Path"):
        AuditLogger(123)  # type: ignore[arg-type]


def test_u1_audit_logger_rejects_non_path_event_stream_type(tmp_path) -> None:
    with pytest.raises(Exception, match="str, Path, or None"):
        AuditLogger(tmp_path / "audit.jsonl", event_stream=123)  # type: ignore[arg-type]


def test_u1_audit_logger_header_rejects_non_positive_dt(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="finite positive real"):
            logger.log_header(n_oscillators=4, dt=0.0)
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_boolean_dt(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="finite positive real"):
            logger.log_header(n_oscillators=4, dt=True)  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_non_finite_dt(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="finite positive real"):
            logger.log_header(n_oscillators=4, dt=float("inf"))
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_boolean_n_oscillators(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="positive integer"):
            logger.log_header(n_oscillators=True, dt=0.1)  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_non_positive_n_oscillators(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="positive integer"):
            logger.log_header(n_oscillators=0, dt=0.1)
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_blank_method(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="non-empty string"):
            logger.log_header(n_oscillators=4, dt=0.1, method=" ")
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_boolean_seed(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="seed must be integer or None"):
            logger.log_header(n_oscillators=4, dt=0.1, seed=True)  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_non_dict_binding_config(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="binding_config"):
            logger.log_header(n_oscillators=4, dt=0.1, binding_config=["bad"])  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_non_bool_amplitude_mode(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="amplitude_mode must be bool"):
            logger.log_header(n_oscillators=4, dt=0.1, amplitude_mode=1)  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_non_dict_binding_summary(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="binding_summary"):
            logger.log_header(
                n_oscillators=4,
                dt=0.1,
                binding_summary=["bad"],  # type: ignore[arg-type]
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_negative_step(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="non-negative integer"):
            logger.log_step(
                -1,
                object(),  # type: ignore[arg-type]
                [],
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_boolean_step(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="non-negative integer"):
            logger.log_step(
                True,  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                [],
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_upde_state(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="UPDEState"):
            logger.log_step(0, object(), [])
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_action_entries(tmp_path) -> None:
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="actions\\[0\\]"):
            logger.log_step(0, state, [object()])  # type: ignore[list-item]
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_list_actions(tmp_path) -> None:
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="actions must be list"):
            logger.log_step(0, state, {})  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_dict_channel_runtime(tmp_path) -> None:
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="channel_runtime"):
            logger.log_step(
                0,
                state,
                [],
                channel_runtime=["bad"],  # type: ignore[arg-type]
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_finite_epsilon(tmp_path) -> None:
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="epsilon must be finite real"):
            logger.log_step(0, state, [], epsilon=float("inf"))
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_boolean_epsilon(tmp_path) -> None:
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="epsilon must be finite real"):
            logger.log_step(0, state, [], epsilon=True)  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_event_rejects_non_string_event_type(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="event_type must be a non-empty string"):
            logger.log_event(1, {})  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_event_rejects_boolean_event_type(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="event_type must be a non-empty string"):
            logger.log_event(True, {})  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_event_rejects_blank_event_type(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="event_type must be a non-empty string"):
            logger.log_event(" ", {})
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_event_rejects_non_dict_data(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="data must be dict"):
            logger.log_event("evt", ["bad"])  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_finite_phases_payload(tmp_path) -> None:
    from scpn_phase_orchestrator.actuation.mapper import ControlAction
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="phases must contain only finite values"):
            logger.log_step(
                0,
                state,
                [ControlAction("K", "global", 0.1, 1.0, "u1-test")],
                phases=np.array([0.0, np.nan], dtype=float),
                omegas=np.array([1.0, 1.0], dtype=float),
                knm=np.ones((2, 2), dtype=float),
                alpha=np.zeros((2, 2), dtype=float),
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_finite_omegas_payload(tmp_path) -> None:
    from scpn_phase_orchestrator.actuation.mapper import ControlAction
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="omegas must contain only finite values"):
            logger.log_step(
                0,
                state,
                [ControlAction("K", "global", 0.1, 1.0, "u1-test")],
                phases=np.array([0.0, 0.1], dtype=float),
                omegas=np.array([1.0, np.inf], dtype=float),
                knm=np.ones((2, 2), dtype=float),
                alpha=np.zeros((2, 2), dtype=float),
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_finite_knm_payload(tmp_path) -> None:
    from scpn_phase_orchestrator.actuation.mapper import ControlAction
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="knm must contain only finite values"):
            logger.log_step(
                0,
                state,
                [ControlAction("K", "global", 0.1, 1.0, "u1-test")],
                phases=np.array([0.0, 0.1], dtype=float),
                omegas=np.array([1.0, 1.0], dtype=float),
                knm=np.array([[1.0, 0.0], [0.0, np.nan]], dtype=float),
                alpha=np.zeros((2, 2), dtype=float),
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_finite_alpha_payload(tmp_path) -> None:
    from scpn_phase_orchestrator.actuation.mapper import ControlAction
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="alpha must contain only finite values"):
            logger.log_step(
                0,
                state,
                [ControlAction("K", "global", 0.1, 1.0, "u1-test")],
                phases=np.array([0.0, 0.1], dtype=float),
                omegas=np.array([1.0, 1.0], dtype=float),
                knm=np.ones((2, 2), dtype=float),
                alpha=np.array([[0.0, 0.0], [0.0, np.inf]], dtype=float),
            )
    finally:
        logger._fh.close()
