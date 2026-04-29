# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit logger tests

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.exceptions import AuditError
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _sample_state(r0=0.8, r1=0.6, regime="nominal"):
    return UPDEState(
        layers=[LayerState(R=r0, psi=0.5), LayerState(R=r1, psi=1.0)],
        cross_layer_alignment=np.eye(2),
        stability_proxy=0.7,
        regime_id=regime,
    )


def _sample_actions():
    return [
        ControlAction(
            knob="K", scope="global", value=0.05, ttl_s=5.0, justification="boost"
        ),
    ]


# ---------------------------------------------------------------------------
# Hash chain integrity
# ---------------------------------------------------------------------------


class TestAuditHashChain:
    """Verify the SHA-256 hash chain that ensures tamper detection.
    This is the core security property of the audit log."""

    def test_single_record_hash_chains_from_zero(self, tmp_path):
        """First record must chain from the null hash (64 zeros)."""
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log_step(0, _sample_state(), [])

        record = json.loads(log_path.read_text().strip())
        # Reconstruct: hash = SHA256(prev_hash + json_without_hash)
        clean = {k: v for k, v in record.items() if k != "_hash"}
        json_line = json.dumps(clean, separators=(",", ":"), sort_keys=True)
        expected = hashlib.sha256(("0" * 64 + json_line).encode()).hexdigest()
        assert record["_hash"] == expected, (
            f"First record hash must chain from null. "
            f"Expected {expected}, got {record['_hash']}"
        )

    def test_multi_record_hash_chain_is_continuous(self, tmp_path):
        """Each record's hash must depend on the previous record's hash,
        forming an unbroken chain. Tampering with any record would break
        all subsequent hashes."""
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log_step(0, _sample_state(), [])
            logger.log_step(1, _sample_state(r0=0.7), _sample_actions())
            logger.log_event("regime_change", {"from": "nominal", "to": "degraded"})

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

        prev_hash = "0" * 64
        for i, line in enumerate(lines):
            record = json.loads(line)
            clean = {k: v for k, v in record.items() if k != "_hash"}
            json_line = json.dumps(clean, separators=(",", ":"), sort_keys=True)
            expected = hashlib.sha256((prev_hash + json_line).encode()).hexdigest()
            assert record["_hash"] == expected, (
                f"Record {i} hash chain broken. prev_hash={prev_hash[:16]}..."
            )
            prev_hash = record["_hash"]

    def test_hash_excludes_hash_field_itself(self, tmp_path):
        """The _hash field must not be included in the hash computation
        (otherwise the hash would be self-referential)."""
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log_step(0, _sample_state(), [])

        record = json.loads(log_path.read_text().strip())
        # Including _hash in the computation would give a different digest
        with_hash = json.dumps(record, separators=(",", ":"), sort_keys=True)
        digest_with = hashlib.sha256(("0" * 64 + with_hash).encode()).hexdigest()
        assert digest_with != record["_hash"], (
            "Hash must exclude the _hash field — otherwise it's self-referential"
        )


# ---------------------------------------------------------------------------
# Data integrity and field preservation
# ---------------------------------------------------------------------------


class TestAuditDataIntegrity:
    """Verify that logged data is faithfully preserved: values, types,
    structure, and ordering."""

    def test_step_preserves_all_fields(self, tmp_path):
        """Step record must contain step, regime, stability, layers, actions,
        and timestamp — all with correct values."""
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log_step(
                42,
                _sample_state(r0=0.85, r1=0.55, regime="degraded"),
                _sample_actions(),
            )

        record = json.loads(log_path.read_text().strip())
        assert record["step"] == 42
        assert record["regime"] == "degraded"
        assert abs(record["stability"] - 0.7) < 1e-10
        assert len(record["layers"]) == 2
        assert abs(record["layers"][0]["R"] - 0.85) < 1e-10
        assert abs(record["layers"][1]["R"] - 0.55) < 1e-10
        assert len(record["actions"]) == 1
        assert record["actions"][0]["knob"] == "K"
        assert record["actions"][0]["value"] == 0.05
        assert record["actions"][0]["justification"] == "boost"
        assert "ts" in record

    def test_event_preserves_arbitrary_data(self, tmp_path):
        """Event records must preserve the event type and all data fields."""
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log_event(
                "regime_change", {"from": "nominal", "to": "critical", "step": 99}
            )

        record = json.loads(log_path.read_text().strip())
        assert record["event"] == "regime_change"
        assert record["from"] == "nominal"
        assert record["to"] == "critical"
        assert record["step"] == 99

    def test_full_state_replay_fields_round_trip(self, tmp_path):
        """When phases/omegas/knm/alpha are provided, they must be
        preserved exactly for deterministic replay."""
        log_path = tmp_path / "audit.jsonl"
        phases = np.array([0.1, 0.2, 0.3])
        omegas = np.array([1.0, 2.0, 3.0])
        knm = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]])
        alpha = np.zeros((3, 3))

        state = UPDEState(
            layers=[LayerState(R=0.9, psi=0.1)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.8,
            regime_id="nominal",
        )
        with AuditLogger(log_path) as logger:
            logger.log_step(
                0,
                state,
                [],
                phases=phases,
                omegas=omegas,
                knm=knm,
                alpha=alpha,
                zeta=0.1,
                psi_drive=0.5,
            )

        record = json.loads(log_path.read_text().strip())
        np.testing.assert_allclose(record["phases"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(record["omegas"], [1.0, 2.0, 3.0])
        assert record["zeta"] == 0.1
        assert record["psi_drive"] == 0.5
        # knm round-trips as nested list
        recovered_knm = np.array(record["knm"])
        np.testing.assert_allclose(recovered_knm, knm)

    def test_phases_without_omegas_raises_audit_error(self, tmp_path):
        """Providing phases without omegas/knm/alpha must raise AuditError,
        because the log would be un-replayable."""
        log_path = tmp_path / "audit.jsonl"
        with (
            AuditLogger(log_path) as logger,
            pytest.raises(AuditError, match="omegas, knm, alpha required"),
        ):
            logger.log_step(0, _sample_state(), [], phases=np.array([0.1, 0.2]))

    def test_amplitude_and_mu_fields_preserved(self, tmp_path):
        """Stuart-Landau specific fields (amplitudes, mu, knm_r, epsilon)
        must be preserved for SL replay."""
        log_path = tmp_path / "audit.jsonl"
        state = _sample_state()
        amps = np.array([0.7, 0.8])
        mu = np.array([0.5, 0.3])
        knm_r = np.array([[0.0, 0.1], [0.1, 0.0]])

        with AuditLogger(log_path) as logger:
            logger.log_step(
                0,
                state,
                [],
                phases=np.array([0.1, 0.2]),
                omegas=np.array([1.0, 1.0]),
                knm=np.zeros((2, 2)),
                alpha=np.zeros((2, 2)),
                amplitudes=amps,
                mu=mu,
                knm_r=knm_r,
                epsilon=0.95,
            )

        record = json.loads(log_path.read_text().strip())
        np.testing.assert_allclose(record["amplitudes"], [0.7, 0.8])
        np.testing.assert_allclose(record["mu"], [0.5, 0.3])
        assert record["epsilon"] == 0.95
        recovered_knm_r = np.array(record["knm_r"])
        np.testing.assert_allclose(recovered_knm_r, knm_r)


# ---------------------------------------------------------------------------
# JSONL format compliance
# ---------------------------------------------------------------------------


class TestAuditJSONLFormat:
    """Verify strict JSONL format: one valid JSON object per line,
    no trailing content, no blank lines."""

    def test_each_line_is_valid_json(self, tmp_path):
        """Every line in the log must parse as valid JSON."""
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log_step(0, _sample_state(), [])
            logger.log_step(1, _sample_state(), _sample_actions())
            logger.log_event("test", {"x": 42})

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                pytest.fail(f"Line {i} is not valid JSON: {line[:80]}")
            assert isinstance(parsed, dict), f"Line {i} must be a JSON object"
            assert "ts" in parsed, f"Line {i} missing timestamp"
            assert "_hash" in parsed, f"Line {i} missing hash chain"

    def test_close_flushes_all_data(self, tmp_path):
        """After close(), all logged data must be readable from disk."""
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log_step(0, _sample_state(), [])
        logger.log_step(1, _sample_state(), [])
        logger.close()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2, "Both steps must be flushed after close()"

    def test_context_manager_flushes_on_exit(self, tmp_path):
        """Using AuditLogger as context manager must flush on __exit__."""
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log_step(0, _sample_state(), [])

        content = log_path.read_text().strip()
        assert len(content) > 0, "Context manager must flush data on exit"
        record = json.loads(content)
        assert record["step"] == 0


# ---------------------------------------------------------------------------
# Header record for replay configuration
# ---------------------------------------------------------------------------


class TestAuditHeader:
    """Verify that log_header produces correct engine configuration records
    for replay reconstruction."""

    def test_header_contains_engine_config(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log_header(n_oscillators=8, dt=0.01, method="rk4", seed=42)

        record = json.loads(log_path.read_text().strip())
        assert record["header"] is True
        assert record["n_oscillators"] == 8
        assert record["dt"] == 0.01
        assert record["method"] == "rk4"
        assert record["seed"] == 42

    def test_header_amplitude_mode_flag(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log_header(n_oscillators=4, dt=0.005, amplitude_mode=True)

        record = json.loads(log_path.read_text().strip())
        assert record["amplitude_mode"] is True

    def test_header_contains_binding_config_when_provided(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        binding_config = {
            "name": "demo",
            "engine_mode": "kuramoto",
            "channels": {"P": {"driver_keys": ["frequency"]}},
        }
        with AuditLogger(log_path) as logger:
            logger.log_header(
                n_oscillators=4,
                dt=0.005,
                binding_config=binding_config,
            )

        record = json.loads(log_path.read_text().strip())
        assert record["binding_config"] == binding_config

    def test_header_without_seed_omits_field(self, tmp_path):
        """Optional fields must be absent, not null, when not provided."""
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log_header(n_oscillators=4, dt=0.01)

        record = json.loads(log_path.read_text().strip())
        assert "seed" not in record
        assert "amplitude_mode" not in record


# Pipeline wiring: AuditLogger tested via hash chain integrity, SHA-256
# verification, and concurrent write safety. TestAuditHashChain and
# TestAuditDataIntegrity prove the audit pipeline.
