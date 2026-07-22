# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL audit-stream round-trip tests

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.monitor.stl import STLMonitor, STLTraceResult
from scpn_phase_orchestrator.runtime import stl_audit_chain
from scpn_phase_orchestrator.runtime.audit_stream import (
    EventStreamWriter,
    read_event_stream,
    verify_event_stream_integrity,
)
from scpn_phase_orchestrator.runtime.stl_audit_chain import (
    STL_AUDIT_EVENT_TYPE,
    append_stl_result,
    read_stl_results,
    write_stl_results,
)


@pytest.fixture(autouse=True)
def _unsigned_audit_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force deterministic unsigned audit mode for the round-trip tests."""
    monkeypatch.delenv("SPO_AUDIT_KEY", raising=False)
    monkeypatch.delenv("SPO_AUDIT_KEYRING", raising=False)


def _sample_results() -> list[STLTraceResult]:
    traces = {"R": [0.9, 0.8, 0.7, 0.5]}
    return [
        STLMonitor("always (R >= 0.3)").evaluate_result(traces),
        STLMonitor("eventually (R >= 0.85)").evaluate_result(traces),
        STLMonitor("always[0,2] (R >= 0.6)").evaluate_result(traces),
    ]


class TestRoundTrip:
    def test_write_then_read_recovers_records(self, tmp_path: Path) -> None:
        results = _sample_results()
        path = write_stl_results(tmp_path / "stl.spoa", results)
        assert path == tmp_path / "stl.spoa"

        recovered = read_stl_results(path)
        assert recovered == results

    def test_recovered_events_pass_integrity_and_carry_event_type(
        self, tmp_path: Path
    ) -> None:
        path = write_stl_results(tmp_path / "stl.spoa", _sample_results())
        events = read_event_stream(path)
        ok, count = verify_event_stream_integrity(events)
        assert ok
        assert count == 3
        assert {event.event_type for event in events} == {STL_AUDIT_EVENT_TYPE}

    def test_write_returns_path_for_empty_batch(self, tmp_path: Path) -> None:
        path = write_stl_results(tmp_path / "empty.spoa", [])
        assert path.exists()
        assert read_stl_results(path) == []

    def test_signed_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPO_AUDIT_KEY", "0" * 32)
        results = _sample_results()
        path = write_stl_results(tmp_path / "signed.spoa", results)
        assert read_stl_results(path) == results


class TestNonStlEventsFiltered:
    def test_only_stl_events_are_replayed(self, tmp_path: Path) -> None:
        writer = EventStreamWriter(tmp_path / "mixed.spoa", stream_id="spo-mixed")
        try:
            append_stl_result(
                writer, STLMonitor("always (R >= 0.3)").evaluate_result({"R": [0.9]})
            )
            writer.write({"unrelated": "payload"}, event_type="other.event")
        finally:
            writer.close()

        recovered = read_stl_results(tmp_path / "mixed.spoa")
        assert len(recovered) == 1
        assert recovered[0].spec == "always (R >= 0.3)"


class TestNonFiniteRobustnessRejected:
    def test_append_rejects_infinite_robustness(self, tmp_path: Path) -> None:
        # A bounded eventually window past the trace end yields -inf.
        vacuous = STLMonitor("eventually[5,6] (R >= 0.3)").evaluate_result(
            {"R": [0.9, 0.8, 0.7]}
        )
        assert vacuous.robustness == float("-inf")
        writer = EventStreamWriter(tmp_path / "reject.spoa")
        try:
            with pytest.raises(ValueError, match="robustness must be finite"):
                append_stl_result(writer, vacuous)
        finally:
            writer.close()

    def test_write_batch_rejects_infinite_robustness(self, tmp_path: Path) -> None:
        forged = STLTraceResult(
            spec="always[9,9] (R >= 0.3)",
            robustness=float("inf"),
            satisfied=True,
            backend="builtin",
        )
        with pytest.raises(ValueError, match="robustness must be finite"):
            write_stl_results(tmp_path / "reject.spoa", [forged])


class TestIntegrityGuard:
    def test_failed_integrity_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write_stl_results(tmp_path / "stl.spoa", _sample_results())
        monkeypatch.setattr(
            stl_audit_chain,
            "verify_event_stream_integrity",
            lambda events: (False, 0),
        )
        with pytest.raises(ValueError, match="failed integrity verification"):
            read_stl_results(path)


class TestPayloadReconstruction:
    def _valid_payload(self) -> dict[str, object]:
        return {
            "spec": "always (R >= 0.3)",
            "robustness": 0.2,
            "satisfied": True,
            "backend": "builtin",
        }

    def test_valid_payload_reconstructs(self) -> None:
        result = stl_audit_chain._stl_result_from_payload(self._valid_payload())
        assert result == STLTraceResult(
            spec="always (R >= 0.3)",
            robustness=0.2,
            satisfied=True,
            backend="builtin",
        )

    def test_integer_robustness_is_coerced_to_float(self) -> None:
        payload = self._valid_payload() | {"robustness": 1}
        result = stl_audit_chain._stl_result_from_payload(payload)
        assert isinstance(result.robustness, float)
        assert result.robustness == 1.0

    def test_missing_field_is_rejected(self) -> None:
        payload = self._valid_payload()
        del payload["backend"]
        with pytest.raises(ValueError, match="must carry exactly"):
            stl_audit_chain._stl_result_from_payload(payload)

    def test_extra_field_is_rejected(self) -> None:
        payload = self._valid_payload() | {"extra": 1}
        with pytest.raises(ValueError, match="must carry exactly"):
            stl_audit_chain._stl_result_from_payload(payload)

    def test_non_string_spec_is_rejected(self) -> None:
        payload = self._valid_payload() | {"spec": 3}
        with pytest.raises(ValueError, match="'spec' and 'backend' must be strings"):
            stl_audit_chain._stl_result_from_payload(payload)

    def test_non_bool_satisfied_is_rejected(self) -> None:
        payload = self._valid_payload() | {"satisfied": 1}
        with pytest.raises(ValueError, match="'satisfied' must be a boolean"):
            stl_audit_chain._stl_result_from_payload(payload)

    def test_boolean_robustness_is_rejected(self) -> None:
        payload = self._valid_payload() | {"robustness": True}
        with pytest.raises(ValueError, match="'robustness' must be a real number"):
            stl_audit_chain._stl_result_from_payload(payload)

    def test_non_numeric_robustness_is_rejected(self) -> None:
        payload = self._valid_payload() | {"robustness": "0.2"}
        with pytest.raises(ValueError, match="'robustness' must be a real number"):
            stl_audit_chain._stl_result_from_payload(payload)
