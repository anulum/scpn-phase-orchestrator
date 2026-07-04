# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Koopman-MPC CLI command tests

"""CLI contract tests for the review-only ``spo koopman-mpc`` command."""

from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import click
import numpy as np
import pytest
from click.testing import CliRunner
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
_MODULE_NAME = "scpn_phase_orchestrator.runtime.cli.koopman_mpc"
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "scpn_phase_orchestrator"
    / "runtime"
    / "cli"
    / "koopman_mpc.py"
)


@dataclass(frozen=True)
class _FakeFinding:
    flagged: bool


@dataclass(frozen=True)
class _FakeEvidence:
    event_id: str
    findings: tuple[_FakeFinding, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return the JSON shape written by the command."""
        return {
            "event_id": self.event_id,
            "finding_count": len(self.findings),
            "flagged_count": sum(finding.flagged for finding in self.findings),
        }


@dataclass(frozen=True)
class _FakeDampingResult:
    before_evidence: _FakeEvidence
    after_evidence: _FakeEvidence
    fit_residual: float
    uncontrolled_damping_ratio: float
    controlled_damping_ratio: float
    damping_improved: bool

    def to_audit_record(self) -> dict[str, object]:
        """Return the combined audit record written by the command."""
        return {
            "schema": "scpn_dvoc_oscillation_damping_audit_v1",
            "claim_boundary": "review_only_offline_no_live_actuation",
            "review_only": True,
            "before": self.before_evidence.to_audit_record(),
            "after": self.after_evidence.to_audit_record(),
            "damping_improved": self.damping_improved,
            "damping_delta": (
                self.controlled_damping_ratio - self.uncontrolled_damping_ratio
            ),
            "content_hash": "a" * 64,
        }


@dataclass(frozen=True)
class _FakePMURingdownResult:
    prc_evidence: _FakeEvidence
    signal_source: str
    sample_count: int
    sampling_rate_hz: float
    source_sha256: str

    def to_audit_record(self) -> dict[str, object]:
        """Return the PMU ringdown audit record written by the command."""
        return {
            "schema": "scpn_pmu_ringdown_prc_audit_v1",
            "claim_boundary": "review_only_offline_no_live_actuation",
            "review_only": True,
            "sample_count": self.sample_count,
            "sampling_rate_hz": self.sampling_rate_hz,
            "source_sha256": self.source_sha256,
            "prc_evidence_hash": "b" * 64,
            "content_hash": "c" * 64,
        }


@dataclass
class _DampingCall:
    state_matrix: FloatArray | None = None
    input_matrix: FloatArray | None = None
    initial_state: FloatArray | None = None
    horizon: int | None = None
    fs: float | None = None


def _placeholder_underdamped_oscillator(
    *, frequency_hz: float, damping_ratio: float, dt: float
) -> tuple[FloatArray, FloatArray]:
    raise AssertionError(
        f"unexpected oscillator call: {frequency_hz}, {damping_ratio}, {dt}"
    )


def _placeholder_damp_oscillation(
    state_matrix: FloatArray,
    input_matrix: FloatArray,
    *,
    initial_state: FloatArray,
    horizon: int,
    fs: float,
    captured_at: str,
) -> _FakeDampingResult:
    raise AssertionError(
        "unexpected damping call: "
        f"{state_matrix.shape}, {input_matrix.shape}, {initial_state.shape}, "
        f"{horizon}, {fs}, {captured_at}"
    )


def _placeholder_screen_pmu_ringdown_csv(*args: object, **kwargs: object) -> object:
    raise AssertionError(f"unexpected PMU ringdown call: {args}, {kwargs}")


def _load_koopman_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[types.ModuleType, click.Group]:
    """Load the Koopman CLI source file with only its command dependencies."""
    main = click.Group()
    cli_package = types.ModuleType("scpn_phase_orchestrator.runtime.cli")
    cli_package.__path__ = [str(_MODULE_PATH.parent)]  # type: ignore[attr-defined]

    app_module = types.ModuleType("scpn_phase_orchestrator.runtime.cli._app")
    app_module.__dict__["main"] = main

    damping_module = types.ModuleType(
        "scpn_phase_orchestrator.runtime.dvoc_oscillation_damping"
    )
    damping_module.__dict__.update(
        {
            "damp_oscillation": _placeholder_damp_oscillation,
            "underdamped_oscillator": _placeholder_underdamped_oscillator,
        }
    )
    pmu_module = types.ModuleType("scpn_phase_orchestrator.runtime.pmu_ringdown")
    pmu_module.__dict__["screen_pmu_ringdown_csv"] = (
        _placeholder_screen_pmu_ringdown_csv
    )

    monkeypatch.setitem(sys.modules, "scpn_phase_orchestrator.runtime.cli", cli_package)
    monkeypatch.setitem(
        sys.modules,
        "scpn_phase_orchestrator.runtime.cli._app",
        app_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "scpn_phase_orchestrator.runtime.dvoc_oscillation_damping",
        damping_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "scpn_phase_orchestrator.runtime.pmu_ringdown",
        pmu_module,
    )

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module, main


def _state_space_fixture() -> tuple[FloatArray, FloatArray]:
    state_matrix = np.asarray([[0.98, 0.02], [-0.1, 0.95]], dtype=np.float64)
    input_matrix = np.asarray([[0.0], [0.05]], dtype=np.float64)
    return state_matrix, input_matrix


def _result_fixture() -> _FakeDampingResult:
    return _FakeDampingResult(
        before_evidence=_FakeEvidence(
            event_id="dvoc-damping-open-loop",
            findings=(_FakeFinding(True), _FakeFinding(False)),
        ),
        after_evidence=_FakeEvidence(
            event_id="dvoc-damping-closed-loop",
            findings=(_FakeFinding(False), _FakeFinding(False)),
        ),
        fit_residual=1.2e-7,
        uncontrolled_damping_ratio=0.0123,
        controlled_damping_ratio=0.2415,
        damping_improved=True,
    )


def test_koopman_mpc_cli_writes_prc_evidence_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The command writes before/after PRC evidence and prints the summary."""
    cli_koopman_mpc, main = _load_koopman_cli(monkeypatch)
    state_matrix, input_matrix = _state_space_fixture()
    captured = _DampingCall()

    def fake_underdamped_oscillator(
        *, frequency_hz: float, damping_ratio: float, dt: float
    ) -> tuple[FloatArray, FloatArray]:
        assert frequency_hz == 0.75
        assert damping_ratio == 0.03
        assert dt == 0.05
        return state_matrix, input_matrix

    def fake_damp_oscillation(
        received_state_matrix: FloatArray,
        received_input_matrix: FloatArray,
        *,
        initial_state: FloatArray,
        horizon: int,
        fs: float,
        captured_at: str,
    ) -> _FakeDampingResult:
        assert captured_at.endswith("+00:00")
        captured.state_matrix = received_state_matrix
        captured.input_matrix = received_input_matrix
        captured.initial_state = initial_state
        captured.horizon = horizon
        captured.fs = fs
        return _result_fixture()

    monkeypatch.setattr(
        cli_koopman_mpc,
        "underdamped_oscillator",
        fake_underdamped_oscillator,
    )
    monkeypatch.setattr(cli_koopman_mpc, "damp_oscillation", fake_damp_oscillation)

    output_path = tmp_path / "prc-evidence.json"
    result = CliRunner().invoke(
        main,
        [
            "koopman-mpc",
            "--frequency-hz",
            "0.75",
            "--damping-ratio",
            "0.03",
            "--dt",
            "0.05",
            "--horizon",
            "12",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "=== Koopman-MPC oscillation damping ===" in result.output
    assert "plant: f=0.75 Hz  open-loop zeta=0.03" in result.output
    assert "[before] weakest damping=0.0123  flagged=1/2" in result.output
    assert "[after ] weakest damping=0.2415  flagged=0/2" in result.output
    assert "damping improved: True" in result.output
    assert f"PRC evidence written to {output_path}" in result.output
    assert captured.state_matrix is state_matrix
    assert captured.input_matrix is input_matrix
    assert captured.initial_state is not None
    np.testing.assert_allclose(captured.initial_state, np.asarray([1.0, 0.0]))
    assert captured.horizon == 12
    assert captured.fs == 20.0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "scpn_dvoc_oscillation_damping_audit_v1"
    assert payload["claim_boundary"] == "review_only_offline_no_live_actuation"
    assert payload["review_only"] is True
    assert payload["before"] == {
        "event_id": "dvoc-damping-open-loop",
        "finding_count": 2,
        "flagged_count": 1,
    }
    assert payload["after"] == {
        "event_id": "dvoc-damping-closed-loop",
        "finding_count": 2,
        "flagged_count": 0,
    }
    assert payload["damping_improved"] is True
    assert payload["damping_delta"] == pytest.approx(0.2292)
    assert len(str(payload["content_hash"])) == 64


def test_koopman_mpc_cli_prints_summary_without_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default command prints review evidence without writing a file."""
    cli_koopman_mpc, main = _load_koopman_cli(monkeypatch)
    monkeypatch.setattr(
        cli_koopman_mpc,
        "underdamped_oscillator",
        lambda **_: _state_space_fixture(),
    )
    monkeypatch.setattr(
        cli_koopman_mpc,
        "damp_oscillation",
        lambda *_, **__: _result_fixture(),
    )

    result = CliRunner().invoke(main, ["koopman-mpc", "--horizon", "4"])

    assert result.exit_code == 0, result.output
    assert "damping improved: True" in result.output
    assert "PRC evidence written" not in result.output


def test_koopman_mpc_cli_reports_invalid_oscillator_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid oscillator parameters are surfaced as Click errors."""
    cli_koopman_mpc, main = _load_koopman_cli(monkeypatch)

    def fake_underdamped_oscillator(
        *, frequency_hz: float, damping_ratio: float, dt: float
    ) -> tuple[FloatArray, FloatArray]:
        raise ValueError(
            f"invalid oscillator: f={frequency_hz}, zeta={damping_ratio}, dt={dt}"
        )

    monkeypatch.setattr(
        cli_koopman_mpc,
        "underdamped_oscillator",
        fake_underdamped_oscillator,
    )

    result = CliRunner().invoke(main, ["koopman-mpc", "--frequency-hz", "-1.0"])

    assert result.exit_code != 0
    assert "Error: invalid oscillator: f=-1.0" in result.output


def test_pmu_ringdown_cli_writes_review_only_prc_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The PMU command screens operator CSV data and writes sealed evidence."""
    cli_koopman_mpc, main = _load_koopman_cli(monkeypatch)
    csv_path = tmp_path / "pmu.csv"
    csv_path.write_text("time_s,frequency_hz\n0.0,60.0\n0.02,60.1\n", encoding="utf-8")
    output_path = tmp_path / "pmu-prc.json"

    def fake_screen_pmu_ringdown_csv(
        path: Path,
        *,
        event_id: str,
        captured_at: str,
        signal_source: str,
        time_column: str,
        frequency_column: str,
        nominal_frequency_hz: float,
    ) -> _FakePMURingdownResult:
        assert path == csv_path
        assert event_id == "PMU-EVT-001"
        assert captured_at == "2026-07-04T10:00:00Z"
        assert signal_source == "PMU/BUS-42/frequency"
        assert time_column == "timestamp"
        assert frequency_column == "freq"
        assert nominal_frequency_hz == 50.0
        return _FakePMURingdownResult(
            prc_evidence=_FakeEvidence(
                event_id="PMU-EVT-001",
                findings=(_FakeFinding(True), _FakeFinding(False)),
            ),
            signal_source="PMU/BUS-42/frequency",
            sample_count=128,
            sampling_rate_hz=25.0,
            source_sha256="d" * 64,
        )

    monkeypatch.setattr(
        cli_koopman_mpc,
        "screen_pmu_ringdown_csv",
        fake_screen_pmu_ringdown_csv,
    )

    result = CliRunner().invoke(
        main,
        [
            "pmu-ringdown",
            str(csv_path),
            "--event-id",
            "PMU-EVT-001",
            "--captured-at",
            "2026-07-04T10:00:00Z",
            "--signal-source",
            "PMU/BUS-42/frequency",
            "--time-column",
            "timestamp",
            "--frequency-column",
            "freq",
            "--nominal-frequency-hz",
            "50.0",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "=== PMU ringdown PRC screening ===" in result.output
    assert "source: PMU/BUS-42/frequency  samples=128  fs=25.0000 Hz" in result.output
    assert "flagged=1/2" in result.output
    assert f"PMU PRC evidence written to {output_path}" in result.output

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "scpn_pmu_ringdown_prc_audit_v1"
    assert payload["claim_boundary"] == "review_only_offline_no_live_actuation"
    assert payload["review_only"] is True
    assert payload["sample_count"] == 128
    assert payload["source_sha256"] == "d" * 64


def test_pmu_ringdown_cli_reports_invalid_operator_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid PMU screening inputs are surfaced as Click errors."""
    cli_koopman_mpc, main = _load_koopman_cli(monkeypatch)
    csv_path = tmp_path / "pmu.csv"
    csv_path.write_text("time_s,frequency_hz\n0.0,60.0\n", encoding="utf-8")

    def fake_screen_pmu_ringdown_csv(*_: object, **__: object) -> object:
        raise ValueError("PMU ringdown CSV is missing required column frequency_hz")

    monkeypatch.setattr(
        cli_koopman_mpc,
        "screen_pmu_ringdown_csv",
        fake_screen_pmu_ringdown_csv,
    )

    result = CliRunner().invoke(
        main,
        [
            "pmu-ringdown",
            str(csv_path),
            "--event-id",
            "PMU-EVT-001",
            "--captured-at",
            "2026-07-04T10:00:00Z",
            "--signal-source",
            "PMU/BUS-42/frequency",
        ],
    )

    assert result.exit_code != 0
    assert "Error: PMU ringdown CSV is missing required column" in result.output
