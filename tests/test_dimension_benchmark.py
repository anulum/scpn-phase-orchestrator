# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Dimension benchmark contract tests

"""Benchmark contract tests for the fractal-dimension polyglot parity gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from benchmarks._dimension_reference import reference_outputs
from benchmarks.dimension_benchmark import (
    BACKEND_ORDER,
    bench_at,
    benchmark_dimension_polyglot_parity_gate,
)
from scpn_phase_orchestrator.monitor import dimension
from tests.test_dimension_julia_absence import run_source_profile


class _Controls(TypedDict, total=False):
    """Name the integer controls whose actual invalid payloads are exercised."""

    t: int
    d: int
    n_k: int
    calls: int
    seed: int
    warmups: int
    repeats: int


@pytest.mark.parametrize(
    ("spectrum", "expected"),
    [
        ((), 0.0),
        ((-1.0, -2.0), 0.0),
        ((0.0, 0.0), 2.0),
        ((1.0, 2.0), 2.0),
        ((1.0, -2.0), 1.5),
    ],
)
def test_reference_oracle_has_analytical_geometry_and_spectrum(
    spectrum: tuple[float, ...], expected: float
) -> None:
    """Check the benchmark oracle on exact 3-4-5 geometry and rational KY cases."""
    trajectory = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 4.0]])
    integral, dimension_value = reference_outputs(
        trajectory, np.array([0.0, 3.0, 4.0, 5.0, 6.0]), np.array(spectrum)
    )
    np.testing.assert_array_equal(integral, [0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert dimension_value == expected


def test_polyglot_parity_gate_records_every_declared_backend() -> None:
    """Verify honest optional slots and required Python numerical evidence."""
    active = dimension.ACTIVE_BACKEND
    out = benchmark_dimension_polyglot_parity_gate(
        t=24,
        d=3,
        n_k=8,
        calls=1,
        seed=19,
    )
    records = json.loads(str(out["backend_records_json"]))

    assert out["suite"] == "dimension_polyglot_parity_gate"
    assert out["backend_count"] == len(BACKEND_ORDER)
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert out["parity_pass_count"] == out["available_backend_count"]
    assert 0.0 <= float(str(out["reference_ci_min"])) <= 1.0
    assert 0.0 <= float(str(out["reference_ci_max"])) <= 1.0
    assert float(str(out["reference_kaplan_yorke"])) == 3.28125
    assert active == dimension.ACTIVE_BACKEND
    assert json.loads(str(out["required_backends_json"])) == ["python"]
    assert out["required_backends_present"] == 1
    assert out["production_timing_claim"] == 0
    assert [record["backend"] for record in records] == list(BACKEND_ORDER)

    python_record = next(record for record in records if record["backend"] == "python")
    assert python_record["status"] == "available"
    assert python_record["ci_max_abs_error"] == 0.0
    assert python_record["ky_abs_error"] == 0.0
    assert python_record["max_abs_error"] == 0.0
    assert python_record["ci_monotonic"] is True
    assert python_record["ci_unit_interval"] is True
    assert python_record["ky_bounds_passed"] is True
    assert python_record["parity_passed"] is True

    for record in records:
        if record["status"] == "available":
            assert record["ms_per_call"] is not None
            assert record["correlation_integral_sha256"] is not None
            assert record["kaplan_yorke_sha256"] is not None
            assert record["ci_max_abs_error"] <= record["tolerance"]
            assert record["ky_abs_error"] <= record["tolerance"]
            assert record["max_abs_error"] <= record["tolerance"]
            assert record["ci_monotonic"] is True
            assert record["ci_unit_interval"] is True
            assert record["ky_bounds_passed"] is True
            assert len(record["repeat_ms_per_call"]) == 3
            continue
        assert record["status"] == "unavailable"
        assert record["unavailable_reason"]
        assert record["correlation_integral_sha256"] is None
        assert record["kaplan_yorke_sha256"] is None
        assert record["parity_passed"] is False


def test_polyglot_parity_gate_rejects_invalid_controls() -> None:
    """Reject invalid integral controls before allocating or starting a runtime."""
    for kwargs in (
        {"t": True},
        {"t": 1},
        {"d": 0},
        {"n_k": 1},
        {"calls": 0},
        {"seed": -1},
        {"warmups": 0},
        {"repeats": 0},
        {"repeats": 1.5},
    ):
        try:
            benchmark_dimension_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid controls were accepted: {kwargs!r}")


@pytest.mark.parametrize("owners", [(), ("auto",), ("Python",), ("python", "python")])
def test_required_owner_names_are_unambiguous(owners: tuple[str, ...]) -> None:
    """Refuse absent, unknown or duplicate comparison requirements."""
    with pytest.raises(ValueError, match="required dimension backends"):
        benchmark_dimension_polyglot_parity_gate(require_backends=owners)


def test_result_hash_excludes_wall_clock_and_tracks_problem() -> None:
    """Repeated actual public calls have a stable numerical evidence hash."""
    first = bench_at(16, 2, 5, 1, repeats=1, seed=7)
    again = bench_at(16, 2, 5, 1, repeats=1, seed=7)
    changed = bench_at(16, 2, 5, 1, repeats=1, seed=8)
    assert first["acceptance_passed"] == again["acceptance_passed"] == 1
    assert len(str(first["benchmark_sha256"])) == 64
    assert first["benchmark_sha256"] == again["benchmark_sha256"]
    assert first["benchmark_sha256"] != changed["benchmark_sha256"]
    assert float(str(first["python_ms_per_call"])) >= 0.0


@pytest.mark.native_runtime
def test_required_all_five_measure_actual_public_owners() -> None:
    """A complete comparison requires real successful results from every owner."""
    out = benchmark_dimension_polyglot_parity_gate(
        t=16, d=2, n_k=5, calls=1, repeats=2, require_backends=BACKEND_ORDER
    )
    assert out["comparison_complete"] == out["acceptance_passed"] == 1
    assert out["available_backend_count"] == out["parity_pass_count"] == 5
    records = json.loads(str(out["backend_records_json"]))
    assert all(record["status"] == "available" for record in records)
    assert all(len(record["repeat_ms_per_call"]) == 2 for record in records)


@pytest.mark.native_runtime
@pytest.mark.parametrize("mode", ["missing", "present"])
def test_actual_absence_cli_never_claims_complete_comparison(
    tmp_path: Path, mode: str
) -> None:
    """Run the actual CLI against absent assets and a real Julia-only profile."""
    run_source_profile(tmp_path, mode, benchmark=True)


def test_independent_benchmark_gate_refuses_corrupted_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real public outputs must fail against a deliberately corrupted oracle."""
    from benchmarks import dimension_benchmark

    def corrupted_reference(
        trajectory: object,
        epsilons: object,
        spectrum: object,
    ) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], float]:
        return np.full(np.asarray(epsilons).size, 0.123456789), 0.0

    monkeypatch.setattr(dimension_benchmark, "reference_outputs", corrupted_reference)
    result = benchmark_dimension_polyglot_parity_gate(t=8, n_k=4, repeats=1)
    assert result["acceptance_passed"] == 0
    assert result["benchmark_sha256"] is None
    records = json.loads(str(result["backend_records_json"]))
    for row in records:
        if row["status"] != "unavailable":
            assert row["status"] == "invalid"
            assert row["ms_per_call"] is None
            assert row["correlation_integral_sha256"] is None
            assert row["kaplan_yorke_sha256"] is None
