# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real-grid early-warning capstone logic tests

"""Tests for the power-grid early-warning capstone adapter on synthetic data.

This capstone is the grid adapter onto the domain-neutral harness (tested in
``tests/test_early_warning_domain.py``): the only grid-specific work is the
signal-processing pipeline that produces the neutral bundle, the PSML CSV
ingestion, the growing-vs-damped scenario classification, and the end-to-end
orchestration. The corpus is citation-only, so every path is pinned here on
**synthetic arrays and synthetic PSML-format scenarios written in the test** —
never the redistributed archive.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bench.early_warning_leadtime_grid import (
    BAND_HZ,
    DETECTORS,
    SEGMENT_SECONDS,
    WINDOW_SECONDS,
    GridPhaseAdapter,
    bus_voltages,
    classify_scenario,
    discover_scenarios,
    grid_observables,
    main,
    oscillation_growth_ratio,
    oscillation_info,
)
from scpn_phase_orchestrator.monitor.early_warning_suite import (
    DomainObservableAdapter,
    SuiteObservables,
)

_TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                           #
# --------------------------------------------------------------------------- #


def _bus_field(
    *,
    n_samples: int,
    fs: float,
    hz: float = 1.0,
    growth: bool = False,
    base_amp: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Return a 23-bus voltage field (rows = buses) around 1.0 p.u.

    A ``growth`` field is a *growing instability*: a common oscillation whose
    amplitude climbs (so the buses lock ever tighter into it — the coherence rise a
    detector can lead) plus a per-bus component that grows with it (so the cross-bus
    deviation grows too — the growth-ratio classifier), over a fixed noise floor.
    Otherwise it is a quiet, damped field — a small fixed-amplitude oscillation over
    the same noise floor, the false-alarm null.
    """
    rng = np.random.default_rng(seed)
    n_bus = 23
    times = np.arange(n_samples) / fs
    noise_floor = 0.06
    common = np.sin(_TWO_PI * hz * times)
    field = np.empty((n_bus, n_samples), dtype=np.float64)
    if growth:
        amp = np.linspace(0.02, 0.7, n_samples)
        for bus in range(n_bus):
            offset = rng.uniform(-np.pi, np.pi)
            scale = 1.0 + 0.3 * rng.standard_normal()
            per_bus = scale * np.sin(_TWO_PI * hz * times + offset)
            field[bus] = (
                1.0
                + amp * (common + 0.35 * per_bus)
                + noise_floor * rng.standard_normal(n_samples)
            )
        return field
    for bus in range(n_bus):
        offset = rng.uniform(-np.pi, np.pi)
        field[bus] = (
            1.0
            + base_amp * np.sin(_TWO_PI * hz * times + offset)
            + noise_floor * rng.standard_normal(n_samples)
        )
    return field


def _write_scenario(
    directory: Path,
    name: str,
    *,
    voltages: np.ndarray,
    fs: float,
    kind: str,
    start: float,
    end: float,
) -> Path:
    """Write a synthetic PSML-format scenario (``trans.csv`` + ``info.csv``)."""
    scenario = directory / name
    scenario.mkdir(parents=True, exist_ok=True)
    n_samples = voltages.shape[1]
    times = (np.arange(n_samples) / fs).reshape(-1, 1)
    table = np.hstack([times, voltages.T])
    header = ["Time(s)"] + [
        f"VOLT {100 + b} [BUS {b}]" for b in range(voltages.shape[0])
    ]
    with (scenario / "trans.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(table.tolist())
    (scenario / "info.csv").write_text(
        f"type, {kind}\nstart, {start}\nend, {end}\nbus1, 102\n", encoding="utf-8"
    )
    return scenario


# --------------------------------------------------------------------------- #
# grid_observables and the GridPhaseAdapter                                    #
# --------------------------------------------------------------------------- #


def test_grid_observables_shapes_and_derived_fields() -> None:
    raw = _bus_field(n_samples=500, fs=100.0, growth=True)
    observables = grid_observables(raw, sampling_rate_hz=100.0)
    assert isinstance(observables, SuiteObservables)
    assert observables.sampling_rate_hz == pytest.approx(100.0)
    assert observables.n_nodes == 23
    assert observables.n_samples == 500
    assert np.allclose(observables.phase_field, np.sin(observables.phases))
    assert np.all(observables.order_parameter >= 0.0)
    assert np.all(observables.order_parameter <= 1.0 + 1.0e-9)


def test_grid_observables_rejects_a_single_bus() -> None:
    with pytest.raises(ValueError, match="at least two buses"):
        grid_observables(np.zeros((1, 500)), sampling_rate_hz=100.0)


def test_grid_observables_rejects_a_non_finite_block() -> None:
    raw = _bus_field(n_samples=500, fs=100.0)
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        grid_observables(raw, sampling_rate_hz=100.0)


def test_grid_phase_adapter_satisfies_the_protocol_and_wraps_the_pipeline() -> None:
    adapter = GridPhaseAdapter(sampling_rate_hz=100.0)
    assert isinstance(adapter, DomainObservableAdapter)
    assert adapter.domain == "power_grid"
    raw = _bus_field(n_samples=500, fs=100.0, growth=True)
    observables = adapter.observables(raw)
    assert isinstance(observables, SuiteObservables)
    assert observables.n_nodes == 23
    direct = grid_observables(raw, sampling_rate_hz=100.0)
    assert np.array_equal(observables.phases, direct.phases)


def test_module_constants_match_the_documented_pipeline() -> None:
    assert BAND_HZ == (0.2, 5.0)
    assert SEGMENT_SECONDS == 2.0
    assert WINDOW_SECONDS == 0.5


# --------------------------------------------------------------------------- #
# PSML ingestion + classification                                             #
# --------------------------------------------------------------------------- #


def test_bus_voltages_reads_field_and_rate(tmp_path: Path) -> None:
    field = _bus_field(n_samples=400, fs=100.0)
    scenario = _write_scenario(
        tmp_path, "s1", voltages=field, fs=100.0, kind="gen_trip", start=0.2, end=3.0
    )
    fs, voltages = bus_voltages(scenario)
    assert fs == pytest.approx(100.0)
    assert voltages.shape == (23, 400)


def test_bus_voltages_rejects_a_single_bus(tmp_path: Path) -> None:
    scenario = tmp_path / "mono"
    scenario.mkdir()
    times = np.arange(50) / 100.0
    with (scenario / "trans.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time(s)", "VOLT 101 [BUS]"])
        writer.writerows(np.column_stack([times, np.ones(50)]).tolist())
    with pytest.raises(ValueError, match="fewer than two buses"):
        bus_voltages(scenario)


def test_oscillation_info_reads_metadata(tmp_path: Path) -> None:
    scenario = _write_scenario(
        tmp_path,
        "s2",
        voltages=_bus_field(n_samples=100, fs=100.0),
        fs=100.0,
        kind="gen_trip",
        start=0.5,
        end=2.0,
    )
    info = oscillation_info(scenario)
    assert info["type"] == "gen_trip"
    assert info["start"] == "0.5"
    assert info["end"] == "2.0"


def test_oscillation_info_skips_lines_without_a_comma(tmp_path: Path) -> None:
    scenario = tmp_path / "s3"
    scenario.mkdir()
    (scenario / "info.csv").write_text(
        "a header line without a comma\ntype, gen_trip\nstart, 0.1\n", encoding="utf-8"
    )
    info = oscillation_info(scenario)
    assert info == {"type": "gen_trip", "start": "0.1"}


def test_growth_ratio_detects_a_growing_oscillation() -> None:
    field = _bus_field(n_samples=500, fs=100.0, growth=True)
    ratio = oscillation_growth_ratio(
        field, sampling_rate_hz=100.0, start_s=0.2, end_s=4.8
    )
    assert ratio > 1.3


def test_growth_ratio_is_zero_for_a_degenerate_window() -> None:
    field = _bus_field(n_samples=500, fs=100.0)
    assert (
        oscillation_growth_ratio(field, sampling_rate_hz=100.0, start_s=1.0, end_s=1.0)
        == 0.0
    )


def test_growth_ratio_is_zero_when_the_early_third_is_flat() -> None:
    # A field whose early third is exactly constant has no early deviation.
    field = np.ones((23, 600), dtype=np.float64)
    field[:, 400:] += np.linspace(0.0, 0.5, 200)  # deviation only in the late third
    ratio = oscillation_growth_ratio(
        field, sampling_rate_hz=100.0, start_s=0.0, end_s=6.0
    )
    assert ratio == 0.0


def test_classify_marks_a_growing_gen_trip_as_a_transition(tmp_path: Path) -> None:
    field = _bus_field(n_samples=500, fs=100.0, growth=True)
    scenario = _write_scenario(
        tmp_path, "grow", voltages=field, fs=100.0, kind="gen_trip", start=0.2, end=4.8
    )
    kind, onset = classify_scenario(scenario, segment_samples=100)
    assert kind == "transition"
    assert onset == 480


def test_classify_skips_a_damped_gen_trip(tmp_path: Path) -> None:
    field = _bus_field(n_samples=500, fs=100.0, growth=False)
    scenario = _write_scenario(
        tmp_path, "flat", voltages=field, fs=100.0, kind="gen_trip", start=0.2, end=4.8
    )
    assert classify_scenario(scenario, segment_samples=100) == ("skip", None)


def test_classify_skips_a_gen_trip_without_room_for_a_segment(tmp_path: Path) -> None:
    field = _bus_field(n_samples=500, fs=100.0, growth=True)
    # end - start = 0.3 s = 30 samples < the 100-sample segment.
    scenario = _write_scenario(
        tmp_path, "tight", voltages=field, fs=100.0, kind="gen_trip", start=0.2, end=0.5
    )
    assert classify_scenario(scenario, segment_samples=100) == ("skip", None)


def test_classify_marks_a_damped_fault_as_a_null(tmp_path: Path) -> None:
    field = _bus_field(n_samples=400, fs=100.0, growth=False)
    scenario = _write_scenario(
        tmp_path, "null", voltages=field, fs=100.0, kind="bus_fault", start=0.2, end=3.0
    )
    assert classify_scenario(scenario, segment_samples=100) == ("null", None)


def test_classify_skips_a_short_damped_fault(tmp_path: Path) -> None:
    field = _bus_field(n_samples=60, fs=100.0, growth=False)
    scenario = _write_scenario(
        tmp_path,
        "short",
        voltages=field,
        fs=100.0,
        kind="bus_fault",
        start=0.1,
        end=0.5,
    )
    assert classify_scenario(scenario, segment_samples=100) == ("skip", None)


def test_classify_skips_an_unhandled_type(tmp_path: Path) -> None:
    field = _bus_field(n_samples=400, fs=100.0)
    scenario = _write_scenario(
        tmp_path, "other", voltages=field, fs=100.0, kind="bus_trip", start=0.2, end=3.0
    )
    assert classify_scenario(scenario, segment_samples=100) == ("skip", None)


def test_classify_skips_a_scenario_without_start_end(tmp_path: Path) -> None:
    scenario = tmp_path / "noinfo"
    _write_scenario(
        scenario,
        "x",
        voltages=_bus_field(n_samples=100, fs=100.0),
        fs=100.0,
        kind="gen_trip",
        start=0.2,
        end=3.0,
    )
    # Overwrite info.csv with a missing end field.
    (scenario / "x" / "info.csv").write_text(
        "type, gen_trip\nstart, 0.2\n", encoding="utf-8"
    )
    assert classify_scenario(scenario / "x", segment_samples=100) == ("skip", None)


def test_discover_scenarios_finds_only_complete_scenarios(tmp_path: Path) -> None:
    _write_scenario(
        tmp_path,
        "a",
        voltages=_bus_field(n_samples=100, fs=100.0),
        fs=100.0,
        kind="gen_trip",
        start=0.2,
        end=1.0,
    )
    # A directory with trans.csv but no info.csv is not a scenario.
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "trans.csv").write_text("Time(s)\n0.0\n", encoding="utf-8")
    found = discover_scenarios(tmp_path)
    assert [p.name for p in found] == ["a"]


# --------------------------------------------------------------------------- #
# main — end-to-end over synthetic PSML scenarios                              #
# --------------------------------------------------------------------------- #


def _build_corpus(data: Path) -> None:
    """Write a small synthetic PSML corpus: growing transitions + damped nulls."""
    fs = 100.0
    # Growing instabilities: a rising common oscillation the buses lock into (some
    # detectors lead) with a growing per-bus component (the growth-ratio classifier).
    for i in range(2):
        _write_scenario(
            data,
            f"grow_{i}",
            voltages=_bus_field(n_samples=300, fs=fs, growth=True, seed=i),
            fs=fs,
            kind="gen_trip",
            start=0.2,
            end=2.6,
        )
    # Damped nulls.
    for i in range(3):
        _write_scenario(
            data,
            f"null_{i}",
            voltages=_bus_field(n_samples=300, fs=fs, growth=False, seed=100 + i),
            fs=fs,
            kind="bus_fault",
            start=0.2,
            end=2.6,
        )
    # An unhandled disturbance type is discovered but skipped (neither transition
    # nor null), so the partition exercises the skip path.
    _write_scenario(
        data,
        "skip_me",
        voltages=_bus_field(n_samples=300, fs=fs, growth=True, seed=42),
        fs=fs,
        kind="bus_trip",
        start=0.2,
        end=2.6,
    )


def test_main_writes_sealed_derived_artefacts(tmp_path: Path) -> None:
    data = tmp_path / "corpus"
    data.mkdir()
    out = tmp_path / "derived"
    _build_corpus(data)

    main(data, out, segment_seconds=1.0, baseline_fraction=1.0 / 3.0)

    aggregate = out / "early_warning_leadtime_grid_results.json"
    assert aggregate.exists()
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "early_warning_leadtime_grid"
    assert set(payload["matched_false_alarm_thresholds"]) == set(DETECTORS)
    assert payload["n_transitions_found"] == 2  # two growing instabilities
    assert payload["n_null_scenarios_found"] == 3
    assert payload["n_transitions_evaluated"] == 2
    assert payload["verdict"]
    # A growing transition leads at least one detector; the members it does not
    # lead exercise the no-lead path too, within the same aggregate.
    all_leads = [
        lead
        for record in payload["transitions"]
        for lead in record["lead_seconds"].values()
    ]
    assert any(lead is not None for lead in all_leads)
    # Every evaluated transition is sealed for every detector.
    for record in payload["transitions"]:
        sealed = json.loads(
            (out / f"{record['record_id']}_early_warning_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(sealed["detectors"]) == set(DETECTORS)
        for detector in DETECTORS:
            assert sealed["detectors"][detector]["content_hash"]


def test_main_caps_the_evaluated_counts(tmp_path: Path) -> None:
    data = tmp_path / "corpus"
    data.mkdir()
    out = tmp_path / "derived"
    _build_corpus(data)

    main(
        data,
        out,
        segment_seconds=1.0,
        baseline_fraction=1.0 / 3.0,
        max_transitions=1,
        max_null_scenarios=2,
    )

    payload = json.loads(
        (out / "early_warning_leadtime_grid_results.json").read_text(encoding="utf-8")
    )
    # Found more than the caps, but only the capped counts are evaluated.
    assert payload["n_transitions_found"] == 2
    assert payload["n_transitions_evaluated"] == 1


def test_main_rejects_an_empty_corpus(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no PSML scenarios"):
        main(tmp_path / "empty", tmp_path / "out")
