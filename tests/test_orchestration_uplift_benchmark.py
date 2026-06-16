# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for the orchestration uplift benchmark

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_BENCH_PATH = Path("benchmarks/orchestration_uplift_benchmark.py")
_spec = importlib.util.spec_from_file_location(
    "orchestration_uplift_benchmark", _BENCH_PATH
)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

DOMAINPACKS = Path("domainpacks")


def _assert_bounded(loop: dict) -> None:
    assert 0.0 <= loop["r_good"] <= 1.0
    assert 0.0 <= loop["r_bad"] <= 1.0
    # r_good, r_bad, and separation are each rounded to 9 decimals independently,
    # so the identity can be off by up to ~2e-9.
    assert loop["separation"] == pytest.approx(loop["r_good"] - loop["r_bad"], abs=2e-9)


class TestMeasurePack:
    def test_record_shape_and_bounds(self) -> None:
        record = bench.measure_pack(
            "sleep_architecture", steps=20, seed=7, domainpacks_root=DOMAINPACKS
        )
        assert record["pack"] == "sleep_architecture"
        assert record["amplitude_mode"] is True
        _assert_bounded(record["open_loop"])
        _assert_bounded(record["closed_loop"])
        assert record["uplift"] == pytest.approx(
            record["closed_loop"]["separation"] - record["open_loop"]["separation"],
            abs=1e-9,
        )

    def test_open_loop_takes_no_actions(self) -> None:
        record = bench.measure_pack(
            "chemical_reactor", steps=60, seed=7, domainpacks_root=DOMAINPACKS
        )
        # The open-loop baseline never actuates; the closed loop may.
        assert record["closed_loop"]["action_total"] >= 0


class TestRunBenchmark:
    def test_covers_all_beachhead_packs(self) -> None:
        report = bench.run_benchmark(steps=15, seed=7, domainpacks_root=DOMAINPACKS)
        assert report["benchmark"] == "orchestration-uplift"
        packs = {r["pack"] for r in report["packs"]}
        assert packs == {pack for _vertical, pack in bench.BEACHHEAD_PACKS}
        verticals = {r["vertical"] for r in report["packs"]}
        assert verticals == {"industrial", "infrastructure", "biosignal"}

    def test_deterministic(self) -> None:
        a = bench.run_benchmark(steps=15, seed=7, domainpacks_root=DOMAINPACKS)
        b = bench.run_benchmark(steps=15, seed=7, domainpacks_root=DOMAINPACKS)
        assert a == b


class TestCommittedSnapshot:
    def test_committed_json_is_wellformed(self) -> None:
        path = Path("benchmarks/results/orchestration_uplift.json")
        data = json.loads(path.read_text())
        assert data["benchmark"] == "orchestration-uplift"
        assert len(data["packs"]) == len(bench.BEACHHEAD_PACKS)
        for record in data["packs"]:
            _assert_bounded(record["open_loop"])
            _assert_bounded(record["closed_loop"])


class TestMain:
    def test_main_writes_json(self, tmp_path: Path) -> None:
        out = tmp_path / "uplift.json"
        code = bench.main(["--steps", "12", "--seed", "7", "--json-out", str(out)])
        assert code == 0
        report = json.loads(out.read_text())
        assert report["steps"] == 12
        assert len(report["packs"]) == len(bench.BEACHHEAD_PACKS)
