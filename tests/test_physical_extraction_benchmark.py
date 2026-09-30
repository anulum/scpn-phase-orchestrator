# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical extraction diagnostic tests

"""Exercise the real physical diagnostic API and subprocess CLI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from typing import cast

import numpy as np
import pytest

from benchmarks.physical_extraction import benchmark_physical_extraction, main


@pytest.mark.parametrize("repeats", [True, 0, -1, 1.5, "2"])
def test_invalid_repeat_counts_refuse_before_measurement(repeats: object) -> None:
    """Non-positive or non-integral repeat counts produce no measurement report."""
    with pytest.raises(ValueError, match="positive integer"):
        benchmark_physical_extraction(repeats=cast(int, repeats))


def test_report_records_real_execution_and_analytic_outputs() -> None:
    """Every observed backend, workload and field agrees with the actual environment."""
    previous = sys.getprofile()
    report = benchmark_physical_extraction(repeats=2)
    assert sys.getprofile() is previous
    native = importlib.util.find_spec("spo_kernel") is not None
    assert report["kernel_present"] is native
    assert (report["native_artifact"] is not None) is native
    assert (
        report["classification"]
        == "non-isolated functional and local timing diagnostic"
    )
    assert report["repeats"] == 2
    assert len(report["cases"]) == 9
    for case in report["cases"]:
        assert case["observed_backend"] == ("native" if native else "python")
        assert case["native_calls"] == (["physical_extract"] if native else [])
        assert case["median_seconds"] > 0
        assert case["p95_seconds"] >= case["median_seconds"]
        assert np.all(np.isfinite(case["fields"]))
        assert 0 <= case["fields"][3] <= 1
        if "modulated" in case["name"]:
            scale = 1e290 if case["name"].startswith("large") else 1.0
            assert case["fields"][2] / scale == pytest.approx(1.0, abs=1e-12)
            assert case["fields"][3] == pytest.approx(1 - 0.6 / np.sqrt(2), abs=1e-12)
    assert len(report["source_sha256"]) == 4
    assert all(len(digest) == 64 for digest in report["source_sha256"].values())


def test_subprocess_cli_emits_one_finite_report() -> None:
    """The actual module command serializes a complete report from a fresh process."""
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.physical_extraction", "--repeats", "2"],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    report = json.loads(completed.stdout)
    assert report["format_version"] == 1
    assert len(report["cases"]) == 9
    assert report["kernel_present"] is (
        importlib.util.find_spec("spo_kernel") is not None
    )


def test_cli_refuses_invalid_counts(capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid CLI counts exit2 and report the concrete input error on stderr."""
    with pytest.raises(SystemExit) as error:
        main(["--repeats", "0"])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "positive integer" in output.err


def test_cli_entry_emits_serialized_public_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public command entry returns success and emits the documented report."""
    assert main(["--repeats", "1"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    report = json.loads(output.out)
    assert report["repeats"] == 1
    assert len(report["cases"]) == 9
