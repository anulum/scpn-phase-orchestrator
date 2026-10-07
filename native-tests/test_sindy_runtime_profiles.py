# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Installed SINDy profiles and actual CLI consumers

"""Qualify actual native and no-kernel installations through public computation.

Fresh processes observe original owners, consume independent physical CSV
samples, and preserve genuine CLI exits. No backend flags or import entries
are replaced. This file belongs to the installed-native test lane.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import venv
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from benchmarks.phase_sindy_benchmark import (
    Backend,
    DiagnosticReport,
    benchmark_phase_sindy,
)
from scpn_phase_orchestrator.binding.types import VALIDATION_TIER_EXTERNALLY_VALIDATED

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.native_runtime


def _run(
    python: Path, arguments: list[str], *, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    """Run a real interpreter without source-path overrides or shell expansion."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [str(python), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture(scope="module")
def python_only(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Prepare an actual installed package in a separate environment without the kernel.

    An explicitly supplied interpreter permits reuse of an already prepared
    real test environment. Every test still verifies its actual computation,
    source identity and absence rather than trusting the path's name.
    """
    supplied = os.environ.get("SPO_SINDY_PYTHON_ONLY")
    if supplied:
        python = Path(supplied).absolute()
        assert python.is_file()
        return python
    temporary = tmp_path_factory.mktemp("sindy-installed-python")
    profile = temporary / "venv"
    venv.EnvBuilder(with_pip=True).create(profile)
    python = profile / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    minor = f"py{sys.version_info.major}{sys.version_info.minor}"
    lock = (
        f"server-lock-windows-{minor}.txt"
        if os.name == "nt"
        else "server-lock-py311.txt"
        if minor == "py311"
        else "server-lock.txt"
    )
    install = _run(
        python,
        [
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--no-deps",
            "-r",
            str(ROOT / "requirements" / lock),
        ],
        timeout=240,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    source = temporary / "source"
    shutil.copytree(
        ROOT / "src",
        source / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    for filename in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "LICENSE-COMMERCIAL",
        "NOTICE",
    ):
        candidate = ROOT / filename
        if candidate.is_file():
            shutil.copy2(candidate, source / filename)
    wheels = temporary / "wheels"
    build = _run(
        Path(sys.executable),
        [
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheels),
            str(source),
        ],
        timeout=180,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    artifacts = list(wheels.glob("scpn_phase_orchestrator-*.whl"))
    assert len(artifacts) == 1
    installed = _run(python, ["-m", "pip", "install", "--no-deps", str(artifacts[0])])
    assert installed.returncode == 0, installed.stdout + installed.stderr
    return python


def test_installed_profiles_execute_original_owners_and_analytic_references(
    python_only: Path,
) -> None:
    """Different installations agree on independent directed, alias and rank oracles."""
    sine = math.sin(0.4)
    inverse_norm = 1.0 / (1.0 + sine * sine)
    expected = {
        "directed_euler": [[1.1, 0.2, 0.35], [1.8, -0.1, 0.13], [2.6, 0.07, -0.18]],
        "multiple_turn_alias": [[1.0]],
        "dependent_features": [
            [inverse_norm, sine * inverse_norm],
            [inverse_norm, -sine * inverse_norm],
        ],
        "empty_support": [[0.0]],
    }
    for python, backend in ((Path(sys.executable), "native"), (python_only, "python")):
        result = _run(
            python,
            [
                str(ROOT / "benchmarks/phase_sindy_benchmark.py"),
                "--repeats",
                "3",
                "--expect-backend",
                backend,
            ],
        )
        assert result.returncode == 0, result.stdout + result.stderr
        report = cast(DiagnosticReport, json.loads(result.stdout))
        assert report["backend"] == backend
        assert (
            report["estimator_artifact"]["sha256"]
            == hashlib.sha256(
                (ROOT / "src/scpn_phase_orchestrator/autotune/sindy.py").read_bytes()
            ).hexdigest()
        )
        assert len(report["cases"]) == len(expected)
        call = (
            "spo_kernel.spo_kernel.sindy_fit_rust"
            if backend == "native"
            else "scipy.linalg.lstsq"
        )
        for case in report["cases"]:
            np.testing.assert_allclose(
                case["coefficients"], expected[case["name"]], rtol=0.0, atol=2e-8
            )
            assert case["dtype"] == "float64"
            assert call in case["observed_calls"]
            assert len(case["durations_seconds"]) == 3
        artifact = report["native_artifact"]
        if backend == "native":
            assert artifact is not None
            binary = Path(artifact["path"])
            assert binary.is_file()
            assert hashlib.sha256(binary.read_bytes()).hexdigest() == artifact["sha256"]
        else:
            assert artifact is None


def _write_phase_csv(path: Path, *, irregular: bool = False) -> None:
    """Write independent directed Euler samples as the real operator CSV input."""
    phases = np.empty((160, 3), dtype=np.float64)
    phases[0] = [0.0, 0.7, 2.1]
    omega = [1.1, 1.8, 2.6]
    coupling = [[0.0, 0.2, 0.35], [-0.1, 0.0, 0.13], [0.07, -0.18, 0.0]]
    for sample in range(1, 160):
        previous = phases[sample - 1]
        for target in range(3):
            derivative = omega[target]
            for source in range(3):
                if source != target:
                    derivative += coupling[target][source] * math.sin(
                        float(previous[source] - previous[target])
                    )
            phases[sample, target] = (previous[target] + 0.02 * derivative) % math.tau
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "theta_0", "theta_1", "theta_2"])
        for index, row in enumerate(phases):
            time = index * 0.02 + (0.003 if irregular and index == 5 else 0.0)
            writer.writerow([time, *row])


def _trace(stderr: str) -> dict[str, object]:
    """Decode the observer's separate record without accepting CLI errors as success."""
    records = [
        line.removeprefix("SINDY_CLI_TRACE=")
        for line in stderr.splitlines()
        if line.startswith("SINDY_CLI_TRACE=")
    ]
    assert len(records) == 1
    value: object = json.loads(records[0])
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_real_csv_cli_preserves_signed_edges_and_observed_profile(
    python_only: Path,
    tmp_path: Path,
) -> None:
    """Installed CLI computation recovers signed directed edges for operator review."""
    csv_path = tmp_path / "directed_phases.csv"
    _write_phase_csv(csv_path)
    expected = {
        (1, 0): 0.2,
        (2, 0): 0.35,
        (0, 1): -0.1,
        (2, 1): 0.13,
        (0, 2): 0.07,
        (1, 2): -0.18,
    }
    source_sha = hashlib.sha256(
        (ROOT / "src/scpn_phase_orchestrator/autotune/sindy.py").read_bytes()
    ).hexdigest()
    for python, native in ((Path(sys.executable), True), (python_only, False)):
        result = _run(
            python,
            [
                str(ROOT / "native-tests/helpers/sindy_cli_probe.py"),
                "auto-bind",
                "time-series-csv",
                str(csv_path),
                "--project-name",
                "sindy_review",
                "--sindy-threshold",
                "0.005",
                "--json-out",
            ],
        )
        assert result.returncode == 0, result.stdout + result.stderr
        record = json.loads(result.stdout)
        phase = record["binding"]["provenance"]["discovery_evidence"]["phase_sindy"]
        assert phase["status"] == "fitted"
        assert phase["sample_count"] == 159
        assert phase["node_count"] == 3
        edges = {
            (
                int(edge["source"].split("_")[-1]),
                int(edge["target"].split("_")[-1]),
            ): edge["coefficient"]
            for edge in phase["coupling_edges"]
        }
        assert edges.keys() == expected.keys()
        for edge, coefficient in expected.items():
            assert edges[edge] == pytest.approx(coefficient, rel=0.0, abs=2e-8)
        assert (
            record["binding"]["provenance"]["discovery_evidence"][
                "phase_sindy_confidence"
            ]["tier"]
            != VALIDATION_TIER_EXTERNALLY_VALIDATED
        )
        trace = _trace(result.stderr)
        assert trace["kernel_present"] is native
        assert trace["estimator_sha256"] == source_sha
        required = (
            "spo_kernel.spo_kernel.sindy_fit_rust" if native else "scipy.linalg.lstsq"
        )
        assert required in cast(list[str], trace["observed_calls"])
        if not native:
            assert str(python_only.parent.parent) in cast(str, trace["estimator_path"])


def test_real_csv_cli_refuses_irregular_sampling_before_phase_regression(
    python_only: Path,
    tmp_path: Path,
) -> None:
    """Refuse irregular operator samples before calling native phase regression."""
    csv_path = tmp_path / "irregular_phases.csv"
    _write_phase_csv(csv_path, irregular=True)
    for python in (Path(sys.executable), python_only):
        result = _run(
            python,
            [
                str(ROOT / "native-tests/helpers/sindy_cli_probe.py"),
                "auto-bind",
                "time-series-csv",
                str(csv_path),
                "--project-name",
                "sindy_refusal",
                "--json-out",
            ],
        )
        assert result.returncode == 1
        assert "regular sampling interval" in result.stderr
        assert result.stdout == ""
        phase_calls = {
            "spo_kernel.spo_kernel.sindy_fit_rust",
            "scipy.linalg.lstsq",
        }
        assert phase_calls.isdisjoint(
            cast(list[str], _trace(result.stderr)["observed_calls"])
        )


@pytest.mark.parametrize("repeats", ["0", "-1"])
def test_benchmark_cli_refuses_nonpositive_repeats(repeats: str) -> None:
    """Reject invalid operator controls with the actual CLI usage status."""
    result = _run(
        Path(sys.executable),
        [str(ROOT / "benchmarks/phase_sindy_benchmark.py"), "--repeats", repeats],
    )
    assert result.returncode == 2
    assert "repeats must be a positive integer" in result.stderr
    assert result.stdout == ""


def test_benchmark_cli_refuses_actual_installation_mismatch(python_only: Path) -> None:
    """A real interpreter cannot report successful proof for the other installation."""
    for python, wrong in ((Path(sys.executable), "python"), (python_only, "native")):
        result = _run(
            python,
            [
                str(ROOT / "benchmarks/phase_sindy_benchmark.py"),
                "--repeats",
                "1",
                "--expect-backend",
                wrong,
            ],
        )
        assert result.returncode == 1
        assert f"expected {wrong}" in result.stderr
        assert result.stdout == ""


@pytest.mark.parametrize(
    ("repeats", "backend", "match"),
    [(True, "auto", "repeats"), (1.5, "auto", "repeats"), (1, "missing", "backend")],
)
def test_benchmark_public_api_refuses_invalid_controls(
    repeats: object, backend: str, match: str
) -> None:
    """The public diagnostic API refuses boolean counts and unknown backend names."""
    with pytest.raises(ValueError, match=match):
        benchmark_phase_sindy(
            repeats=cast(int, repeats), expect_backend=cast(Backend, backend)
        )


@pytest.mark.parametrize(
    ("provider", "attribute", "match"),
    [
        ("scipy.linalg", "lstsq", "no Python call identity"),
        ("spo_kernel", "sindy_fit_rust", "No actual.*invocation"),
    ],
)
def test_benchmark_refuses_delegated_public_abi_identity(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    attribute: str,
    match: str,
) -> None:
    """Reject unsupported observation identities without fabricating numerical results.

    This rejection-only ABI control exposes a delegating partial of the genuine
    owner. Availability and cached numerical owners stay unchanged; the wrapper
    returns real solver results if invoked. It never qualifies native success.
    """
    module = importlib.import_module(provider)
    monkeypatch.setattr(module, attribute, partial(getattr(module, attribute)))
    with pytest.raises(RuntimeError, match=match):
        benchmark_phase_sindy(repeats=1, expect_backend="native")
