# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual dimension runtime asset profiles

"""Exercise public selection with actual Julia and isolated native assets."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pytest


def _qualify(
    exported: Path, mode: str, source_hash: str, benchmark: bool = False
) -> None:
    """Verify genuine Rust absence before starting the actual Julia interpreter."""
    sys.path.insert(0, str(exported / "src"))
    assert find_spec("spo_kernel") is None, "Profile requires actual Rust absence"
    main = import_module("juliacall").Main
    evaluate = cast("Callable[[str], object]", main.seval)
    assert evaluate("1 + 1") == 2

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests._dimension_profile_coverage import measure_profile

    with measure_profile(exported, f"julia-{mode}"):
        _exercise(exported, mode, source_hash, benchmark)


def _exercise(exported: Path, mode: str, source_hash: str, benchmark: bool) -> None:
    """Exercise selected public owners after actual Julia initialization."""
    from scpn_phase_orchestrator.monitor import dimension

    source = Path(dimension.__file__).resolve()
    assert source.is_relative_to(exported)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    selected = "julia" if mode == "present" else "python"
    if mode == "unusable-mojo":
        selected = "mojo"
    assert selected == dimension.ACTIVE_BACKEND
    assert (
        [selected, "python"] if selected != "python" else ["python"]
    ) == dimension.AVAILABLE_BACKENDS

    if mode == "unusable-mojo":
        executable = exported / "mojo/dimension_mojo"
        executable.chmod(0o600)

    trajectory = np.array([[0.0], [1.0], [3.0]])
    radii = np.array([1.0, 2.0, 4.0])
    spectrum = np.array([1.0, -2.0])
    for _ in range(2):
        np.testing.assert_allclose(
            dimension.correlation_integral(trajectory, radii, max_pairs=10),
            [0.0, 1.0 / 3.0, 1.0],
            rtol=0.0,
            atol=1e-15,
        )
        assert dimension.kaplan_yorke_dimension(spectrum) == 1.5

    if mode == "present":
        np.testing.assert_allclose(
            dimension.correlation_integral(trajectory, radii, backend="julia"),
            [0.0, 1.0 / 3.0, 1.0],
            rtol=0.0,
            atol=1e-15,
        )
        assert dimension.kaplan_yorke_dimension(spectrum, backend="julia") == 1.5
    else:
        owner = "mojo" if mode == "unusable-mojo" else "julia"
        with pytest.raises(RuntimeError, match=f"dimension backend '{owner}'"):
            dimension.correlation_integral(trajectory, radii, backend=owner)
        with pytest.raises(RuntimeError, match=f"dimension backend '{owner}'"):
            dimension.kaplan_yorke_dimension(spectrum, backend=owner)

    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    circle = np.column_stack([np.sin(angles), np.cos(angles)])
    actual = dimension.correlation_dimension(circle, n_epsilons=12, max_pairs=1000)
    expected = dimension.correlation_dimension(
        circle, n_epsilons=12, max_pairs=1000, backend="python"
    )
    assert pytest.approx(expected.D2, abs=1e-14) == actual.D2
    for value, reference in (
        (actual.epsilons, expected.epsilons),
        (actual.C_eps, expected.C_eps),
        (actual.slope, expected.slope),
    ):
        np.testing.assert_allclose(value, reference, rtol=0.0, atol=1e-14)
    if benchmark:
        # Fresh native CLI processes see only actual assets in this export.
        # No finder/module/function is replaced to manufacture absence.
        owners = ["julia", "python"] if mode == "present" else ["python"]
        for selection, expected_exit in (([], 1), (["--require-backends", *owners], 0)):
            destination = exported / f"benchmark-result-{expected_exit}.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--benchmark-cli",
                    str(exported),
                    f"julia-{mode}-cli-{'all' if not selection else 'requested'}",
                    "--parity-gate",
                    "--T-list",
                    "8",
                    "--n-k",
                    "4",
                    "--calls",
                    "1",
                    "--repeats",
                    "1",
                    "--output",
                    str(destination),
                    *selection,
                ],
                env=_profile_environment(),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            assert completed.returncode == expected_exit, (
                completed.stdout + completed.stderr
            )
            result = json.loads(completed.stdout)
            assert destination.read_text().strip() == completed.stdout.strip()
            assert result["comparison_complete"] == 0
            assert result["acceptance_passed"] == 1 - expected_exit
            assert result["required_backends_present"] == 1 - expected_exit
            records = json.loads(result["backend_records_json"])
            for record in records:
                if record["backend"] in owners:
                    assert record["status"] == "available"
                else:
                    assert record["status"] == "unavailable"
                    assert record["ms_per_call"] is None
                    assert record["correlation_integral_sha256"] is None
                    assert record["kaplan_yorke_sha256"] is None
            if expected_exit:
                assert result["benchmark_sha256"] is None
            else:
                assert len(result["benchmark_sha256"]) == 64
        invalid_destination = exported / "invalid-arguments.json"
        invalid = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--benchmark-cli",
                str(exported),
                f"julia-{mode}-cli-invalid",
                "--parity-gate",
                "--T-list",
                "8",
                "--calls",
                "0",
                "--output",
                str(invalid_destination),
                "--require-backends",
                "python",
            ],
            env=_profile_environment(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert invalid.returncode == 2, invalid.stdout + invalid.stderr
        assert "calls must be at least 1" in invalid.stderr
        assert not invalid.stdout.strip()
        assert not invalid_destination.exists()
    print(
        json.dumps(
            {
                "mode": mode,
                "selected_before_failure": selected,
                "available": dimension.AVAILABLE_BACKENDS,
                "source_sha256": source_hash,
                "rust_spec_absent": True,
                "actual_julia": True,
            }
        )
    )


def _profile_environment() -> dict[str, str]:
    """Use the retained parent package as the exact child source destination."""
    from tests._dimension_profile_coverage import profile_environment

    return profile_environment()


def _benchmark_cli(exported: Path, label: str, arguments: list[str]) -> None:
    """Run the actual benchmark CLI with Julia initialized before tracing."""
    sys.path[:0] = [
        str(exported),
        str(exported / "src"),
        str(Path(__file__).resolve().parents[1]),
    ]
    from tests._dimension_profile_coverage import measure_profile

    main = import_module("juliacall").Main
    evaluate = cast("Callable[[str], object]", main.seval)
    assert evaluate("1 + 1") == 2
    with measure_profile(exported, label):
        sys.argv = ["benchmarks.dimension_benchmark", *arguments]
        runpy.run_module("benchmarks.dimension_benchmark", run_name="__main__")


def run_source_profile(tmp_path: Path, mode: str, *, benchmark: bool = False) -> None:
    """Run a finite, isolated profile without changing process-global loaders.

    Missing/broken/non-exporting assets are negative admission fixtures. The
    present asset is the actual Julia solver. The unusable-Mojo case revokes
    execution on an owned copy of the real binary after successful admission.

    Parameters
    ----------
    tmp_path : Path
        Owned parent for the finite export, removed after the child has exited.
    mode : str
        Actual missing, broken, missing-entry, present or unusable-Mojo profile.
    benchmark : bool
        Also exercise fresh CLI processes, JSON persistence and argument refusal.
    """
    if mode not in {"missing", "broken", "missing-entry", "present", "unusable-mojo"}:
        raise ValueError(f"Unknown dimension runtime profile: {mode}")
    root = Path(__file__).resolve().parents[1]
    source = root / "src/scpn_phase_orchestrator"
    source_hash = hashlib.sha256(
        (source / "monitor/dimension.py").read_bytes()
    ).hexdigest()
    interpreter = os.environ.get("SPO_DIMENSION_JULIA_ONLY_PYTHON", sys.executable)
    with TemporaryDirectory(prefix="dimension-profile-", dir=tmp_path) as temporary:
        exported = Path(temporary)
        shutil.copytree(
            source,
            exported / "src/scpn_phase_orchestrator",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        if benchmark:
            (exported / "benchmarks").mkdir()
            for name in ("dimension_benchmark.py", "_dimension_reference.py"):
                shutil.copyfile(
                    root / "benchmarks" / name, exported / "benchmarks" / name
                )
        if mode in {"present", "broken", "missing-entry"}:
            (exported / "julia").mkdir()
            asset = exported / "julia/dimension.jl"
            if mode == "present":
                shutil.copyfile(root / "julia/dimension.jl", asset)
            elif mode == "missing-entry":
                asset.write_text("module DimensionJL\nconst value = 1\nend\n")
            else:
                asset.write_text("module DimensionJL\nfunction broken(\nend\n")
        if mode == "unusable-mojo":
            (exported / "mojo").mkdir()
            binary = Path(
                os.environ.get(
                    "SPO_DIMENSION_MOJO_EXE", str(root / "mojo/dimension_mojo")
                )
            )
            shutil.copy2(binary, exported / "mojo/dimension_mojo")
        completed = subprocess.run(
            [
                interpreter,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                str(exported),
                mode,
                source_hash,
                "benchmark" if benchmark else "numerical",
            ],
            env=_profile_environment(),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        receipt = json.loads(completed.stdout.splitlines()[-1])
        assert receipt["mode"] == mode
        assert receipt["source_sha256"] == source_hash


if __name__ == "__main__":
    if sys.argv[1] == "--benchmark-cli":
        _benchmark_cli(Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
    else:
        _qualify(
            Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "benchmark"
        )
