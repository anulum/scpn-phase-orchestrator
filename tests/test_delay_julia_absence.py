# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual Julia delay asset admission

"""Qualify real running Julia with absent, broken and valid source artifacts."""

from __future__ import annotations

import hashlib
import os
import runpy
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pytest
from coverage import Coverage, CoverageData


def _repository_omit() -> list[str]:
    """Return the measurement omit patterns declared in pyproject.toml."""
    document = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    patterns = document["tool"]["coverage"]["run"]["omit"]
    if not isinstance(patterns, list) or not all(
        isinstance(pattern, str) for pattern in patterns
    ):
        raise TypeError("tool.coverage.run.omit must be a list of strings")
    return list(patterns)


def _qualify(exported: Path, mode: str, source_hash: str) -> None:
    """Start actual Julia before measuring genuine asset-admission contracts."""
    sys.path.insert(0, str(exported / "src"))
    sys.path.insert(0, str(exported))
    assert find_spec("spo_kernel") is None, "This profile requires actual Rust absence"
    main = import_module("juliacall").Main
    evaluate = cast("Callable[[str], object]", main.seval)
    assert evaluate("1 + 1") == 2
    coverage_file = os.environ.get("COVERAGE_FILE")
    measurement = (
        Coverage(
            config_file=False,
            data_file=str(exported / "profile-coverage"),
            branch=True,
            source=[str(exported / "src/scpn_phase_orchestrator")],
            # config_file=False drops the repository's omit list; apply it so
            # unimported generated stubs are not recorded as unexecuted source.
            omit=_repository_omit(),
        )
        if coverage_file
        else None
    )
    if measurement is not None:
        measurement.start()
    try:
        _exercise(exported, mode, source_hash)
    finally:
        if measurement is not None and coverage_file is not None:
            measurement.stop()
            _save_profile_measurement(measurement, exported, mode, Path(coverage_file))


def _save_profile_measurement(
    measurement: Coverage, exported: Path, mode: str, database: Path
) -> None:
    """Bind measured child sources by exact bytes before retiring their export."""
    original = exported / "src/scpn_phase_orchestrator"
    destination = Path(
        os.environ.get(
            "SPO_DELAY_COVERAGE_SOURCE_ROOT",
            str(Path(__file__).resolve().parents[1] / "src/scpn_phase_orchestrator"),
        )
    )

    def bind_source(filename: str) -> str:
        """Map an executed source only after verifying identical retained bytes."""
        source = Path(filename)
        retained = destination / source.relative_to(original)
        assert source.read_bytes() == retained.read_bytes(), filename
        return str(retained)

    bound = CoverageData(basename=str(database.parent / f".coverage.delay-{mode}"))
    bound.update(measurement.get_data(), map_path=bind_source)
    bound.write()


def _exercise(exported: Path, mode: str, source_hash: str) -> None:
    """Exercise public selection and comparison CLI against the actual Julia asset."""
    from scpn_phase_orchestrator.upde import delay

    source = Path(delay.__file__).resolve()
    assert source.is_relative_to(exported)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    phases = np.array([0.4, 1.1])
    omega = np.ones(2)
    coupling = np.array([[0.0, 0.5], [0.2, 0.0]])
    alpha = np.full((2, 2), 0.01)
    engine = delay.DelayedEngine(2, dt=0.01, delay_steps=3)
    result = engine.run(phases, omega, coupling, 0.2, 0.3, alpha, n_steps=4)
    expected = [0.45155943106653923, 1.1289796554407023]
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-13)
    if mode == "present":
        assert delay.AVAILABLE_BACKENDS == ["julia", "python"]
        assert delay.ACTIVE_BACKEND == "julia"
        for _ in range(2):
            actual = engine.run(
                phases,
                omega,
                coupling,
                0.2,
                0.3,
                alpha,
                n_steps=4,
                backend="julia",
            )
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)
        endpoint = delay.DelayedEngine(1, dt=0.01).run(
            np.array([-np.nextafter(0.0, 1.0)]),
            np.zeros(1),
            np.zeros((1, 1)),
            n_steps=1,
            backend="julia",
        )
        np.testing.assert_array_equal(endpoint, [0.0])
        assert not np.any(np.signbit(endpoint))
    else:
        assert delay.AVAILABLE_BACKENDS == ["python"]
        assert delay.ACTIVE_BACKEND == "python"
        with pytest.raises(ImportError, match="delay backend 'julia' is unavailable"):
            engine.run(phases, omega, coupling, n_steps=4, backend="julia")
    np.testing.assert_array_equal(phases, [0.4, 1.1])

    from benchmarks.delay_benchmark import benchmark_delay_polyglot_parity_gate

    comparison = benchmark_delay_polyglot_parity_gate(
        n=2, n_steps=4, calls=1, require_backends=("python", "julia")
    )
    assert comparison["acceptance_passed"] == int(mode == "present")
    if mode in {"missing", "present"}:
        sys.argv = [
            "benchmarks.delay_benchmark",
            "--parity-gate",
            "--n",
            "2",
            "--n-steps",
            "4",
            "--calls",
            "1",
            "--require-backends",
            "python",
            "julia",
        ]
        with pytest.raises(SystemExit) as cli:
            runpy.run_module("benchmarks.delay_benchmark", run_name="__main__")
        assert cli.value.code == (0 if mode == "present" else 1)


def run_source_profile(tmp_path: Path, mode: str) -> None:
    """Qualify public selection in a source export with the requested actual asset.

    Parameters
    ----------
    tmp_path : Path
        Parent for the finite owned export; the child finishes before its removal.
    mode : str
        Missing, broken, missing-entry or present Julia side-file profile.
    """
    root = Path(__file__).resolve().parents[1]
    source = root / "src/scpn_phase_orchestrator"
    source_hash = hashlib.sha256((source / "upde/delay.py").read_bytes()).hexdigest()
    interpreter = os.environ.get("SPO_DELAY_JULIA_ONLY_PYTHON", sys.executable)
    with TemporaryDirectory(prefix="delay-julia-", dir=tmp_path) as temporary:
        exported = Path(temporary)
        shutil.copytree(
            source,
            exported / "src/scpn_phase_orchestrator",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (exported / "benchmarks").mkdir()
        shutil.copyfile(
            root / "benchmarks/delay_benchmark.py",
            exported / "benchmarks/delay_benchmark.py",
        )
        if mode != "missing":
            (exported / "julia").mkdir()
            destination = exported / "julia/delay.jl"
            if mode == "present":
                shutil.copyfile(root / "julia/delay.jl", destination)
            elif mode == "missing-entry":
                destination.write_text("module DelayJL\nconst value = 1\nend\n")
            else:
                destination.write_text("module DelayJL\nfunction broken(\nend\n")
        completed = subprocess.run(
            [
                interpreter,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                str(exported),
                mode,
                source_hash,
            ],
            env=dict(os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.native_runtime
@pytest.mark.parametrize("mode", ["broken", "missing-entry", "present"])
def test_public_julia_delay_asset_admission(tmp_path: Path, mode: str) -> None:
    """Only the actual numerical side-file admits Julia to public delay selection."""
    run_source_profile(tmp_path, mode)


if __name__ == "__main__":
    _qualify(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
