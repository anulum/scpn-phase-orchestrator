# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Original chimera benchmark and CLI contracts

"""Qualify real public comparison data and refuse invalid or missing-owner evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks import chimera_benchmark as legacy
from benchmarks import chimera_comparison as comparison
from benchmarks.chimera_benchmark import (
    bench_at,
    benchmark_chimera_polyglot_parity_gate,
    main,
)
from benchmarks.chimera_comparison import benchmark_comparison, require_owners
from scpn_phase_orchestrator.monitor import chimera


@pytest.mark.native_runtime
def test_real_all_owner_comparison_retains_raw_samples_and_models() -> None:
    """Measure every original owner plus genuine completed JAX calls on one graph."""
    result = benchmark_comparison([3], calls=20, density=1.0, seed=17)
    rows = cast("list[dict[str, object]]", result["results"])
    assert [row["owner"] for row in rows] == [
        "rust",
        "mojo",
        "julia",
        "go",
        "python",
        "jax_nn",
    ]
    for row in rows:
        samples = cast("list[float]", row["sample_seconds"])
        assert row["sample_count"] == 20 and len(samples) == 20
        assert all(value > 0.0 for value in samples)
        assert float(cast(float, row["max_abs_error"])) <= float(
            cast(float, row["tolerance"])
        )
        assert float(cast(float, row["mean_seconds"])) == pytest.approx(
            float(np.mean(samples))
        )
        assert (
            float(cast(float, row["min_seconds"]))
            <= float(cast(float, row["p50_seconds"]))
            <= float(cast(float, row["max_seconds"]))
        )
    assert rows[-1]["operation"] == "nn.local_order_parameter_jit"
    assert "nonzero signed/self" in str(rows[-1]["model"])
    hashes = cast("dict[str, str]", result["source_hashes"])
    assert all(len(value) == 64 for value in hashes.values())
    assert {"rust_binary", "go_binary", "mojo_binary", "julia_loaded_source"} <= set(
        hashes
    )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("sizes", []),
        ("sizes", [2]),
        ("sizes", [True]),
        ("sizes", [3.0]),
        ("calls", 19),
        ("calls", True),
        ("calls", 20.0),
        ("seed", -1),
        ("seed", False),
        ("seed", 1.0),
        ("density", True),
        ("density", "0.3"),
        ("density", -0.1),
        ("density", 1.1),
        ("density", float("nan")),
    ],
)
def test_comparison_control_refusal_precedes_measurement(
    field: str, bad: object
) -> None:
    """Reject original invalid controls without normalizing aliases into success."""
    options: dict[str, object] = {"sizes": [3], "calls": 20, "seed": 17, "density": 0.3}
    options[field] = bad
    with pytest.raises(ValueError):
        benchmark_comparison(
            cast("list[int]", options["sizes"]),
            calls=cast(int, options["calls"]),
            seed=cast(int, options["seed"]),
            density=cast(float, options["density"]),
        )


def test_unsupported_benchmark_owner_is_refused() -> None:
    """An unsupported label cannot be reported as an executed implementation."""
    with pytest.raises(ValueError, match="Unknown chimera benchmark owner"):
        require_owners(["not-an-owner"])
    require_owners(["python"])


@pytest.mark.parametrize(
    "field,bad",
    [
        ("n", 2),
        ("n", True),
        ("calls", 0),
        ("calls", True),
        ("density", False),
        ("density", float("inf")),
    ],
)
def test_legacy_timing_controls_refuse_original_invalid_values(
    field: str, bad: object
) -> None:
    """The original quick timing API keeps strict measurement metadata admission."""
    options: dict[str, object] = {"n": 3, "calls": 1, "density": 0.3}
    options[field] = bad
    with pytest.raises(ValueError):
        bench_at(
            cast(int, options["n"]),
            cast(float, options["density"]),
            cast(int, options["calls"]),
        )


@pytest.mark.native_runtime
def test_legacy_gate_preserves_current_consumer_schema() -> None:
    """Exercise original native slots against the independent scalar equation."""
    result = benchmark_chimera_polyglot_parity_gate(n=3, density=1.0, calls=1, seed=7)
    assert result["backend_count"] == 5 and result["available_backend_count"] == 5
    assert result["acceptance_passed"] == 1
    rows = cast(
        "list[dict[str, object]]", json.loads(str(result["backend_records_json"]))
    )
    assert all(
        row["parity_passed"] and row["reference_contracts_passed"] for row in rows
    )
    assert all(len(cast("list[float]", row["sample_seconds"])) == 1 for row in rows)


@pytest.mark.native_runtime
@pytest.mark.parametrize("mode", ["missing_required", "available_parity", "comparison"])
def test_real_installed_missing_owner_cli_refuses_without_false_result(
    tmp_path: Path, mode: str
) -> None:
    """Report actual installed absence and refuse explicitly required missing owners."""
    python = os.environ.get("SPO_CHIMERA_ABSENT_PYTHON")
    assert python is not None
    repo = Path(__file__).resolve().parents[1]
    script = """
import os, sys, runpy
sys.path.insert(0, sys.argv.pop(1))
from scpn_phase_orchestrator.monitor import chimera as c
assert c.AVAILABLE_BACKENDS == ['python']
configuration = os.environ.get('COVERAGE_PROCESS_START')
measurement = None
if configuration:
    from coverage import Coverage
    measurement = Coverage(config_file=configuration, data_suffix=True)
    measurement.start()
try:
    runpy.run_module('benchmarks.chimera_benchmark', run_name='__main__')
finally:
    if measurement:
        measurement.stop()
        measurement.save()
"""
    output = tmp_path / "actual.json"
    arguments = [
        python,
        "-I",
        "-B",
        "-c",
        script,
        str(repo),
        "--sizes",
        "3",
        "--calls",
        "20",
        "--output",
        str(output),
    ]
    if mode == "comparison":
        arguments.append("--comparison")
    else:
        arguments.append("--parity-gate")
        if mode == "missing_required":
            arguments.extend(["--require-backends", "rust"])
    result = subprocess.run(
        arguments, text=True, capture_output=True, check=False, timeout=60
    )
    if mode == "available_parity":
        assert result.returncode == 0, result.stderr
        data = json.loads(output.read_text())
        assert data["available_backend_count"] == 1
        assert data["unavailable_backend_count"] == 4
        rows = json.loads(data["backend_records_json"])
        assert all(
            row["status"] == "unavailable"
            and not row["parity_passed"]
            and row["unavailable_reason"]
            for row in rows
            if row["backend"] != "python"
        )
    else:
        assert result.returncode == 2
        assert "Required chimera benchmark owners unavailable:" in result.stderr
        assert not output.exists()


def test_cli_invalid_controls_fail_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invalid CLI metadata is a parser refusal rather than partial benchmark output."""
    output = tmp_path / "invalid.json"
    monkeypatch.setattr(
        sys, "argv", ["chimera_benchmark", "--calls", "0", "--output", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not output.exists()


@pytest.mark.native_runtime
@pytest.mark.parametrize("mode", ["quick", "--parity-gate", "--comparison"])
@pytest.mark.parametrize("save", [False, True])
def test_original_cli_modes_emit_actual_owner_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    save: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise every original CLI route and its saved consumer schema."""
    output = tmp_path / "actual.json"
    arguments = [
        "chimera_benchmark",
        "--sizes",
        "3",
        "--calls",
        "20",
        "--require-backends",
        "rust",
        "go",
        "julia",
        "mojo",
        "python",
    ]
    if save:
        arguments.extend(["--output", str(output)])
    if mode != "quick":
        arguments.append(mode)
    monkeypatch.setattr(sys, "argv", arguments)
    assert main() == 0
    stdout = capsys.readouterr().out
    assert output.exists() == save
    if mode == "quick" and not save:
        assert "Active:" in stdout and "rust_ms" in stdout
        return
    data = json.loads(output.read_text() if save else stdout)
    if mode == "--comparison":
        assert len(data["results"]) == 6
        assert all(row["sample_count"] == 20 for row in data["results"])
    elif mode == "--parity-gate":
        assert data["acceptance_passed"] == 1
        assert data["available_backend_count"] == 5
    else:
        assert data["results"][0]["calls"] == 20
        assert len(data["results"][0]["available"]) == 5


@pytest.mark.native_runtime
@pytest.mark.parametrize(
    "fault_at,expected", [(1, "parity failed for rust"), (22, "output drift for rust")]
)
def test_comparison_refuses_corrupted_actual_cpu_output(
    monkeypatch: pytest.MonkeyPatch, fault_at: int, expected: str
) -> None:
    """Corrupt a completed real Rust call solely to qualify comparison refusal."""
    original = chimera.local_order_parameter
    count = 0

    def corrupted(
        phases: NDArray[np.float64],
        coupling: NDArray[np.float64],
        *,
        backend: str | None = None,
    ) -> NDArray[np.float64]:
        """Invoke the original public runtime before injecting the negative fault."""
        nonlocal count
        actual = original(phases, coupling, backend=backend)
        if backend == "rust":
            count += 1
            if count == fault_at:
                return actual * 0.5
        return actual

    monkeypatch.setattr(chimera, "local_order_parameter", corrupted)
    with pytest.raises(RuntimeError, match=expected):
        benchmark_comparison([3], density=1.0)
    assert count == fault_at


@pytest.mark.native_runtime
@pytest.mark.parametrize(
    "fault_at,expected",
    [(1, "scalar parity failed"), (22, "JAX measured output drift")],
)
def test_comparison_refuses_corrupted_actual_jax_output(
    monkeypatch: pytest.MonkeyPatch, fault_at: int, expected: str
) -> None:
    """Corrupt a genuine JIT result solely as a negative comparison control."""
    compile_original = jax.jit
    count = 0

    def compile_with_fault(
        function: Callable[[jax.Array, jax.Array], jax.Array],
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        """Keep genuine JAX compilation and inject only a returned-vector fault."""
        compiled = cast(
            "Callable[[jax.Array, jax.Array], jax.Array]",
            compile_original(function),
        )

        def operation(phases: jax.Array, coupling: jax.Array) -> jax.Array:
            """Complete the actual original JIT operation before the negative fault."""
            nonlocal count
            actual = compiled(phases, coupling).block_until_ready()
            count += 1
            return actual * 0.5 if count == fault_at else actual

        return operation

    monkeypatch.setattr(jax, "jit", compile_with_fault)
    with pytest.raises(RuntimeError, match=expected):
        benchmark_comparison([3], density=1.0)
    assert count == fault_at


@pytest.mark.native_runtime
def test_comparison_refuses_metadata_drift_after_actual_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject mismatched evidence metadata only after all original owners execute."""
    hash_original = comparison._hashes
    count = 0

    def inconsistent_metadata() -> dict[str, str]:
        """Read actual source/artifact hashes and corrupt one final metadata field."""
        nonlocal count
        observed = hash_original()
        count += 1
        if count == 2:
            observed["go_binary"] = "deliberate negative custody control"
        return observed

    monkeypatch.setattr(comparison, "_hashes", inconsistent_metadata)
    with pytest.raises(RuntimeError, match="changed during measurement"):
        benchmark_comparison([3], density=1.0)
    assert count == 2


@pytest.mark.native_runtime
def test_parity_cli_failed_numerics_emit_failure_and_exit_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A corrupted real owner cannot produce a successful parity-gate exit."""
    original = chimera.local_order_parameter
    calls = 0

    def corrupted(
        phases: NDArray[np.float64],
        coupling: NDArray[np.float64],
        *,
        backend: str | None = None,
    ) -> NDArray[np.float64]:
        """Execute the original API before applying a deliberate negative fault."""
        nonlocal calls
        actual = original(phases, coupling, backend=backend)
        if backend == "rust":
            calls += 1
            return actual * 0.5
        return actual

    output = tmp_path / "failed.json"
    monkeypatch.setattr(legacy, "local_order_parameter", corrupted)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chimera_benchmark",
            "--parity-gate",
            "--sizes",
            "3",
            "--calls",
            "1",
            "--require-backends",
            "rust",
            "--output",
            str(output),
        ],
    )
    assert main() == 1
    record = json.loads(output.read_text())
    assert record["acceptance_passed"] == 0 and calls > 0
    rows = json.loads(record["backend_records_json"])
    rust = next(row for row in rows if row["backend"] == "rust")
    assert rust["status"] == "available" and not rust["parity_passed"]


@pytest.mark.native_runtime
@pytest.mark.parametrize("mode", ["--comparison", "--parity-gate"])
def test_original_native_cli_emits_complete_json_through_small_pipe(
    tmp_path: Path, mode: str
) -> None:
    """Exercise actual native owners under real output backpressure."""
    import fcntl

    output = tmp_path / "pipe.json"
    command = [
        sys.executable,
        "-B",
        "-m",
        "benchmarks.chimera_benchmark",
        mode,
        "--sizes",
        "3",
        "--calls",
        "20",
        "--require-backends",
        "rust",
        "go",
        "julia",
        "mojo",
        "python",
        "--output",
        str(output),
    ]
    configuration = os.environ.get("COVERAGE_PROCESS_START")
    if configuration:
        command[2:2] = ["-m", "coverage", "run", "--rcfile=" + configuration]
    with subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stdout is not None
        fcntl.fcntl(child.stdout.fileno(), fcntl.F_SETPIPE_SZ, 4096)
        try:
            stdout, stderr = child.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            raise
        assert child.returncode == 0, stderr
    assert len(stdout) > 4096
    record = json.loads(stdout)
    assert record == json.loads(output.read_text())
    if mode == "--comparison":
        assert len(record["results"]) == 6
        assert all(row["sample_count"] == 20 for row in record["results"])
    else:
        assert record["acceptance_passed"] == 1
        assert record["available_backend_count"] == 5
