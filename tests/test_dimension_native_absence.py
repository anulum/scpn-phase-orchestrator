# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual Rust and Go dimension asset admission

"""Exercise native packaging/ELF faults without replacing Python dispatchers."""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.native_runtime
MODES = (
    "absent",
    "rust-unusable",
    "rust-missing-entry",
    "rust-present",
    "go-unusable",
    "go-missing-entry",
    "go-present",
    "rust-broken-go-present",
)


def _exercise(exported: Path, mode: str, hashes: dict[str, str]) -> None:
    """Check actual artifact identities and public numerical selection/refusal."""
    rust_spec = find_spec("spo_kernel")
    if mode.startswith("rust-"):
        assert rust_spec is not None
        if mode == "rust-missing-entry":
            # The genuine extension is present and executable, but its native
            # package has lost the actual __init__.py re-export facade.
            package = import_module("spo_kernel")
            assert not hasattr(package, "correlation_integral_rust")
            extension = import_module("spo_kernel.spo_kernel")
            assert inspect.isbuiltin(extension.correlation_integral_rust)
            filename = extension.__file__
            assert isinstance(filename, str)
            assert Path(filename).resolve().is_relative_to(exported)
        elif mode == "rust-present":
            extension = import_module("spo_kernel")
            assert inspect.isbuiltin(extension.correlation_integral_rust)
            filename = extension.__file__
            assert isinstance(filename, str)
            assert Path(filename).resolve().is_relative_to(exported)
        else:
            with pytest.raises(ImportError):
                import_module("spo_kernel")
    else:
        assert rust_spec is None
    if mode == "go-missing-entry":
        loaded = ctypes.CDLL(str(exported / "go/libdimension.so"))
        assert not hasattr(loaded, "CorrelationIntegral")
        assert not hasattr(loaded, "KaplanYorkeDimension")
    elif mode == "go-unusable":
        with pytest.raises(OSError):
            ctypes.CDLL(str(exported / "go/libdimension.so"))

    from scpn_phase_orchestrator.monitor import dimension

    assert Path(dimension.__file__).resolve().is_relative_to(exported)
    for relative, digest in hashes.items():
        assert (
            hashlib.sha256(
                (exported / "src/scpn_phase_orchestrator" / relative).read_bytes()
            ).hexdigest()
            == digest
        )
    selected = "python"
    if mode == "rust-present":
        selected = "rust"
    elif mode in {"go-present", "rust-broken-go-present"}:
        selected = "go"
    assert selected == dimension.ACTIVE_BACKEND
    assert (
        [selected, "python"] if selected != "python" else ["python"]
    ) == dimension.AVAILABLE_BACKENDS
    trajectory = np.array([[0.0], [1.0], [3.0]])
    radii = np.array([1.0, 2.0, 4.0])
    spectrum = np.array([1.0, -2.0])
    for owner in ("auto", selected):
        np.testing.assert_allclose(
            dimension.correlation_integral(trajectory, radii, backend=owner),
            [0.0, 1.0 / 3.0, 1.0],
            rtol=0.0,
            atol=1e-15,
        )
        assert dimension.kaplan_yorke_dimension(spectrum, backend=owner) == 1.5
    for owner in ("rust", "go"):
        if owner == selected:
            continue
        with pytest.raises(RuntimeError, match=f"dimension backend '{owner}'"):
            dimension.correlation_integral(trajectory, radii, backend=owner)
        with pytest.raises(RuntimeError, match=f"dimension backend '{owner}'"):
            dimension.kaplan_yorke_dimension(spectrum, backend=owner)
        with pytest.raises(RuntimeError, match=f"dimension backend '{owner}'"):
            dimension.correlation_dimension(trajectory, backend=owner)
    actual = dimension.correlation_dimension(trajectory, n_epsilons=8, backend=selected)
    np.testing.assert_allclose(
        actual.C_eps,
        [
            sum(distance < radius for distance in (1.0, 2.0, 3.0)) / 3.0
            for radius in actual.epsilons
        ],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(trajectory, [[0.0], [1.0], [3.0]])
    np.testing.assert_array_equal(radii, [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(spectrum, [1.0, -2.0])

    from benchmarks.dimension_benchmark import benchmark_dimension_polyglot_parity_gate

    out = benchmark_dimension_polyglot_parity_gate(
        t=8,
        d=2,
        n_k=4,
        repeats=1,
        require_backends=(selected,),
    )
    assert out["acceptance_passed"] == 1
    assert out["comparison_complete"] == 0
    records = json.loads(str(out["backend_records_json"]))
    for record in records:
        if record["backend"] not in {selected, "python"}:
            assert record["status"] == "unavailable"
            assert record["ms_per_call"] is None
            assert record["correlation_integral_sha256"] is None
            assert record["kaplan_yorke_sha256"] is None
    print(json.dumps({"mode": mode, "selected": selected, "source_hashes": hashes}))


def _qualify(exported: Path, mode: str, hashes: dict[str, str]) -> None:
    """Initialize actual Julia before measuring native admission source branches."""
    sys.path[:0] = [str(exported), str(exported / "src"), str(ROOT)]
    from tests._dimension_profile_coverage import measure_profile

    main = import_module("juliacall").Main
    evaluate_raw = main.seval
    assert callable(evaluate_raw)
    evaluate: Callable[[str], object] = evaluate_raw
    assert evaluate("1 + 1") == 2
    with measure_profile(exported, f"native-{mode}"):
        _exercise(exported, mode, hashes)


def _export(root: Path, mode: str) -> dict[str, str]:
    """Copy current sources and requested actual artifacts into an owned profile."""
    package = ROOT / "src/scpn_phase_orchestrator"
    shutil.copytree(
        package,
        root / "src/scpn_phase_orchestrator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (root / "benchmarks").mkdir()
    for name in ("dimension_benchmark.py", "_dimension_reference.py"):
        shutil.copyfile(ROOT / "benchmarks" / name, root / "benchmarks" / name)
    if mode.startswith("rust-"):
        kernel_file = import_module("spo_kernel").__file__
        assert isinstance(kernel_file, str)
        kernel = Path(kernel_file).resolve().parent
        shutil.copytree(
            kernel,
            root / "spo_kernel",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        binary = next((root / "spo_kernel").glob("*.so"))
        if mode in {"rust-unusable", "rust-broken-go-present"}:
            binary.write_bytes(b"deliberately unusable native ELF artifact\n")
        elif mode == "rust-missing-entry":
            (root / "spo_kernel/__init__.py").unlink()
    if mode.startswith("go-") or mode == "rust-broken-go-present":
        (root / "go").mkdir()
        if mode == "go-unusable":
            (root / "go/libdimension.so").write_bytes(
                b"deliberately unusable native ELF artifact\n"
            )
        else:
            name = (
                "SPO_DIMENSION_MISSING_EXPORT_LIB"
                if mode == "go-missing-entry"
                else "SPO_DIMENSION_GO_LIB"
            )
            fallback = (
                ROOT
                / "go"
                / ("libhodge.so" if mode == "go-missing-entry" else "libdimension.so")
            )
            asset = Path(os.environ.get(name, str(fallback)))
            shutil.copyfile(asset, root / "go/libdimension.so")
    relative = ["monitor/dimension.py", "monitor/_dimension_numerics.py"]
    relative += [
        f"experimental/accelerators/monitor/_dimension_{name}.py"
        for name in ("validation", "go", "julia", "mojo")
    ]
    return {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in relative
    }


def _run(
    exported: Path, mode: str, hashes: dict[str, str], environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a genuine isolated interpreter; no ambient Rust installation may leak in."""
    interpreter = os.environ.get("SPO_DIMENSION_JULIA_ONLY_PYTHON", sys.executable)
    return subprocess.run(
        [
            interpreter,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            str(exported),
            mode,
            json.dumps(hashes),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.parametrize("mode", MODES)
def test_actual_native_asset_selection_and_refusal(tmp_path: Path, mode: str) -> None:
    """Admit real numerical owners and refuse actual absent/bad/package-missing assets.

    Parameters
    ----------
    tmp_path : Path
        Owned export parent, removed only after the child has completed.
    mode : str
        Actual native artifact profile; numerical callbacks are never replaced.
    """
    from tests._dimension_profile_coverage import profile_environment

    with TemporaryDirectory(prefix=f"dimension-{mode}-", dir=tmp_path) as directory:
        exported = Path(directory)
        hashes = _export(exported, mode)
        completed = _run(exported, mode, hashes, profile_environment())
        assert completed.returncode == 0, completed.stdout + completed.stderr
        receipt = json.loads(completed.stdout.splitlines()[-1])
        assert receipt["mode"] == mode
        assert receipt["source_hashes"] == hashes


def test_child_coverage_refuses_retained_source_drift(tmp_path: Path) -> None:
    """A real executed source mismatch prevents publication of child coverage."""
    from tests._dimension_profile_coverage import profile_environment

    with TemporaryDirectory(
        prefix="dimension-source-drift-", dir=tmp_path
    ) as directory:
        exported = Path(directory)
        hashes = _export(exported, "absent")
        retained = exported / "retained-package"
        shutil.copytree(exported / "src/scpn_phase_orchestrator", retained)
        path = retained / "monitor/dimension.py"
        path.write_bytes(
            path.read_bytes() + b"\n# deliberate retained-source identity fault\n"
        )
        database = tmp_path / "rejected-measurements/.coverage"
        environment = profile_environment()
        environment.update(
            COVERAGE_FILE=str(database),
            SPO_DIMENSION_COVERAGE_SOURCE_ROOT=str(retained),
        )
        completed = _run(exported, "absent", hashes, environment)
        assert completed.returncode != 0
        assert "dimension child source differs from retained source" in completed.stderr
        assert not list(database.parent.glob(".coverage.dimension-*"))


if __name__ == "__main__":
    _qualify(Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3]))
