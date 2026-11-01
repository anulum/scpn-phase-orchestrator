# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia Hodge side-file admission contracts

"""Public fallback and recovery in a genuine Julia-only interpreter profile."""

from __future__ import annotations

import hashlib
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


def _check_profile(
    exported_root: Path, side_file_present: bool, expected_source_sha256: str
) -> None:
    """Prove live Julia and genuine Rust absence before public selection."""
    sys.path.insert(0, str(exported_root / "src"))
    assert find_spec("spo_kernel") is None, "This profile requires genuine Rust absence"
    main = import_module("juliacall").Main
    evaluate_raw = main.seval
    assert callable(evaluate_raw)
    evaluate: Callable[[str], object] = evaluate_raw
    assert evaluate("1 + 1") == 2

    from scpn_phase_orchestrator.coupling import hodge

    source = Path(hodge.__file__).resolve()
    assert source.is_relative_to(exported_root)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_source_sha256
    assert (exported_root / "julia/hodge.jl").is_file() == side_file_present

    coupling = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    phases = np.array([0.0, 0.4, 0.9, 1.2])
    actual = [hodge.hodge_decomposition(coupling, phases)]
    reference = hodge.hodge_decomposition(coupling, phases, backend="python")
    assert reference.betti_one == 1
    assert np.linalg.norm(reference.harmonic) > 0.0
    if side_file_present:
        actual.append(hodge.hodge_decomposition(coupling, phases, backend="julia"))
        assert hodge.AVAILABLE_BACKENDS == ["julia", "python"]
        assert hodge.ACTIVE_BACKEND == "julia"
    else:
        with pytest.raises(ImportError, match="Hodge backend 'julia' is unavailable"):
            hodge.hodge_decomposition(coupling, phases, backend="julia")
        assert hodge.AVAILABLE_BACKENDS == ["python"]
        assert hodge.ACTIVE_BACKEND == "python"
    for result in actual:
        for obtained, expected in (
            (result.gradient, reference.gradient),
            (result.curl, reference.curl),
            (result.harmonic, reference.harmonic),
            (result.flow, reference.flow),
            (result.potential, reference.potential),
        ):
            np.testing.assert_allclose(obtained, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            result.gradient + result.curl + result.harmonic,
            coupling * np.sin(phases[None, :] - phases[:, None]),
            atol=1e-12,
        )
        assert result.betti_one == reference.betti_one


@pytest.mark.native_runtime
@pytest.mark.parametrize("side_file_present", [False, True], ids=["absent", "present"])
def test_public_hodge_julia_sidefile_admission(
    tmp_path: Path, side_file_present: bool
) -> None:
    """Missing Julia assets fall back; an actual side-file enables the Julia owner."""
    root = Path(__file__).resolve().parents[1]
    package = root / "src/scpn_phase_orchestrator"
    source_sha256 = hashlib.sha256(
        (package / "coupling/hodge.py").read_bytes()
    ).hexdigest()
    interpreter = os.environ.get("SPO_HODGE_JULIA_ONLY_PYTHON", sys.executable)
    with TemporaryDirectory(prefix="hodge-julia-", dir=tmp_path) as temporary:
        exported = Path(temporary)
        shutil.copytree(
            package,
            exported / "src/scpn_phase_orchestrator",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        if side_file_present:
            (exported / "julia").mkdir()
            shutil.copyfile(root / "julia/hodge.jl", exported / "julia/hodge.jl")
        completed = subprocess.run(
            [
                interpreter,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                str(exported),
                str(side_file_present),
                source_sha256,
            ],
            env=dict(os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == "__main__":
    _check_profile(Path(sys.argv[1]), sys.argv[2] == "True", sys.argv[3])
