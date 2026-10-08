# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Connectome installed comparison qualification

"""Exercise real profile admission, rejection and finite allocator recovery."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from benchmarks.connectome_benchmark import run_profile
from benchmarks.connectome_reference import reference_connectome

pytestmark = pytest.mark.native_runtime


@pytest.mark.parametrize("owner", ["python", "rust"])
def test_original_installed_profile_matches_oracle_and_native_observation(
    owner: str,
) -> None:
    """Each actual installed owner preserves its own noise law and native count."""
    executable = Path(os.environ["SPO_CONNECTOME_" + owner.upper() + "_PROFILE"])
    record = run_profile(executable, {"owner": owner, "n_regions": 7, "seed": 19})
    assert record["native_calls"] == (1 if owner == "rust" else 0)
    np.testing.assert_allclose(
        np.asarray(record["matrix"], dtype=np.float64),
        reference_connectome(7, 19, owner),
        atol=3e-14,
    )


def test_profile_refuses_other_executable_before_invocation() -> None:
    """The actual Git executable cannot be selected as a trusted Python profile."""
    executable = shutil.which("git")
    assert executable is not None
    with pytest.raises(ValueError, match="trusted Python binary"):
        run_profile(Path(executable), {"owner": "python", "n_regions": 3})


def test_required_native_owner_refuses_genuine_absence() -> None:
    """A required native comparison fails in the actual kernel-absent profile."""
    executable = Path(os.environ["SPO_CONNECTOME_PYTHON_PROFILE"])
    with pytest.raises(ValueError, match="AssertionError"):
        run_profile(executable, {"owner": "rust", "n_regions": 3})


def test_profile_refuses_changed_installed_production_source() -> None:
    """A different actual installed package hash is refused after real execution."""
    executable = Path(os.environ["SPO_CONNECTOME_PYTHON_PROFILE"])
    record = run_profile(executable, {"owner": "python", "n_regions": 3})
    source = Path(str(record["module"]))
    assert source.is_relative_to(executable.parent.parent)
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n# profile-source-admission-negative-case\n")
        with pytest.raises(ValueError, match="differs from current production source"):
            run_profile(executable, {"owner": "python", "n_regions": 3})
    finally:
        source.write_bytes(original)


@pytest.mark.parametrize("field", ["module", "admission_module"])
def test_profile_refuses_byte_identical_source_symlink_outside_installation(
    field: str,
) -> None:
    """Source identity cannot turn an external source symlink into an installation."""
    executable = Path(os.environ["SPO_CONNECTOME_PYTHON_PROFILE"])
    record = run_profile(executable, {"owner": "python", "n_regions": 3})
    source = Path(str(record[field]))
    target = (
        Path(__file__).resolve().parents[1]
        / "src/scpn_phase_orchestrator/coupling"
        / source.name
    )
    original = source.read_bytes()
    assert target.read_bytes() == original and not source.is_symlink()
    mode = source.stat().st_mode & 0o777
    try:
        source.unlink()
        source.symlink_to(target)
        with pytest.raises(ValueError, match="physically installed package"):
            run_profile(executable, {"owner": "python", "n_regions": 3})
    finally:
        source.unlink()
        source.write_bytes(original)
        source.chmod(mode)


@pytest.mark.parametrize("owner", ["python", "rust"])
def test_linux_actual_allocator_refusal_preserves_small_public_generation(
    owner: str,
) -> None:
    """Finite Linux address-space limits refuse dense allocation with recovery."""
    executable = Path(os.environ["SPO_CONNECTOME_" + owner.upper() + "_PROFILE"])
    record = run_profile(
        executable,
        {"owner": owner, "n_regions": 16384, "allocation_budget": 64 * 1024 * 1024},
    )
    error = cast("dict[str, str]", record["error"])
    assert error["type"] == "ValueError"
    assert "cannot allocate connectome matrix" in error["message"]
    np.testing.assert_allclose(
        np.asarray(record["recovery_matrix"], dtype=np.float64),
        [[0.0, 4.95], [4.95, 0.0]],
    )


def test_profile_refuses_original_kernel_symlink_outside_installation(
    tmp_path: Path,
) -> None:
    """Even unchanged original kernel bytes must physically belong to the profile."""
    executable = Path(os.environ["SPO_CONNECTOME_RUST_PROFILE"])
    record = run_profile(executable, {"owner": "rust", "n_regions": 3})
    source = Path(str(record["binary"]))
    original = source.read_bytes()
    target = tmp_path / source.name
    target.write_bytes(original)
    assert not source.is_symlink()
    mode = source.stat().st_mode & 0o777
    try:
        source.unlink()
        source.symlink_to(target)
        with pytest.raises(ValueError, match="physically installed kernel"):
            run_profile(executable, {"owner": "rust", "n_regions": 3})
    finally:
        source.unlink()
        source.write_bytes(original)
        source.chmod(mode)
