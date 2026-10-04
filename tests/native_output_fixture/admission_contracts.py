# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual branch artifact admission contracts

"""Exercise the public gate on real downloaded profile databases.

This file is selected explicitly after the obligatory producer artifacts arrive;
it is outside default test filename discovery. The inputs are actual measured
databases, not fabricated coverage maps. Negative cases mutate copied artifact
metadata or generate an actual incomplete/statement-only measurement. No
numerical backend or production capability is replaced.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from coverage import Coverage

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("native", "absent", "defective-output", "defective-missing")


@pytest.fixture
def measured_inputs() -> Path:
    """Require all actual producer artifacts instead of a pre-baked success fixture."""
    root = Path(os.environ["SPO_BRANCH_PROFILE_INPUTS"])
    for profile in PROFILES:
        assert (root / profile / "profile.json").is_file()
    return root


def _invoke(
    inputs: Path,
    destination: Path,
    *,
    revision: str,
    root: Path = ROOT,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run the same public aggregate CLI used by the required CI job.

    ``root`` is the checkout that the aggregate judges; ``cwd`` is where the
    caller stands. The tool itself is always the one of this checkout.
    """
    command = [
        sys.executable,
        "-m",
        "tools.branch_coverage_profiles",
        "combine",
        "--root",
        str(root),
        "--revision",
        revision,
        "--destination",
        str(destination),
    ]
    for profile in PROFILES:
        command.extend(["--" + profile, str(inputs / profile)])
    return subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
        text=True,
        capture_output=True,
    )


def test_actual_profile_union_preserves_inputs_and_qualifies_only_reviewed_guards(
    measured_inputs: Path,
    tmp_path: Path,
) -> None:
    """Join real observations and preserve originals while qualifying the guards."""
    originals = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in measured_inputs.rglob("*")
        if path.is_file()
    }
    revision = json.loads((measured_inputs / "native/profile.json").read_text())[
        "revision"
    ]
    result = _invoke(measured_inputs, tmp_path / "accepted", revision=revision)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "accepted/admission.json").read_text())
    assert report["exact_arc_union_verified"]
    assert report["originals_unchanged"]
    assert report["guard_coverage"] == "100 % with three real profiles, 0 exclusions"
    for key in ("genuine", "all"):
        database = Path(report[key + "_data"])
        assert database.is_file()
        assert (
            hashlib.sha256(database.read_bytes()).hexdigest()
            == report[key + "_data_sha256"]
        )
    coverage = json.loads((tmp_path / "accepted/all.json").read_text())
    residuals = json.loads((ROOT / "tools/branch_profile_residuals.json").read_text())
    for name in residuals["files"]:
        assert not coverage["files"][name]["missing_lines"]
        assert not coverage["files"][name]["missing_branches"]
        assert not coverage["files"][name]["excluded_lines"]
    assert {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in measured_inputs.rglob("*")
        if path.is_file()
    } == originals


def _actual_incomplete_data(database: Path, *, branch: bool) -> None:
    """Measure a real public consumer in a fresh process without synthetic hits."""
    program = f"""
from coverage import Coverage
measurement = Coverage(
    config_file={str(ROOT / "pyproject.toml")!r},
    data_file={str(database)!r}, branch={branch!r},
    source=[
        "scpn_phase_orchestrator.coupling.ei_balance",
        "scpn_phase_orchestrator.upde.sheaf_engine",
        "scpn_phase_orchestrator.upde.sparse_engine",
    ],
)
measurement.start()
import numpy as np
from scpn_phase_orchestrator.coupling.ei_balance import compute_ei_balance
from scpn_phase_orchestrator.upde.sheaf_engine import SheafUPDEEngine
from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine
coupling = np.array([[0.0, 2.0], [1.0, 0.0]])
assert compute_ei_balance(coupling, [0], [1]).ratio == 2.0
phases = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6]])
np.testing.assert_array_equal(
    SheafUPDEEngine(3,2,0.125).step(phases,np.zeros_like(phases),np.zeros((3,3,2,2)),0.0,np.zeros(2)),
    phases,
)
vector = np.array([0.1,0.2,0.3])
np.testing.assert_array_equal(
    SparseUPDEEngine(3,0.125).step(vector,np.zeros(3),np.zeros(4,dtype=np.int64),
        np.empty(0,dtype=np.int64),np.empty(0),0.0,0.0,np.empty(0)),
    vector,
)
measurement.stop()
measurement.save()
"""
    subprocess.run([sys.executable, "-c", program], cwd=ROOT, check=True)
    # The required subprocess patch gives real child databases unique suffixes.
    # Combine those actual pieces instead of assuming a single unsuffixed file.
    pieces = [
        str(path)
        for path in database.parent.glob(database.name + ".*")
        if path.suffix != ".license"
    ]
    assert pieces
    collected = Coverage(
        config_file=str(ROOT / "pyproject.toml"), data_file=str(database)
    )
    collected.combine(data_paths=pieces, strict=True, keep=True)
    collected.save()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing-profile", "No such file"),
        ("stale-revision", "stale or malformed"),
        ("statement-only", "statement-only database"),
        ("database-hash", "database hash mismatch"),
        ("fixture-marker", "fixture provenance"),
        ("fixture-variant", "fixture variant"),
        ("absence-proof", "No such file"),
        ("lock-mismatch", "configuration or environment mismatch"),
        ("package-root-type", "malformed package_root"),
        ("absence-environment-type", "malformed absent-kernel observation"),
        ("unreviewed-gap", "unreviewed"),
    ],
)
def test_actual_artifact_faults_fail_closed(
    measured_inputs: Path,
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    """Reject genuine artifact loss, drift, wrong modes and hidden success-path gaps."""
    inputs = tmp_path / "inputs"
    revision = json.loads((measured_inputs / "native/profile.json").read_text())[
        "revision"
    ]
    shutil.copytree(measured_inputs, inputs)
    profile = "absent" if failure == "lock-mismatch" else "defective-output"
    if failure in {
        "stale-revision",
        "database-hash",
        "statement-only",
        "unreviewed-gap",
    }:
        profile = "native"
    receipt_path = inputs / profile / "profile.json"
    receipt = json.loads(receipt_path.read_text())
    database = receipt_path.parent / receipt["database"]
    if failure == "missing-profile":
        shutil.rmtree(inputs / "defective-missing")
    elif failure == "stale-revision":
        receipt["revision"] = "0" * 40
    elif failure == "database-hash":
        with database.open("ab") as stream:
            stream.write(b"changed-artifact")
    elif failure == "fixture-marker":
        receipt["fixture"]["marker"] = "unmarked-producer"
    elif failure == "fixture-variant":
        receipt["kernel"]["variant"] = "wrong-variant"
    elif failure == "absence-proof":
        (receipt_path.parent / receipt["fixture"]["before_install"]).unlink()
    elif failure == "lock-mismatch":
        receipt["input_hashes"]["requirements/dev-lock.txt"] = "0" * 64
    elif failure == "package-root-type":
        receipt["package_root"] = 5
    elif failure == "absence-environment-type":
        proof_path = receipt_path.parent / receipt["fixture"]["before_install"]
        proof = json.loads(proof_path.read_text())
        proof["environment"] = 5
        proof_path.write_text(json.dumps(proof))
        receipt["fixture"]["before_install_sha256"] = hashlib.sha256(
            proof_path.read_bytes()
        ).hexdigest()
    else:
        database.unlink()
        _actual_incomplete_data(database, branch=failure == "unreviewed-gap")
        receipt["database_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
        package = importlib.import_module("scpn_phase_orchestrator")
        assert package.__file__ is not None
        receipt["package_root"] = str(Path(package.__file__).resolve().parent)
    receipt_path.write_text(json.dumps(receipt))
    result = _invoke(inputs, tmp_path / "refused", revision=revision)
    assert result.returncode == 1, result.stdout
    assert "branch profile refusal:" in result.stderr
    assert message in result.stderr
    assert not (tmp_path / "refused/admission.json").exists()


def test_aggregate_does_not_depend_on_the_working_directory(
    measured_inputs: Path,
    tmp_path: Path,
) -> None:
    """Admit the same real profiles when the caller stands outside the checkout.

    The reports name sources relative to the working directory and the reviewed
    disposition names them relative to the checkout; from another directory
    the two did not meet and every valid aggregate was refused.
    """
    revision = json.loads((measured_inputs / "native/profile.json").read_text())[
        "revision"
    ]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _invoke(
        measured_inputs, tmp_path / "accepted", revision=revision, cwd=elsewhere
    )
    assert result.returncode == 0, result.stderr
    reported = json.loads((tmp_path / "accepted/all.json").read_text())["files"]
    residuals = json.loads((ROOT / "tools/branch_profile_residuals.json").read_text())
    assert set(residuals["files"]) <= set(reported)
    assert not list(elsewhere.iterdir())


def test_aggregate_takes_relative_paths_from_the_callers_directory(
    measured_inputs: Path,
    tmp_path: Path,
) -> None:
    """Resolve relative inputs and destination where the caller stands.

    The caller stands in the directory of the downloaded profiles and names
    them and the destination relatively. Nothing may be written into the
    checkout, and the profiles must be found.
    """
    revision = json.loads((measured_inputs / "native/profile.json").read_text())[
        "revision"
    ]
    destination = tmp_path / "accepted"
    relative = os.path.relpath(destination, measured_inputs)
    command = [
        sys.executable,
        "-m",
        "tools.branch_coverage_profiles",
        "combine",
        "--root",
        str(ROOT),
        "--revision",
        revision,
        "--destination",
        relative,
    ]
    for profile in PROFILES:
        command.extend(["--" + profile, profile])
    result = subprocess.run(
        command,
        cwd=measured_inputs,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (destination / "admission.json").is_file()
    assert not (ROOT / relative).exists()


def _checkout_copy(destination: Path) -> Path:
    """Copy exactly the checkout members that the aggregate reads."""
    for name in (
        "src",
        "tools",
        "requirements",
        "spo-kernel/crates",
        "tests/native_output_fixture",
    ):
        shutil.copytree(
            ROOT / name,
            destination / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    for name in ("pyproject.toml", "spo-kernel/Cargo.toml", "spo-kernel/Cargo.lock"):
        shutil.copy2(ROOT / name, destination / name)
    return destination


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("changed-member", "aggregate checkout source mismatch"),
        ("linked-member", "exact raw source membership and arc union"),
    ],
)
def test_actual_checkout_faults_fail_closed(
    measured_inputs: Path,
    tmp_path: Path,
    fault: str,
    message: str,
) -> None:
    """Reject a checkout that changed after recording or resolves elsewhere.

    The second case keeps the bytes of the measured source and replaces the
    file by a symbolic link: the combined data then names the link target,
    which is not a member of the checkout.
    """
    revision = json.loads((measured_inputs / "native/profile.json").read_text())[
        "revision"
    ]
    checkout = _checkout_copy(tmp_path / "checkout")
    member = checkout / "src/scpn_phase_orchestrator/coupling/ei_balance.py"
    if fault == "changed-member":
        with member.open("a", encoding="utf-8") as stream:
            stream.write("\n# changed after the profiles were recorded\n")
    else:
        outside = tmp_path / "outside-the-checkout.py"
        shutil.copy2(member, outside)
        member.unlink()
        member.symlink_to(outside)
    result = _invoke(
        measured_inputs, tmp_path / "refused", revision=revision, root=checkout
    )
    assert result.returncode == 1, result.stdout
    assert "branch profile refusal:" in result.stderr
    assert message in result.stderr
    assert not (tmp_path / "refused/admission.json").exists()
