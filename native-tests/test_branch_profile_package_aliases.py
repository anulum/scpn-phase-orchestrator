# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual installed branch-profile aliases

"""Bind coverage from a genuine second installation without weakening source scope."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from tools.branch_profile_provenance import ProfileReceipt

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.native_runtime


def test_actual_absent_installation_requires_explicit_unchanged_provenance(
    tmp_path: Path,
) -> None:
    """Real child coverage is admitted only with absence and complete byte identity."""
    absent = os.environ.get("SPO_CONNECTOME_PYTHON_PROFILE")
    assert absent is not None
    assert Path(absent).is_file()
    environment = os.environ.copy()
    environment.pop("COVERAGE_PROCESS_START", None)
    database = tmp_path / ".coverage.child"
    program = """
from coverage import Coverage
import numpy as np
coverage = Coverage(config_file=False, branch=True,
    source=["scpn_phase_orchestrator"], data_file=__import__("sys").argv[1])
coverage.start()
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy
values = PhaseSINDy(threshold=0).fit(np.arange(12).reshape(-1, 1) * 0.2, 0.1)
np.testing.assert_allclose(values, [[2.0]], atol=1e-12, rtol=0)
coverage.stop()
coverage.save()
"""
    measured = subprocess.run(
        [absent, "-I", "-B", "-c", program, str(database)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert measured.returncode == 0, measured.stderr
    original_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    command = [
        sys.executable,
        "-m",
        "tools.branch_coverage_profiles",
        "record",
        "--root",
        str(ROOT),
        "--profile",
        "native",
        "--revision",
        "alias-test",
        "--database",
        str(database),
    ]
    refused = subprocess.run(
        [*command, "--output", str(tmp_path / "undeclared.json")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert refused.returncode == 1
    assert "outside the installed package" in refused.stderr
    output = tmp_path / "declared.json"
    admitted = subprocess.run(
        [*command, "--absent-interpreter", absent, "--output", str(output)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert admitted.returncode == 0, admitted.stderr
    value: object = json.loads(output.read_text())
    assert isinstance(value, dict)
    proof = cast("ProfileReceipt", value)
    additional = proof.get("additional_absent_profiles", [])
    assert len(additional) == 1
    assert additional[0]["kernel_status"] == "absent"
    assert hashlib.sha256(database.read_bytes()).hexdigest() == original_hash
    installed = Path(additional[0]["package_root"])
    source = installed / "autotune/sindy.py"
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n# deliberate source-identity fault\n")
        changed = subprocess.run(
            [
                *command,
                "--absent-interpreter",
                absent,
                "--output",
                str(tmp_path / "changed.json"),
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert changed.returncode == 1
        assert "additional installed package source mismatch" in changed.stderr
    finally:
        source.write_bytes(original)
    assert source.read_bytes() == original
    assert hashlib.sha256(database.read_bytes()).hexdigest() == original_hash
