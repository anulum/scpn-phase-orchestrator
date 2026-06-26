# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spo-kernel installer tool tests

"""CLI tests for ``tools/install_spo_kernel.py``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return the repository root containing this test file."""
    return Path(__file__).resolve().parents[1]


def _tool() -> Path:
    """Return the installer tool path."""
    return _repo_root() / "tools" / "install_spo_kernel.py"


def test_install_spo_kernel_dry_run_uses_selected_python() -> None:
    """Dry-run emits the exact maturin command without building Rust."""
    completed = subprocess.run(
        [
            sys.executable,
            str(_tool()),
            "--dry-run",
            "--json",
            "--python",
            sys.executable,
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=_repo_root(),
    )

    payload = json.loads(completed.stdout)

    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["editable"] is True
    assert payload["release"] is True
    assert payload["command"][:3] == [sys.executable, "-m", "maturin"]
    assert "develop" in payload["command"]
    assert (
        Path(payload["manifest"])
        .as_posix()
        .endswith("spo-kernel/crates/spo-ffi/Cargo.toml")
    )


def test_install_spo_kernel_check_only_uses_target_interpreter() -> None:
    """Check-only verifies imports through the selected interpreter."""
    completed = subprocess.run(
        [
            sys.executable,
            str(_tool()),
            "--check-only",
            "--verify-module",
            "sys",
            "--json",
            "--python",
            sys.executable,
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=_repo_root(),
    )

    payload = json.loads(completed.stdout)

    assert payload["ok"] is True
    assert payload["check_only"] is True
    assert payload["module"] == "sys"
    assert payload["stdout"] == "sys"


def test_install_spo_kernel_rejects_invalid_verify_module() -> None:
    """The check-only path rejects invalid module expressions."""
    completed = subprocess.run(
        [
            sys.executable,
            str(_tool()),
            "--check-only",
            "--verify-module",
            "sys;raise SystemExit",
            "--json",
            "--python",
            sys.executable,
        ],
        check=False,
        text=True,
        capture_output=True,
        cwd=_repo_root(),
    )

    payload = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert payload["ok"] is False
    assert "invalid module name" in payload["error"]
