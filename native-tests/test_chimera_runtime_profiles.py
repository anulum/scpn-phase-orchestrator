# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Genuine installed chimera artifact failure profiles

"""Exercise original installed public APIs against actual obsolete/broken assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from scpn_phase_orchestrator.monitor import chimera

pytestmark = pytest.mark.native_runtime
REPO = Path(__file__).resolve().parents[1]

PROBE = r"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scpn_phase_orchestrator.monitor import chimera as c
owner = sys.argv[1]
p = np.array([0.0, 0.0, np.pi])
k = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
record = dict(module=c.__file__, available=c.AVAILABLE_BACKENDS,
              source_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest())
try:
    local = c.local_order_parameter(p, k, backend=owner)
    state = c.detect_chimera(p, k, backend=owner)
except ImportError as error:
    record.update(status='unavailable', error=str(error), cause=str(error.__cause__))
else:
    record.update(status='computed', local=local.tolist(),
                  coherent=state.coherent_indices, incoherent=state.incoherent_indices,
                  index=state.chimera_index)
print(json.dumps(record))
"""


def _install_target(tmp_path: Path) -> tuple[Path, Path]:
    """Install the actual current wheel into an isolated native asset layout.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Owner-provided pytest directory under the registered task workspace.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Installed site-packages and its Python-version native asset root.
    """
    wheel_value = os.environ.get("SPO_CHIMERA_PROFILE_WHEEL")
    assert wheel_value is not None, "A source-qualified current wheel is required"
    wheel = Path(wheel_value)
    assert wheel.is_file()
    expected_wheel = os.environ.get("SPO_CHIMERA_PROFILE_WHEEL_SHA256")
    assert expected_wheel is not None
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == expected_wheel
    root = (
        tmp_path
        / "install"
        / "lib"
        / ("python" + str(sys.version_info.major) + "." + str(sys.version_info.minor))
    )
    site = root / "site-packages"
    env = os.environ.copy()
    env["PIP_NO_CACHE_DIR"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--target",
            str(site),
            str(wheel),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = hashlib.sha256(Path(chimera.__file__).read_bytes()).hexdigest()
    assert (
        hashlib.sha256(
            (site / "scpn_phase_orchestrator/monitor/chimera.py").read_bytes()
        ).hexdigest()
        == expected
    )
    return site, root


def _probe(site: Path, owner: str) -> dict[str, object]:
    """Run a fresh real interpreter and verify the installed public source.

    Parameters
    ----------
    site : pathlib.Path
        Actual pip target installation of the current wheel.
    owner : str
        Explicit Go or Julia owner; no resolution globals are modified.

    Returns
    -------
    dict[str, object]
        Actual owner refusal or numerical results with module provenance.
    """
    assert owner in {"go", "julia"}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(site)
    result = subprocess.run(
        [sys.executable, "-B", "-c", PROBE, owner],
        cwd=site.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    raw: object = json.loads(result.stdout)
    assert isinstance(raw, dict)
    record = cast("dict[str, object]", raw)
    assert Path(str(record["module"])).is_relative_to(site)
    assert (
        record["source_sha256"]
        == hashlib.sha256(Path(chimera.__file__).read_bytes()).hexdigest()
    )
    return record


def _assert_computed(record: dict[str, object], owner: str) -> None:
    """Check analytic nonempty output from the actual recovered native owner."""
    assert record["status"] == "computed"
    assert owner in cast("list[str]", record["available"])
    local = cast("list[float]", record["local"])
    assert len(local) == 3 and abs(local[0]) <= 1e-12
    assert local[1] == 0.0 and abs(local[2] - 1.0) <= 1e-12
    assert record["coherent"] == [2] and record["incoherent"] == [0, 1]
    assert record["index"] == 0.0


def test_actual_obsolete_go_abi_refusal_and_recovery(tmp_path: Path) -> None:
    """Refuse the preserved original ABI and recover using the current real binary."""
    site, root = _install_target(tmp_path)
    old_value = os.environ.get("SPO_CHIMERA_OLD_GO_LIB")
    assert old_value is not None, "Original pre-task Go artifact is required"
    old = Path(old_value)
    current = REPO / "go/libchimera.so"
    assert old.is_file() and current.is_file()
    assert (
        hashlib.sha256(old.read_bytes()).hexdigest()
        != hashlib.sha256(current.read_bytes()).hexdigest()
    )
    target = root / "go/libchimera.so"
    target.parent.mkdir()
    shutil.copy2(old, target)
    refused = _probe(site, "go")
    assert refused["status"] == "unavailable"
    assert "go" not in cast("list[str]", refused["available"])
    assert "LocalOrderParameterV2" in str(refused["cause"])
    shutil.copy2(current, target)
    _assert_computed(_probe(site, "go"), "go")


def test_actual_julia_missing_broken_source_and_recovery(tmp_path: Path) -> None:
    """Refuse missing/truncated assets in actual Julia, then run valid source."""
    site, root = _install_target(tmp_path)
    refused = _probe(site, "julia")
    assert refused["status"] == "unavailable"
    assert "side-file not found" in str(refused["cause"])
    target = root / "julia/chimera.jl"
    target.parent.mkdir()
    source = (REPO / "julia/chimera.jl").read_text()
    target.write_text(source[: source.rfind("end")])
    broken = _probe(site, "julia")
    assert broken["status"] == "unavailable"
    assert "source cannot load" in str(broken["cause"])
    target.write_text(source)
    _assert_computed(_probe(site, "julia"), "julia")
