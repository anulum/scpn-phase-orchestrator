# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Every domainpack runner runs to completion

"""Each ``domainpacks/<pack>/run.py`` runs end to end as a script.

Static checks cannot see a runner whose hard-coded frequencies fall behind
its spec, or that imports a module moved since: the package's lazy
``__getattr__`` makes mypy treat an unknown submodule as ``Any``. Each
runner is therefore executed as ``__main__`` on a copy of its pack, so
plots and state files land in a temporary directory, never in the source
tree or the user's home.
"""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = sorted(ROOT.glob("domainpacks/*/run.py"))


def test_every_pack_has_a_runner_under_test() -> None:
    """The parametrisation covers every pack directory that ships a runner."""
    assert len(RUNNERS) >= 36
    assert {runner.parent.name for runner in RUNNERS} >= {
        "agent_coordination",
        "swarm_robotics",
        "traffic_flow",
        "power_safety_nchannel",
        "edge_consensus_nchannel",
        "digital_twin_nchannel",
    }


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda path: path.parent.name)
def test_runner_completes_as_a_script(
    runner: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The runner executes as ``__main__`` without error and reports output."""
    pack = tmp_path / runner.parent.name
    shutil.copytree(runner.parent, pack)
    monkeypatch.setenv("SPO_IDENTITY_IMPRINT_DIR", str(tmp_path / "imprint"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(pack)

    runpy.run_path(str(pack / "run.py"), run_name="__main__")

    assert capsys.readouterr().out.strip()
