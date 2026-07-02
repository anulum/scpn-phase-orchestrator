# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — direct experimental Mojo runtime contracts

"""Direct experimental Mojo bridge executable-availability contracts."""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

MojoExecutableProbe = Callable[[], Path]
MojoExecutableValidator = Callable[[Path], Path]
MojoProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
MojoProcessLauncher = Callable[..., subprocess.CompletedProcess[str]]

DIRECT_MOJO_MODULES: tuple[str, ...] = (
    "scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.coupling._spatial_modulator_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.coupling._spectral_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._chimera_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._dimension_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._embedding_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._entropy_prod_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._itpc_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._koopman_edmd_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._lyapunov_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._npe_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._opt_entropy_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._pid_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._poincare_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._psychedelic_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._recurrence_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._te_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._twin_confidence_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._winding_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._basin_stability_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._delay_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._engine_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._envelope_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._geometric_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._hypergraph_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._inertial_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._market_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._order_params_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pac_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._reduction_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._simplicial_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._splitting_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._swarmalator_mojo",
)


def _direct_executable_probe(module: ModuleType) -> MojoExecutableProbe:
    """Return the private direct Mojo executable probe for ``module``."""
    return cast(MojoExecutableProbe, vars(module)["_ensure_exe"])


@pytest.mark.parametrize(
    "module_name",
    DIRECT_MOJO_MODULES,
    ids=[name.rsplit(".", maxsplit=1)[-1] for name in DIRECT_MOJO_MODULES],
)
def test_direct_mojo_bridges_reject_non_executable_artefact(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Direct Mojo probes must reject a present artefact without execute bits."""
    module = importlib.import_module(module_name)
    fake_executable = tmp_path / "mojo-backend"
    fake_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_executable.chmod(0o600)

    monkeypatch.setattr(module, "_EXE_PATH", fake_executable)

    with pytest.raises(ImportError, match="not executable"):
        _direct_executable_probe(module)()


def test_direct_mojo_runtime_probe_accepts_executable_artefact(
    tmp_path: Path,
) -> None:
    """The shared Mojo runtime probe must return an executable artefact path."""
    runtime_module = importlib.import_module(
        "scpn_phase_orchestrator.experimental.accelerators._mojo_runtime"
    )
    executable = tmp_path / "mojo-backend"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    validator = cast(
        MojoExecutableValidator,
        runtime_module.require_mojo_executable,
    )

    assert validator(executable) == executable


def test_direct_mojo_runtime_runner_invokes_checked_subprocess(
    tmp_path: Path,
) -> None:
    """The shared Mojo runner must invoke the supplied process launcher."""
    runtime_module = importlib.import_module(
        "scpn_phase_orchestrator.experimental.accelerators._mojo_runtime"
    )
    executable = tmp_path / "mojo-backend"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)

    def run_process(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert args == ([str(executable)],)
        assert kwargs == {
            "input": "RUN\n",
            "capture_output": True,
            "text": True,
            "check": False,
        }
        return subprocess.CompletedProcess(
            args=[str(executable)],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )

    launcher = cast(MojoProcessLauncher, runtime_module.run_mojo_executable)

    assert (
        launcher(
            executable, "RUN\n", runner=cast(MojoProcessRunner, run_process)
        ).stdout
        == "ok\n"
    )


def test_direct_mojo_runtime_runner_demotes_host_execution_error(
    tmp_path: Path,
) -> None:
    """The shared Mojo runner must demote host execution errors to ImportError."""
    runtime_module = importlib.import_module(
        "scpn_phase_orchestrator.experimental.accelerators._mojo_runtime"
    )
    executable = tmp_path / "mojo-backend"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)

    def fail_process(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise OSError("Exec format error")

    launcher = cast(MojoProcessLauncher, runtime_module.run_mojo_executable)

    with pytest.raises(ImportError, match="could not be executed"):
        launcher(executable, "RUN\n", runner=cast(MojoProcessRunner, fail_process))
