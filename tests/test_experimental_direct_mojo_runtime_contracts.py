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

import scpn_phase_orchestrator.experimental.accelerators._mojo_runtime as mojo_runtime

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
    """Direct Mojo probes must delegate present artefacts to the runtime guard."""
    module = importlib.import_module(module_name)
    fake_executable = tmp_path / "mojo-backend"
    fake_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    checked_paths: list[Path] = []

    def reject_executable(executable_path: Path) -> Path:
        checked_paths.append(executable_path)
        raise ImportError(
            f"{executable_path.name} exists at {executable_path} but is not executable"
        )

    monkeypatch.setattr(module, "_EXE_PATH", fake_executable)
    monkeypatch.setattr(module, "require_mojo_executable", reject_executable)
    with pytest.raises(ImportError, match="not executable"):
        _direct_executable_probe(module)()
    assert checked_paths == [fake_executable]


def test_direct_mojo_runtime_probe_rejects_non_executable_artefact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The shared Mojo runtime probe must reject denied execution access."""
    executable = tmp_path / "mojo-backend"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def deny_access(_path: Path, _mode: int) -> bool:
        return False

    monkeypatch.setattr(mojo_runtime.os, "access", deny_access)

    with pytest.raises(ImportError, match="not executable"):
        mojo_runtime.require_mojo_executable(executable)


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


def _loader_failure_executable(tmp_path: Path, diagnostic: str, status: int) -> Path:
    """Write a real executable that ends like a program ld.so cannot link."""
    message = tmp_path / "loader_diagnostic.txt"
    message.write_text(diagnostic + "\n", encoding="utf-8")
    executable = tmp_path / "stale_mojo_backend"
    executable.write_text(
        f"#!/bin/sh\ncat >/dev/null\ncat '{message}' >&2\nexit {status}\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


_SYMBOL_LOOKUP_ERROR = (
    "./mojo/order_params_mojo: symbol lookup error: ./mojo/order_params_mojo: "
    "undefined symbol: KGEN_CompilerRT_AsyncRT_ReleaseRuntime"
)


@pytest.mark.parametrize(
    "diagnostic",
    [
        _SYMBOL_LOOKUP_ERROR,
        "./mojo/pac_mojo: error while loading shared libraries: "
        "libKGENCompilerRTShared.so: cannot open shared object file: "
        "No such file or directory",
        "./mojo/pac_mojo: /lib/x86_64-linux-gnu/libc.so.6: "
        "version `GLIBC_2.99' not found (required by ./mojo/pac_mojo)",
    ],
    ids=["undefined-symbol", "missing-shared-library", "missing-symbol-version"],
)
def test_direct_mojo_runtime_runner_demotes_dynamic_loader_failure(
    tmp_path: Path, diagnostic: str
) -> None:
    """A program the dynamic loader cannot start is an unavailable backend."""
    executable = _loader_failure_executable(tmp_path, diagnostic, 127)

    with pytest.raises(ImportError, match="cannot be loaded by this host") as excinfo:
        mojo_runtime.run_mojo_executable(executable, "R 1 0.0\n", runner=subprocess.run)
    assert diagnostic in str(excinfo.value)


@pytest.mark.parametrize(
    ("diagnostic", "status"),
    [
        ("backend rejected the payload", 127),
        (_SYMBOL_LOOKUP_ERROR, 1),
    ],
    ids=["status-127-without-loader-text", "loader-text-with-other-status"],
)
def test_direct_mojo_runtime_runner_keeps_backend_failures(
    tmp_path: Path, diagnostic: str, status: int
) -> None:
    """Only the loader signature is demoted; other exits reach the bridge."""
    executable = _loader_failure_executable(tmp_path, diagnostic, status)

    process = mojo_runtime.run_mojo_executable(
        executable, "R 1 0.0\n", runner=subprocess.run
    )
    assert process.returncode == status
    assert diagnostic in process.stderr


def test_unloadable_mojo_order_parameter_backend_is_excluded_from_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The import-time timing probe must not fail on an unloadable Mojo build."""
    bridge = importlib.import_module(
        "scpn_phase_orchestrator.experimental.accelerators.upde._order_params_mojo"
    )
    order_params = importlib.import_module("scpn_phase_orchestrator.upde.order_params")
    executable = _loader_failure_executable(tmp_path, _SYMBOL_LOOKUP_ERROR, 127)
    monkeypatch.setattr(bridge, "_EXE_PATH", executable)

    with pytest.raises(ImportError, match="cannot be loaded by this host"):
        bridge.order_parameter_mojo(order_params.np.array([0.0, 1.0, 2.0]))
    probe = vars(order_params)["_order_parameter_probe_seconds"]
    assert probe("mojo") == float("inf")


def test_public_winding_falls_back_when_the_mojo_build_cannot_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale Mojo build selected as active backend degrades to the reference."""
    bridge = importlib.import_module(
        "scpn_phase_orchestrator.experimental.accelerators.monitor._winding_mojo"
    )
    winding = importlib.import_module("scpn_phase_orchestrator.monitor.winding")
    executable = _loader_failure_executable(tmp_path, _SYMBOL_LOOKUP_ERROR, 127)
    monkeypatch.setattr(bridge, "_EXE_PATH", executable)
    monkeypatch.setattr(winding, "ACTIVE_BACKEND", "mojo")
    monkeypatch.setattr(winding, "AVAILABLE_BACKENDS", ["mojo", "python"])
    monkeypatch.setattr(winding, "_BACKEND_CACHE", {})
    history = winding.np.linspace(0.0, 20.0, 50)[:, None] * winding.np.ones((1, 3))

    result = winding.winding_numbers(history)

    reference = vars(winding)["_winding_reference"](history)
    assert result.tolist() == reference.tolist()


def test_executable_guard_rejects_a_build_the_loader_cannot_link(
    tmp_path: Path,
) -> None:
    """Backend admission fails closed on a stale build before any request."""
    executable = _loader_failure_executable(tmp_path, _SYMBOL_LOOKUP_ERROR, 127)

    with pytest.raises(ImportError, match="cannot be loaded by this host"):
        mojo_runtime.require_mojo_executable(executable)


def _counting_executable(tmp_path: Path, body: str) -> tuple[Path, Path]:
    """Write a real executable that records each launch in a counter file."""
    counter = tmp_path / "launches.txt"
    executable = tmp_path / "mojo_backend"
    executable.write_text(
        f"#!/bin/sh\necho launch >> '{counter}'\ncat >/dev/null\n{body}\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, counter


def _launches(counter: Path) -> int:
    return len(counter.read_text(encoding="utf-8").splitlines())


def test_executable_guard_checks_each_build_once(tmp_path: Path) -> None:
    """A loadable build is launched once; a rebuilt artefact is checked again."""
    executable, counter = _counting_executable(tmp_path, "exit 1")

    assert mojo_runtime.require_mojo_executable(executable) == executable
    assert mojo_runtime.require_mojo_executable(executable) == executable
    assert _launches(counter) == 1

    executable.write_text(
        executable.read_text(encoding="utf-8") + "# rebuilt\n", encoding="utf-8"
    )
    assert mojo_runtime.require_mojo_executable(executable) == executable
    assert _launches(counter) == 2


def test_executable_guard_accepts_a_build_that_outlives_the_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A program still running at the probe deadline has been linked."""
    executable, counter = _counting_executable(tmp_path, "sleep 5")
    monkeypatch.setattr(mojo_runtime, "_LOADER_PROBE_TIMEOUT_SECONDS", 0.2)

    assert mojo_runtime.require_mojo_executable(executable) == executable
    assert mojo_runtime.require_mojo_executable(executable) == executable
    assert _launches(counter) == 1


def test_executable_guard_demotes_launch_errors(tmp_path: Path) -> None:
    """An artefact the kernel refuses to start is an unavailable backend."""
    executable = tmp_path / "mojo_backend"
    executable.write_text("#!/nonexistent/interpreter\n", encoding="utf-8")
    executable.chmod(0o700)

    with pytest.raises(ImportError, match="could not be executed"):
        mojo_runtime.require_mojo_executable(executable)


def test_backend_resolution_excludes_a_build_the_loader_cannot_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Module backend resolution must not admit a stale Mojo build."""
    bridge = importlib.import_module(
        "scpn_phase_orchestrator.experimental.accelerators.monitor._winding_mojo"
    )
    winding = importlib.import_module("scpn_phase_orchestrator.monitor.winding")
    executable = _loader_failure_executable(tmp_path, _SYMBOL_LOOKUP_ERROR, 127)
    monkeypatch.setattr(bridge, "_EXE_PATH", executable)
    monkeypatch.setattr(winding, "_BACKEND_CACHE", {})

    active, available = vars(winding)["_resolve_backends"]()

    assert "mojo" not in available
    assert active != "mojo"
    assert available[-1] == "python"
