# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — direct Mojo runtime executable probe

"""Shared executable probes for direct experimental Mojo bridges."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

__all__ = ["require_mojo_executable", "run_mojo_executable"]


class MojoProcess(Protocol):
    """Text-mode process result returned by a direct Mojo bridge launch."""

    returncode: int
    stdout: str
    stderr: str


MojoProcessRunner = Callable[..., MojoProcess]

_LOADER_EXIT_STATUS = 127
"""Exit status the glibc dynamic loader uses when it cannot start a program."""

_LOADER_FAILURE_MARKERS: tuple[str, ...] = (
    "symbol lookup error",
    "error while loading shared libraries",
    "cannot open shared object file",
)
"""glibc ``ld.so`` diagnostics for a program the host cannot link at start-up.

A Mojo executable built with one toolchain and run against another runtime
fails this way, for example ``undefined symbol:
KGEN_CompilerRT_AsyncRT_ReleaseRuntime`` for a Mojo 0.26 build started under
Mojo 1.0.
"""


def _is_loader_failure(process: MojoProcess) -> bool:
    """Return whether the dynamic loader, not the backend, ended the process."""
    if process.returncode != _LOADER_EXIT_STATUS:
        return False
    stderr = process.stderr
    if any(marker in stderr for marker in _LOADER_FAILURE_MARKERS):
        return True
    return "version `" in stderr and "' not found" in stderr


def require_mojo_executable(executable_path: Path) -> Path:
    """Return a compiled Mojo backend path only when the host can execute it.

    Parameters
    ----------
    executable_path:
        Absolute path to the compiled Mojo executable owned by a direct
        accelerator bridge.

    Returns
    -------
    pathlib.Path
        The same path after host executable-permission validation.

    Raises
    ------
    ImportError
        Raised when the artefact exists but cannot be executed by the current
        process, or when the dynamic loader cannot link it against the
        installed runtime. Missing-file diagnostics stay in the owning bridge
        so each module can preserve its specific build command.

    Notes
    -----
    Backend resolution calls this before admitting the Mojo backend, so the
    loader check runs the executable once with empty standard input. Every
    direct Mojo program reads its request with ``input()`` and exits at end of
    input; only a loader failure (status 127 with a glibc ``ld.so``
    diagnostic) rejects the artefact. The verdict is cached per path, size and
    modification time, so a rebuilt executable is checked again.
    """
    if not os.access(executable_path, os.X_OK):
        raise ImportError(
            f"{executable_path.name} exists at {executable_path} but is not "
            "executable; rebuild the Mojo backend or restore execute permissions"
        )
    _require_loadable(executable_path)
    return executable_path


_LOADER_PROBE_TIMEOUT_SECONDS = 10.0
_LOADABLE_ARTEFACTS: set[tuple[str, int, int]] = set()


def _require_loadable(executable_path: Path) -> None:
    """Reject an executable the dynamic loader cannot start, once per build.

    ``executable_path`` is the bridge's own compiled artefact. The launcher is
    looked up as ``subprocess.run`` at call time, the same process boundary the
    bridges pass to :func:`run_mojo_executable`.
    """
    status = executable_path.stat()
    identity = (str(executable_path), status.st_size, status.st_mtime_ns)
    if identity in _LOADABLE_ARTEFACTS:
        return
    runner: MojoProcessRunner = subprocess.run
    try:
        process = runner(
            [str(executable_path)],
            input="",
            capture_output=True,
            text=True,
            check=False,
            timeout=_LOADER_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        _LOADABLE_ARTEFACTS.add(identity)
        return
    except OSError as exc:
        raise ImportError(
            f"{executable_path.name} could not be executed from "
            f"{executable_path}: {exc}"
        ) from exc
    if _is_loader_failure(process):
        raise ImportError(_loader_failure_message(executable_path, process))
    _LOADABLE_ARTEFACTS.add(identity)


def _loader_failure_message(executable_path: Path, process: MojoProcess) -> str:
    """Describe a loader failure with the last ``ld.so`` diagnostic line."""
    diagnostic = process.stderr.strip().splitlines()[-1]
    return (
        f"{executable_path.name} at {executable_path} cannot be loaded by "
        f"this host; rebuild it against the Mojo runtime this host provides: "
        f"{diagnostic}"
    )


def run_mojo_executable(
    executable_path: Path,
    payload: str,
    *,
    runner: MojoProcessRunner,
) -> MojoProcess:
    """Run a direct Mojo executable and demote host execution failures.

    Parameters
    ----------
    executable_path:
        Absolute path returned by :func:`require_mojo_executable`.
    payload:
        Text protocol payload sent to the backend executable on standard input.
    runner:
        Process-launch callable. Bridges pass their module-local process
        launcher so existing tests can still monkeypatch the real production
        subprocess boundary.

    Returns
    -------
    MojoProcess
        The completed process with captured text stdout and stderr.

    Raises
    ------
    ImportError
        Raised when the operating system rejects the executable at launch time,
        for example because the file format is invalid for this host, or when
        the dynamic loader exits with status 127 and a glibc ``ld.so``
        diagnostic because the executable cannot be linked against the
        installed runtime. Callers treat the backend as unavailable. A non-zero
        exit without a loader diagnostic is returned unchanged so the bridge
        still reports it as a backend failure.
    """
    try:
        process = runner(
            [str(executable_path)],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ImportError(
            f"{executable_path.name} could not be executed from "
            f"{executable_path}: {exc}"
        ) from exc
    if _is_loader_failure(process):
        raise ImportError(_loader_failure_message(executable_path, process))
    return process
