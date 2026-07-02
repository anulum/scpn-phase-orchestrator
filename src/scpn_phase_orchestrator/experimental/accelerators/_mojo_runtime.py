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
        process. Missing-file diagnostics stay in the owning bridge so each
        module can preserve its specific build command.
    """
    if not os.access(executable_path, os.X_OK):
        raise ImportError(
            f"{executable_path.name} exists at {executable_path} but is not "
            "executable; rebuild the Mojo backend or restore execute permissions"
        )
    return executable_path


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
        for example because the file format is invalid for this host.
    """
    try:
        return runner(
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
