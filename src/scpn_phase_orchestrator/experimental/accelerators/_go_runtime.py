# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — direct Go runtime loader contracts

"""Shared runtime probes for direct Go shared-library accelerator bridges."""

from __future__ import annotations

import ctypes
from pathlib import Path

__all__ = ["load_go_library"]


def load_go_library(library_path: Path) -> ctypes.CDLL:
    """Load a Go ``c-shared`` accelerator library.

    Parameters
    ----------
    library_path:
        Absolute path to the compiled ``.so`` artifact owned by the direct Go
        accelerator bridge.

    Returns
    -------
    ctypes.CDLL
        Loaded shared-library handle ready for per-module symbol configuration.

    Raises
    ------
    ImportError
        Raised when the artifact is present but the host dynamic loader rejects
        it. Missing-file diagnostics stay in the owning bridge so each module
        can preserve its specific build command.
    """
    try:
        return ctypes.CDLL(str(library_path))
    except OSError as exc:
        raise ImportError(
            f"{library_path.name} could not be loaded from {library_path}: {exc}"
        ) from exc
