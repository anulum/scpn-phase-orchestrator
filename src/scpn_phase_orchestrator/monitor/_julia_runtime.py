# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — monitor Julia runtime probe

"""Shared Julia runtime probe for monitor optional backend loaders."""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["require_juliacall_main"]


def require_juliacall_main() -> ModuleType:
    """Return ``juliacall`` only when the Julia runtime is initialised.

    Returns
    -------
        Imported ``juliacall`` module with its runtime-injected ``Main`` symbol.

    Raises
    ------
        ImportError: Raised when ``juliacall`` is missing, or when the package
            imported without the ``Main`` symbol required by Julia bridges.
    """
    try:
        juliacall = importlib.import_module("juliacall")
    except ImportError:
        raise
    except Exception as exc:
        # juliacall and juliapkg report a broken Julia set-up (a missing
        # executable, an inconsistent environment) with ValueError or a bare
        # Exception; the backend is then unavailable, not the package.
        raise ImportError(
            f"juliacall could not start Julia: {type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(juliacall, "Main"):
        raise ImportError("juliacall.Main unavailable; Julia runtime not initialised")
    return juliacall
