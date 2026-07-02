# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — experimental Julia runtime probe

"""Shared Julia runtime probe for direct experimental accelerator bridges."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["require_julia_main"]


def require_julia_main() -> Any:
    """Return the runtime-injected ``juliacall.Main`` symbol.

    Returns
    -------
    Any
        The dynamic ``juliacall.Main`` object used to include Julia side-files
        and access exported Julia modules.

    Raises
    ------
    ImportError
        Raised when ``juliacall`` is unavailable, or when it imports without
        the ``Main`` symbol required by direct Julia bridge loaders.
    """
    juliacall = importlib.import_module("juliacall")
    main = getattr(juliacall, "Main", None)
    if main is None:
        raise ImportError("juliacall.Main unavailable; Julia runtime not initialised")
    return main
