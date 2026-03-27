# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling templates

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

__all__ = ["KnmTemplate", "KnmTemplateSet"]


@dataclass(frozen=True)
class KnmTemplate:
    """Named K_nm coupling matrix with associated phase-lag matrix."""

    name: str
    knm: NDArray
    alpha: NDArray
    description: str


class KnmTemplateSet:
    """Registry of named K_nm templates for runtime switching."""

    def __init__(self) -> None:
        self._templates: dict[str, KnmTemplate] = {}

    def add(self, template: KnmTemplate) -> None:
        """Register a template, overwriting any existing one with the same name."""
        self._templates[template.name] = template

    def get(self, name: str) -> KnmTemplate:
        """Retrieve a template by name. Raises KeyError if not found."""
        try:
            return self._templates[name]
        except KeyError:
            available = ", ".join(sorted(self._templates)) or "(none)"
            msg = f"Unknown template {name!r}; available: {available}"
            raise KeyError(msg) from None

    def list_names(self) -> list[str]:
        """Return all registered template names."""
        return list(self._templates.keys())
