# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling templates

"""Named coupling-template registry for runtime K/alpha switching.

Templates bundle a phase-coupling matrix, phase-lag matrix, description, and
stable name. The registry is intentionally in-memory and deterministic:
retrieval fails explicitly with available names when a requested template is
not registered, leaving persistence and validation to the binding/template
owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["KnmTemplate", "KnmTemplateSet"]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class KnmTemplate:
    """Named K_nm coupling matrix with associated phase-lag matrix."""

    name: str
    knm: FloatArray
    alpha: FloatArray
    description: str


class KnmTemplateSet:
    """Registry of named K_nm templates for runtime switching."""

    def __init__(self) -> None:
        self._templates: dict[str, KnmTemplate] = {}

    def add(self, template: KnmTemplate) -> None:
        """Register a template, overwriting any existing one with the same name."""
        if not isinstance(template, KnmTemplate):
            raise TypeError(f"template must be KnmTemplate, got {template!r}")
        normalized_name = template.name.strip()
        if not normalized_name:
            raise ValueError("template name must be a non-empty string")
        if not isinstance(template.description, str) or not template.description.strip():
            raise ValueError("template description must be a non-empty string")
        if template.knm.ndim != 2 or template.alpha.ndim != 2:
            raise ValueError("template knm/alpha must be 2D matrices")
        if not np.issubdtype(template.knm.dtype, np.floating) or not np.issubdtype(
            template.alpha.dtype, np.floating
        ):
            raise ValueError("template knm/alpha must use floating-point dtypes")
        if template.knm.shape != template.alpha.shape:
            raise ValueError("template knm and alpha must have identical shapes")
        if template.knm.shape[0] != template.knm.shape[1]:
            raise ValueError("template knm must be square")
        if not np.isfinite(template.knm).all() or not np.isfinite(template.alpha).all():
            raise ValueError("template knm/alpha must contain only finite values")
        if any(
            isinstance(v, bool) or not isinstance(v, Real)
            for v in (float(np.min(template.knm)), float(np.max(template.knm)))
        ):
            raise ValueError("template knm must contain numeric real values")
        if not isfinite(float(np.min(template.alpha))) or not isfinite(
            float(np.max(template.alpha))
        ):
            raise ValueError("template alpha must contain finite real values")
        self._templates[normalized_name] = KnmTemplate(
            name=normalized_name,
            knm=template.knm,
            alpha=template.alpha,
            description=template.description,
        )

    def get(self, name: str) -> KnmTemplate:
        """Retrieve a template by name. Raises KeyError if not found."""
        if not isinstance(name, str) or not name.strip():
            raise KeyError(f"template name must be a non-empty string, got {name!r}")
        normalized = name.strip()
        try:
            return self._templates[normalized]
        except KeyError:
            available = ", ".join(sorted(self._templates)) or "(none)"
            msg = f"Unknown template {name!r}; available: {available}"
            raise KeyError(msg) from None

    def list_names(self) -> list[str]:
        """Return all registered template names."""
        return list(self._templates.keys())
