# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GeometryConstraint",
    "SymmetryConstraint",
    "NonNegativeConstraint",
    "project_knm",
]


class GeometryConstraint(ABC):
    @abstractmethod
    def project(self, knm: NDArray) -> NDArray: ...


class SymmetryConstraint(GeometryConstraint):
    def project(self, knm: NDArray) -> NDArray:
        result: NDArray = 0.5 * (knm + knm.T)
        return result


class NonNegativeConstraint(GeometryConstraint):
    def project(self, knm: NDArray) -> NDArray:
        result: NDArray = np.maximum(knm, 0.0)
        return result


def project_knm(knm: NDArray, constraints: list[GeometryConstraint]) -> NDArray:
    result = knm.copy()
    for c in constraints:
        result = c.project(result)
    np.fill_diagonal(result, 0.0)
    return result
