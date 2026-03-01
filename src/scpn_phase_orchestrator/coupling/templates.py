# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass(frozen=True)
class KnmTemplate:
    name: str
    knm: NDArray
    alpha: NDArray
    description: str


class KnmTemplateSet:
    def __init__(self):
        self._templates: dict[str, KnmTemplate] = {}

    def add(self, template: KnmTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> KnmTemplate:
        return self._templates[name]

    def list_names(self) -> list[str]:
        return list(self._templates.keys())
