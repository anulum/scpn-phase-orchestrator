# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray


@dataclass
class LockSignature:
    source_layer: int
    target_layer: int
    plv: float
    mean_lag: float


@dataclass
class LayerState:
    R: float
    psi: float
    lock_signatures: dict[str, LockSignature] = field(default_factory=dict)


@dataclass
class UPDEState:
    layers: list[LayerState]
    cross_layer_alignment: NDArray
    stability_proxy: float
    regime_id: str
