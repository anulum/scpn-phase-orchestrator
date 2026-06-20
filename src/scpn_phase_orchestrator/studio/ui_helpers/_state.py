# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio knob and replay-result state

"""Studio knob and replay-result dataclasses plus domainpack discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from scpn_phase_orchestrator.studio.workflow import (
    ExportManifest,
    StudioProjectState,
)

from ._shared import _finite_range


@dataclass(frozen=True, slots=True)
class StudioKnobState:
    """Review-only knob state used by Studio replay controls."""

    K: float = 1.0
    alpha: float = 0.0
    zeta: float = 0.0
    Psi: float = 0.0

    def __post_init__(self) -> None:
        _finite_range(self.K, "K", low=0.1, high=10.0)
        _finite_range(self.alpha, "alpha", low=0.0, high=5.0)
        _finite_range(self.zeta, "zeta", low=0.0, high=5.0)
        _finite_range(self.Psi, "Psi", low=0.0, high=10.0)

    def to_audit_record(self) -> dict[str, float]:
        """Return a JSON-safe knob record.

        Returns
        -------
        dict[str, float]
            A JSON-safe knob record.
        """
        return {
            "K": float(self.K),
            "alpha": float(self.alpha),
            "zeta": float(self.zeta),
            "Psi": float(self.Psi),
        }


@dataclass(frozen=True, slots=True)
class StudioReplayResult:
    """Replay output rendered by SPO Studio."""

    project_state: StudioProjectState
    r_history: tuple[float, ...]
    regime_history: tuple[str, ...]
    layer_table: tuple[dict[str, object], ...]
    oscillator_table: tuple[dict[str, object], ...]
    canvas_graph: Mapping[str, object]
    connector_plan: Mapping[str, object]
    export_manifests: tuple[ExportManifest, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe replay audit record.

        Returns
        -------
        dict[str, object]
            A JSON-safe replay audit record.
        """
        return {
            "project": self.project_state.to_audit_record(),
            "r_history": list(self.r_history),
            "regime_history": list(self.regime_history),
            "layer_table": list(self.layer_table),
            "oscillator_table": list(self.oscillator_table),
            "canvas_graph": dict(self.canvas_graph),
            "connector_plan": dict(self.connector_plan),
            "exports": [
                manifest.to_audit_record() for manifest in self.export_manifests
            ],
        }


def discover_domainpacks(domainpack_dir: Path) -> tuple[str, ...]:
    """Return domainpack names containing a binding spec.

    Parameters
    ----------
    domainpack_dir : Path
        Directory containing domainpacks.

    Returns
    -------
    tuple[str, ...]
        Domainpack names containing a binding spec.
    """
    if not domainpack_dir.exists():
        return ()
    return tuple(
        sorted(
            path.name
            for path in domainpack_dir.iterdir()
            if path.is_dir() and (path / "binding_spec.yaml").exists()
        )
    )


def apply_knob_update(
    knobs: StudioKnobState,
    *,
    K: float | None = None,
    alpha: float | None = None,
    zeta: float | None = None,
    Psi: float | None = None,
) -> StudioKnobState:
    """Return validated knobs after a UI edit.

    Parameters
    ----------
    knobs : StudioKnobState
        The Studio knob state.
    K : float | None
        Coupling-strength knob value, or ``None``.
    alpha : float | None
        Phase-lag knob value, or ``None``.
    zeta : float | None
        Drive-strength knob value, or ``None``.
    Psi : float | None
        Drive reference-phase knob value, or ``None``.

    Returns
    -------
    StudioKnobState
        Validated knobs after a UI edit.
    """
    return StudioKnobState(
        K=knobs.K if K is None else K,
        alpha=knobs.alpha if alpha is None else alpha,
        zeta=knobs.zeta if zeta is None else zeta,
        Psi=knobs.Psi if Psi is None else Psi,
    )
