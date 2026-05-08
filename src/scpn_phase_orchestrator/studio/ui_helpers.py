# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio pure UI helpers

"""Pure helper layer for the SPO Studio Streamlit surface."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding import validate_binding_spec
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.server import SimulationState
from scpn_phase_orchestrator.studio.workflow import (
    BindingProposal,
    ExportManifest,
    ImportedSourceSummary,
    RuntimeSnapshot,
    StudioProjectState,
)

__all__ = [
    "StudioKnobState",
    "StudioReplayResult",
    "apply_knob_update",
    "binding_spec_project_state",
    "build_export_manifests",
    "build_layer_table",
    "build_oscillator_edit_artifact",
    "build_oscillator_table",
    "build_regime_chart_payload",
    "build_runtime_snapshot",
    "build_series_chart_payload",
    "disabled_export_reasons",
    "discover_domainpacks",
    "run_binding_spec_replay",
]


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
        """Return a JSON-safe knob record."""
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
    export_manifests: tuple[ExportManifest, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe replay audit record."""
        return {
            "project": self.project_state.to_audit_record(),
            "r_history": list(self.r_history),
            "regime_history": list(self.regime_history),
            "layer_table": list(self.layer_table),
            "oscillator_table": list(self.oscillator_table),
            "exports": [
                manifest.to_audit_record() for manifest in self.export_manifests
            ],
        }


def discover_domainpacks(domainpack_dir: Path) -> tuple[str, ...]:
    """Return domainpack names containing a binding spec."""
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
    """Return validated knobs after a UI edit."""
    return StudioKnobState(
        K=knobs.K if K is None else K,
        alpha=knobs.alpha if alpha is None else alpha,
        zeta=knobs.zeta if zeta is None else zeta,
        Psi=knobs.Psi if Psi is None else Psi,
    )


def build_series_chart_payload(
    label: str,
    values: Sequence[float],
) -> list[dict[str, float | int]]:
    """Return dense chart rows for a scalar time-series."""
    _require_non_empty_text(label, "label")
    return [
        {"step": index, label: _finite_number(value, label)}
        for index, value in enumerate(values, 1)
    ]


def build_regime_chart_payload(regimes: Sequence[str]) -> list[dict[str, object]]:
    """Return deterministic chart rows for regime timelines."""
    regime_levels = {
        "critical": 0.0,
        "degraded": 1.0,
        "recovery": 1.5,
        "nominal": 2.0,
    }
    rows: list[dict[str, object]] = []
    for index, regime in enumerate(regimes, 1):
        regime_text = _require_non_empty_text(regime, "regime")
        rows.append(
            {
                "step": index,
                "regime": regime_text,
                "regime_level": regime_levels.get(regime_text, 0.0),
            }
        )
    return rows


def build_layer_table(spec: BindingSpec) -> tuple[dict[str, object], ...]:
    """Return editable layer rows for the Studio oscillator canvas."""
    return tuple(
        {
            "index": int(layer.index),
            "name": layer.name,
            "oscillator_count": len(layer.oscillator_ids),
            "family": layer.family or "",
            "omega_count": len(layer.omegas or ()),
        }
        for layer in sorted(spec.layers, key=lambda item: item.index)
    )


def build_oscillator_table(spec: BindingSpec) -> tuple[dict[str, object], ...]:
    """Return oscillator rows suitable for Streamlit data editing."""
    family_channels = {
        family_name: family.channel
        for family_name, family in spec.oscillator_families.items()
    }
    rows: list[dict[str, object]] = []
    for layer in sorted(spec.layers, key=lambda item: item.index):
        channel = family_channels.get(layer.family or "", "")
        for oscillator_id in layer.oscillator_ids:
            rows.append(
                {
                    "layer": layer.name,
                    "layer_index": int(layer.index),
                    "oscillator_id": oscillator_id,
                    "family": layer.family or "",
                    "channel": channel,
                }
            )
    return tuple(rows)


def build_runtime_snapshot(
    *,
    final_state: Mapping[str, object],
    knobs: StudioKnobState,
    hierarchy_watermarks: Mapping[str, int] | None = None,
    replay_status: str = "not_started",
) -> RuntimeSnapshot:
    """Build a workflow runtime snapshot from a simulation state dict."""
    layers = _layer_metrics(final_state.get("layers", ()))
    return RuntimeSnapshot(
        R=_finite_number(final_state.get("R_global", 0.0), "R_global"),
        Psi=knobs.Psi,
        K=knobs.K,
        alpha=knobs.alpha,
        zeta=knobs.zeta,
        regime=_require_non_empty_text(final_state.get("regime", "unknown"), "regime"),
        layer_metrics=layers,
        hierarchy_watermarks=dict(hierarchy_watermarks or {}),
        replay_status=replay_status,
    )


def binding_spec_project_state(
    *,
    project_name: str,
    spec_path: Path,
    knobs: StudioKnobState,
    runtime: RuntimeSnapshot,
) -> StudioProjectState:
    """Create a Studio project state from an existing binding spec file."""
    yaml_text = spec_path.read_text(encoding="utf-8")
    spec = load_binding_spec(spec_path)
    validation_errors = tuple(validate_binding_spec(spec))
    source = ImportedSourceSummary.from_payload(
        source_kind="binding_spec_yaml",
        payload=yaml_text.encode("utf-8"),
        channel_count=max(1, len(spec.used_channels())),
        sample_count=sum(len(layer.oscillator_ids) for layer in spec.layers),
    )
    binding = BindingProposal(
        yaml_text=yaml_text,
        validation_errors=validation_errors,
        inferred_channels=tuple(sorted(spec.used_channels())),
        confidence_factors={
            "validator_acceptance": 1.0 if not validation_errors else 0.0,
            "layer_coverage": 1.0 if spec.layers else 0.0,
        },
        provenance={
            "source_path": str(spec_path),
            "knobs": knobs.to_audit_record(),
            "validator": "validate_binding_spec",
        },
    )
    exports = build_export_manifests(
        project_name=project_name,
        binding_yaml=yaml_text,
        audit_payload={
            "project_name": project_name,
            "runtime": runtime.to_audit_record(),
        },
        validation_errors=validation_errors,
    )
    return StudioProjectState(
        project_name=project_name,
        source=source,
        binding=binding,
        runtime=runtime,
        exports=exports,
        metadata={
            "domainpack": project_name,
            "safety": "local_replay_only",
        },
    )


def build_export_manifests(
    *,
    project_name: str,
    binding_yaml: str,
    audit_payload: Mapping[str, object],
    validation_errors: Sequence[str],
) -> tuple[ExportManifest, ...]:
    """Build review-only export manifests for Studio."""
    deploy_warnings = disabled_export_reasons(validation_errors)
    audit_export_payload = {
        **dict(audit_payload),
        "enabled": not deploy_warnings,
        "disabled_reasons": list(deploy_warnings),
    }
    audit_json = json.dumps(audit_export_payload, sort_keys=True, indent=2)
    docker_payload = json.dumps(
        {
            "project_name": project_name,
            "image": "scpn-phase-orchestrator:local",
            "command": "spo run binding_spec.yaml --audit audit.jsonl",
            "enabled": not deploy_warnings,
            "disabled_reasons": list(deploy_warnings),
        },
        sort_keys=True,
        indent=2,
    )
    wasm_payload = json.dumps(
        {
            "project_name": project_name,
            "target": "wasm_review_manifest",
            "enabled": not deploy_warnings,
            "disabled_reasons": list(deploy_warnings),
        },
        sort_keys=True,
        indent=2,
    )
    return (
        ExportManifest.review_artifact(
            target_kind="binding_spec",
            file_name="binding_spec.yaml",
            payload=binding_yaml,
            command="spo run binding_spec.yaml --audit audit.jsonl",
            warnings=deploy_warnings,
        ),
        ExportManifest.review_artifact(
            target_kind="audit_summary",
            file_name="spo_studio_audit.json",
            payload=audit_json,
            command="spo audit summary spo_studio_audit.json",
            warnings=deploy_warnings,
        ),
        ExportManifest.review_artifact(
            target_kind="docker_manifest",
            file_name="docker_manifest.json",
            payload=docker_payload,
            command="docker compose config",
            warnings=deploy_warnings,
        ),
        ExportManifest.review_artifact(
            target_kind="wasm_manifest",
            file_name="wasm_manifest.json",
            payload=wasm_payload,
            command="spo export wasm --manifest wasm_manifest.json",
            warnings=deploy_warnings,
        ),
    )


def build_oscillator_edit_artifact(
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
) -> ExportManifest:
    """Build a review artefact from edited oscillator table rows."""
    before = _normalise_table_rows(before_rows, "before_rows")
    after = _normalise_table_rows(after_rows, "after_rows")
    payload = json.dumps(
        {
            "artifact": "oscillator_edit_review",
            "changed": before != after,
            "row_count_before": len(before),
            "row_count_after": len(after),
            "rows_before": before,
            "rows_after": after,
        },
        sort_keys=True,
        indent=2,
    )
    return ExportManifest.review_artifact(
        target_kind="oscillator_edit_review",
        file_name="oscillator_edit_review.json",
        payload=payload,
        command="review oscillator_edit_review.json before updating binding_spec.yaml",
    )


def disabled_export_reasons(validation_errors: Sequence[str]) -> tuple[str, ...]:
    """Return reasons deploy-like exports must stay review-only."""
    errors = tuple(str(error) for error in validation_errors)
    if not errors:
        return ()
    return (
        "binding validation must pass before deploy manifests are enabled",
        *errors,
    )


def run_binding_spec_replay(
    spec_path: Path,
    *,
    steps: int,
    knobs: StudioKnobState,
) -> StudioReplayResult:
    """Run a local binding-spec replay and return Studio-ready payloads."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    spec = load_binding_spec(spec_path)
    sim = SimulationState(spec)
    _apply_replay_knobs(sim, knobs)

    r_history: list[float] = []
    regime_history: list[str] = []
    final_state: Mapping[str, object] = sim.snapshot()
    for _ in range(steps):
        final_state = sim.step()
        r_history.append(_finite_number(final_state["R_global"], "R_global"))
        regime_history.append(
            _require_non_empty_text(final_state["regime"], "regime")
        )

    runtime = build_runtime_snapshot(
        final_state=final_state,
        knobs=knobs,
        replay_status="completed",
    )
    project_state = binding_spec_project_state(
        project_name=spec.name,
        spec_path=spec_path,
        knobs=knobs,
        runtime=runtime,
    )
    return StudioReplayResult(
        project_state=project_state,
        r_history=tuple(r_history),
        regime_history=tuple(regime_history),
        layer_table=build_layer_table(spec),
        oscillator_table=build_oscillator_table(spec),
        export_manifests=project_state.exports,
    )


def _apply_replay_knobs(sim: SimulationState, knobs: StudioKnobState) -> None:
    scaled_knm = np.asarray(sim.coupling.knm, dtype=np.float64) * knobs.K
    alpha = np.asarray(sim.coupling.alpha, dtype=np.float64).copy()
    if knobs.alpha:
        alpha = alpha + knobs.alpha
        np.fill_diagonal(alpha, 0.0)
    knm_r = None
    if sim.coupling.knm_r is not None:
        knm_r = np.asarray(sim.coupling.knm_r, dtype=np.float64) * knobs.K
    sim.coupling = CouplingState(
        knm=scaled_knm,
        alpha=alpha,
        active_template=f"{sim.coupling.active_template}:studio_replay",
        knm_r=knm_r,
    )
    if knobs.zeta or knobs.Psi:
        sim.omegas = np.asarray(sim.omegas, dtype=np.float64) + knobs.zeta * knobs.Psi


def _layer_metrics(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    rows: list[tuple[str, float]] = []
    for index, layer in enumerate(value):
        if not isinstance(layer, Mapping):
            continue
        name = _require_non_empty_text(layer.get("name", f"layer_{index}"), "layer")
        rows.append((name, _finite_number(layer.get("R", 0.0), "layer.R")))
    return tuple(rows)


def _normalise_table_rows(
    rows: Sequence[Mapping[str, object]],
    field_name: str,
) -> list[dict[str, object]]:
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise ValueError(f"{field_name} must be a sequence of mappings")
    normalised: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{field_name}[{index}] must be a mapping")
        safe_row: dict[str, object] = {}
        for key, value in row.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name}[{index}] contains a non-string key")
            if value is None or isinstance(value, str | int | float | bool):
                if isinstance(value, float) and not isfinite(value):
                    raise ValueError(f"{field_name}[{index}].{key} must be finite")
                safe_row[key] = value
            else:
                raise ValueError(f"{field_name}[{index}].{key} must be JSON-safe")
        normalised.append(safe_row)
    return normalised


def _finite_range(value: object, name: str, *, low: float, high: float) -> float:
    number = _finite_number(value, name)
    if not low <= number <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return number


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_non_empty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
