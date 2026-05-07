# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Morphogenetic topology field supervisor

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "MorphogeneticFieldPolicy",
    "MorphogeneticFieldResult",
    "MorphogeneticFieldSnapshot",
    "MorphogeneticFieldSVG",
    "MorphogeneticFieldState",
    "MorphogeneticTopologySupervisor",
    "build_morphogenetic_field_snapshot",
    "render_morphogenetic_field_svg",
]


@dataclass(frozen=True)
class MorphogeneticFieldPolicy:
    """Knobs for reaction-diffusion-style topology field evolution."""

    growth_rate: float = 0.2
    shrink_rate: float = 0.15
    diffusion_rate: float = 0.1
    coherence_target: float = 0.75
    max_delta: float = 0.05
    max_coupling: float = 10.0

    def __post_init__(self) -> None:
        _require_unit_interval(self.growth_rate, "growth_rate")
        _require_unit_interval(self.shrink_rate, "shrink_rate")
        _require_unit_interval(self.diffusion_rate, "diffusion_rate")
        _require_unit_interval(self.coherence_target, "coherence_target")
        _require_non_negative(self.max_delta, "max_delta")
        _require_non_negative(self.max_coupling, "max_coupling")


@dataclass(frozen=True)
class MorphogeneticFieldState:
    """Persistent topology field carried between supervisor ticks."""

    field: FloatArray

    def to_audit_snapshot(self) -> dict[str, object]:
        """Return compact, serialisable field statistics for audit logs."""
        return {
            "shape": list(self.field.shape),
            "mean": float(np.mean(self.field)),
            "minimum": float(np.min(self.field)),
            "maximum": float(np.max(self.field)),
            "l2_norm": float(np.linalg.norm(self.field)),
        }


@dataclass(frozen=True)
class MorphogeneticFieldResult:
    """Output of one morphogenetic topology field step."""

    knm: FloatArray
    field_state: MorphogeneticFieldState
    grown_edges: tuple[tuple[int, int, float], ...]
    shrunk_edges: tuple[tuple[int, int, float], ...]
    delta_norm: float
    global_coherence: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable topology-field audit payload."""
        return {
            "global_coherence": self.global_coherence,
            "delta_norm": self.delta_norm,
            "grown_edges": [
                {"source": src, "target": dst, "delta": delta}
                for src, dst, delta in self.grown_edges
            ],
            "shrunk_edges": [
                {"source": src, "target": dst, "delta": delta}
                for src, dst, delta in self.shrunk_edges
            ],
            "field": self.field_state.to_audit_snapshot(),
        }


@dataclass(frozen=True)
class MorphogeneticFieldSnapshot:
    """Compact visual snapshot of a morphogenetic topology field."""

    shape: tuple[int, int]
    mean: float
    minimum: float
    maximum: float
    l2_norm: float
    heatmap_rows: tuple[str, ...]
    top_edges: tuple[tuple[int, int, float], ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe field snapshot for docs, reports, and audits."""
        return {
            "shape": list(self.shape),
            "mean": self.mean,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "l2_norm": self.l2_norm,
            "heatmap_rows": list(self.heatmap_rows),
            "top_edges": [
                {"source": src, "target": dst, "weight": weight}
                for src, dst, weight in self.top_edges
            ],
        }


@dataclass(frozen=True)
class MorphogeneticFieldSVG:
    """Dependency-free SVG rendering of a morphogenetic topology field."""

    svg: str
    width: int
    height: int
    snapshot: MorphogeneticFieldSnapshot

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe SVG artefact record for review tooling."""
        return {
            "format": "svg",
            "width": self.width,
            "height": self.height,
            "snapshot": self.snapshot.to_audit_record(),
            "svg": self.svg,
        }


class MorphogeneticTopologySupervisor:
    """Grow or shrink pairwise topology from a persistent coherence field."""

    def __init__(self, policy: MorphogeneticFieldPolicy | None = None) -> None:
        self.policy = policy or MorphogeneticFieldPolicy()
        self.last_result: MorphogeneticFieldResult | None = None

    def step(
        self,
        phases: FloatArray,
        knm: FloatArray,
        field_state: MorphogeneticFieldState | None = None,
    ) -> MorphogeneticFieldResult:
        """Evolve the topology field and return the next pairwise coupling."""
        phases_arr = _validate_phases(phases)
        knm_arr = _validate_knm(knm, phases_arr.size)
        field = (
            _initial_field(knm_arr, self.policy.max_coupling)
            if field_state is None
            else _validate_field(field_state.field, phases_arr.size)
        )
        alignment = _pairwise_phase_alignment(phases_arr)
        diffused = _incident_diffusion(field)
        reaction = alignment - self.policy.coherence_target
        next_field = np.clip(
            field
            + self.policy.diffusion_rate * (diffused - field)
            + self.policy.growth_rate * np.maximum(reaction, 0.0)
            - self.policy.shrink_rate * np.maximum(-reaction, 0.0),
            0.0,
            1.0,
        )
        np.fill_diagonal(next_field, 0.0)
        field_delta = next_field - field
        coupling_delta = self.policy.max_delta * field_delta
        next_knm = np.clip(knm_arr + coupling_delta, 0.0, self.policy.max_coupling)
        np.fill_diagonal(next_knm, 0.0)
        result = MorphogeneticFieldResult(
            knm=next_knm,
            field_state=MorphogeneticFieldState(next_field),
            grown_edges=_edge_deltas(coupling_delta, positive=True),
            shrunk_edges=_edge_deltas(coupling_delta, positive=False),
            delta_norm=float(np.linalg.norm(next_knm - knm_arr)),
            global_coherence=_order_parameter(phases_arr),
        )
        self.last_result = result
        return result

    def reset(self) -> None:
        """Clear the cached result; caller owns persistent field snapshots."""
        self.last_result = None


def build_morphogenetic_field_snapshot(
    field_state: MorphogeneticFieldState | MorphogeneticFieldResult,
    *,
    top_k: int = 5,
    palette: str = " .:-=+*#%@",
) -> MorphogeneticFieldSnapshot:
    """Build a compact visual snapshot for a topology field.

    The snapshot is dependency-free and audit-oriented: it exposes summary
    statistics, ASCII heatmap rows, and the strongest non-diagonal field edges.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    _require_non_empty(palette, "palette")
    source_state = (
        field_state.field_state
        if isinstance(field_state, MorphogeneticFieldResult)
        else field_state
    )
    field = _validate_square_field(source_state.field)
    return MorphogeneticFieldSnapshot(
        shape=(int(field.shape[0]), int(field.shape[1])),
        mean=float(np.mean(field)),
        minimum=float(np.min(field)),
        maximum=float(np.max(field)),
        l2_norm=float(np.linalg.norm(field)),
        heatmap_rows=_field_heatmap_rows(field, palette),
        top_edges=_top_field_edges(field, top_k),
    )


def render_morphogenetic_field_svg(
    field_state: MorphogeneticFieldState | MorphogeneticFieldResult,
    *,
    top_k: int = 5,
    cell_size: int = 28,
    title: str = "Morphogenetic topology field",
) -> MorphogeneticFieldSVG:
    """Render a dependency-free SVG heatmap for a topology field.

    The renderer is passive: it produces a review artefact from an already
    computed field and does not mutate policy, coupling, or actuation state.
    """
    if cell_size < 8:
        raise ValueError("cell_size must be at least 8")
    _require_non_empty(title, "title")
    source_state = (
        field_state.field_state
        if isinstance(field_state, MorphogeneticFieldResult)
        else field_state
    )
    field = _validate_square_field(source_state.field)
    snapshot = build_morphogenetic_field_snapshot(source_state, top_k=top_k)
    n = int(field.shape[0])
    label_band = 84
    legend_band = 24
    width = cell_size * n
    height = label_band + cell_size * n + legend_band
    escaped_title = escape(title, quote=True)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
            f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        ),
        f"<title>{escaped_title}</title>",
        f'<rect width="{width}" height="{height}" fill="#fbf7ef"/>',
        (
            f'<text x="0" y="18" font-family="monospace" font-size="13" '
            f'fill="#24302f">{escaped_title}</text>'
        ),
        (
            f'<text x="0" y="38" font-family="monospace" font-size="11" '
            f'fill="#5f6b64">mean={snapshot.mean:.4f} '
            f"max={snapshot.maximum:.4f} l2={snapshot.l2_norm:.4f}</text>"
        ),
    ]
    y0 = label_band
    for row_idx, row in enumerate(field):
        for col_idx, value in enumerate(row):
            opacity = 0.10 + 0.85 * float(value)
            x = col_idx * cell_size
            y = y0 + row_idx * cell_size
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="#0e7c66" fill-opacity="{opacity:.4f}" '
                f'stroke="#f3eadc" stroke-width="1"/>'
            )
    for src, dst, weight in snapshot.top_edges:
        label_x = dst * cell_size + cell_size / 2.0
        label_y = y0 + src * cell_size + cell_size / 2.0
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y + 3.5:.1f}" text-anchor="middle" '
            f'font-family="monospace" font-size="{max(8, cell_size // 3)}" '
            f'fill="#10221f">{src}->{dst}:{weight:.2f}</text>'
        )
    parts.append(
        f'<text x="0" y="{height - 7}" font-family="monospace" font-size="10" '
        f'fill="#5f6b64">top_edges={len(snapshot.top_edges)} '
        f"shape={snapshot.shape[0]}x{snapshot.shape[1]}</text>"
    )
    parts.append("</svg>")
    return MorphogeneticFieldSVG(
        svg="".join(parts),
        width=width,
        height=height,
        snapshot=snapshot,
    )


def _validate_phases(phases: FloatArray) -> FloatArray:
    arr = np.asarray(phases, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("phases must be a one-dimensional array")
    if arr.size < 1:
        raise ValueError("phases must contain at least one oscillator")
    if not np.all(np.isfinite(arr)):
        raise ValueError("phases must be finite")
    return arr


def _validate_knm(knm: FloatArray, n: int) -> FloatArray:
    arr = np.asarray(knm, dtype=np.float64)
    if arr.shape != (n, n):
        raise ValueError(f"knm must have shape ({n}, {n})")
    if not np.all(np.isfinite(arr)):
        raise ValueError("knm must be finite")
    if np.any(arr < 0.0):
        raise ValueError("knm must be non-negative")
    return arr


def _validate_field(field: FloatArray, n: int) -> FloatArray:
    arr = np.asarray(field, dtype=np.float64)
    if arr.shape != (n, n):
        raise ValueError(f"field must have shape ({n}, {n})")
    if not np.all(np.isfinite(arr)):
        raise ValueError("field must be finite")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("field values must be in [0, 1]")
    return arr


def _validate_square_field(field: FloatArray) -> FloatArray:
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("field must be a square matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError("field must be finite")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("field values must be in [0, 1]")
    return arr


def _initial_field(knm: FloatArray, max_coupling: float) -> FloatArray:
    if max_coupling <= 0.0:
        return np.zeros_like(knm)
    field = np.clip(knm / max_coupling, 0.0, 1.0)
    np.fill_diagonal(field, 0.0)
    return field


def _pairwise_phase_alignment(phases: FloatArray) -> FloatArray:
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    alignment = 0.5 * (np.cos(diffs) + 1.0)
    np.fill_diagonal(alignment, 0.0)
    result: FloatArray = alignment
    return result


def _incident_diffusion(field: FloatArray) -> FloatArray:
    n = field.shape[0]
    if n == 1:
        return np.zeros_like(field)
    row_mean = np.sum(field, axis=1) / max(1, n - 1)
    col_mean = np.sum(field, axis=0) / max(1, n - 1)
    diffused = 0.25 * (
        row_mean[:, np.newaxis]
        + row_mean[np.newaxis, :]
        + col_mean[:, np.newaxis]
        + col_mean[np.newaxis, :]
    )
    np.fill_diagonal(diffused, 0.0)
    result: FloatArray = diffused
    return result


def _edge_deltas(
    delta: FloatArray,
    *,
    positive: bool,
) -> tuple[tuple[int, int, float], ...]:
    mask = delta > 1e-12 if positive else delta < -1e-12
    edges = []
    for src, dst in np.argwhere(mask):
        if src == dst:
            continue
        edges.append((int(src), int(dst), float(delta[src, dst])))
    return tuple(edges)


def _order_parameter(phases: FloatArray) -> float:
    return float(np.abs(np.mean(np.exp(1j * phases))))


def _field_heatmap_rows(field: FloatArray, palette: str) -> tuple[str, ...]:
    if len(palette) == 1:
        return tuple(palette * field.shape[1] for _ in range(field.shape[0]))
    scale = len(palette) - 1
    rows = []
    for row in field:
        rows.append("".join(palette[int(round(float(value) * scale))] for value in row))
    return tuple(rows)


def _top_field_edges(
    field: FloatArray,
    top_k: int,
) -> tuple[tuple[int, int, float], ...]:
    edges: list[tuple[int, int, float]] = []
    for src, dst in np.argwhere(field > 0.0):
        if src == dst:
            continue
        edges.append((int(src), int(dst), float(field[src, dst])))
    edges.sort(key=lambda item: (-item[2], item[0], item[1]))
    return tuple(edges[:top_k])


def _require_unit_interval(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _require_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
