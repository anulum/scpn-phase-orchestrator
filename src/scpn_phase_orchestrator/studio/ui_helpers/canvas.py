# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio canvas graph and edit builders

"""Canvas graph, edit, layout, topology, and binding-rewrite builders."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from pathlib import Path

from scpn_phase_orchestrator.binding import validate_binding_spec
from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.studio.workflow import (
    ExportManifest,
)

from ._shared import (
    _canvas_channel_id,
    _canvas_graph_count,
    _finite_number,
    _finite_range,
    _is_sha256_digest,
    _require_non_empty_text,
)
from ._state import StudioReplayResult


def build_canvas_graph(spec: BindingSpec) -> dict[str, object]:
    """Return a deterministic layer/coupling graph for Studio canvas review.

    Parameters
    ----------
    spec : BindingSpec
        The binding specification.

    Returns
    -------
    dict[str, object]
        A deterministic layer/coupling graph for Studio canvas review.
    """
    family_channels = {
        family_name: family.channel
        for family_name, family in spec.oscillator_families.items()
    }
    channel_order = {
        channel: index for index, channel in enumerate(sorted(spec.used_channels()))
    }
    channels = tuple(sorted(channel_order))
    nodes: list[dict[str, object]] = []
    for layer in sorted(spec.layers, key=lambda item: item.index):
        family = layer.family or ""
        channel = family_channels.get(family, "")
        nodes.append(
            {
                "id": f"layer_{layer.index}",
                "label": layer.name,
                "kind": "layer",
                "layer_index": int(layer.index),
                "family": family,
                "channel": channel,
                "oscillator_count": len(layer.oscillator_ids),
                "x": float(layer.index) * 220.0,
                "y": float(channel_order.get(channel, 0)) * 140.0,
            }
        )
    for index, channel in enumerate(channels):
        nodes.append(
            {
                "id": _canvas_channel_id(channel),
                "label": channel,
                "kind": "channel",
                "channel": channel,
                "layer_index": -1,
                "family": "",
                "oscillator_count": 0,
                "x": float(index) * 220.0,
                "y": 420.0,
            }
        )

    edges = [
        {
            "id": f"cross_channel_{index}",
            "source": _canvas_channel_id(coupling.source),
            "target": _canvas_channel_id(coupling.target),
            "kind": "cross_channel_coupling",
            "source_channel": coupling.source,
            "target_channel": coupling.target,
            "strength": float(coupling.strength),
            "mode": coupling.mode,
            "template": coupling.template or "",
        }
        for index, coupling in enumerate(spec.cross_channel_couplings, 1)
        if coupling.source in channel_order and coupling.target in channel_order
    ]
    return {
        "canvas_kind": "layer_coupling_graph",
        "node_count": len(nodes),
        "layer_count": len(spec.layers),
        "channel_count": len(channels),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def build_canvas_edit_artifact(
    before_graph: Mapping[str, object],
    after_graph: Mapping[str, object],
) -> ExportManifest:
    """Build a review artefact from edited Studio canvas graph rows.

    Parameters
    ----------
    before_graph : Mapping[str, object]
        The canvas graph before the change.
    after_graph : Mapping[str, object]
        The edited canvas graph after the change.

    Returns
    -------
    ExportManifest
        A review artefact from edited Studio canvas graph rows.
    """
    before_nodes, before_edges = _normalise_canvas_graph(before_graph, "before_graph")
    after_nodes, after_edges = _normalise_canvas_graph(after_graph, "after_graph")
    payload = json.dumps(
        {
            "artifact": "canvas_edit_review",
            "changed": (before_nodes, before_edges) != (after_nodes, after_edges),
            "node_count_before": len(before_nodes),
            "node_count_after": len(after_nodes),
            "edge_count_before": len(before_edges),
            "edge_count_after": len(after_edges),
            "nodes_before": before_nodes,
            "nodes_after": after_nodes,
            "edges_before": before_edges,
            "edges_after": after_edges,
        },
        sort_keys=True,
        indent=2,
    )
    return ExportManifest.review_artifact(
        target_kind="canvas_edit_review",
        file_name="canvas_edit_review.json",
        payload=payload,
        command="review canvas_edit_review.json before updating binding_spec.yaml",
    )


def build_canvas_layout_manifest(
    *,
    project_name: str,
    graph: Mapping[str, object],
) -> ExportManifest:
    """Build a deterministic canvas layout manifest from node positions.

    Parameters
    ----------
    project_name : str
        Name of the project.
    graph : Mapping[str, object]
        The canvas graph mapping.

    Returns
    -------
    ExportManifest
        A deterministic canvas layout manifest from node positions.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    nodes, edges = _normalise_canvas_graph(graph, "canvas_layout")
    positions = []
    for node in sorted(nodes, key=lambda item: str(item.get("id", ""))):
        try:
            node_id = _require_non_empty_text(node.get("id"), "canvas layout id")
            kind = _require_non_empty_text(node.get("kind"), "canvas layout kind")
            label = _require_non_empty_text(node.get("label"), "canvas layout label")
            x = _finite_number(node.get("x"), "canvas layout x")
            y = _finite_number(node.get("y"), "canvas layout y")
        except ValueError as exc:
            raise ValueError(f"canvas layout node is invalid: {exc}") from exc
        positions.append({"id": node_id, "kind": kind, "label": label, "x": x, "y": y})
    payload = json.dumps(
        {
            "manifest_kind": "canvas_layout_manifest",
            "project_name": _require_non_empty_text(project_name, "project_name"),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "positions": positions,
        },
        sort_keys=True,
        indent=2,
    )
    return ExportManifest.review_artifact(
        target_kind="canvas_layout_manifest",
        file_name="canvas_layout_manifest.json",
        payload=payload,
        command="review canvas_layout_manifest.json before restoring Studio layout",
    )


def build_canvas_topology_patch(
    *,
    project_name: str,
    before_graph: Mapping[str, object],
    after_graph: Mapping[str, object],
) -> ExportManifest:
    """Build a review patch for persistent Studio topology edits.

    Parameters
    ----------
    project_name : str
        Name of the project.
    before_graph : Mapping[str, object]
        The canvas graph before the change.
    after_graph : Mapping[str, object]
        The edited canvas graph after the change.

    Returns
    -------
    ExportManifest
        A review patch for persistent Studio topology edits.
    """
    before_nodes, before_edges = _normalise_canvas_graph(before_graph, "before_graph")
    after_nodes, after_edges = _normalise_canvas_graph(after_graph, "after_graph")
    _validate_canvas_edge_endpoints(after_nodes, after_edges)

    node_changes = _canvas_item_changes(
        before_nodes,
        after_nodes,
        fields=("id", "kind", "label", "x", "y"),
    )
    edge_changes = _canvas_item_changes(
        before_edges,
        after_edges,
        fields=("id", "kind", "source", "target"),
    )
    changed = any(node_changes[key] or edge_changes[key] for key in node_changes)
    payload = json.dumps(
        {
            "patch_kind": "canvas_topology_patch",
            "project_name": _require_non_empty_text(project_name, "project_name"),
            "status": "review_required",
            "changed": changed,
            "node_count_before": len(before_nodes),
            "node_count_after": len(after_nodes),
            "edge_count_before": len(before_edges),
            "edge_count_after": len(after_edges),
            "node_changes": node_changes,
            "edge_changes": edge_changes,
            "safety": {
                "binding_spec_rewritten": False,
                "actuation_permitted": False,
            },
        },
        sort_keys=True,
        indent=2,
    )
    return ExportManifest.review_artifact(
        target_kind="canvas_topology_patch",
        file_name="canvas_topology_patch.json",
        payload=payload,
        command="review canvas_topology_patch.json before rewriting binding_spec.yaml",
    )


def build_canvas_interaction_state(
    *,
    canvas_artifact: ExportManifest,
    canvas_layout: ExportManifest,
    canvas_patch: ExportManifest,
    canvas_rewrite: Mapping[str, object],
    operator_signoff: bool,
) -> dict[str, object]:
    """Summarise Canvas browser controls for deterministic operator feedback.

    Parameters
    ----------
    canvas_artifact : ExportManifest
        The canvas edit artefact manifest.
    canvas_layout : ExportManifest
        The canvas layout manifest.
    canvas_patch : ExportManifest
        The canvas topology patch manifest.
    canvas_rewrite : Mapping[str, object]
        The canvas binding-rewrite candidate.
    operator_signoff : bool
        Whether the operator has signed off.

    Returns
    -------
    dict[str, object]
        Canvas browser controls for deterministic operator feedback.
    """
    record = json.loads(canvas_artifact.payload)
    changed = bool(record.get("changed"))
    rewrite_status = _require_non_empty_text(
        canvas_rewrite.get("status"),
        "rewrite_status",
    )
    validation_errors = _string_list(
        canvas_rewrite.get("validation_errors", ()),
        "validation_errors",
    )
    disabled_reasons: list[str] = []
    if rewrite_status != "review_ready":
        disabled_reasons.append("binding rewrite candidate is blocked")
    disabled_reasons.extend(validation_errors)
    if not operator_signoff:
        disabled_reasons.append("operator sign-off required")
    apply_enabled = not disabled_reasons
    return {
        "state_kind": "studio_canvas_interaction_state",
        "changed": changed,
        "rewrite_status": rewrite_status,
        "apply_enabled": apply_enabled,
        "disabled_reasons": disabled_reasons,
        "next_action": _canvas_next_action(
            changed=changed,
            rewrite_status=rewrite_status,
            operator_signoff=operator_signoff,
            apply_enabled=apply_enabled,
        ),
        "status_message": (
            "Canvas edits need review before apply."
            if changed
            else "Canvas graph matches the current binding."
        ),
        "download_manifest": [
            canvas_artifact.file_name,
            canvas_layout.file_name,
            canvas_patch.file_name,
            "binding_rewrite_candidate.yaml",
        ],
        "candidate_yaml_sha256": canvas_rewrite.get("candidate_yaml_sha256", ""),
    }


def build_canvas_binding_rewrite_candidate(
    result: StudioReplayResult,
    *,
    after_graph: Mapping[str, object],
) -> dict[str, object]:
    """Build validated binding YAML candidate from reviewed canvas edits.

    Parameters
    ----------
    result : StudioReplayResult
        The Studio replay result.
    after_graph : Mapping[str, object]
        The edited canvas graph after the change.

    Returns
    -------
    dict[str, object]
        Validated binding YAML candidate from reviewed canvas edits.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not isinstance(result, StudioReplayResult):
        raise ValueError("replay result must be a StudioReplayResult")
    _, after_edges = _normalise_canvas_graph(after_graph, "after_graph")
    before_yaml = result.project_state.binding.yaml_text
    before_digest = sha256(before_yaml.encode("utf-8")).hexdigest()
    unsupported = [
        _require_non_empty_text(edge.get("id"), "canvas edge id")
        for edge in after_edges
        if edge.get("kind") != "cross_channel_coupling"
    ]
    if unsupported:
        return _blocked_binding_rewrite_candidate(
            result,
            before_digest,
            ["only cross_channel_coupling edges can rewrite binding YAML"],
        )

    try:
        candidate_yaml = _rewrite_binding_cross_channel_couplings(
            before_yaml,
            after_edges,
        )
    except ValueError as exc:
        return _blocked_binding_rewrite_candidate(result, before_digest, [str(exc)])

    validation_errors = _validate_candidate_binding_yaml(candidate_yaml)
    return {
        "candidate_kind": "canvas_binding_rewrite_candidate",
        "project_name": result.project_state.project_name,
        "status": "blocked" if validation_errors else "review_ready",
        "binding_spec_rewritten": False,
        "actuation_permitted": False,
        "network_opened": False,
        "before_yaml_sha256": before_digest,
        "candidate_yaml_sha256": sha256(candidate_yaml.encode("utf-8")).hexdigest(),
        "coupling_count_before": _canvas_graph_count(result, "edge_count"),
        "coupling_count_after": len(after_edges),
        "validation_errors": validation_errors,
        "candidate_yaml": candidate_yaml,
    }


def apply_canvas_binding_rewrite_candidate(
    candidate: Mapping[str, object],
    *,
    binding_spec_path: str | Path,
    operator_signoff: bool,
    create_backup: bool = True,
) -> dict[str, object]:
    """Apply a reviewed canvas binding candidate with hash and validation gates.

    Parameters
    ----------
    candidate : Mapping[str, object]
        The binding rewrite candidate mapping.
    binding_spec_path : str | Path
        Path to the binding-spec file.
    operator_signoff : bool
        Whether the operator has signed off.
    create_backup : bool
        Whether to write a backup before applying.

    Returns
    -------
    dict[str, object]
        A reviewed canvas binding candidate with hash and validation gates.
    """
    path = Path(binding_spec_path)
    candidate_yaml = _require_non_empty_payload(
        candidate.get("candidate_yaml"),
        "candidate_yaml",
    )
    before_digest = _require_sha256_digest(
        candidate.get("before_yaml_sha256"),
        "before_yaml_sha256",
    )
    candidate_digest = _require_sha256_digest(
        candidate.get("candidate_yaml_sha256"),
        "candidate_yaml_sha256",
    )
    blocked_reasons = _binding_apply_blocked_reasons(
        candidate,
        path,
        candidate_yaml,
        before_digest,
        candidate_digest,
        operator_signoff=operator_signoff,
    )
    if blocked_reasons:
        return _binding_apply_record(
            candidate,
            path,
            status="blocked",
            before_digest=before_digest,
            after_digest="",
            backup_path="",
            blocked_reasons=blocked_reasons,
        )

    current_yaml = path.read_text(encoding="utf-8")
    backup_path = ""
    if create_backup:
        backup = _next_binding_backup_path(path, before_digest)
        backup.write_text(current_yaml, encoding="utf-8")
        backup_path = str(backup)
    _atomic_write_text(path, candidate_yaml)
    after_digest = sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    return _binding_apply_record(
        candidate,
        path,
        status="applied",
        before_digest=before_digest,
        after_digest=after_digest,
        backup_path=backup_path,
        blocked_reasons=[],
    )


def build_oscillator_edit_artifact(
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
) -> ExportManifest:
    """Build a review artefact from edited oscillator table rows.

    Parameters
    ----------
    before_rows : Sequence[Mapping[str, object]]
        The table rows before the change.
    after_rows : Sequence[Mapping[str, object]]
        The edited table rows after the change.

    Returns
    -------
    ExportManifest
        A review artefact from edited oscillator table rows.
    """
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


def _canvas_next_action(
    *,
    changed: bool,
    rewrite_status: str,
    operator_signoff: bool,
    apply_enabled: bool,
) -> str:
    """Return the recommended next operator action for the canvas review state."""
    if apply_enabled:
        return "apply reviewed binding rewrite or download artefacts"
    if rewrite_status != "review_ready":
        return "fix blocked canvas rewrite before apply"
    if changed and not operator_signoff:
        return "review artefacts and sign off before apply"
    if not changed:
        return "download artefacts or continue replay review"
    return "review canvas artefacts"


def _string_list(value: object, name: str) -> list[str]:
    """Return ``value`` as a list of non-empty strings, else raise ``ValueError``."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    items: list[str] = []
    for item in value:
        items.append(_require_non_empty_text(item, name))
    return items


def _blocked_binding_rewrite_candidate(
    result: StudioReplayResult,
    before_digest: str,
    validation_errors: Sequence[str],
) -> dict[str, object]:
    """Build a blocked canvas binding-rewrite candidate record (unchanged YAML)."""
    yaml_text = result.project_state.binding.yaml_text
    return {
        "candidate_kind": "canvas_binding_rewrite_candidate",
        "project_name": result.project_state.project_name,
        "status": "blocked",
        "binding_spec_rewritten": False,
        "actuation_permitted": False,
        "network_opened": False,
        "before_yaml_sha256": before_digest,
        "candidate_yaml_sha256": before_digest,
        "coupling_count_before": _canvas_graph_count(result, "edge_count"),
        "coupling_count_after": _canvas_graph_count(result, "edge_count"),
        "validation_errors": list(validation_errors),
        "candidate_yaml": yaml_text,
    }


def _binding_apply_blocked_reasons(
    candidate: Mapping[str, object],
    path: Path,
    candidate_yaml: str,
    before_digest: str,
    candidate_digest: str,
    *,
    operator_signoff: bool,
) -> list[str]:
    """Return the reasons that block applying a canvas binding rewrite.

    Checks the candidate kind and review-ready status, operator sign-off, that the
    target path exists, that the candidate YAML matches its digest and validates,
    and that the on-disk source still matches the recorded before-digest.
    """
    blocked: list[str] = []
    if candidate.get("candidate_kind") != "canvas_binding_rewrite_candidate":
        blocked.append("candidate_kind must be canvas_binding_rewrite_candidate")
    if candidate.get("status") != "review_ready":
        blocked.append("candidate status must be review_ready")
    if operator_signoff is not True:
        blocked.append("operator_signoff must be true")
    if not path.exists() or not path.is_file():
        blocked.append("binding_spec_path must point to an existing file")
    if sha256(candidate_yaml.encode("utf-8")).hexdigest() != candidate_digest:
        blocked.append("candidate YAML SHA-256 does not match candidate metadata")

    validation_errors = _validate_candidate_binding_yaml(candidate_yaml)
    blocked.extend(validation_errors)

    if path.exists() and path.is_file():
        current_yaml = path.read_text(encoding="utf-8")
        current_digest = sha256(current_yaml.encode("utf-8")).hexdigest()
        if current_digest != before_digest:
            blocked.append(
                "current binding_spec.yaml SHA-256 does not match candidate source"
            )
    return blocked


def _binding_apply_record(
    candidate: Mapping[str, object],
    path: Path,
    *,
    status: str,
    before_digest: str,
    after_digest: str,
    backup_path: str,
    blocked_reasons: Sequence[str],
) -> dict[str, object]:
    """Build the canvas binding-rewrite apply audit record."""
    return {
        "apply_kind": "canvas_binding_rewrite_apply",
        "candidate_kind": candidate.get("candidate_kind", ""),
        "project_name": candidate.get("project_name", ""),
        "status": status,
        "binding_spec_path": str(path),
        "backup_path": backup_path,
        "binding_spec_rewritten": status == "applied",
        "actuation_permitted": False,
        "network_opened": False,
        "before_yaml_sha256": before_digest,
        "after_yaml_sha256": after_digest,
        "candidate_yaml_sha256": candidate.get("candidate_yaml_sha256", ""),
        "blocked_reasons": list(blocked_reasons),
    }


def _next_binding_backup_path(path: Path, before_digest: str) -> Path:
    """Return the next free ``.studio-backup-<digest>.bak`` path for the binding."""
    stem = f"{path.name}.studio-backup-{before_digest[:12]}.bak"
    backup = path.with_name(stem)
    if not backup.exists():
        return backup
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}.{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate binding backup path")


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path`` via a same-directory temp file."""
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = handle.name
            handle.write(text)
            handle.flush()
        Path(tmp_path).replace(path)
    except OSError:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        raise


def _rewrite_binding_cross_channel_couplings(
    yaml_text: str,
    edges: Sequence[Mapping[str, object]],
) -> str:
    """Rewrite the binding YAML's ``cross_channel_couplings`` from canvas edges."""
    import yaml

    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise ValueError("binding YAML must contain a mapping")
    raw["cross_channel_couplings"] = [
        _canvas_edge_to_cross_channel_coupling(edge) for edge in edges
    ]
    rendered: str = yaml.safe_dump(raw, sort_keys=False)
    return rendered


def _canvas_edge_to_cross_channel_coupling(
    edge: Mapping[str, object],
) -> dict[str, object]:
    """Convert a canvas edge to a cross-channel coupling (distinct source/target)."""
    source = _canvas_edge_channel(edge, "source")
    target = _canvas_edge_channel(edge, "target")
    if source == target:
        raise ValueError("cross-channel coupling source and target must differ")
    return {
        "source": source,
        "target": target,
        "strength": _finite_range(
            edge.get("strength", 0.0),
            "cross-channel coupling strength",
            low=0.0,
            high=100.0,
        ),
        "mode": _require_non_empty_text(edge.get("mode", "directed"), "mode"),
        "template": _require_non_empty_text(
            edge.get("template", "studio_canvas_review"),
            "template",
        ),
    }


def _canvas_edge_channel(edge: Mapping[str, object], endpoint: str) -> str:
    """Resolve the channel name for an edge endpoint, else raise ``ValueError``."""
    explicit = edge.get(f"{endpoint}_channel")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    raw_endpoint = _require_non_empty_text(edge.get(endpoint), endpoint)
    if raw_endpoint.startswith("channel_"):
        return raw_endpoint.removeprefix("channel_")
    raise ValueError(f"{endpoint} must reference a channel node")


def _validate_candidate_binding_yaml(candidate_yaml: str) -> list[str]:
    """Return validation errors for candidate binding YAML, loaded in a temp dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "binding_spec.yaml"
        spec_path.write_text(candidate_yaml, encoding="utf-8")
        try:
            spec = load_binding_spec(spec_path)
        except (BindingLoadError, ValueError, TypeError, OSError) as exc:
            return [f"candidate binding failed to load: {type(exc).__name__}"]
        return list(validate_binding_spec(spec))


def _require_sha256_digest(value: object, field_name: str) -> str:
    """Return ``value`` if it is a SHA-256 digest, else raise ``ValueError``."""
    if not _is_sha256_digest(value):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return str(value)


def _require_non_empty_payload(value: object, name: str) -> str:
    """Return ``value`` if it is a non-empty string payload, else raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalise_table_rows(
    rows: Sequence[Mapping[str, object]],
    field_name: str,
) -> list[dict[str, object]]:
    """Return table rows as mappings with string keys and JSON-safe finite values."""
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


def _normalise_canvas_graph(
    graph: Mapping[str, object],
    field_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return the validated canvas graph's node and edge row lists."""
    if not isinstance(graph, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    nodes = graph.get("nodes", ())
    edges = graph.get("edges", ())
    if isinstance(nodes, str | bytes) or not isinstance(nodes, Sequence):
        raise ValueError("canvas nodes must be a sequence of mappings")
    if isinstance(edges, str | bytes) or not isinstance(edges, Sequence):
        raise ValueError("canvas edges must be a sequence of mappings")
    return (
        _normalise_table_rows(nodes, "canvas nodes"),
        _normalise_table_rows(edges, "canvas edges"),
    )


def _validate_canvas_edge_endpoints(
    nodes: Sequence[Mapping[str, object]],
    edges: Sequence[Mapping[str, object]],
) -> None:
    """Assert every canvas edge's source and target reference a known node id."""
    node_ids = {
        _require_non_empty_text(node.get("id"), "canvas node id") for node in nodes
    }
    for edge in edges:
        edge_id = _require_non_empty_text(edge.get("id"), "canvas edge id")
        source = _require_non_empty_text(edge.get("source"), "canvas edge source")
        target = _require_non_empty_text(edge.get("target"), "canvas edge target")
        if source not in node_ids or target not in node_ids:
            raise ValueError(f"canvas edge {edge_id!r} references unknown endpoint")


def _canvas_item_changes(
    before_items: Sequence[Mapping[str, object]],
    after_items: Sequence[Mapping[str, object]],
    *,
    fields: Sequence[str],
) -> dict[str, list[dict[str, object]]]:
    """Return the added, removed, and modified items between two canvas snapshots."""
    before = _canvas_item_index(before_items, fields=fields)
    after = _canvas_item_index(after_items, fields=fields)
    before_ids = set(before)
    after_ids = set(after)
    common_ids = before_ids & after_ids
    return {
        "added": [after[item_id] for item_id in sorted(after_ids - before_ids)],
        "removed": [before[item_id] for item_id in sorted(before_ids - after_ids)],
        "modified": [
            {"id": item_id, "before": before[item_id], "after": after[item_id]}
            for item_id in sorted(common_ids)
            if before[item_id] != after[item_id]
        ],
    }


def _canvas_item_index(
    items: Sequence[Mapping[str, object]],
    *,
    fields: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Index canvas items by their id, keeping only the selected fields."""
    indexed: dict[str, dict[str, object]] = {}
    for item in items:
        item_id = _require_non_empty_text(item.get("id"), "canvas item id")
        if item_id in indexed:
            raise ValueError(f"canvas item id {item_id!r} must be unique")
        indexed[item_id] = {
            field: item[field]
            for field in fields
            if field in item and item[field] is not None
        }
    return indexed
