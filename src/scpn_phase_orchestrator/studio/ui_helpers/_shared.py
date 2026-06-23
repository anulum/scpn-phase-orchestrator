# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio shared value-coercion primitives

"""Value-coercion and validation primitives shared across SPO Studio UI helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.studio.workflow import (
    ExportManifest,
)

if TYPE_CHECKING:
    from ._state import StudioReplayResult


def _deployment_blocked_reasons(
    exports: Sequence[ExportManifest],
) -> tuple[str, ...]:
    """Return the de-duplicated warnings that block deployment across exports."""
    reasons: list[str] = []
    for manifest in exports:
        for warning in manifest.warnings:
            if warning not in reasons:
                reasons.append(warning)
    return tuple(reasons)


def _blocked_target(
    target: str,
    blocked_reasons: Sequence[str],
) -> dict[str, object]:
    """Return a blocked deployment-target record carrying its blocking reasons."""
    return {
        "target": target,
        "status": "blocked",
        "required_artifacts": (),
        "commands": (),
        "operator_action": "resolve blocked reasons before packaging",
        "blocked_reasons": list(blocked_reasons),
    }


def _optional_sha256_hex(value: object, name: str) -> str | None:
    """Return ``None`` for a null value, else the validated SHA-256 hex digest."""
    if value is None:
        return None
    return _require_sha256_hex(value, name)


def _optional_non_negative_int(value: object, name: str) -> int | None:
    """Return ``None`` for a null value, else the validated non-negative integer."""
    if value is None:
        return None
    return _non_negative_int(value, name)


def _normalise_optional_text_sequence(value: object, name: str) -> tuple[str, ...]:
    """Return an empty tuple for a null value, else the validated text tuple."""
    if value is None:
        return ()
    return _normalise_text_sequence(value, name)


def _required_bool(value: object, name: str) -> bool:
    """Return ``value`` if it is a real boolean, else raise ``ValueError``."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _non_negative_float(value: object, name: str) -> float:
    """Return ``value`` as a finite non-negative float, else raise ``ValueError``."""
    if isinstance(value, (bool, complex)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real value")
    scalar = float(value)
    if not isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real value")
    return scalar


def _unit_interval_number(value: object, name: str) -> float:
    """Return ``value`` as a float in [0, 1], else raise ``ValueError``."""
    scalar = _non_negative_float(value, name)
    if scalar > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return scalar


def _positive_int(value: object, name: str, *, minimum: int) -> int:
    """Return ``value`` as an integer at least ``minimum``, else raise."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    checked = int(value)
    if checked < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return checked


def _non_negative_int(value: object, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be non-negative")
    return checked


def _normalise_text_sequence(value: object, name: str) -> tuple[str, ...]:
    """Return ``value`` as a tuple of non-empty strings, else raise ``ValueError``."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    return tuple(_require_non_empty_text(item, name) for item in value)


def _normalise_float_sequence(value: object, name: str) -> tuple[float, ...]:
    """Return ``value`` as a non-empty tuple of finite floats, else raise."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of finite numbers")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return tuple(_finite_number(item, name) for item in value)


def _normalise_information_geometry_gradient(
    value: object,
    name: str,
) -> tuple[tuple[str, float], ...]:
    """Return ``value`` as a non-empty tuple of (knob, finite-value) pairs."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of knob/value pairs")
    if not value:
        raise ValueError(f"{name} must not be empty")
    gradient: list[tuple[str, float]] = []
    for item in value:
        if isinstance(item, str | bytes) or not isinstance(item, Sequence):
            raise ValueError(f"{name} entries must be knob/value pairs")
        if len(item) != 2:
            raise ValueError(f"{name} entries must be knob/value pairs")
        knob, raw_value = item
        gradient.append(
            (
                _require_non_empty_text(knob, f"{name} knob"),
                _finite_number(raw_value, f"{name} value"),
            )
        )
    return tuple(gradient)


def _require_sha256_hex(value: object, name: str) -> str:
    """Return ``value`` if it is a lowercase 64-char SHA-256 hex digest, else raise."""
    text = _require_non_empty_text(value, name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


def _connector_by_transport(
    connector_plan: Mapping[str, object],
    transport: str,
) -> dict[str, object]:
    """Return the connector in ``connector_plan`` matching ``transport``, else raise."""
    connectors = connector_plan.get("connectors", ())
    if isinstance(connectors, str | bytes) or not isinstance(connectors, Sequence):
        raise ValueError("connectors must be a sequence")
    for connector in connectors:
        if not isinstance(connector, Mapping):
            raise ValueError("connector entries must be mappings")
        if connector.get("transport") == transport:
            return dict(connector)
    raise ValueError(f"connector transport {transport!r} not found")


def _mapping_count(mapping: Mapping[str, object], name: str) -> int:
    """Return ``mapping[name]`` validated as a non-negative integer count."""
    return _non_negative_int(mapping.get(name), name)


def _canvas_graph_count(result: StudioReplayResult, name: str) -> int:
    """Return the named non-negative count from a replay result's canvas graph."""
    return _mapping_count(result.canvas_graph, name)


def _require_sequence(value: object, name: str) -> Sequence[object]:
    """Return ``value`` if it is a non-string sequence, else raise ``ValueError``."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _is_sha256_digest(value: object) -> bool:
    """Return whether ``value`` is a 64-character hexadecimal SHA-256 digest."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _layer_metrics(value: object) -> tuple[tuple[str, float], ...]:
    """Return (layer-name, R) pairs from a layer table, skipping non-mapping rows."""
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    rows: list[tuple[str, float]] = []
    for index, layer in enumerate(value):
        if not isinstance(layer, Mapping):
            continue
        name = _require_non_empty_text(layer.get("name", f"layer_{index}"), "layer")
        rows.append((name, _finite_number(layer.get("R", 0.0), "layer.R")))
    return tuple(rows)


def _canvas_channel_id(channel: str) -> str:
    """Return a stable ``channel_<slug>`` id, non-alphanumerics replaced by ``_``."""
    safe = "".join(character if character.isalnum() else "_" for character in channel)
    return f"channel_{safe}"


def _finite_range(value: object, name: str, *, low: float, high: float) -> float:
    """Return ``value`` as a finite float within ``[low, high]``, else raise."""
    number = _finite_number(value, name)
    if not low <= number <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return number


def _finite_number(value: object, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_float(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    number = _finite_number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_non_empty_text(value: object, name: str) -> str:
    """Return ``value`` stripped if it is a non-empty string, else raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
