# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit report summary builder

"""Reusable audit report summaries for CLI, notebooks, and tools."""

from __future__ import annotations

from math import isfinite
from numbers import Real

__all__ = ["build_audit_report_summary"]


def build_audit_report_summary(
    entries: list[dict[str, object]],
    *,
    hash_chain_ok: bool,
    hash_chain_verified: int,
) -> dict[str, object]:
    """Build a JSON-ready report summary from audit log entries.

    Parameters
    ----------
    entries : list[dict[str, object]]
        Audit-log entries.
    hash_chain_ok : bool
        Whether the hash chain verified.
    hash_chain_verified : int
        Whether the hash chain verified.

    Returns
    -------
    dict[str, object]
        A JSON-ready report summary from audit log entries.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    records = _record_entries(entries)
    steps = [entry for entry in records if "step" in entry and "layers" in entry]
    events = [entry for entry in records if "event" in entry]
    header = _load_header(records)
    if not steps:
        raise ValueError("audit report requires at least one step record")

    n_steps = len(steps)
    n_layers = max(len(_layers(step)) for step in steps)
    r_series = [
        [_layer_r(_layers(step)[index]) for step in steps if index < len(_layers(step))]
        for index in range(n_layers)
    ]
    regime_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for step in steps:
        regime = str(step.get("regime", "NOMINAL"))
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        for action in _actions(step):
            knob = str(action.get("knob", "?"))
            action_counts[knob] = action_counts.get(knob, 0) + 1

    summary: dict[str, object] = {
        "steps": n_steps,
        "layers": n_layers,
        "amplitude_mode": bool(header and header.get("amplitude_mode")),
        "final_regime": steps[-1].get("regime", "unknown"),
        "final_stability": _numeric_value(steps[-1], "stability"),
        "layer_r_mean": [
            round(sum(series) / len(series), 4) if series else 0.0
            for series in r_series
        ],
        "layer_r_final": [
            round(series[-1], 4) if series else 0.0 for series in r_series
        ],
        "regime_counts": regime_counts,
        "action_counts": action_counts,
        "events": len(events),
        "hash_chain_ok": hash_chain_ok,
        "hash_chain_verified": hash_chain_verified,
    }
    integrated_information = _integrated_information_summary(records)
    if integrated_information is not None:
        summary["integrated_information"] = integrated_information
    if header is not None:
        binding_summary = header.get("binding_summary") or header.get("binding_config")
        if isinstance(binding_summary, dict):
            summary["binding_summary"] = binding_summary
            channel_algebra = binding_summary.get("channel_algebra")
            if isinstance(channel_algebra, dict):
                summary["channel_algebra"] = channel_algebra
    return summary


def _record_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the audit record entries from a report payload, else raise."""
    return [entry for entry in entries if isinstance(entry, dict)]


def _load_header(entries: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the audit report header fields."""
    for entry in entries:
        if entry.get("header") is True:
            return entry
    return None


def _layers(step: dict[str, object]) -> list[dict[str, object]]:
    """Return the per-layer records from a step entry."""
    layers = step.get("layers", [])
    if not isinstance(layers, list):
        return []
    return [layer for layer in layers if isinstance(layer, dict)]


def _layer_r(layer: dict[str, object]) -> float:
    """Return the order parameter (R) for a layer record."""
    return _numeric_value(layer, "R")


def _actions(step: dict[str, object]) -> list[dict[str, object]]:
    """Return the action records from a step entry."""
    actions = step.get("actions", [])
    if not isinstance(actions, list):
        return []
    return [action for action in actions if isinstance(action, dict)]


def _integrated_information_summary(
    entries: list[dict[str, object]],
) -> dict[str, object] | None:
    """Return a summary of integrated-information metrics from the records."""
    records = [
        entry
        for entry in entries
        if entry.get("monitor") == "integrated_information"
        and _is_finite_real(entry.get("phi"))
        and _is_finite_real(entry.get("normalised_phi"))
    ]
    if not records:
        return None

    latest = records[-1]
    phi_series = [_numeric_value(record, "phi") for record in records]
    normalised_series = [_numeric_value(record, "normalised_phi") for record in records]
    total_series = [
        _numeric_value(record, "total_integration")
        for record in records
        if _is_finite_real(record.get("total_integration"))
    ]
    summary: dict[str, object] = {
        "records": len(records),
        "latest_phi": _numeric_value(latest, "phi"),
        "latest_normalised_phi": _numeric_value(latest, "normalised_phi"),
        "phi_mean": round(sum(phi_series) / len(phi_series), 6),
        "normalised_phi_mean": round(
            sum(normalised_series) / len(normalised_series),
            6,
        ),
        "phi_series": phi_series,
        "normalised_phi_series": normalised_series,
        "claim_boundary": latest.get(
            "claim_boundary",
            "engineering_proxy_not_theoretical_iit",
        ),
    }
    if total_series:
        summary["latest_total_integration"] = total_series[-1]
        summary["total_integration_mean"] = round(
            sum(total_series) / len(total_series),
            6,
        )
    return summary


def _numeric_value(record: dict[str, object], key: str) -> float:
    """Return a named numeric field from a mapping, else raise."""
    value = record.get(key, 0.0)
    if isinstance(value, Real) and not isinstance(value, bool):
        parsed = float(value)
        if isfinite(parsed):
            return parsed
    return 0.0


def _is_finite_real(value: object) -> bool:
    """Return whether ``value`` is a finite real number."""
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
