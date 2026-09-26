# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — twin-confidence Prometheus export

"""Prometheus exposition for scored digital-twin confidence summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scpn_phase_orchestrator.monitor.twin_confidence import TwinConfidenceSummary

__all__ = ["twin_confidence_prometheus_text"]

_STATUS_LEVELS: dict[str, int] = {"healthy": 0, "warning": 1, "critical": 2}


def twin_confidence_prometheus_text(
    summary: TwinConfidenceSummary,
    *,
    prefix: str = "spo",
) -> str:
    """Render a twin-confidence summary as Prometheus exposition text.

    Parameters
    ----------
    summary : TwinConfidenceSummary
        The operator-facing aggregate to export.
    prefix : str, optional
        Metric-name prefix (default ``"spo"``).

    Returns
    -------
    str
        Prometheus exposition text with confidence gauges, per-status counters,
        and a numeric worst-status level gauge.

    Raises
    ------
    ValueError
        If ``prefix`` is not a non-empty string.
    """
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    lines = [
        f"# HELP {prefix}_twin_confidence_mean Mean twin confidence over scored ticks",
        f"# TYPE {prefix}_twin_confidence_mean gauge",
        f"{prefix}_twin_confidence_mean {summary.mean_confidence}",
        f"# HELP {prefix}_twin_confidence_min Minimum twin confidence over ticks",
        f"# TYPE {prefix}_twin_confidence_min gauge",
        f"{prefix}_twin_confidence_min {summary.min_confidence}",
        f"# HELP {prefix}_twin_confidence_latest Most recent twin confidence",
        f"# TYPE {prefix}_twin_confidence_latest gauge",
        f"{prefix}_twin_confidence_latest {summary.latest_confidence}",
        f"# HELP {prefix}_twin_confidence_tick_count Scored twin-confidence ticks",
        f"# TYPE {prefix}_twin_confidence_tick_count gauge",
        f"{prefix}_twin_confidence_tick_count {summary.tick_count}",
        (
            f"# HELP {prefix}_twin_confidence_status_total "
            "Twin-confidence ticks per operator status"
        ),
        f"# TYPE {prefix}_twin_confidence_status_total counter",
        (
            f'{prefix}_twin_confidence_status_total{{status="healthy"}} '
            f"{summary.healthy_count}"
        ),
        (
            f'{prefix}_twin_confidence_status_total{{status="warning"}} '
            f"{summary.warning_count}"
        ),
        (
            f'{prefix}_twin_confidence_status_total{{status="critical"}} '
            f"{summary.critical_count}"
        ),
        (
            f"# HELP {prefix}_twin_confidence_worst_status_level "
            "Worst operator status (0 healthy, 1 warning, 2 critical)"
        ),
        f"# TYPE {prefix}_twin_confidence_worst_status_level gauge",
        (
            f"{prefix}_twin_confidence_worst_status_level "
            f"{_STATUS_LEVELS[summary.worst_status]}"
        ),
    ]
    return "\n".join(lines) + "\n"
