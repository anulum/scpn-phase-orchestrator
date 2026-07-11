# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — monitor validation-status guide drift guard

"""Keep the monitor validation-status guide in sync with the registry.

The ``docs/guide/monitor_validation_status.md`` guide promises an honest,
per-monitor validation posture. This drift guard fails if a promoted monitor
(externally-validated or synthetic-only) is missing from the guide, if a tier
token disappears, or if the published per-tier counts drift from the registry —
so the honest posture in prose can never fall out of step with the code.
"""

from __future__ import annotations

from pathlib import Path

from scpn_phase_orchestrator.monitor.validation_status import (
    MONITOR_VALIDATION,
    MonitorValidationStatus,
    validation_summary,
)

GUIDE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "guide"
    / "monitor_validation_status.md"
)


def _promoted_monitors() -> tuple[str, ...]:
    return tuple(
        record.monitor
        for record in MONITOR_VALIDATION.values()
        if record.status is not MonitorValidationStatus.RESEARCH
    )


def test_guide_lists_every_promoted_monitor() -> None:
    doc = GUIDE.read_text(encoding="utf-8")
    missing = {monitor for monitor in _promoted_monitors() if monitor not in doc}
    assert not missing, f"promoted monitors missing from the guide: {missing}"


def test_guide_states_every_tier_token() -> None:
    doc = GUIDE.read_text(encoding="utf-8")
    for status in MonitorValidationStatus:
        assert f"`{status.value}`" in doc, status.value


def test_guide_publishes_the_registry_counts() -> None:
    doc = GUIDE.read_text(encoding="utf-8")
    summary = validation_summary()
    for status in MonitorValidationStatus:
        count = summary[status]
        assert f"**{count}**" in doc, f"count {count} for {status.value} not in guide"


def test_guide_states_the_validated_niche() -> None:
    doc = GUIDE.read_text(encoding="utf-8")
    # The one externally-validated niche and the honesty caveat must remain.
    assert "grid modal damping" in doc
    assert "at chance" in doc
