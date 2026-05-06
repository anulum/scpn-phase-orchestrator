# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit report summary tests

from __future__ import annotations

from typing import cast, get_type_hints

import pytest

from scpn_phase_orchestrator.reporting import build_audit_report_summary


def test_audit_report_summary_contract_is_typed() -> None:
    hints = get_type_hints(build_audit_report_summary)

    assert "list" in str(hints["entries"])
    assert hints["hash_chain_ok"] is bool
    assert hints["hash_chain_verified"] is int
    assert "dict" in str(hints["return"])


def test_audit_report_summary_includes_core_report_fields() -> None:
    entries: list[dict[str, object]] = [
        {
            "step": 0,
            "regime": "nominal",
            "stability": 0.7,
            "layers": [{"R": 0.8}, {"R": 0.6}],
            "actions": [{"knob": "K"}],
        },
        {
            "step": 1,
            "regime": "degraded",
            "stability": 0.65,
            "layers": [{"R": 0.9}],
            "actions": [],
        },
        {"event": "boundary_violation"},
    ]

    summary = build_audit_report_summary(
        entries,
        hash_chain_ok=True,
        hash_chain_verified=3,
    )

    assert summary["steps"] == 2
    assert summary["layers"] == 2
    assert summary["layer_r_mean"] == [0.85, 0.6]
    assert summary["layer_r_final"] == [0.9, 0.6]
    assert summary["regime_counts"] == {"nominal": 1, "degraded": 1}
    assert summary["action_counts"] == {"K": 1}
    assert summary["events"] == 1
    assert summary["hash_chain_ok"] is True
    assert summary["hash_chain_verified"] == 3


def test_audit_report_summary_preserves_binding_channel_algebra() -> None:
    channel_algebra = {
        "channels": ["P", "Risk"],
        "derived_channels": ["Risk"],
        "missing_required_channels": [],
    }
    entries: list[dict[str, object]] = [
        {
            "header": True,
            "binding_config": {
                "name": "nchannel",
                "channel_algebra": channel_algebra,
            },
        },
        {"step": 0, "layers": [{"R": 0.8}], "regime": "nominal"},
    ]

    summary = build_audit_report_summary(
        entries,
        hash_chain_ok=True,
        hash_chain_verified=2,
    )
    binding_summary = cast("dict[str, object]", summary["binding_summary"])

    assert binding_summary["name"] == "nchannel"
    assert summary["channel_algebra"] == channel_algebra


def test_audit_report_summary_includes_integrated_information_records() -> None:
    entries: list[dict[str, object]] = [
        {"step": 0, "layers": [{"R": 0.8}], "regime": "nominal"},
        {
            "monitor": "integrated_information",
            "phi": 0.12,
            "normalised_phi": 0.24,
            "total_integration": 0.5,
            "claim_boundary": "engineering_proxy_not_theoretical_iit",
        },
        {
            "monitor": "integrated_information",
            "phi": 0.18,
            "normalised_phi": 0.36,
            "total_integration": 0.7,
            "claim_boundary": "engineering_proxy_not_theoretical_iit",
        },
    ]

    summary = build_audit_report_summary(
        entries,
        hash_chain_ok=True,
        hash_chain_verified=3,
    )
    phi_summary = cast("dict[str, object]", summary["integrated_information"])

    assert phi_summary["records"] == 2
    assert phi_summary["latest_phi"] == 0.18
    assert phi_summary["latest_normalised_phi"] == 0.36
    assert phi_summary["latest_total_integration"] == 0.7
    assert phi_summary["phi_mean"] == 0.15
    assert phi_summary["normalised_phi_mean"] == 0.3
    assert phi_summary["total_integration_mean"] == 0.6
    assert phi_summary["phi_series"] == [0.12, 0.18]


def test_audit_report_summary_rejects_logs_without_steps() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        build_audit_report_summary(
            [{"event": "only-event"}],
            hash_chain_ok=True,
            hash_chain_verified=1,
        )
