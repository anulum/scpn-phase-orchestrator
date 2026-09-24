# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio product manifest declared-gate tests

"""The product manifest refuses a panel that enables any gate it declares.

The manifest states product-wide that QPU execution, hardware writes, and
network access are not permitted. A panel record that declares one of those
gates as enabled must be refused, not published under a manifest that
contradicts it.
"""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.studio.product import (
    STUDIO_REVIEW_PANEL_REGISTRY,
    PanelRecord,
    build_studio_product_manifest,
)


def _panel(panel_id: str) -> PanelRecord:
    """Return a copy of the shipped registry record for ``panel_id``."""
    for panel in STUDIO_REVIEW_PANEL_REGISTRY:
        if panel["panel_id"] == panel_id:
            return dict(panel)
    raise AssertionError(f"{panel_id} is not in the shipped registry")


@pytest.mark.parametrize(
    ("panel_id", "gate", "value"),
    [
        ("hybrid_order_parameters", "qpu_execution_permitted", True),
        ("topos_semantic_binding", "formal_proof_claim_permitted", True),
        ("intergenerational_policy_inheritance", "direct_hot_patch_permitted", True),
        ("integrated_information_monitor", "hardware_write_permitted", True),
        ("integrated_information_monitor", "network_opened", True),
        ("hybrid_order_parameters", "qpu_execution_permitted", "false"),
    ],
)
def test_enabled_declared_gate_is_refused(
    panel_id: str, gate: str, value: object
) -> None:
    """A declared gate that is anything but ``False`` refuses the manifest."""
    panel = {**_panel(panel_id), gate: value}

    with pytest.raises(ValueError, match=f"{panel_id} {gate} invalid"):
        build_studio_product_manifest(panel_registry=(panel,))


def test_shipped_registry_declares_every_gate_disabled() -> None:
    """Every gate the shipped panels declare is disabled in the manifest."""
    manifest = build_studio_product_manifest()

    panels = manifest["review_panels"]
    assert isinstance(panels, tuple)
    declared = [
        (panel["panel_id"], field, value)
        for panel in panels
        for field, value in panel.items()
        if field.endswith("_permitted") or field == "network_opened"
    ]
    assert declared
    assert all(value is False for _, _, value in declared)
    assert manifest["qpu_execution_permitted"] is False
    assert manifest["hardware_write_permitted"] is False
