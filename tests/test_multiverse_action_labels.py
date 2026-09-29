# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Multiverse audit action precision tests

"""Action values survive real rollouts, JSON audit records and Studio review."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.studio import build_multiverse_counterfactual_studio_panel
from scpn_phase_orchestrator.supervisor.multiverse import (
    MultiverseCounterfactualManifest,
    simulate_multiverse_counterfactual_branches,
)
from scpn_phase_orchestrator.supervisor.multiverse_risk import (
    evaluate_multiverse_branch_risk,
)


def _rollout(backend: str, knob: str, value: float) -> MultiverseCounterfactualManifest:
    """Simulate neighbouring interventions on the same two-oscillator graph."""
    return simulate_multiverse_counterfactual_branches(
        phases=np.array([0.1, 0.4]),
        omegas=np.array([0.03, -0.02]),
        baseline_k=np.array([[0.0, 0.15], [0.15, 0.0]]),
        baseline_alpha=np.zeros((2, 2)),
        branch_action_sets=(
            (ControlAction(knob, "global", value, 1.0, "review intervention"),),
            (
                ControlAction(
                    knob,
                    "global",
                    math.nextafter(value, math.inf),
                    1.0,
                    "neighbouring intervention",
                ),
            ),
        ),
        horizon=2,
        backend=backend,
    )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("knob", ["K", "alpha", "zeta", "Psi", "psi"])
@pytest.mark.parametrize("value", [0.12345678901234567, -0.23456789012345678, 1e-100])
def test_action_labels_round_trip_through_json_and_studio(
    backend: str, knob: str, value: float
) -> None:
    """Adjacent representable actions remain distinguishable in review records."""
    manifest = _rollout(backend, knob, value)
    audit = json.loads(json.dumps(manifest.to_audit_record(), allow_nan=False))
    labels = [record["action_labels"][0] for record in audit["branch_records"]]
    for label, expected in zip(
        labels, (value, math.nextafter(value, math.inf)), strict=True
    ):
        prefix, literal = label.rsplit(":", 1)
        assert prefix == f"{knob}:global"
        assert float(literal).hex() == expected.hex()
    assert labels[0] != labels[1]
    assert manifest.manifest_hash == _rollout(backend, knob, value).manifest_hash
    risk = evaluate_multiverse_branch_risk(audit)
    panel = build_multiverse_counterfactual_studio_panel(audit, risk.to_audit_record())
    rows = panel["branch_rows"]
    assert isinstance(rows, tuple)
    assert [row["action_labels"][0] for row in rows] == labels
    assert panel["actuation_permitted"] is False
    assert panel["non_actuating"] is True
    assert panel["execution_disabled"] is True


@pytest.mark.parametrize("value", [-0.0, math.ulp(0.0)])
def test_action_labels_preserve_signed_zero_and_subnormal_values(value: float) -> None:
    """Audit text retains the input even when its dynamical effect rounds away."""
    manifest = _rollout("numpy", "zeta", value)
    label = manifest.branch_records[0].action_labels[0]
    assert float(label.rsplit(":", 1)[1]).hex() == value.hex()


def test_action_label_precision_is_shared_by_numpy_and_jax() -> None:
    """Backend selection cannot change the action's audit representation."""
    numpy_manifest = _rollout("numpy", "K", 0.12345678901234567)
    jax_manifest = _rollout("jax", "K", 0.12345678901234567)
    for numpy_record, jax_record in zip(
        numpy_manifest.branch_records, jax_manifest.branch_records, strict=True
    ):
        assert numpy_record.action_labels == jax_record.action_labels
        assert numpy_record.branch_hash == jax_record.branch_hash
        np.testing.assert_allclose(numpy_record.final_R, jax_record.final_R, atol=1e-10)
