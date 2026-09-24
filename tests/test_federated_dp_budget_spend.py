# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — a NaN privacy spend must not bypass the DP budget

"""Per-node epsilon spend must be a finite non-negative real.

``epsilon_spent = NaN`` made the summed spend NaN, so ``spent > epsilon``
never fired and the request was admitted whatever the other nodes had spent;
the CLI's JSON loader accepts ``NaN``, so a preflight bundle was issued for
it. ``True`` counted as 1.0 and a string raised ``TypeError``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli._app import main
from scpn_phase_orchestrator.supervisor.federated_dp_noise_service import (
    DpNoiseNodePrivacyBudget,
    DpNoiseServiceRequestManifest,
)

_BASE = {
    "epsilon": 2.5,
    "delta": 1e-6,
    "sensitivity": 1.75,
    "noise_multiplier": 0.9,
    "node_count": 2,
    "seed_hash": "a" * 64,
    "policy_keys": ("alpha", "beta"),
}


@pytest.mark.parametrize(
    ("spent", "match"),
    [
        (math.nan, "finite non-negative real"),
        (math.inf, "finite non-negative real"),
        (True, "finite non-negative real"),
        ("0.9", "finite non-negative real"),
        (-0.1, "must be non-negative"),
    ],
)
def test_invalid_spend_is_refused(spent: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DpNoiseServiceRequestManifest(
            **_BASE,
            node_budgets=(
                DpNoiseNodePrivacyBudget("node-a", spent),  # type: ignore[arg-type]
                DpNoiseNodePrivacyBudget("node-b", 0.1),
            ),
        )


def test_budget_is_still_enforced_for_real_spends() -> None:
    with pytest.raises(ValueError, match="privacy budget exceeded"):
        DpNoiseServiceRequestManifest(
            **_BASE,
            node_budgets=(
                DpNoiseNodePrivacyBudget("node-a", 2.0),
                DpNoiseNodePrivacyBudget("node-b", 1.0),
            ),
        )


def test_cli_refuses_a_nan_spend(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    # json.dumps writes the non-standard NaN literal the CLI loader accepts
    request.write_text(
        '{"epsilon": 2.5, "delta": 1e-06, "sensitivity": 1.75, '
        '"noise_multiplier": 0.9, "node_count": 2, "seed_hash": "' + "a" * 64 + '", '
        '"policy_keys": ["alpha"], "node_budgets": ['
        '{"node_id": "node-a", "epsilon_spent": NaN}, '
        '{"node_id": "node-b", "epsilon_spent": 99.0}]}',
        encoding="utf-8",
    )
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        '{"mechanism_label": "mechanism-v1", "privacy_accountant_owner": "a", '
        '"seed_custody_label": "s", "budget_issuer_label": "b", '
        '"service_endpoint_label": "https://dp-noise.internal", '
        '"operator_approved": true}',
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        main, ["federated-dp-noise-service-preflight", str(request), str(deployment)]
    )
    assert result.exit_code != 0
    assert "finite non-negative real" in result.output
