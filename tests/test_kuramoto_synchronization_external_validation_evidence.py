# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Kuramoto external-validation sealed evidence integrity

"""Integrity tests over the committed Kuramoto synchronisation artefact.

These pin the sealed result without re-integrating the model: the ``content_hash``
recomputes from the committed rows, both mean-field observables track the analytic
eigenvalue in rank, and only the signed ``Re(Z)`` recovers it in magnitude (a near-unit
fitted slope), while the folded ``|Z|`` sizes it at roughly twice the slope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_ARTEFACT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "real_data"
    / "kuramoto_synchronization"
    / "kuramoto_synchronization_external_validation.json"
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def _correlation(payload: dict[str, Any], label: str) -> dict[str, Any]:
    return payload["record"]["observables"][label]["correlation"]


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    sealed = dict(payload)
    stored = sealed.pop("content_hash")
    assert canonical_record_hash(sealed) == stored


def test_benchmark_and_two_observables_sealed(payload: dict[str, Any]) -> None:
    assert payload["benchmark"] == "kuramoto_synchronization_external_validation"
    observables = set(payload["record"]["observables"])
    assert observables == {"mean_field_real", "order_parameter_amplitude"}


def test_critical_coupling_matches_noisy_mean_field(payload: dict[str, Any]) -> None:
    # K_c = 2(γ + D) for the sealed γ = D = 0.5
    assert payload["record"]["critical_coupling"] == pytest.approx(
        2.0 * (payload["gamma"] + payload["diffusion"])
    )


@pytest.mark.parametrize("label", ["mean_field_real", "order_parameter_amplitude"])
def test_both_observables_track_lambda_in_rank(
    payload: dict[str, Any], label: str
) -> None:
    # the collective coordinate undergoes critical slowing down on both observables
    assert _correlation(payload, label)["spearman"] > 0.9


def test_signed_component_recovers_lambda_in_magnitude(
    payload: dict[str, Any],
) -> None:
    signed = _correlation(payload, "mean_field_real")
    # Re(Z) fits λ with near-unit slope: it sizes the eigenvalue, not just ranks it
    assert signed["slope"] == pytest.approx(1.0, abs=0.4)


def test_amplitude_only_ranks_lambda(payload: dict[str, Any]) -> None:
    amplitude = _correlation(payload, "order_parameter_amplitude")
    # the folded |Z| ranks λ but its fitted slope is far from one (near two)
    assert amplitude["slope"] > 1.6
    assert (
        amplitude["mean_abs_gap"]
        > _correlation(payload, "mean_field_real")["mean_abs_gap"]
    )


def test_verdict_confirms_collective_slowing_down(payload: dict[str, Any]) -> None:
    assert "recovers λ in magnitude" in payload["verdict"]
    assert "first-principles" in payload["verdict"]
