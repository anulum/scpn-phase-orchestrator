# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — bimodal Kuramoto external-validation evidence integrity

"""Integrity tests over the committed bimodal Kuramoto Hopf artefact.

These pin the sealed result without re-integrating the model: the ``content_hash``
recomputes from the committed rows, both families track the analytic complex eigenvalue
in rank, only the envelope-growth family recovers it in magnitude (a near-unit fitted
slope) while the autocorrelation family is confounded by the oscillation (a
far-from-unit slope), and the measured oscillation frequency lies in the physical band,
consistent with a genuinely complex eigenvalue.
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
    / "kuramoto_bimodal"
    / "kuramoto_bimodal_hopf_external_validation.json"
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def _correlation(payload: dict[str, Any], label: str) -> dict[str, Any]:
    return payload["record"]["families"][label]["correlation"]


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    sealed = dict(payload)
    stored = sealed.pop("content_hash")
    assert canonical_record_hash(sealed) == stored


def test_benchmark_and_two_families_sealed(payload: dict[str, Any]) -> None:
    assert payload["benchmark"] == "kuramoto_bimodal_hopf_external_validation"
    families = set(payload["record"]["families"])
    assert families == {"envelope_growth", "autocorrelation"}


def test_critical_coupling_is_four_delta(payload: dict[str, Any]) -> None:
    assert payload["record"]["critical_coupling"] == pytest.approx(
        4.0 * payload["delta"]
    )


@pytest.mark.parametrize("label", ["envelope_growth", "autocorrelation"])
def test_both_families_track_eigenvalue_in_rank(
    payload: dict[str, Any], label: str
) -> None:
    # the collective damped oscillation slows down as K -> K_c on both families
    assert _correlation(payload, label)["spearman"] > 0.9


def test_envelope_family_recovers_eigenvalue_in_magnitude(
    payload: dict[str, Any],
) -> None:
    envelope = _correlation(payload, "envelope_growth")
    # the sub-population modulus fits Re(lambda) with near-unit slope
    assert envelope["slope"] == pytest.approx(1.0, abs=0.4)


def test_autocorrelation_family_only_ranks(payload: dict[str, Any]) -> None:
    autocorr = _correlation(payload, "autocorrelation")
    # the oscillation confounds the AR1: its fitted slope is far from one
    assert autocorr["slope"] > 1.6
    assert (
        autocorr["mean_abs_gap"]
        > _correlation(payload, "envelope_growth")["mean_abs_gap"]
    )


def test_measured_frequency_lies_in_physical_band(payload: dict[str, Any]) -> None:
    # the collective mode oscillates below ~1.5 omega0, confirming a complex eigenvalue
    measured = payload["record"]["frequency"]["measured"]
    bound = 1.5 * payload["omega0"]
    assert all(0.0 < f <= bound for f in measured)


def test_measured_frequency_matches_analytic_in_value(payload: dict[str, Any]) -> None:
    # the analytic Omega is nearly constant across the sweep, so the check is on value:
    # the measured frequency matches the analytic Omega to a small absolute error
    frequency = payload["record"]["frequency"]
    analytic = frequency["analytic"]
    assert max(analytic) - min(analytic) < 0.2  # Omega is nearly constant here
    assert frequency["mean_abs_error"] < 0.3


def test_verdict_confirms_envelope_sizes_the_eigenvalue(
    payload: dict[str, Any],
) -> None:
    assert "recovers Re(λ) in magnitude" in payload["verdict"]
    assert "first-principles" in payload["verdict"]
