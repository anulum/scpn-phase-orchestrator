# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hopf-bridge external-validation sealed evidence integrity

"""Integrity tests over the committed Hopf-bridge external-validation artefact.

These pin the sealed result without re-integrating: the ``content_hash`` recomputes from
the committed rows; the envelope-growth family recovers α in magnitude while the
autocorrelation family's magnitude is confounded; the recovery is
frequency-invariant and its SNR degradation is recorded.
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
    / "hopf_bridge"
    / "hopf_bridge_external_validation.json"
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    sealed = dict(payload)
    stored = sealed.pop("content_hash")
    assert canonical_record_hash(sealed) == stored


def test_benchmark_identity(payload: dict[str, Any]) -> None:
    assert payload["benchmark"] == "hopf_bridge_external_validation"


def test_envelope_family_recovers_alpha_in_magnitude(payload: dict[str, Any]) -> None:
    env = payload["record"]["families"]["envelope_growth"]["correlation"]
    assert env["spearman"] > 0.9
    assert env["mean_abs_error"] < 0.1  # tight magnitude recovery


def test_autocorrelation_family_magnitude_is_confounded(
    payload: dict[str, Any],
) -> None:
    ac = payload["record"]["families"]["autocorrelation"]["correlation"]
    # it may correlate, but its magnitude gap is an order larger than the envelope's
    env_mae = payload["record"]["families"]["envelope_growth"]["correlation"][
        "mean_abs_error"
    ]
    assert ac["mean_abs_error"] > 10.0 * env_mae


def test_recovery_is_frequency_invariant(payload: dict[str, Any]) -> None:
    assert payload["frequency_invariance"]["pearson_spread"] < 0.05


def test_snr_degradation_is_recorded(payload: dict[str, Any]) -> None:
    # the envelope recovery falls monotonically as the ringdown noise rises — a physical
    # floor, recorded rather than hidden
    curve = payload["snr_robustness"]
    sigmas = [e["ring_sigma"] for e in curve]
    pearsons = [e["pearson"] for e in curve]
    assert sigmas == sorted(sigmas)
    assert pearsons[0] > pearsons[-1]


def test_verdict_states_the_bridge(payload: dict[str, Any]) -> None:
    assert "regime-dependent" in payload["verdict"]
    assert "MAGNITUDE" in payload["verdict"]
