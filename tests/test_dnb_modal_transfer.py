# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — DNB modal-growth transfer benchmark tests

"""Tests for the DNB modal-growth transfer benchmark's pure core.

The growth-form characterisation, the resolution-limit verdict, and the payload sealing
are exercised on synthetic index trajectories: a well-fit exponential the gate keeps, a
flat limb with no growth, and a step whose poor fit the gate rejects. The corpus loading
(the in-code single-cell limbs and the GSE2565 bulk I/O) is pragma-excluded, so only the
tested pure functions are measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.dnb_modal_transfer import (
    MIN_RESOLVABLE_POINTS,
    dnb_transfer_payload,
    dnb_transfer_verdict,
    growth_form_record,
)

# --------------------------------------------------------------------------- #
# growth_form_record                                                          #
# --------------------------------------------------------------------------- #


def test_growth_form_record_keeps_a_well_fit_exponential() -> None:
    record = growth_form_record(np.exp(0.3 * np.arange(8)), label="exp")
    assert record["label"] == "exp"
    assert record["n_points"] == 8
    assert record["growth_rate"] == pytest.approx(0.3, abs=1e-6)
    assert record["exponential_fit_r2"] == pytest.approx(1.0, abs=1e-9)
    assert record["gate_keeps"] is True
    assert record["gated_growth_rate"] == record["growth_rate"]


def test_growth_form_record_does_not_keep_a_flat_limb() -> None:
    record = growth_form_record(np.full(8, 1.0), label="flat")
    assert record["growth_rate"] == pytest.approx(0.0, abs=1e-9)
    assert record["gate_keeps"] is False  # no growth to keep


def test_growth_form_record_rejects_a_poorly_fit_step() -> None:
    # a growth trend is present, but the exponential fit is poor, so the gate clamps it
    record = growth_form_record(np.array([0.1, 8, 8, 8, 8, 8, 8, 8.0]), label="step")
    assert record["growth_rate"] > 0.0
    assert record["exponential_fit_r2"] < 0.5
    assert record["gated_growth_rate"] == 0.0
    assert record["gate_keeps"] is False  # kept only when the fit passes the gate


def test_growth_form_record_rejects_too_short_a_trajectory() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        growth_form_record([1.0], label="one")


# --------------------------------------------------------------------------- #
# verdict + payload                                                           #
# --------------------------------------------------------------------------- #


def _record(label: str, *, n_points: int, r2: float) -> dict[str, object]:
    return {
        "label": label,
        "n_points": n_points,
        "slope": 0.1,
        "growth_rate": 0.5,
        "exponential_fit_r2": r2,
        "gated_growth_rate": 0.5,
        "gate_keeps": True,
    }


def test_dnb_transfer_verdict_states_the_resolution_limit() -> None:
    single_cell = [_record(f"lin{i}", n_points=3, r2=0.7 + 0.05 * i) for i in range(3)]
    bulk = _record("bulk", n_points=4, r2=0.80)
    verdict = dnb_transfer_verdict(single_cell, bulk)
    assert "cannot be posed" in verdict
    assert f"{MIN_RESOLVABLE_POINTS}" in verdict  # the resolution threshold
    assert "4 points at most" in verdict  # the longest rising limb


def test_dnb_transfer_payload_seals_a_reproducible_hash() -> None:
    single_cell = [_record("lin0", n_points=3, r2=0.82)]
    bulk = _record("bulk", n_points=4, r2=0.80)
    payload = dnb_transfer_payload(
        single_cell=single_cell,
        bulk=bulk,
        corpus={"single_cell_source": "Mojtahedi", "bulk_source": "GSE2565"},
    )
    assert payload["benchmark"] == "dnb_modal_transfer"
    assert "cannot be posed" in payload["verdict"]

    from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

    sealed = dict(payload)
    stored = sealed.pop("content_hash")
    assert stored == canonical_record_hash(sealed)
