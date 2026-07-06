# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CSD external-validation sealed evidence integrity

"""Integrity tests over the committed CSD external-validation artefact.

These pin the sealed result without re-running the normal forms: the ``content_hash``
recomputes from the committed rows, and on both independent bifurcation classes the
detector's autocorrelation channel recovers the true Jacobian eigenvalue λ while the
variance channel rises in step.
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
    / "csd_bifurcation"
    / "csd_bifurcation_external_validation.json"
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def _record(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return next(b for b in payload["bifurcations"] if b["name"] == name)


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    sealed = dict(payload)
    stored = sealed.pop("content_hash")
    assert canonical_record_hash(sealed) == stored


def test_two_independent_bifurcation_classes_are_sealed(
    payload: dict[str, Any],
) -> None:
    assert payload["benchmark"] == "csd_bifurcation_external_validation"
    names = {b["name"] for b in payload["bifurcations"]}
    assert names == {"fold", "pitchfork"}


@pytest.mark.parametrize("name", ["fold", "pitchfork"])
def test_autocorrelation_recovers_true_lambda(
    payload: dict[str, Any], name: str
) -> None:
    record = _record(payload, name)
    rho = record["indicators"]["autocorrelation"]["correlation"]["spearman"]
    # the AR1-implied rate tracks the true eigenvalue tightly on both classes
    assert rho > 0.9


@pytest.mark.parametrize("name", ["fold", "pitchfork"])
def test_variance_rises_in_step(payload: dict[str, Any], name: str) -> None:
    record = _record(payload, name)
    rho = record["indicators"]["variance"]["correlation"]["spearman"]
    # variance rises as |λ| falls, so it too correlates positively with λ
    assert rho > 0.9


def test_autocorrelation_estimates_lambda_in_magnitude(payload: dict[str, Any]) -> None:
    # ln(AR1)/Δt is a genuine rate: the Pearson correlation (magnitude, not just rank)
    # is high, which the rank-only grid eigenvalue test could not assert
    fold = _record(payload, "fold")
    pearson = fold["indicators"]["autocorrelation"]["correlation"]["pearson"]
    assert pearson > 0.9


def test_verdict_confirms_slowing_down(payload: dict[str, Any]) -> None:
    assert "recovers" in payload["verdict"]
    assert "first-principles" in payload["verdict"]
