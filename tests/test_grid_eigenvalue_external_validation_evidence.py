# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — cross-system external-validation sealed evidence integrity

"""Integrity tests over the committed cross-system external-validation artefact.

These pin the sealed result without re-running ANDES: the ``content_hash`` recomputes
from the committed rows, the coherent (mean/spatial) aggregation recovers the true
eigenvalue σ on both independent systems, and the focal aggregation — the PSML winner —
does not, so the best aggregation is regime-dependent.
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
    / "grid_external_validation"
    / "grid_eigenvalue_external_validation.json"
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def _system(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in payload["systems"] if s["name"] == name)


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    sealed = dict(payload)
    stored = sealed.pop("content_hash")
    assert canonical_record_hash(sealed) == stored


def test_two_independent_systems_are_sealed(payload: dict[str, Any]) -> None:
    assert payload["benchmark"] == "grid_eigenvalue_external_validation"
    names = {s["name"] for s in payload["systems"]}
    assert names == {"ieee39", "kundur"}


@pytest.mark.parametrize("name", ["ieee39", "kundur"])
def test_coherent_aggregation_recovers_true_sigma(
    payload: dict[str, Any], name: str
) -> None:
    system = _system(payload, name)
    mean = system["aggregations"]["mean"]["correlation"]["spearman"]
    spatial = system["aggregations"]["spatial"]["correlation"]["spearman"]
    # the coherent aggregation tracks the true sigma (positive rank correlation)
    assert max(mean, spatial) > 0.4


@pytest.mark.parametrize("name", ["ieee39", "kundur"])
def test_focal_aggregation_does_not_transfer(
    payload: dict[str, Any], name: str
) -> None:
    # the PSML winner (focal) fails to track the true sigma on the slow inter-area mode
    focal = _system(payload, name)["aggregations"]["focal"]["correlation"]["spearman"]
    assert focal < 0.0


def test_ieee39_is_the_strongest_recovery(payload: dict[str, Any]) -> None:
    mean = _system(payload, "ieee39")["aggregations"]["mean"]["correlation"]["spearman"]
    assert mean > 0.8  # the clean monotonic IEEE-39 sweep gives the tightest tracking


def test_verdict_states_regime_dependent_aggregation(payload: dict[str, Any]) -> None:
    assert "regime-dependent" in payload["verdict"]
    assert "generalises" in payload["verdict"]
