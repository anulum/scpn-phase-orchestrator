# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — digital-twin operator-evidence guard tests

"""Guard coverage for digital-twin operator-evidence summarisation.

These tests drive the remaining rejection and fold paths of
:mod:`scpn_phase_orchestrator.binding.digital_twin.evidence`: the residual extractor
(non-finite value, absent key) and the threshold validator (non-real, negative or
non-finite) directly, and the rejected-validation / reason-less-rejection folds of
``build_digital_twin_operator_evidence`` through a real contract and envelope.
"""

from __future__ import annotations

import math

import pytest

from scpn_phase_orchestrator.binding import (
    build_digital_twin_binding_contract,
    build_digital_twin_operator_evidence,
    build_digital_twin_sync_envelope,
    load_binding_spec,
)
from scpn_phase_orchestrator.binding.digital_twin.envelope import (
    DigitalTwinTransportValidation,
)
from scpn_phase_orchestrator.binding.digital_twin.evidence import (
    _extract_twin_residual,
    _validated_residual_threshold,
)

_SPEC_PATH = "domainpacks/digital_twin_nchannel/binding_spec.yaml"


def _contract() -> object:
    """Build the digital-twin binding contract from the packaged spec."""
    return build_digital_twin_binding_contract(load_binding_spec(_SPEC_PATH))


# --- _extract_twin_residual -----------------------------------------------


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_residual_extraction_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="must be a finite real value"):
        _extract_twin_residual({"residual": value})


def test_residual_extraction_returns_none_when_absent() -> None:
    assert _extract_twin_residual({"unrelated": 1.0}) is None


def test_residual_extraction_skips_none_valued_keys() -> None:
    assert _extract_twin_residual({"residual": None}) is None


# --- _validated_residual_threshold ----------------------------------------


@pytest.mark.parametrize("value", ["0.1", True, None])
def test_threshold_rejects_non_real(value: object) -> None:
    with pytest.raises(ValueError, match="finite non-negative real value"):
        _validated_residual_threshold(value, "residual_warning_threshold")


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_threshold_rejects_negative_or_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative real value"):
        _validated_residual_threshold(value, "residual_critical_threshold")


# --- build_digital_twin_operator_evidence folds ---------------------------


def test_evidence_folds_a_rejected_validation_reason() -> None:
    contract = _contract()
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=3,
        payload={"layer": "machine_cells", "R": 0.9},
    )
    rejected_validation = DigitalTwinTransportValidation(
        accepted=False, reason="stale_sequence", envelope=envelope
    )

    evidence = build_digital_twin_operator_evidence(contract, [rejected_validation])

    assert "stale_sequence" in evidence.mismatch_reasons
    assert evidence.accepted_count == 0
    assert evidence.rejected_count == 1
    assert evidence.status == "degraded"


def test_evidence_defaults_a_reason_less_rejection() -> None:
    contract = _contract()

    evidence = build_digital_twin_operator_evidence(
        contract, [], rejected=[{"note": "no reason field"}]
    )

    assert "rejected" in evidence.mismatch_reasons
    assert evidence.rejected_count == 1
