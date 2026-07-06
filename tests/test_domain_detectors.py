# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — domain-detector registry tests

"""Tests for the first-class domain-specific detector registry.

The registry catalogues the three domain-specific detectors with their certified status
(functional vs at-chance), maps each to its scoring callable and sealed evidence, and
supports discovery by name and by domain. These tests pin the catalogue's contents, its
honest status ledger, and every lookup path including the error path, and check that the
claimed sealed evidence artefacts actually exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.dnb_detector import dnb_significance
from bench.domain_detectors import (
    DomainDetector,
    detectors_for_domain,
    functional_detectors,
    get_detector,
    registered_detectors,
)
from bench.grid_oscillation_detector import modal_growth_significance
from bench.seizure_detector import seizure_significance

_REPO = Path(__file__).resolve().parents[1]


def test_registry_catalogues_the_three_domain_detectors() -> None:
    detectors = registered_detectors()
    assert [detector.name for detector in detectors] == [
        "dnb_transition_index",
        "eeg_spectral_rise",
        "grid_modal_growth",
    ]  # ordered by name
    assert all(isinstance(detector, DomainDetector) for detector in detectors)


def test_each_detector_maps_to_its_scoring_callable() -> None:
    by_name = {detector.name: detector for detector in registered_detectors()}
    assert by_name["grid_modal_growth"].significance is modal_growth_significance
    assert by_name["eeg_spectral_rise"].significance is seizure_significance
    assert by_name["dnb_transition_index"].significance is dnb_significance
    assert all(callable(detector.significance) for detector in registered_detectors())


def test_status_ledger_is_honest() -> None:
    by_name = {detector.name: detector for detector in registered_detectors()}
    # only the grid detector clears the bar; the other two are honestly at chance
    assert by_name["grid_modal_growth"].status == "functional"
    assert by_name["eeg_spectral_rise"].status == "at_chance"
    assert by_name["dnb_transition_index"].status == "at_chance"
    assert {detector.status for detector in registered_detectors()} <= {
        "functional",
        "at_chance",
    }


def test_get_detector_returns_the_named_detector() -> None:
    detector = get_detector("grid_modal_growth")
    assert detector.domain == "power_grid"
    assert "σ" in detector.quantity
    assert detector.references


def test_get_detector_raises_for_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="no domain detector registered as 'nope'"):
        get_detector("nope")


def test_detectors_for_domain_filters_by_domain() -> None:
    grid = detectors_for_domain("power_grid")
    assert [detector.name for detector in grid] == ["grid_modal_growth"]
    assert detectors_for_domain("mycology") == ()  # an unserved domain


def test_functional_detectors_are_only_those_that_clear_the_bar() -> None:
    functional = functional_detectors()
    assert [detector.name for detector in functional] == ["grid_modal_growth"]
    assert all(detector.status == "functional" for detector in functional)


def test_sealed_evidence_paths_exist_where_claimed() -> None:
    by_name = {detector.name: detector for detector in registered_detectors()}
    # the grid and single-cell DNB detectors point at committed sealed artefacts
    for name in ("grid_modal_growth", "dnb_transition_index"):
        evidence = by_name[name].evidence
        assert evidence is not None
        assert (_REPO / evidence).is_file()
    # the at-chance EEG detector has no sealed capstone artefact
    assert by_name["eeg_spectral_rise"].evidence is None
