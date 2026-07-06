# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — first-class registry of the domain-specific detectors

"""A first-class registry of the domain-specific early-warning detectors.

The generic suite (:mod:`bench.early_warning_domain`) is modality-neutral, which is what
makes the four-domain replication possible — and is why its *detection* is a commodity.
The domain-specific detectors each read the raw physical quantity whose growth *is* the
instability, and each is certified through the same matched-false-alarm + permutation
moat, so a domain-specific p-value is directly comparable to the generic suite.

This module is the single first-class catalogue of that suite. For each detector it
records the domain and modality it serves, the physical quantity it reads, the callable
that scores it through the moat, its literature references, and — the point of an honest
registry — its **certified status**: ``"functional"`` where the domain carries a
deterministic signature that clears the operational bar, ``"at_chance"`` where it does
not, each with a pointer to the sealed evidence. The registry is therefore both the
discovery surface a real-time runtime looks a detector up through and the honest ledger
of what each detector has been *shown* to do — never a claim beyond the sealed result.

The three significance callables have deliberately different signatures (raw multi-bus
voltages for the grid, multichannel signals for the EEG, cross-sample index trajectories
for the DNB), because a domain-specific detector reads its own modality; the registry
catalogues them, it does not force a uniform call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from bench.dnb_detector import dnb_significance
from bench.grid_oscillation_detector import modal_growth_significance
from bench.seizure_detector import seizure_significance

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Callable

#: A detector's honest certified verdict from the sealed evidence.
DetectorStatus = Literal["functional", "at_chance"]

__all__ = [
    "DomainDetector",
    "detectors_for_domain",
    "functional_detectors",
    "get_detector",
    "registered_detectors",
]


@dataclass(frozen=True)
class DomainDetector:
    """A registered domain-specific detector and its certified status.

    Attributes
    ----------
    name : str
        The registry key, e.g. ``"grid_modal_growth"``.
    domain : str
        The domain served, e.g. ``"power_grid"``.
    modality : str
        The data modality the detector reads.
    quantity : str
        The physical quantity whose growth the detector measures.
    status : DetectorStatus
        ``"functional"`` if the detector clears the matched-false-alarm bar on real
        data, ``"at_chance"`` if the honest test finds no operational skill.
    summary : str
        The one-line certified result, straight from the sealed evidence.
    references : tuple of str
        The literature the detector's feature and corpus come from.
    significance : Callable
        The callable that scores the detector through the shared moat and returns a
        result carrying a ``significance`` permutation record. Its signature is
        modality-specific.
    evidence : str or None
        Repository-relative path to the sealed evidence artefact, if one exists.
    """

    name: str
    domain: str
    modality: str
    quantity: str
    status: DetectorStatus
    summary: str
    references: tuple[str, ...]
    significance: Callable[..., object]
    evidence: str | None


_REGISTRY: dict[str, DomainDetector] = {
    detector.name: detector
    for detector in (
        DomainDetector(
            name="grid_modal_growth",
            domain="power_grid",
            modality="wide-area PMU bus-voltage magnitudes",
            quantity=(
                "exponential growth rate σ of the most unstable bus's cross-bus "
                "voltage deviation (the dominant mode's eigenvalue real part)"
            ),
            status="functional",
            summary=(
                "leads 36/90 growing-instability transitions (p=0.0001) where every "
                "generic member is at chance (best 13/90); held-out 24/45 (p=0.0002)"
            ),
            references=("Zheng et al. 2021 (PSML)", "Kundur 1994"),
            significance=modal_growth_significance,
            evidence="examples/real_data/psml_modal_growth/grid_modal_head_to_head.json",
        ),
        DomainDetector(
            name="eeg_spectral_rise",
            domain="scalp_eeg",
            modality="multichannel scalp EEG",
            quantity=(
                "rising beta (13–30 Hz) / delta (0.5–4 Hz) band-power ratio "
                "(Kendall-τ trend over the pre-onset segment)"
            ),
            status="at_chance",
            summary=(
                "1/6 preictal seizures led (p≈0.56, both aggregations); the preictal "
                "state is murky and the corpus small — a domain detector need not win"
            ),
            references=("Shoeb 2009 (CHB-MIT)", "Mormann et al. 2007"),
            significance=seizure_significance,
            evidence=None,
        ),
        DomainDetector(
            name="dnb_transition_index",
            domain="single_cell",
            modality="single-cell / bulk transcriptomics (cross-sample statistic)",
            quantity=(
                "dynamical-network-biomarker transition index over the critical module "
                "(slope of its pre-transition rising limb)"
            ),
            status="at_chance",
            summary=(
                "single-cell 1/3 lineages clear (p=0.27); bulk GSE2565 rise is a "
                "selection artefact under a re-selecting surrogate null (p=0.39)"
            ),
            references=("Chen et al. 2012", "Mojtahedi et al. 2016"),
            significance=dnb_significance,
            evidence="examples/real_data/mojtahedi_fate/early_warning_dnb_mojtahedi.json",
        ),
    )
}


def registered_detectors() -> tuple[DomainDetector, ...]:
    """Return every registered domain-specific detector, ordered by name."""
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def get_detector(name: str) -> DomainDetector:
    """Return the registered detector with ``name``.

    Parameters
    ----------
    name : str
        The registry key.

    Returns
    -------
    DomainDetector
        The registered detector.

    Raises
    ------
    KeyError
        If no detector is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no domain detector registered as {name!r}") from None


def detectors_for_domain(domain: str) -> tuple[DomainDetector, ...]:
    """Return the registered detectors serving ``domain``, ordered by name."""
    return tuple(
        detector for detector in registered_detectors() if detector.domain == domain
    )


def functional_detectors() -> tuple[DomainDetector, ...]:
    """Return the functional-certified detectors (they clear the bar on real data)."""
    return tuple(
        detector
        for detector in registered_detectors()
        if detector.status == "functional"
    )
