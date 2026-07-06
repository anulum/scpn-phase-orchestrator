# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — single-cell DNB early-warning capstone (Mojtahedi fate)

"""Single-cell DNB early-warning capstone: matched false alarm on a fate bifurcation.

The fifth domain the early-warning design is proven on, and the first molecular one. The
four physical domains (brain, heart, grid, climate) are *many time-samples × few
oscillator nodes*; a cell-fate transition is *few timepoints × many genes*, read by the
dynamical-network-biomarker index (:mod:`bench.dnb_detector`) rather than a
sliding-window monitor. The canonical single-cell demonstration is Mojtahedi et al. 2016
(*PLoS Biol* 14:e2000640): EML multipotent progenitors pushed toward the erythroid or
myeloid fate pass through a bifurcation, and their transition index — the ratio of mean
gene–gene to mean cell–cell correlation — rises sharply to a peak at the tipping point
(shared Day-0 baseline 0.143 rising to a Day-3 peak of 0.40 in the erythroid arm, a
2.8-fold rise). This capstone asks the programme's operational question of that
celebrated result: held to a *matched false-alarm* operating point and a
label-permutation test — the identical protocol the SCPN suite and the AR(1)-Kendall-τ
competitor are scored by — does the DNB rise beat chance across the lineages?

Corpus and its honest limits
----------------------------
The transition-index trajectory is taken **from the published supplement** (Mojtahedi et
al. 2016, Table S2: the per-lineage transition index and its bootstrap standard error at
Days 0, 1, 3, 6), not re-derived from the raw single-cell qPCR. That is a deliberate,
honest choice: the raw Ct matrix carries heavy non-detects and an unspecified
normalisation, so a re-derivation reproduces the published rise only qualitatively; the
authors' peer-reviewed index values are the sounder input. The index definition itself
is reproduced and validated in code — ``single_cell_transition_index`` in
:mod:`bench.dnb_detector` computes exactly the published form (mean signed gene–gene
over mean signed cell–cell correlation; the published index equals that ratio of the
two published correlations). Three limits belong on this proof, not buried:

* **A published-summary resampling.** With one trajectory per lineage there is no
  separate no-transition recording, so the matched-false-alarm null is a *temporally
  shuffled surrogate*: each day's index is resampled from its published mean and
  standard error, then the day order is shuffled, destroying the trend. This is the
  standard early-warning surrogate, driven by the paper's own bootstrap error.
* **A tiny corpus.** Three lineages give three real transitions, so the test has low
  power by construction; a silence bounds demonstrated skill, it is not a proof of
  impossibility.
* **A coarse temporal resolution.** Four published timepoints make a rank-correlation
  trend too discrete to resolve a 10 % operating point, so the rising limb is scored by
  a continuous least-squares slope (:func:`bench.dnb_detector.dnb_regression_slope`) —
  the temporal resolution, not the effect size, is the binding constraint.

The rising limb (Days 0, 1, 3, up to the peak) of each lineage is the operational
pre-transition window; its slope is the score; the matched false-alarm threshold is
calibrated on the pooled shuffled surrogates; and the transition alarm count is tested
by the shared label-permutation core, so the p-value is directly comparable to the four
physical domains. Only the derived, hash-sealed artefact is written; no raw data is.

References
----------
* Mojtahedi, Skupin, Zhou, Castaño, Leong-Quong, Chang, … Huang 2016, *PLoS Biol*
  14:e2000640 — the single-cell transition index at a leukaemic fate bifurcation
  (Table S2: transition index and bootstrap SE per lineage per day).
* Chen, Liu, Liu & Aihara 2012, *Sci Rep* 2:342 — the dynamical network biomarker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bench.dnb_detector import DnbSignificance, dnb_regression_slope, dnb_significance
from bench.early_warning_domain import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]

#: Number of leading days that make the pre-transition rising limb (Days 0, 1, 3 up to
#: the Day-3 peak); the falling post-peak Day-6 point is excluded from the rise score.
RISING_LIMB_DAYS = 3
#: Target false-alarm rate the slope threshold is held at or below.
TARGET_FALSE_ALARM = DEFAULT_TARGET_FALSE_ALARM
#: Surrogate resamples drawn per lineage for the matched-false-alarm null distribution.
DEFAULT_RESAMPLES = 2000
#: Seed of the surrogate resampling, so the sealed artefact is byte-reproducible.
DEFAULT_SEED = DEFAULT_PERMUTATION_SEED

__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "MOJTAHEDI_LINEAGES",
    "RISING_LIMB_DAYS",
    "TARGET_FALSE_ALARM",
    "MojtahediLineage",
    "dnb_corpus",
    "evaluate_mojtahedi",
    "main",
    "mojtahedi_verdict",
    "surrogate_trajectories",
]


@dataclass(frozen=True)
class MojtahediLineage:
    """One lineage's published transition-index trajectory and its provenance.

    Attributes
    ----------
    lineage_id : str
        Fate-driving condition, e.g. ``erythroid_epo``.
    day_labels : tuple of int
        The measurement days (0, 1, 3, 6).
    index_means : tuple of float
        The published transition index at each day (Mojtahedi 2016 Table S2).
    index_ses : tuple of float
        The published bootstrap standard error of the index at each day.
    citation : str
        Original data-source citation, carried into the sealed provenance.
    """

    lineage_id: str
    day_labels: tuple[int, ...]
    index_means: tuple[float, ...]
    index_ses: tuple[float, ...]
    citation: str

    def rising_limb(self) -> tuple[FloatArray, FloatArray]:
        """Return the (means, standard errors) of the pre-transition rising limb.

        Returns
        -------
        tuple[FloatArray, FloatArray]
            The first :data:`RISING_LIMB_DAYS` days' index means and standard errors —
            the operational pre-transition window up to the Day-3 peak.
        """
        means = np.asarray(self.index_means[:RISING_LIMB_DAYS], dtype=np.float64)
        ses = np.asarray(self.index_ses[:RISING_LIMB_DAYS], dtype=np.float64)
        return means, ses


#: The three Mojtahedi et al. 2016 lineages, transcribed from the published Table S2
#: (transition index and its bootstrap standard error at Days 0, 1, 3, 6). The Day-0
#: baseline is shared — all arms start from the same untreated progenitor population.
MOJTAHEDI_LINEAGES: tuple[MojtahediLineage, ...] = (
    MojtahediLineage(
        lineage_id="erythroid_epo",
        day_labels=(0, 1, 3, 6),
        index_means=(
            0.142912793234395,
            0.148789966145399,
            0.399752586178157,
            0.251935631232389,
        ),
        index_ses=(0.0112, 0.0102, 0.0087, 0.0112),
        citation=(
            "Mojtahedi et al. 2016, PLoS Biol 14:e2000640, Table S2 — EPO-treated "
            "erythroid arm transition index (mean signed gene-gene over cell-cell "
            "correlation) and bootstrap SE"
        ),
    ),
    MojtahediLineage(
        lineage_id="myeloid_gmcsf_il3",
        day_labels=(0, 1, 3, 6),
        index_means=(
            0.142912793234395,
            0.156885888646943,
            0.327320823946196,
            0.164584194345332,
        ),
        index_ses=(0.0112, 0.0099, 0.0126, 0.0118),
        citation=(
            "Mojtahedi et al. 2016, PLoS Biol 14:e2000640, Table S2 — GM-CSF/IL-3-"
            "treated myeloid arm transition index and bootstrap SE"
        ),
    ),
    MojtahediLineage(
        lineage_id="combined_epo_gmcsf",
        day_labels=(0, 1, 3, 6),
        index_means=(
            0.142912793234395,
            0.131008857018087,
            0.199224655628324,
            0.106658433664927,
        ),
        index_ses=(0.0112, 0.0108, 0.0110, 0.0088),
        citation=(
            "Mojtahedi et al. 2016, PLoS Biol 14:e2000640, Table S2 — combined "
            "EPO + GM-CSF arm transition index and bootstrap SE"
        ),
    ),
)


def surrogate_trajectories(
    means: FloatArray,
    ses: FloatArray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> list[FloatArray]:
    """Return temporally-shuffled surrogate resamples of one rising limb.

    The matched-false-alarm null: each day's index is drawn from its published mean and
    standard error, then the day order is shuffled, destroying the temporal trend while
    preserving the marginal spread. The slope of a shuffled surrogate regresses toward
    zero, so the pooled surrogates form the no-trend null the threshold calibrates on.

    Parameters
    ----------
    means : FloatArray
        The rising-limb index means, shape ``(RISING_LIMB_DAYS,)``.
    ses : FloatArray
        The rising-limb index standard errors, same shape.
    n_resamples : int
        Number of surrogate trajectories to draw.
    rng : numpy.random.Generator
        The seeded generator, so the surrogate set is reproducible.

    Returns
    -------
    list[FloatArray]
        ``n_resamples`` shuffled, resampled trajectories.
    """
    return [rng.permutation(rng.normal(means, ses)) for _ in range(n_resamples)]


def dnb_corpus(
    lineages: Sequence[MojtahediLineage],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Build the transition and surrogate-null corpus from the published trajectories.

    The observed transition trajectories are the lineages' real-order rising limbs (the
    published means); the null is the pooled temporally-shuffled surrogates across all
    lineages, so a matched false alarm is calibrated on many no-trend trajectories while
    the significance test sees exactly the real transitions.

    Parameters
    ----------
    lineages : sequence of MojtahediLineage
        The corpus of lineage trajectories.
    n_resamples : int
        Surrogate resamples drawn per lineage.
    seed : int
        Seed of the surrogate resampling.

    Returns
    -------
    tuple[list[FloatArray], list[FloatArray]]
        The observed transition trajectories and the pooled surrogate null trajectories.
    """
    rng = np.random.default_rng(seed)
    transitions: list[FloatArray] = []
    nulls: list[FloatArray] = []
    for lineage in lineages:
        means, ses = lineage.rising_limb()
        transitions.append(means)
        nulls.extend(
            surrogate_trajectories(means, ses, n_resamples=n_resamples, rng=rng)
        )
    return transitions, nulls


def mojtahedi_verdict(significance: DnbSignificance, n_lineages: int) -> str:
    """Return a one-sentence honest reading of the sealed result.

    Parameters
    ----------
    significance : DnbSignificance
        The sealed matched-false-alarm significance result.
    n_lineages : int
        Number of lineages (real transitions) tested.

    Returns
    -------
    str
        A clinical summary naming the alarm count, the operating point, and the p-value.
    """
    led = significance.significance.observed_led
    p_value = significance.significance.p_value
    reach = "beats chance" if p_value <= 0.05 else "does not reach significance"
    return (
        f"{led} of {n_lineages} lineages' DNB rise cleared the "
        f"{TARGET_FALSE_ALARM:.0%} matched false-alarm operating point; the DNB "
        f"transition index rises toward the fate bifurcation but {reach} across the "
        f"corpus (permutation p={p_value:.3f})."
    )


def evaluate_mojtahedi(
    lineages: Sequence[MojtahediLineage] = MOJTAHEDI_LINEAGES,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    target_fa: float = TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
) -> dict[str, object]:
    """Score the single-cell DNB corpus and return the hash-sealed result payload.

    Builds the transition/surrogate corpus (:func:`dnb_corpus`), scores each rising limb
    by its least-squares slope, calibrates a matched false alarm on the pooled
    surrogates, tests the transition alarm count by the shared label-permutation core,
    and seals the whole result with the canonical-JSON SHA-256 the assurance layer uses.

    Parameters
    ----------
    lineages : sequence of MojtahediLineage
        The corpus of lineage trajectories.
    n_resamples : int
        Surrogate resamples drawn per lineage.
    seed : int
        Seed of the surrogate resampling.
    target_fa : float
        Target false-alarm rate the slope threshold is held at or below.
    n_permutations : int
        Number of random relabellings for the significance test.

    Returns
    -------
    dict[str, object]
        The JSON-safe, hash-sealed result payload (with a ``content_hash`` field).
    """
    transitions, nulls = dnb_corpus(lineages, n_resamples=n_resamples, seed=seed)
    result = dnb_significance(
        transitions,
        nulls,
        score=dnb_regression_slope,
        target_fa=target_fa,
        n_permutations=n_permutations,
        seed=seed,
    )
    threshold = result.score_threshold
    lineage_records = [
        {
            "lineage_id": lineage.lineage_id,
            "day_labels": list(lineage.day_labels),
            "index_means": list(lineage.index_means),
            "rising_limb_slope": dnb_regression_slope(lineage.rising_limb()[0]),
            "alarmed": bool(
                dnb_regression_slope(lineage.rising_limb()[0]) >= threshold
            ),
            "citation": lineage.citation,
        }
        for lineage in lineages
    ]
    payload: dict[str, object] = {
        "benchmark": "early_warning_dnb_mojtahedi",
        "corpus": (
            "Mojtahedi et al. 2016 single-cell leukaemic fate bifurcation "
            "(published transition index, Table S2)"
        ),
        "modality": (
            "single-cell dynamical network biomarker (cross-cell transition index)"
        ),
        "index_definition": (
            "mean signed gene-gene / mean signed cell-cell correlation (Mojtahedi 2016)"
        ),
        "score": "least-squares slope of the transition index over the rising limb",
        "rising_limb_days": [
            lineages[0].day_labels[day] for day in range(RISING_LIMB_DAYS)
        ],
        "target_false_alarm": target_fa,
        "n_resamples_per_lineage": n_resamples,
        "n_null_trajectories": len(nulls),
        "seed": seed,
        "matched_false_alarm_threshold": threshold,
        "achieved_false_alarm": result.achieved_false_alarm,
        "lineages": lineage_records,
        "permutation_significance": result.significance.to_audit_record(),
        "verdict": mojtahedi_verdict(result, len(lineages)),
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def main(output_dir: str | Path) -> None:
    """Run the single-cell DNB capstone and write the sealed artefact.

    All input is the embedded published summary, so this runs with no external data and
    writes one hash-sealed aggregate JSON to ``output_dir``.

    Parameters
    ----------
    output_dir : str or Path
        Directory the sealed derived artefact is written to.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = evaluate_mojtahedi()
    (out / "early_warning_dnb_mojtahedi.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    significance = payload["permutation_significance"]
    assert isinstance(significance, dict)
    print(payload["verdict"])
    print(
        f"permutation p-value {significance['p_value']:.3f} "
        f"({significance['observed_led']}/{significance['n_transitions']} led vs "
        f"{significance['expected_led']:.2f} expected by chance)"
    )
    print(f"content hash {payload['content_hash']}")
    print(f"results written to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over the tested logic
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="directory for the sealed derived output")
    arguments = parser.parse_args()
    main(arguments.output_dir)
