# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — single-cell DNB early-warning capstone tests

"""Tests for the Mojtahedi single-cell DNB early-warning capstone.

The corpus builder, the surrogate resampler, the verdict, and the sealed evaluation are
exercised on the embedded published summary — the capstone needs no external data, so
every path runs here. The sealed result is checked for its honest reading (one of three
lineages clears the matched operating point, not significant) and for byte-reproducible
hashing, and ``main`` is run end-to-end to a temporary directory.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench.dnb_detector import DnbSignificance
from bench.early_warning_dnb import (
    MOJTAHEDI_LINEAGES,
    RISING_LIMB_DAYS,
    MojtahediLineage,
    dnb_corpus,
    evaluate_mojtahedi,
    main,
    mojtahedi_verdict,
    surrogate_trajectories,
)
from bench.early_warning_domain import PermutationSignificance
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash


def test_rising_limb_takes_the_pre_peak_days() -> None:
    lineage = MOJTAHEDI_LINEAGES[0]
    means, ses = lineage.rising_limb()
    assert means.shape == (RISING_LIMB_DAYS,)
    assert ses.shape == (RISING_LIMB_DAYS,)
    assert means[0] == pytest.approx(lineage.index_means[0])
    assert means[-1] > means[0]  # the limb rises toward the peak


def test_surrogate_trajectories_are_shuffled_resamples() -> None:
    rng = np.random.default_rng(0)
    means = np.array([0.1, 0.2, 0.4])
    ses = np.array([0.01, 0.01, 0.01])
    surrogates = surrogate_trajectories(means, ses, n_resamples=5, rng=rng)
    assert len(surrogates) == 5
    for surrogate in surrogates:
        assert surrogate.shape == (3,)
        # resampled near the (permuted) means, so the sorted values track the means
        assert np.allclose(np.sort(surrogate), np.sort(means), atol=0.1)


def test_dnb_corpus_pairs_real_limbs_with_pooled_surrogates() -> None:
    transitions, nulls = dnb_corpus(MOJTAHEDI_LINEAGES, n_resamples=100, seed=0)
    assert len(transitions) == len(MOJTAHEDI_LINEAGES)
    assert len(nulls) == len(MOJTAHEDI_LINEAGES) * 100
    # transitions are the real-order means, so the erythroid arm rises monotonically
    assert transitions[0][-1] > transitions[0][0]


def test_mojtahedi_verdict_reports_a_reached_significance() -> None:
    significant = DnbSignificance(
        score_threshold=0.1,
        achieved_false_alarm=0.1,
        significance=PermutationSignificance(
            observed_led=3,
            n_transitions=3,
            pooled_alarm_rate=0.1,
            expected_led=0.3,
            p_value=0.01,
            n_permutations=100,
            seed=0,
        ),
    )
    assert "beats chance" in mojtahedi_verdict(significant, 3)


def test_mojtahedi_verdict_reports_a_silence() -> None:
    silent = DnbSignificance(
        score_threshold=0.1,
        achieved_false_alarm=0.1,
        significance=PermutationSignificance(
            observed_led=1,
            n_transitions=3,
            pooled_alarm_rate=0.1,
            expected_led=0.3,
            p_value=0.27,
            n_permutations=100,
            seed=0,
        ),
    )
    verdict = mojtahedi_verdict(silent, 3)
    assert "does not reach significance" in verdict
    assert "1 of 3" in verdict


def test_evaluate_mojtahedi_seals_the_honest_result() -> None:
    payload = evaluate_mojtahedi(n_resamples=2000, seed=0)
    significance = payload["permutation_significance"]
    assert isinstance(significance, dict)
    assert significance["observed_led"] == 1  # only the erythroid arm clears
    assert significance["n_transitions"] == 3
    assert significance["p_value"] > 0.05  # not significant on three lineages
    lineages = payload["lineages"]
    assert isinstance(lineages, list)
    assert lineages[0]["alarmed"] is True
    assert lineages[2]["alarmed"] is False


def test_evaluate_mojtahedi_hash_seals_the_payload() -> None:
    payload = evaluate_mojtahedi(n_resamples=500, seed=0)
    sealed = payload.pop("content_hash")
    assert sealed == canonical_record_hash(payload)


def test_evaluate_mojtahedi_is_reproducible() -> None:
    first = evaluate_mojtahedi(n_resamples=500, seed=0)
    second = evaluate_mojtahedi(n_resamples=500, seed=0)
    assert first == second


def test_evaluate_accepts_a_custom_lineage() -> None:
    # A single synthetic lineage with a flat trajectory: no rise, nothing alarms.
    flat = MojtahediLineage(
        lineage_id="flat_control",
        day_labels=(0, 1, 3, 6),
        index_means=(0.2, 0.2, 0.2, 0.2),
        index_ses=(0.01, 0.01, 0.01, 0.01),
        citation="synthetic flat control",
    )
    payload = evaluate_mojtahedi([flat], n_resamples=500, seed=1)
    significance = payload["permutation_significance"]
    assert isinstance(significance, dict)
    assert significance["n_transitions"] == 1


def test_main_writes_the_sealed_artefact(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    main(tmp_path)
    artefact = tmp_path / "early_warning_dnb_mojtahedi.json"
    assert artefact.exists()
    payload = json.loads(artefact.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "early_warning_dnb_mojtahedi"
    sealed = payload.pop("content_hash")
    assert sealed == canonical_record_hash(payload)
