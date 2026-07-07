# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real-detector auditor head-to-head tests

from __future__ import annotations

import numpy as np

import bench.auditor_detector_head_to_head as head_to_head
from bench.auditor_detector_head_to_head import (
    N_BUSES,
    SEGMENT_SAMPLES,
    audit_regime,
    competitor_ar1_score,
    main,
    monotone_block,
    monotone_corpus,
    oscillatory_block,
    oscillatory_corpus,
    run,
    scpn_growth_score,
)


class TestSyntheticBlocks:
    def test_oscillatory_block_shape_and_growth(self):
        rng = np.random.default_rng(0)
        block = oscillatory_block(rng, 0.6)
        assert block.shape == (N_BUSES, SEGMENT_SAMPLES)
        # A growing mode scores a positive envelope growth rate.
        assert scpn_growth_score(block) > 0.0

    def test_monotone_block_shape_and_rising_autocorrelation(self):
        rng = np.random.default_rng(0)
        rising = monotone_block(rng, 0.95)
        flat = monotone_block(rng, 0.0)
        assert rising.shape == (N_BUSES, SEGMENT_SAMPLES)
        # Rising autocorrelation scores a higher AR(1)/Kendall-tau trend than noise.
        assert competitor_ar1_score(rising) > competitor_ar1_score(flat)


class TestCorpora:
    def test_oscillatory_corpus_sizes(self):
        events, nulls = oscillatory_corpus(seed=0)
        assert len(events) == 40
        assert len(nulls) == 120
        assert all(block.shape == (N_BUSES, SEGMENT_SAMPLES) for block in events)

    def test_monotone_corpus_sizes(self):
        events, nulls = monotone_corpus(seed=1)
        assert len(events) == 40
        assert len(nulls) == 120


class TestAuditRegime:
    def test_audits_both_real_detectors(self):
        events, nulls = oscillatory_corpus(seed=0)
        audits = audit_regime(events, nulls, n_permutations=200)
        assert set(audits) == {"scpn-modal-growth", "ar1-kendall-tau"}
        # Both real detectors carry skill on the oscillatory regime.
        assert audits["scpn-modal-growth"].beats_chance is True
        assert audits["ar1-kendall-tau"].beats_chance is True


class TestRun:
    def test_auditor_surfaces_regime_dependent_skill(self):
        table = run(n_permutations=200)
        osc = table["oscillatory"]
        mono = table["monotone"]
        # The envelope-growth detector leads on the oscillatory regime; the
        # AR(1)/Kendall-tau competitor leads on the monotone regime. Ranked by the
        # detection rate at a matched false alarm (the p-value saturates).
        assert (
            osc["scpn-modal-growth"].detection_rate
            > osc["ar1-kendall-tau"].detection_rate
        )
        assert (
            mono["ar1-kendall-tau"].detection_rate
            > mono["scpn-modal-growth"].detection_rate
        )
        # Every cell holds the matched false-alarm target and beats chance.
        for audits in table.values():
            for audit in audits.values():
                assert audit.achieved_false_alarm <= 0.12
                assert audit.beats_chance is True


class TestMain:
    def test_main_prints_regime_winners(self, monkeypatch, capsys):
        # Compute the table once cheaply; main() only has to format and seal it.
        table = run(n_permutations=200)
        monkeypatch.setattr(head_to_head, "run", lambda *, n_permutations=10000: table)
        main()
        out = capsys.readouterr().out
        assert "oscillatory" in out
        assert "monotone" in out
        assert "scpn-modal-growth" in out
        assert "ar1-kendall-tau" in out
