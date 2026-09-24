# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — early-warning evaluation input contracts

"""Prove the evaluation layer refuses inputs that would flatter a detector.

A NaN score compares false with everything, so it never alarms: a NaN null
lowered the reported false-alarm rate, and a NaN statistic ranked below every
surrogate and earned the smallest possible p-value. ``bool("False")`` is
``True``, text was parsed as a score, and the meta-analysis read a *fraction* of
recordings as a yes/no verdict. Every case drives the public entry points with
the offending value itself.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.evaluation import (
    AuditRecord,
    audit_detector,
    audit_scoring_detector,
    benjamini_hochberg,
    calibrate_score_threshold,
    matched_false_alarm_rate,
    permutation_significance_from_alarms,
    seal_detector_audit,
    surrogate_rank_pvalue,
)
from scpn_phase_orchestrator.evaluation.cross_domain_transfer import (
    ScorePair,
    audit_cross_domain_transfer,
    classify_transfer_verdict,
)
from scpn_phase_orchestrator.evaluation.detector_meta_analysis import (
    extract_evidence,
    generate_report,
    run_analysis,
)
from scpn_phase_orchestrator.evaluation.detector_meta_analysis import (
    main as meta_main,
)

NAN = float("nan")
NULLS = [0.05 * index for index in range(20)]
EVENTS = [2.0] * 20


def _audit(**overrides: object) -> object:
    arguments: dict[str, object] = {
        "event_scores": EVENTS,
        "null_scores": NULLS,
        "n_permutations": 200,
    }
    arguments.update(overrides)
    return audit_detector(**arguments)  # type: ignore[arg-type]


class TestSurrogateRankPValue:
    def test_nan_statistic_is_refused_not_maximally_significant(self) -> None:
        with pytest.raises(ValueError, match=r"observed\[0\] must not be NaN"):
            surrogate_rank_pvalue(NAN, [0.1] * 99)

    def test_nan_surrogate_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"surrogates\[2\] must not be NaN"):
            surrogate_rank_pvalue(0.5, [0.1, 0.2, NAN])

    @pytest.mark.parametrize("bad", ["0.9", True, None])
    def test_non_numeric_surrogate_is_refused(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"surrogates\[0\] must be a real number"):
            surrogate_rank_pvalue(0.5, [bad])

    def test_infinities_keep_their_order(self) -> None:
        assert surrogate_rank_pvalue(math.inf, [0.0, 1.0, 2.0]) == 0.25
        assert surrogate_rank_pvalue(0.5, [-math.inf, math.inf]) == 2 / 3


class TestCalibration:
    def test_nan_null_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"null_scores\[1\] must not be NaN"):
            calibrate_score_threshold([0.1, NAN, 0.3])

    @pytest.mark.parametrize("bad", ["0.5", True, np.bool_(True)])
    def test_non_numeric_null_is_refused(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"null_scores\[0\] must be a real"):
            calibrate_score_threshold([bad, 0.1])

    @pytest.mark.parametrize("bad", [True, "0.1", NAN])
    def test_target_must_be_a_real_probability(self, bad: object) -> None:
        with pytest.raises(ValueError, match="target_fa must"):
            calibrate_score_threshold(NULLS, target_fa=bad)  # type: ignore[arg-type]

    def test_numpy_scores_are_accepted(self) -> None:
        threshold = calibrate_score_threshold(np.asarray(NULLS, dtype=np.float32))
        assert threshold > 0.85


class TestMatchedFalseAlarm:
    def test_nan_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="threshold must not be NaN"):
            matched_false_alarm_rate([0.1, 0.9], NAN)

    def test_nan_null_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"null_scores\[0\] must not be NaN"):
            matched_false_alarm_rate([NAN, 0.9], 0.5)

    def test_open_gate_counts_every_null(self) -> None:
        assert matched_false_alarm_rate([0.1, 0.9], -math.inf) == 1.0


class TestPermutationSignificance:
    @pytest.mark.parametrize("bad", ["False", NAN, 1, 0.0, None])
    def test_alarm_must_be_a_boolean(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"event_alarms\[1\] must be a boolean"):
            permutation_significance_from_alarms([True, bad], [False], n_permutations=5)

    def test_null_alarm_must_be_a_boolean(self) -> None:
        with pytest.raises(ValueError, match=r"null_alarms\[0\] must be a boolean"):
            permutation_significance_from_alarms([True], ["True"], n_permutations=5)

    def test_numpy_booleans_are_accepted(self) -> None:
        result = permutation_significance_from_alarms(
            np.array([True, True]), np.array([False, False]), n_permutations=20
        )
        assert result.observed_alarms == 2

    @pytest.mark.parametrize("bad", [None, 1.5, -1, True, "0"])
    def test_seed_must_be_a_non_negative_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            permutation_significance_from_alarms(
                [True],
                [False],
                n_permutations=5,
                seed=bad,  # type: ignore[arg-type]
            )

    def test_numpy_integer_seed_is_recorded_as_int(self) -> None:
        result = permutation_significance_from_alarms(
            [True], [False], n_permutations=5, seed=np.int64(7)
        )
        assert result.seed == 7
        assert type(result.seed) is int


class TestBenjaminiHochberg:
    @pytest.mark.parametrize("bad", ["0.5", True])
    def test_non_numeric_p_value_is_refused(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"p_values\[0\] must be a real number"):
            benjamini_hochberg([bad])


class TestAuditDetector:
    def test_nan_nulls_are_refused_not_reported_as_quiet(self) -> None:
        """NaN nulls used to give threshold nan, false alarm 0.0 and p = 1."""
        with pytest.raises(ValueError, match=r"null_scores\[0\] must not be NaN"):
            _audit(null_scores=[NAN] * 18 + [0.1, 0.2])

    def test_nan_events_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"event_scores\[0\] must not be NaN"):
            _audit(event_scores=[NAN] * 20)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("target_false_alarm", True, "target_fa must be a real number"),
            ("alpha", True, "alpha must be a real number"),
            ("alpha", "0.05", "alpha must be a real number"),
            ("detector_name", 5, "detector_name must be a non-blank string"),
            ("detector_name", " ", "detector_name must be a non-blank string"),
        ],
    )
    def test_parameters_are_typed(
        self, field: str, value: object, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _audit(**{field: value})

    def test_positive_infinite_gate_seals_as_strict_json(self) -> None:
        audit = _audit(null_scores=[math.inf] * 20)
        assert audit.matched_threshold == math.inf
        assert audit.to_record()["matched_threshold"] == "inf"

        record = seal_detector_audit(audit, corpus_id="c", captured_at="t")
        assert record.verify()
        assert json.loads(json.dumps(record.to_record(), allow_nan=False))

    def test_scoring_callable_returning_text_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"event_scores\[0\] must be a real"):
            audit_scoring_detector(
                score=lambda series: str(series[0]),
                event_series=[[0.9]] * 5,
                null_series=[[0.1]] * 5,
                n_permutations=20,
            )

    def test_scoring_callable_returning_numpy_scalars_is_accepted(self) -> None:
        audit = audit_scoring_detector(
            score=lambda series: np.float32(series[0]),
            event_series=[[0.9]] * 5,
            null_series=[[0.1 * index] for index in range(5)],
            n_permutations=20,
        )
        assert audit.detection_rate == 1.0


class TestAuditRecord:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("corpus_id", 5),
            ("corpus_id", "  "),
            ("captured_at", None),
            ("framework", ""),
            ("disclaimer", 1),
        ],
    )
    def test_provenance_fields_are_non_blank_strings(
        self, field: str, value: object
    ) -> None:
        fields: dict[str, object] = {"corpus_id": "c", "captured_at": "t", "audit": {}}
        fields[field] = value
        with pytest.raises(ValueError, match=f"{field} must not be empty"):
            AuditRecord(**fields)  # type: ignore[arg-type]

    def test_audit_must_be_a_mapping(self) -> None:
        with pytest.raises(ValueError, match="audit must be the verdict mapping"):
            AuditRecord(corpus_id="c", captured_at="t", audit=[1])  # type: ignore[arg-type]

    @pytest.mark.parametrize(("signature", "key_id"), [("ab", None), (None, "kid")])
    def test_signature_and_key_id_come_together(
        self, signature: str | None, key_id: str | None
    ) -> None:
        with pytest.raises(ValueError, match="must be given together"):
            AuditRecord(
                corpus_id="c",
                captured_at="t",
                audit={},
                signature=signature,
                signing_key_id=key_id,
            )

    @pytest.mark.parametrize(
        ("signature", "key_id", "field"),
        [(b"ab", "kid", "signature"), ("ab", 7, "signing_key_id")],
    )
    def test_signature_fields_are_strings(
        self, signature: object, key_id: object, field: str
    ) -> None:
        with pytest.raises(ValueError, match=f"{field} must be a string"):
            AuditRecord(
                corpus_id="c",
                captured_at="t",
                audit={},
                signature=signature,  # type: ignore[arg-type]
                signing_key_id=key_id,  # type: ignore[arg-type]
            )

    def test_seal_refuses_a_blank_corpus_id(self) -> None:
        with pytest.raises(ValueError, match="corpus_id must not be empty"):
            seal_detector_audit(_audit(), corpus_id=" ", captured_at="t")


class TestCrossDomainTransfer:
    def test_nan_null_in_a_score_pair_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"null_scores\[1\] must not be NaN"):
            ScorePair(event_scores=(0.9,), null_scores=(0.1, NAN))

    def test_text_score_in_a_score_pair_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"event_scores\[0\] must be a real"):
            ScorePair(event_scores=("0.9",), null_scores=(0.1,))  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["False", 1, None])
    def test_verdict_gates_must_be_booleans(self, bad: object) -> None:
        with pytest.raises(ValueError, match="transfer_above_floor must be a boolean"):
            classify_transfer_verdict(
                transfer_beats_own_null=True,
                transfer_above_floor=bad,  # type: ignore[arg-type]
                within_domain_detectable=True,
            )

    @pytest.mark.parametrize(
        ("field", "value"), [("source_domain", ""), ("target_domain", 3)]
    )
    def test_domain_labels_are_non_blank_strings(
        self, field: str, value: object
    ) -> None:
        pair = ScorePair(event_scores=tuple(EVENTS), null_scores=tuple(NULLS))
        with pytest.raises(ValueError, match=f"{field} must be a non-blank string"):
            audit_cross_domain_transfer(
                transfer=pair,
                within_domain=pair,
                shuffled_source=[pair],
                n_permutations=20,
                **{field: value},  # type: ignore[arg-type]
            )


def _write(root: Path, domain: str, name: str, payload: object) -> Path:
    directory = root / domain
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _honest(**detector: object) -> dict[str, object]:
    return {"per_recording": [], "detector_a": detector}


def _leadtime(**stats: object) -> dict[str, object]:
    return {"permutation_significance": {"detector_a": stats}}


class TestMetaAnalysisExtraction:
    def test_a_fraction_of_recordings_is_not_a_verdict(self, tmp_path: Path) -> None:
        """0.75 of recordings beating chance with p = 0.147 does not beat chance."""
        path = _write(
            tmp_path,
            "d",
            "x_aggregate.json",
            _honest(
                mean_detection_rate=0.25,
                geometric_mean_p_value=0.147,
                fraction_beats_chance=0.75,
            ),
        )
        (row,) = extract_evidence(path)
        assert row.beats_chance is False

    def test_significant_p_beats_chance_whatever_the_fraction(
        self, tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path,
            "d",
            "x_aggregate.json",
            _honest(
                mean_detection_rate=0.5,
                geometric_mean_p_value=0.01,
                fraction_beats_chance=0.0,
            ),
        )
        (row,) = extract_evidence(path)
        assert row.beats_chance is True

    def test_committed_csd_baseline_no_longer_beats_chance(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "real_data"
            / "csd_variant_synthetic"
            / "csd_variant_synthetic_results.json"
        )
        rows = {row.detector: row for row in extract_evidence(path)}
        baseline = rows["critical_slowing_down_baseline"]
        assert baseline.p_value > 0.05
        assert baseline.beats_chance is False

    @pytest.mark.parametrize(
        ("detector", "key"),
        [
            ({"mean_detection_rate": 0.5, "geometric_mean_p_value": "0.01"}, "p_value"),
            ({"mean_detection_rate": 0.5, "geometric_mean_p_value": True}, "p_value"),
            ({"mean_detection_rate": 0.5, "geometric_mean_p_value": 1.5}, "p_value"),
            ({"mean_detection_rate": 1.5}, "mean_detection_rate"),
            ({"mean_detection_rate": "0.5"}, "mean_detection_rate"),
        ],
    )
    def test_honest_values_must_lie_in_the_unit_interval(
        self, tmp_path: Path, detector: dict[str, object], key: str
    ) -> None:
        path = _write(tmp_path, "d", "x_aggregate.json", _honest(**detector))
        with pytest.raises(
            ValueError, match=rf"x_aggregate\.json: detector_a\.\w*{key}"
        ):
            extract_evidence(path)

    def test_nan_literal_p_value_is_refused(self, tmp_path: Path) -> None:
        directory = tmp_path / "d"
        directory.mkdir()
        path = directory / "x_aggregate.json"
        path.write_text(
            '{"per_recording": [], "detector_a": '
            '{"mean_detection_rate": 0.5, "geometric_mean_p_value": NaN}}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="must be in"):
            extract_evidence(path)

    @pytest.mark.parametrize(
        ("stats", "message"),
        [
            ({"observed_led": 3, "n_transitions": 2, "p_value": 0.1}, "3 of 2"),
            ({"observed_led": 0, "n_transitions": 0, "p_value": 0.1}, "0 of 0"),
            ({"observed_led": "3", "n_transitions": 4, "p_value": 0.1}, "observed_led"),
            (
                {"observed_led": True, "n_transitions": 4, "p_value": 0.1},
                "observed_led",
            ),
            ({"observed_led": 2.0, "n_transitions": 4, "p_value": 0.1}, "observed_led"),
            ({"observed_led": 1, "n_transitions": -4, "p_value": 0.1}, "n_transitions"),
            ({"observed_led": 1, "n_transitions": 4}, "p_value"),
        ],
    )
    def test_leadtime_counts_and_p_value_are_validated(
        self, tmp_path: Path, stats: dict[str, object], message: str
    ) -> None:
        path = _write(tmp_path, "d", "x_results.json", _leadtime(**stats))
        with pytest.raises(ValueError, match=message):
            extract_evidence(path)

    @pytest.mark.parametrize(
        ("flag", "detector"),
        [
            (True, "critical_slowing_down_multiscale"),
            ("false", "critical_slowing_down"),
            (1, "critical_slowing_down"),
        ],
    )
    def test_only_a_literal_true_names_the_multiscale_variant(
        self, tmp_path: Path, flag: object, detector: str
    ) -> None:
        payload = {
            "multiscale": flag,
            "permutation_significance": {
                "observed_led": 1,
                "n_transitions": 2,
                "p_value": 0.2,
            },
        }
        path = _write(tmp_path, "d", "x_results.json", payload)
        (row,) = extract_evidence(path)
        assert row.detector == detector

    def test_a_non_object_payload_is_unsupported(self, tmp_path: Path) -> None:
        _write(tmp_path, "d", "x_aggregate.json", ["per_recording"])
        rows, _, _, sources, unsupported = run_analysis(tmp_path)
        assert rows == []
        assert unsupported == sources


def _flat_leadtime(*, multiscale: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "permutation_significance": {
            "observed_led": 2,
            "n_transitions": 3,
            "p_value": 0.01,
        }
    }
    if multiscale:
        payload["multiscale"] = True
    return payload


class TestMetaAnalysisCorpusShapes:
    def test_a_missing_root_holds_no_aggregates(self, tmp_path: Path) -> None:
        rows, _, _, sources, _ = run_analysis(tmp_path / "absent")
        assert rows == [] and sources == []

    def test_files_beside_the_domain_folders_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "stray_aggregate.json").write_text("{}", encoding="utf-8")
        _, _, _, sources, _ = run_analysis(tmp_path)
        assert sources == []

    def test_a_non_mapping_significance_block_is_unsupported(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, "d", "x_results.json", {"permutation_significance": [1]})
        rows, _, _, sources, unsupported = run_analysis(tmp_path)
        assert rows == [] and unsupported == sources

    def test_an_empty_honest_block_falls_through_to_the_leadtime_schema(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "per_recording": [],
            **_leadtime(observed_led=1, n_transitions=2, p_value=0.3),
        }
        path = _write(tmp_path, "d", "x_results.json", payload)
        (row,) = extract_evidence(path)
        assert row.detection_rate == 0.5

    def test_backlog_recommends_refining_the_baseline_family(
        self, tmp_path: Path
    ) -> None:
        """Baseline CSD wins two domains; the multiscale variant only one."""
        _write(
            tmp_path, "climate_a", "a_results.json", _flat_leadtime(multiscale=False)
        )
        _write(
            tmp_path, "climate_b", "b_results.json", _flat_leadtime(multiscale=False)
        )
        _write(tmp_path, "climate_c", "c_results.json", _flat_leadtime(multiscale=True))
        _write(
            tmp_path,
            "novel_domain",
            "n_aggregate.json",
            {
                "per_recording": [],
                "novel_detector": {
                    "mean_detection_rate": 0.4,
                    "geometric_mean_p_value": 0.02,
                },
            },
        )

        report = generate_report(tmp_path)

        assert "**Refine the `critical_slowing_down` family** — it wins in 2" in report
        assert "**Complete the PhysioNet corpora**" in report
        assert "**Study `novel_detector` transferability**" in report

    def test_cli_reports_a_missing_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert meta_main(["--root", str(tmp_path / "absent")]) == 1
        assert "root directory not found" in capsys.readouterr().err
