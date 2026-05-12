# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for cross-domain meta-transfer

from __future__ import annotations

import json
from typing import get_type_hints

import pytest

from scpn_phase_orchestrator.meta import (
    CrossDomainMetaTransfer,
    MetaPolicyRecord,
    MetaTrainingSummary,
    MetaTransferProposal,
    records_from_audit_directory,
    records_from_audit_jsonl,
)


def _records() -> tuple[MetaPolicyRecord, ...]:
    return (
        MetaPolicyRecord(
            domain="power_grid",
            features={"R_global": 0.4, "stability_proxy": 0.3, "event_rate": 0.8},
            knobs={"K": 0.08, "zeta": 0.02},
            reward=0.9,
        ),
        MetaPolicyRecord(
            domain="cardiac",
            features={"R_global": 0.85, "stability_proxy": 0.8, "event_rate": 0.1},
            knobs={"K": 0.02, "zeta": 0.08},
            reward=0.7,
        ),
        MetaPolicyRecord(
            domain="traffic",
            features={"R_global": 0.5, "stability_proxy": 0.45, "event_rate": 0.6},
            knobs={"K": 0.06, "alpha": 0.01},
            reward=0.8,
        ),
    )


class TestMetaTransferContracts:
    def test_public_contracts_are_typed(self) -> None:
        hints = get_type_hints(CrossDomainMetaTransfer.propose)

        assert "float" in str(hints["features"])
        assert hints["return"] is MetaTransferProposal

    def test_record_requires_non_empty_domain(self) -> None:
        with pytest.raises(ValueError, match="domain"):
            MetaPolicyRecord("", {"R": 0.5}, {"K": 0.1})

    def test_record_rejects_non_finite_features(self) -> None:
        with pytest.raises(ValueError, match="features"):
            MetaPolicyRecord("bad", {"R": float("nan")}, {"K": 0.1})

    def test_record_rejects_non_finite_reward_and_non_string_keys(self) -> None:
        with pytest.raises(ValueError, match="reward"):
            MetaPolicyRecord("bad_reward", {"R": 0.5}, {"K": 0.1}, reward=float("inf"))

        with pytest.raises(ValueError, match="features keys"):
            MetaPolicyRecord("bad_feature_key", {1: 0.5}, {"K": 0.1})

    def test_model_requires_records(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CrossDomainMetaTransfer.fit(())

    def test_fit_audit_history_requires_positive_min_records(self, tmp_path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        audit_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="min_records"):
            CrossDomainMetaTransfer.fit_audit_history([audit_path], min_records=0)

    def test_k_neighbours_must_be_positive(self) -> None:
        model = CrossDomainMetaTransfer.fit(_records())

        with pytest.raises(ValueError, match="k_neighbours"):
            model.propose({"R_global": 0.5}, k_neighbours=0)


class TestMetaTransferBehaviour:
    def test_similar_domain_dominates_policy_proposal(self) -> None:
        model = CrossDomainMetaTransfer.fit(_records())

        proposal = model.propose(
            {"R_global": 0.42, "stability_proxy": 0.31, "event_rate": 0.78},
            k_neighbours=1,
        )

        assert proposal.neighbours[0][0] == "power_grid"
        assert proposal.knobs["K"] == pytest.approx(0.08)
        assert proposal.knobs["zeta"] == pytest.approx(0.02)
        assert 0.0 <= proposal.confidence <= 1.0

    def test_weighted_neighbours_mix_available_knobs(self) -> None:
        model = CrossDomainMetaTransfer.fit(_records())

        proposal = model.propose(
            {"R_global": 0.48, "stability_proxy": 0.4, "event_rate": 0.65},
            k_neighbours=2,
        )

        assert "K" in proposal.knobs
        assert proposal.knobs["K"] > 0.0
        assert proposal.neighbours
        assert proposal.feature_keys == ("R_global", "event_rate", "stability_proxy")

    def test_zero_query_has_low_confidence_but_returns_knobs(self) -> None:
        model = CrossDomainMetaTransfer.fit(_records())

        proposal = model.propose(
            {"R_global": 0.0, "stability_proxy": 0.0, "event_rate": 0.0},
            k_neighbours=2,
        )

        assert proposal.confidence == pytest.approx(0.0)
        assert proposal.knobs

    def test_audit_record_is_serialisable(self) -> None:
        model = CrossDomainMetaTransfer.fit(_records())
        proposal = model.propose({"R_global": 0.5, "stability_proxy": 0.4})

        record = proposal.to_audit_record()

        assert record["method"] == "cosine_nearest_policy_transfer"
        assert isinstance(record["neighbours"], list)
        assert isinstance(record["feature_keys"], list)
        assert "knobs" in record

    def test_training_summary_describes_replay_corpus(self) -> None:
        model = CrossDomainMetaTransfer.fit(_records())

        summary = model.training_summary
        record = summary.to_audit_record()

        assert isinstance(summary, MetaTrainingSummary)
        assert summary.record_count == 3
        assert summary.domain_count == 3
        assert summary.domains == ("cardiac", "power_grid", "traffic")
        assert summary.knob_keys == ("K", "alpha", "zeta")
        assert record["reward_min"] == pytest.approx(0.7)
        assert record["reward_max"] == pytest.approx(0.9)

    def test_loads_records_from_audit_jsonl(self, tmp_path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        payloads = [
            "",
            {
                "domain": "alpha",
                "metrics": {"R_global": 0.2, "stability_proxy": 0.1},
                "actions": [
                    "manual-review-note",
                    {"knob": "K", "value": 0.04},
                    {"knob": "alpha", "value": 0.02},
                    {"knob": "unknown", "value": 1.0},
                    {"knob": "zeta", "value": float("nan")},
                ],
                "reward": 0.5,
            },
            {
                "domainpack": "beta",
                "features": {"R_global": 0.8, "stability_proxy": 0.7},
                "knobs": {"zeta": 0.03},
            },
        ]
        audit_path.write_text(
            "\n".join(
                payload if isinstance(payload, str) else json.dumps(payload)
                for payload in payloads
            ),
            encoding="utf-8",
        )

        records = records_from_audit_jsonl(audit_path)

        assert len(records) == 2
        assert records[0].domain == "alpha"
        assert records[0].knobs == {"K": 0.04, "alpha": 0.02}
        assert records[1].domain == "beta"

    def test_fit_audit_history_aggregates_multiple_jsonl_files(self, tmp_path) -> None:
        first = tmp_path / "first.jsonl"
        second = tmp_path / "second.jsonl"
        first.write_text(
            json.dumps(
                {
                    "domain": "alpha",
                    "metrics": {"R_global": 0.3},
                    "actions": [{"knob": "K", "value": 0.05}],
                    "reward": 0.4,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        second.write_text(
            json.dumps(
                {
                    "domain": "beta",
                    "features": {"R_global": 0.9},
                    "knobs": {"zeta": 0.02},
                    "reward": 0.8,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        model = CrossDomainMetaTransfer.fit_audit_history(
            [first, second],
            min_records=2,
        )
        proposal = model.propose({"R_global": 0.85}, k_neighbours=1)

        assert model.training_summary.record_count == 2
        assert model.training_summary.domains == ("alpha", "beta")
        assert proposal.neighbours[0][0] == "beta"

    def test_fit_audit_directory_discovers_nested_jsonl_corpus(self, tmp_path) -> None:
        first_dir = tmp_path / "grid"
        second_dir = tmp_path / "cardiac" / "nested"
        first_dir.mkdir()
        second_dir.mkdir(parents=True)
        (first_dir / "audit.jsonl").write_text(
            json.dumps(
                {
                    "domain": "grid",
                    "metrics": {"R_global": 0.2, "event_rate": 0.9},
                    "actions": [{"knob": "K", "value": 0.08}],
                    "reward": 0.6,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (second_dir / "audit.jsonl").write_text(
            json.dumps(
                {
                    "domain": "cardiac",
                    "features": {"R_global": 0.9, "event_rate": 0.1},
                    "knobs": {"zeta": 0.07},
                    "reward": 0.9,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        records = records_from_audit_directory(tmp_path, min_records=2)
        model = CrossDomainMetaTransfer.fit_audit_directory(
            tmp_path,
            min_records=2,
        )
        proposal = model.propose({"R_global": 0.88, "event_rate": 0.12})

        assert {record.domain for record in records} == {"grid", "cardiac"}
        assert model.training_summary.record_count == 2
        assert model.training_summary.domain_count == 2
        assert proposal.neighbours[0][0] == "cardiac"

    def test_records_from_audit_directory_validates_empty_corpus(
        self, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="min_records"):
            records_from_audit_directory(tmp_path, min_records=0)

        with pytest.raises(ValueError, match="no JSONL files"):
            records_from_audit_directory(tmp_path)

        with pytest.raises(ValueError, match="audit directory must exist"):
            records_from_audit_directory(tmp_path / "missing")

    def test_fit_audit_history_enforces_min_records(self, tmp_path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        audit_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="audit history yielded 0 records"):
            CrossDomainMetaTransfer.fit_audit_history([audit_path], min_records=1)

    def test_json_package_round_trip_preserves_proposals(self) -> None:
        model = CrossDomainMetaTransfer.fit(_records())

        package = model.to_json_package()
        restored = CrossDomainMetaTransfer.from_json_package(package)
        original = model.propose({"R_global": 0.42, "event_rate": 0.78})
        loaded = restored.propose({"R_global": 0.42, "event_rate": 0.78})

        assert json.loads(package)["schema"] == "scpn_meta_transfer_package_v1"
        assert restored.training_summary.to_audit_record() == (
            model.training_summary.to_audit_record()
        )
        assert loaded.knobs == pytest.approx(original.knobs)
        assert loaded.neighbours == original.neighbours

    def test_json_package_rejects_unknown_schema(self) -> None:
        with pytest.raises(ValueError, match="schema"):
            CrossDomainMetaTransfer.from_json_package('{"schema":"wrong"}')

    @pytest.mark.parametrize(
        ("payload", "expected_error"),
        [
            ("[]", "JSON object"),
            ('{"schema":"scpn_meta_transfer_package_v1"}', "records must be a list"),
            (
                '{"schema":"scpn_meta_transfer_package_v1","records":[1]}',
                "records must be objects",
            ),
        ],
    )
    def test_json_package_rejects_invalid_record_package_shape(
        self,
        payload: str,
        expected_error: str,
    ) -> None:
        with pytest.raises(ValueError, match=expected_error):
            CrossDomainMetaTransfer.from_json_package(payload)

    @pytest.mark.parametrize(
        ("payload", "expected_error"),
        [
            (
                {"domain": "bad", "features": [("R", 0.2)], "knobs": {"K": 0.1}},
                "features/metrics",
            ),
            (
                {"domain": "bad", "features": {"R": 0.2}, "knobs": [("K", 0.1)]},
                "knobs/actions",
            ),
            (
                {"domain": "bad", "features": {"R": 0.2}, "actions": "K=0.1"},
                "knobs",
            ),
        ],
    )
    def test_audit_jsonl_rejects_invalid_feature_and_knob_payloads(
        self,
        tmp_path,
        payload: dict[str, object],
        expected_error: str,
    ) -> None:
        audit_path = tmp_path / "audit.jsonl"
        audit_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match=expected_error):
            records_from_audit_jsonl(audit_path)

    def test_audit_directory_enforces_minimum_replay_count(self, tmp_path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        audit_path.write_text(
            json.dumps(
                {
                    "domain": "single",
                    "metrics": {"R_global": 0.4},
                    "actions": [{"knob": "K", "value": 0.05}],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="min_records=2"):
            records_from_audit_directory(tmp_path, min_records=2)
