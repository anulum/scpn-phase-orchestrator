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
    MetaTransferProposal,
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

    def test_model_requires_records(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CrossDomainMetaTransfer.fit(())

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

    def test_loads_records_from_audit_jsonl(self, tmp_path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        payloads = [
            {
                "domain": "alpha",
                "metrics": {"R_global": 0.2, "stability_proxy": 0.1},
                "actions": [{"knob": "K", "value": 0.04}],
                "reward": 0.5,
            },
            {
                "domainpack": "beta",
                "features": {"R_global": 0.8, "stability_proxy": 0.7},
                "knobs": {"zeta": 0.03},
            },
        ]
        audit_path.write_text(
            "\n".join(json.dumps(payload) for payload in payloads),
            encoding="utf-8",
        )

        records = records_from_audit_jsonl(audit_path)

        assert len(records) == 2
        assert records[0].domain == "alpha"
        assert records[0].knobs == {"K": 0.04}
        assert records[1].domain == "beta"
