# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-domain meta-transfer prototype

"""Replay-backed cross-domain policy proposal utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CrossDomainMetaTransfer",
    "MetaPolicyRecord",
    "MetaTrainingSummary",
    "MetaTransferProposal",
    "records_from_audit_directory",
    "records_from_audit_jsonl",
]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.intp]

_KNOB_ORDER = ("K", "alpha", "zeta", "Psi")


@dataclass(frozen=True)
class MetaPolicyRecord:
    """One replay-derived domain policy example."""

    domain: str
    features: dict[str, float]
    knobs: dict[str, float]
    reward: float = 1.0

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("domain must be non-empty")
        _validate_float_mapping(self.features, "features", allow_empty=False)
        _validate_float_mapping(self.knobs, "knobs", allow_empty=False)
        if not np.isfinite(self.reward):
            raise ValueError("reward must be finite")


@dataclass(frozen=True)
class MetaTransferProposal:
    """Initial policy proposal for a new domain signature."""

    knobs: dict[str, float]
    confidence: float
    neighbours: tuple[tuple[str, float], ...]
    feature_keys: tuple[str, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable proposal record."""
        return {
            "knobs": dict(self.knobs),
            "confidence": self.confidence,
            "neighbours": [
                {"domain": domain, "similarity": similarity}
                for domain, similarity in self.neighbours
            ],
            "feature_keys": list(self.feature_keys),
            "method": "cosine_nearest_policy_transfer",
        }


@dataclass(frozen=True)
class MetaTrainingSummary:
    """Audit-ready summary of the replay corpus used for meta-transfer."""

    record_count: int
    domain_count: int
    domains: tuple[str, ...]
    feature_keys: tuple[str, ...]
    knob_keys: tuple[str, ...]
    reward_mean: float
    reward_min: float
    reward_max: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe training corpus summary."""
        return {
            "record_count": self.record_count,
            "domain_count": self.domain_count,
            "domains": list(self.domains),
            "feature_keys": list(self.feature_keys),
            "knob_keys": list(self.knob_keys),
            "reward_mean": self.reward_mean,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
        }


class CrossDomainMetaTransfer:
    """Nearest-neighbour policy transfer over replay-derived embeddings."""

    def __init__(self, records: tuple[MetaPolicyRecord, ...]) -> None:
        if not records:
            raise ValueError("at least one meta-policy record is required")
        self.records = records
        self.feature_keys = _feature_keys(records)
        self._matrix = np.vstack(
            [_feature_vector(record.features, self.feature_keys) for record in records]
        )
        self.training_summary = _training_summary(records, self.feature_keys)

    @classmethod
    def fit(
        cls, records: list[MetaPolicyRecord] | tuple[MetaPolicyRecord, ...]
    ) -> CrossDomainMetaTransfer:
        """Construct a meta-transfer model from replay-derived records."""
        return cls(tuple(records))

    @classmethod
    def fit_audit_history(
        cls,
        paths: list[str | Path] | tuple[str | Path, ...],
        *,
        min_records: int = 1,
    ) -> CrossDomainMetaTransfer:
        """Fit a model from one or more audit JSONL files."""
        if min_records < 1:
            raise ValueError("min_records must be at least 1")
        records: list[MetaPolicyRecord] = []
        for path in paths:
            records.extend(records_from_audit_jsonl(path))
        if len(records) < min_records:
            raise ValueError(
                f"audit history yielded {len(records)} records; "
                f"min_records={min_records}"
            )
        return cls.fit(tuple(records))

    @classmethod
    def fit_audit_directory(
        cls,
        root: str | Path,
        *,
        pattern: str = "**/*.jsonl",
        min_records: int = 1,
    ) -> CrossDomainMetaTransfer:
        """Fit a model from a recursively discovered audit JSONL corpus."""
        records = records_from_audit_directory(
            root,
            pattern=pattern,
            min_records=min_records,
        )
        return cls.fit(records)

    def propose(
        self,
        features: dict[str, float],
        *,
        k_neighbours: int = 3,
    ) -> MetaTransferProposal:
        """Propose initial policy knobs for a new domain signature."""
        _validate_float_mapping(features, "features", allow_empty=False)
        if k_neighbours < 1:
            raise ValueError("k_neighbours must be at least 1")
        query = _feature_vector(features, self.feature_keys)
        similarities = np.array(
            [_cosine_similarity(query, row) for row in self._matrix],
            dtype=np.float64,
        )
        order: IntArray = np.asarray(
            np.argsort(similarities)[::-1][: min(k_neighbours, len(self.records))],
            dtype=np.intp,
        )
        weights = _proposal_weights(similarities[order], self.records, order)
        knobs = _weighted_knobs(self.records, order, weights)
        confidence = float(np.clip(np.mean(similarities[order]), 0.0, 1.0))
        neighbours = tuple(
            (self.records[index].domain, float(similarities[index])) for index in order
        )
        return MetaTransferProposal(
            knobs=knobs,
            confidence=confidence,
            neighbours=neighbours,
            feature_keys=self.feature_keys,
        )

    def to_json_package(self) -> str:
        """Serialise records and training summary for reviewable reuse."""
        package = {
            "schema": "scpn_meta_transfer_package_v1",
            "training_summary": self.training_summary.to_audit_record(),
            "records": [
                {
                    "domain": record.domain,
                    "features": dict(record.features),
                    "knobs": dict(record.knobs),
                    "reward": record.reward,
                }
                for record in self.records
            ],
        }
        return json.dumps(package, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json_package(cls, payload: str) -> CrossDomainMetaTransfer:
        """Restore a packaged meta-transfer model."""
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("package must be a JSON object")
        if data.get("schema") != "scpn_meta_transfer_package_v1":
            raise ValueError("unsupported meta-transfer package schema")
        records_payload = data.get("records")
        if not isinstance(records_payload, list):
            raise ValueError("package records must be a list")
        records = [
            _record_from_payload(record, index)
            for index, record in enumerate(records_payload, start=1)
            if isinstance(record, dict)
        ]
        if len(records) != len(records_payload):
            raise ValueError("package records must be objects")
        return cls.fit(tuple(records))


def records_from_audit_jsonl(path: str | Path) -> tuple[MetaPolicyRecord, ...]:
    """Load meta-policy records from audit-style JSONL lines.

    Each line may provide either explicit ``features`` and ``knobs`` mappings
    or the common SPO audit shape with ``metrics`` plus ``actions``.
    """
    records: list[MetaPolicyRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            records.append(_record_from_payload(payload, line_number))
    return tuple(records)


def records_from_audit_directory(
    root: str | Path,
    *,
    pattern: str = "**/*.jsonl",
    min_records: int = 1,
) -> tuple[MetaPolicyRecord, ...]:
    """Load replay records from a nested audit-history directory."""
    if min_records < 1:
        raise ValueError("min_records must be at least 1")
    base = Path(root)
    if not base.exists() or not base.is_dir():
        raise ValueError("audit directory must exist")
    paths = tuple(sorted(path for path in base.glob(pattern) if path.is_file()))
    if not paths:
        raise ValueError("audit directory yielded no JSONL files")
    records: list[MetaPolicyRecord] = []
    for path in paths:
        records.extend(records_from_audit_jsonl(path))
    if len(records) < min_records:
        raise ValueError(
            f"audit directory yielded {len(records)} records; min_records={min_records}"
        )
    return tuple(records)


def _record_from_payload(payload: dict[str, Any], line_number: int) -> MetaPolicyRecord:
    domain = str(payload.get("domain") or payload.get("domainpack") or "unknown")
    feature_payload = payload.get("features", payload.get("metrics", {}))
    knob_payload = payload.get("knobs")
    if knob_payload is None:
        knob_payload = _knobs_from_actions(payload.get("actions", ()))
    if not isinstance(feature_payload, dict):
        raise ValueError(f"line {line_number}: features/metrics must be a mapping")
    if not isinstance(knob_payload, dict):
        raise ValueError(f"line {line_number}: knobs/actions must define a mapping")
    reward = float(payload.get("reward", payload.get("R_global", 1.0)))
    return MetaPolicyRecord(
        domain=domain,
        features=_finite_float_dict(feature_payload, "features"),
        knobs=_finite_float_dict(knob_payload, "knobs"),
        reward=reward,
    )


def _knobs_from_actions(actions: object) -> dict[str, float]:
    if not isinstance(actions, list):
        return {}
    by_knob: dict[str, list[float]] = {}
    for action in actions:
        if not isinstance(action, dict):
            continue
        knob = action.get("knob")
        value = action.get("value")
        if not isinstance(knob, str) or knob not in _KNOB_ORDER:
            continue
        if not isinstance(value, int | float) or not np.isfinite(value):
            continue
        by_knob.setdefault(knob, []).append(float(value))
    return {knob: float(np.mean(values)) for knob, values in by_knob.items()}


def _feature_keys(records: tuple[MetaPolicyRecord, ...]) -> tuple[str, ...]:
    keys = sorted({key for record in records for key in record.features})
    if not keys:
        raise ValueError("records must contain at least one feature")
    return tuple(keys)


def _training_summary(
    records: tuple[MetaPolicyRecord, ...],
    feature_keys: tuple[str, ...],
) -> MetaTrainingSummary:
    rewards = np.array([record.reward for record in records], dtype=np.float64)
    knob_keys = sorted({knob for record in records for knob in record.knobs})
    domains = tuple(sorted({record.domain for record in records}))
    return MetaTrainingSummary(
        record_count=len(records),
        domain_count=len(domains),
        domains=domains,
        feature_keys=feature_keys,
        knob_keys=tuple(knob_keys),
        reward_mean=float(np.mean(rewards)),
        reward_min=float(np.min(rewards)),
        reward_max=float(np.max(rewards)),
    )


def _feature_vector(
    features: dict[str, float],
    keys: tuple[str, ...],
) -> FloatArray:
    result: FloatArray = np.array(
        [features.get(key, 0.0) for key in keys],
        dtype=np.float64,
    )
    return result


def _cosine_similarity(left: FloatArray, right: FloatArray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.clip(similarity, 0.0, 1.0))


def _proposal_weights(
    similarities: FloatArray,
    records: tuple[MetaPolicyRecord, ...],
    order: IntArray,
) -> FloatArray:
    rewards = np.array(
        [max(0.0, records[int(index)].reward) for index in order],
        dtype=np.float64,
    )
    weights = np.maximum(similarities, 0.0) * np.maximum(rewards, 1e-12)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / max(1, weights.size), dtype=np.float64)
    result: FloatArray = weights / total
    return result


def _weighted_knobs(
    records: tuple[MetaPolicyRecord, ...],
    order: IntArray,
    weights: FloatArray,
) -> dict[str, float]:
    knobs = sorted({knob for index in order for knob in records[int(index)].knobs})
    proposal: dict[str, float] = {}
    for knob in knobs:
        value = 0.0
        weight_sum = 0.0
        for local_index, record_index in enumerate(order):
            record = records[int(record_index)]
            if knob not in record.knobs:
                continue
            weight = float(weights[local_index])
            value += weight * record.knobs[knob]
            weight_sum += weight
        if weight_sum > 0.0:
            proposal[knob] = float(value / weight_sum)
    return proposal


def _finite_float_dict(values: dict[str, Any], name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        if not isinstance(value, int | float) or not np.isfinite(value):
            raise ValueError(f"{name}.{key} must be finite")
        result[key] = float(value)
    return result


def _validate_float_mapping(
    values: dict[str, float],
    name: str,
    *,
    allow_empty: bool,
) -> None:
    if not allow_empty and not values:
        raise ValueError(f"{name} must be non-empty")
    _finite_float_dict(values, name)
