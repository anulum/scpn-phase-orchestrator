# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio hybrid-order cosimulation panel

"""Hybrid-order quantum cosimulation review panel builder."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import cast

from ._shared import _finite_number, _require_non_empty_text, _require_sha256_hex

_HYBRID_ORDER_CLAIM_BOUNDARY = "quantum_cosimulation_monitor_not_qpu_execution"


_HYBRID_ORDER_BACKENDS = frozenset(
    {
        "numpy_statevector_density_matrix",
        "numpy_statevector",
        "numpy_density_matrix",
    }
)


def build_hybrid_order_studio_panel(
    records: Sequence[Mapping[str, object]],
    *,
    scenarios: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return a Studio panel for hybrid classical-quantum order evidence.

    The helper renders local simulator evidence only. It preserves the
    explicit quantum co-simulation claim boundary, validates deterministic
    record hashes, summarises statevector/density-matrix simulator backends,
    and folds deterministic scenario fixtures into candidate review rows.
    Nothing in the payload permits live QPU execution or actuation.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        The records to summarise.
    scenarios : Sequence[Mapping[str, object]]
        Scenario records.

    Returns
    -------
    dict[str, object]
        A Studio panel for hybrid classical-quantum order evidence.
    """
    normalised_records = _normalise_hybrid_order_records(records)
    normalised_scenarios, candidate_rows = _normalise_hybrid_order_scenarios(scenarios)
    entropies = [
        cast("float", record["entanglement_entropy"]) for record in normalised_records
    ]
    normalised_entropies = [
        cast("float", record["normalised_entanglement_entropy"])
        for record in normalised_records
    ]
    participation_ratios = [
        cast("float", record["participation_ratio"]) for record in normalised_records
    ]
    strongest = max(
        normalised_records,
        key=lambda record: cast("float", record["entanglement_entropy"]),
    )
    backends = tuple(
        sorted({cast("str", record["backend"]) for record in normalised_records})
    )
    scenario_domains = tuple(
        sorted({cast("str", scenario["domain"]) for scenario in normalised_scenarios})
    )
    return {
        "panel_kind": "studio_hybrid_order_panel",
        "monitor": "hybrid_entanglement_order_parameter",
        "record_count": len(normalised_records),
        "scenario_count": len(normalised_scenarios),
        "claim_boundary": _HYBRID_ORDER_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "actuation_permitted": False,
        "qpu_execution_permitted": False,
        "simulator_backends": backends,
        "scenario_domains": scenario_domains,
        "latest": normalised_records[-1],
        "strongest_entanglement": strongest,
        "series": normalised_records,
        "candidate_rows": candidate_rows,
        "entropy_range": {
            "minimum": min(entropies),
            "maximum": max(entropies),
        },
        "normalised_entanglement_range": {
            "minimum": min(normalised_entropies),
            "maximum": max(normalised_entropies),
        },
        "participation_ratio_range": {
            "minimum": min(participation_ratios),
            "maximum": max(participation_ratios),
        },
        "operator_summary": (
            "hybrid order review: "
            f"{len(normalised_records)} monitor records across "
            f"{len(backends)} local simulator backend(s); "
            f"max entropy {max(entropies):.6g}"
        ),
        "operator_action": (
            "render as local quantum co-simulation evidence only; compare "
            "classical R/Psi with entanglement entropy and keep QPU execution, "
            "actuation, and backend promotion behind separate evidence gates"
        ),
    }


def _normalise_hybrid_order_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate and normalise hybrid quantum-classical order records for the panel.

    Each record must keep the review-safe claim boundary and non-actuating flags,
    name a supported backend, and carry finite R/Psi, bounded entropy and
    participation metrics, a qubit-covering bipartition, and a payload-matching
    hash. Raises ``ValueError`` on any malformed or boundary-violating record.
    """
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or not records:
        raise ValueError("hybrid-order records must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("hybrid-order record must be a mapping")
        label = f"hybrid-order record {index}"
        claim_boundary = _require_non_empty_text(
            record.get("claim_boundary"), f"{label} claim_boundary"
        )
        if claim_boundary != _HYBRID_ORDER_CLAIM_BOUNDARY:
            raise ValueError(f"{label} claim boundary is not review-safe")
        if record.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        if record.get("execution_disabled") is not True:
            raise ValueError(f"{label} execution_disabled must be true")
        backend = _require_non_empty_text(record.get("backend"), f"{label} backend")
        if backend not in _HYBRID_ORDER_BACKENDS:
            raise ValueError(f"{label} backend is not supported")
        qubit_count = _hybrid_positive_int(
            record.get("qubit_count"), f"{label} qubit_count"
        )
        normalised_record = {
            "R": _hybrid_unit_interval(
                _finite_number(record.get("R"), f"{label} R"),
                f"{label} R",
            ),
            "Psi": _finite_number(record.get("Psi"), f"{label} Psi"),
            "entanglement_entropy": _hybrid_non_negative(
                _finite_number(
                    record.get("entanglement_entropy"),
                    f"{label} entanglement_entropy",
                ),
                f"{label} entanglement_entropy",
            ),
            "normalised_entanglement_entropy": _hybrid_unit_interval(
                _finite_number(
                    record.get("normalised_entanglement_entropy"),
                    f"{label} normalised_entanglement_entropy",
                ),
                f"{label} normalised_entanglement_entropy",
            ),
            "participation_ratio": _hybrid_positive_float(
                _finite_number(
                    record.get("participation_ratio"),
                    f"{label} participation_ratio",
                ),
                f"{label} participation_ratio",
            ),
            "qubit_count": qubit_count,
            "bipartition": _normalise_hybrid_bipartition(
                record.get("bipartition"),
                qubit_count=qubit_count,
                label=f"{label} bipartition",
            ),
            "backend": backend,
            "claim_boundary": claim_boundary,
            "non_actuating": True,
            "execution_disabled": True,
            "record_hash": _validated_hybrid_record_hash(record, label),
        }
        normalised.append(normalised_record)
    return tuple(normalised)


def _normalise_hybrid_order_scenarios(
    scenarios: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Validate hybrid-order scenarios and flatten their state candidates.

    Returns the per-scenario summary rows and the flattened candidate rows; every
    scenario and candidate must keep the review-safe claim boundary and
    non-actuating flags and carry a supported state type and bounded metrics.
    """
    if isinstance(scenarios, Mapping) or not isinstance(scenarios, Sequence):
        raise ValueError("hybrid-order scenarios must be a sequence")
    normalised_scenarios: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            raise ValueError("hybrid-order scenario must be a mapping")
        label = f"hybrid-order scenario {scenario_index}"
        claim_boundary = _require_non_empty_text(
            scenario.get("claim_boundary"), f"{label} claim_boundary"
        )
        if claim_boundary != _HYBRID_ORDER_CLAIM_BOUNDARY:
            raise ValueError(f"{label} claim boundary is not review-safe")
        if scenario.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        if scenario.get("execution_disabled") is not True:
            raise ValueError(f"{label} execution_disabled must be true")
        domain = _require_non_empty_text(scenario.get("domain"), f"{label} domain")
        scenario_id = _require_non_empty_text(
            scenario.get("scenario_id"), f"{label} scenario_id"
        )
        scenario_hash = _require_sha256_hex(
            scenario.get("scenario_hash"), f"{label} scenario_hash"
        )
        qubit_count = _hybrid_positive_int(
            scenario.get("qubit_count"), f"{label} qubit_count"
        )
        candidates = scenario.get("state_candidates")
        if (
            isinstance(candidates, Mapping)
            or not isinstance(candidates, Sequence)
            or not candidates
        ):
            raise ValueError(f"{label} state_candidates must be a non-empty sequence")
        normalised_scenarios.append(
            {
                "domain": domain,
                "scenario_id": scenario_id,
                "scenario_hash": scenario_hash,
                "qubit_count": qubit_count,
                "claim_boundary": claim_boundary,
                "non_actuating": True,
                "execution_disabled": True,
            }
        )
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"{label} candidate must be a mapping")
            candidate_label = f"{label} candidate {candidate_index}"
            candidate_claim = _require_non_empty_text(
                candidate.get("claim_boundary"), f"{candidate_label} claim_boundary"
            )
            if candidate_claim != _HYBRID_ORDER_CLAIM_BOUNDARY:
                raise ValueError(f"{candidate_label} claim boundary is not review-safe")
            if candidate.get("non_actuating") is not True:
                raise ValueError(f"{candidate_label} non_actuating must be true")
            if candidate.get("execution_disabled") is not True:
                raise ValueError(f"{candidate_label} execution_disabled must be true")
            state_type = _require_non_empty_text(
                candidate.get("state_type"), f"{candidate_label} state_type"
            )
            if state_type not in {"product", "entangled"}:
                raise ValueError(f"{candidate_label} state_type is not supported")
            candidate_rows.append(
                {
                    "domain": domain,
                    "scenario_id": scenario_id,
                    "scenario_hash": scenario_hash,
                    "state_id": _require_non_empty_text(
                        candidate.get("state_id"), f"{candidate_label} state_id"
                    ),
                    "state_type": state_type,
                    "entanglement_entropy": _hybrid_non_negative(
                        _finite_number(
                            candidate.get("entanglement_entropy"),
                            f"{candidate_label} entanglement_entropy",
                        ),
                        f"{candidate_label} entanglement_entropy",
                    ),
                    "order_metric_r": _hybrid_unit_interval(
                        _finite_number(
                            candidate.get("order_metric_r"),
                            f"{candidate_label} order_metric_r",
                        ),
                        f"{candidate_label} order_metric_r",
                    ),
                    "order_metric_psi": _finite_number(
                        candidate.get("order_metric_psi"),
                        f"{candidate_label} order_metric_psi",
                    ),
                }
            )
    return tuple(normalised_scenarios), tuple(candidate_rows)


def _validated_hybrid_record_hash(record: Mapping[str, object], label: str) -> str:
    """Return the record's SHA-256 hash after checking it matches its payload."""
    record_hash = _require_sha256_hex(record.get("record_hash"), f"{label} record_hash")
    payload = dict(record)
    payload.pop("record_hash", None)
    expected_hash = sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if expected_hash != record_hash:
        raise ValueError(f"{label} record_hash does not match payload")
    return record_hash


def _normalise_hybrid_bipartition(
    value: object,
    *,
    qubit_count: int,
    label: str,
) -> list[list[int]]:
    """Return a two-group qubit bipartition covering each qubit exactly once."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a two-group bipartition")
    if len(value) != 2:
        raise ValueError(f"{label} must contain two groups")
    groups: list[list[int]] = []
    merged: list[int] = []
    for group_index, group in enumerate(value):
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            raise ValueError(f"{label} group {group_index} must be a sequence")
        if not group:
            raise ValueError(f"{label} group {group_index} must be non-empty")
        indices: list[int] = []
        for item in group:
            index = _hybrid_non_bool_int(item, f"{label} index")
            if index < 0 or index >= qubit_count:
                raise ValueError(f"{label} index out of range")
            indices.append(index)
            merged.append(index)
        groups.append(indices)
    if len(set(merged)) != len(merged) or len(merged) != qubit_count:
        raise ValueError(f"{label} must cover each qubit exactly once")
    return groups


def _hybrid_non_bool_int(value: object, name: str) -> int:
    """Return ``value`` as an integer, rejecting booleans, else raise."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _hybrid_positive_int(value: object, name: str) -> int:
    """Return ``value`` as a strictly positive integer, else raise."""
    result = _hybrid_non_bool_int(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _hybrid_non_negative(value: float, name: str) -> float:
    """Return ``value`` if it is non-negative, else raise ``ValueError``."""
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _hybrid_positive_float(value: float, name: str) -> float:
    """Return ``value`` if it is strictly positive, else raise ``ValueError``."""
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _hybrid_unit_interval(value: float, name: str) -> float:
    """Return ``value`` if it lies in [0, 1], else raise ``ValueError``."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value
