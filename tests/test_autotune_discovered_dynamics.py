# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — discovered-dynamics record tests

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune.discovered_dynamics import (
    DiscoveredDynamics,
    discovered_dynamics_from_block,
)
from scpn_phase_orchestrator.autotune.discovery import discover_time_series_structure
from scpn_phase_orchestrator.autotune.sindy_confidence import (
    POSTURE_DISCOVERED,
    POSTURE_REFUSED,
    SindyConfidence,
    SindyConfidencePolicy,
)
from scpn_phase_orchestrator.binding.types import (
    VALIDATION_TIER_PARTIAL,
    VALIDATION_TIER_SCAFFOLD,
)

_HEX_DIGITS = set("0123456789abcdef")


def _planted_two_node_kuramoto(
    *,
    omega: tuple[float, float],
    coupling: float,
    steps: int,
    dt: float,
) -> np.ndarray:
    """Integrate a two-node Kuramoto system into a phase table (a simulation)."""
    natural = np.asarray(omega, dtype=np.float64)
    phases = np.zeros((steps, 2), dtype=np.float64)
    state = np.asarray([0.0, 0.4], dtype=np.float64)
    for index in range(steps):
        phases[index] = state
        drift = np.asarray(
            [
                natural[0] + coupling * np.sin(state[1] - state[0]),
                natural[1] + coupling * np.sin(state[0] - state[1]),
            ]
        )
        state = state + dt * drift
    return phases


def _clean_report() -> object:
    phases = _planted_two_node_kuramoto(
        omega=(1.0, 1.3), coupling=0.8, steps=260, dt=0.02
    )
    return discover_time_series_structure(
        phases,
        columns=("theta_0", "theta_1"),
        sample_period_s=0.02,
    )


def test_record_from_clean_kuramoto_is_discovered_with_recovered_edges() -> None:
    report = _clean_report()
    record = discovered_dynamics_from_block(report.phase_sindy)

    assert record.library == "kuramoto_sine_phase_differences"
    assert record.status == "fitted"
    assert record.confidence.posture == POSTURE_DISCOVERED
    assert record.confidence.tier == VALIDATION_TIER_PARTIAL
    assert record.equations
    assert record.coupling_edges

    # The recovered coupling must land near the planted K = 0.8. This is the
    # edge-recovery-within-tolerance check on clean planted data.
    coefficients = [abs(float(edge["coefficient"])) for edge in record.coupling_edges]
    assert coefficients
    assert min(coefficients) == pytest.approx(0.8, abs=0.2)


def test_record_content_hash_is_lowercase_hex_sha256() -> None:
    report = _clean_report()
    record = discovered_dynamics_from_block(report.phase_sindy)

    digest = record.content_hash
    assert len(digest) == 64
    assert set(digest) <= _HEX_DIGITS


def test_record_hash_is_stable_and_tamper_evident() -> None:
    report = _clean_report()
    first = discovered_dynamics_from_block(report.phase_sindy)
    second = discovered_dynamics_from_block(report.phase_sindy)

    assert first.content_hash == second.content_hash

    mutated = DiscoveredDynamics(
        library=first.library,
        status="tampered",
        equations=first.equations,
        coupling_edges=first.coupling_edges,
        confidence=first.confidence,
    )
    assert mutated.content_hash != first.content_hash


def test_record_to_audit_record_is_json_safe_and_carries_provenance() -> None:
    report = _clean_report()
    record = discovered_dynamics_from_block(report.phase_sindy)

    audit = record.to_audit_record()
    round_tripped = json.loads(json.dumps(audit))

    assert round_tripped["content_hash"] == record.content_hash
    assert round_tripped["confidence"]["tier"] == VALIDATION_TIER_PARTIAL
    assert round_tripped["confidence"]["posture"] == POSTURE_DISCOVERED
    assert round_tripped["library"] == "kuramoto_sine_phase_differences"
    assert isinstance(round_tripped["equations"], list)
    assert isinstance(round_tripped["coupling_edges"], list)


def test_record_from_skipped_block_refuses_without_equations() -> None:
    samples = np.asarray(
        [[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0]],
        dtype=np.float64,
    )
    report = discover_time_series_structure(
        samples,
        columns=("temperature", "pressure"),
        sample_period_s=1.0,
    )

    record = discovered_dynamics_from_block(report.phase_sindy)

    assert record.confidence.posture == POSTURE_REFUSED
    assert record.confidence.tier == VALIDATION_TIER_SCAFFOLD
    assert record.equations == ()
    assert record.coupling_edges == ()
    # A refused record still hashes cleanly.
    assert len(record.content_hash) == 64


def test_record_respects_a_custom_confidence_policy() -> None:
    report = _clean_report()
    strict = SindyConfidencePolicy(min_samples_per_parameter=1_000.0)

    record = discovered_dynamics_from_block(report.phase_sindy, policy=strict)

    # An unreachable determination threshold must downgrade the very same fit.
    assert record.confidence.posture != POSTURE_DISCOVERED
    assert record.confidence.tier == VALIDATION_TIER_SCAFFOLD


def test_record_from_empty_block_is_refused() -> None:
    record = discovered_dynamics_from_block({})

    assert isinstance(record.confidence, SindyConfidence)
    assert record.confidence.posture == POSTURE_REFUSED
    assert record.library == ""
    assert record.status == ""
    assert record.equations == ()


def test_report_method_matches_block_builder_and_honours_policy() -> None:
    report = _clean_report()

    from_method = report.discovered_dynamics()
    from_block = discovered_dynamics_from_block(report.phase_sindy)
    assert from_method.content_hash == from_block.content_hash
    assert from_method.confidence.posture == POSTURE_DISCOVERED

    strict = SindyConfidencePolicy(min_samples_per_parameter=1_000.0)
    downgraded = report.discovered_dynamics(policy=strict)
    assert downgraded.confidence.posture != POSTURE_DISCOVERED
    assert downgraded.confidence.tier == VALIDATION_TIER_SCAFFOLD


def test_discovered_dynamics_is_frozen() -> None:
    record = discovered_dynamics_from_block({})

    with pytest.raises(AttributeError):
        record.status = "mutated"  # type: ignore[misc]
