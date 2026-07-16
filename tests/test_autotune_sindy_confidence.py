# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — phase-SINDy discovery confidence tests

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from scpn_phase_orchestrator.autotune.discovery import discover_time_series_structure
from scpn_phase_orchestrator.autotune.sindy_confidence import (
    DEFAULT_SINDY_CONFIDENCE_POLICY,
    POSTURE_DISCOVERED,
    POSTURE_INSUFFICIENT_EVIDENCE,
    POSTURE_REFUSED,
    SindyConfidence,
    SindyConfidencePolicy,
    classify_phase_sindy_block,
    classify_phase_sindy_confidence,
)
from scpn_phase_orchestrator.binding.types import (
    VALID_VALIDATION_TIERS,
    VALIDATION_TIER_EXTERNALLY_VALIDATED,
    VALIDATION_TIER_PARTIAL,
    VALIDATION_TIER_SCAFFOLD,
)


def _strong_fit_kwargs(**overrides: object) -> dict[str, object]:
    """Return keyword arguments describing a strong, well-determined fit."""
    base: dict[str, object] = {
        "status": "fitted",
        "r_squared": 0.99,
        "sample_count": 100,
        "node_count": 3,
        "active_terms": 4,
        "total_terms": 9,
        "sparsity": 0.55,
    }
    base.update(overrides)
    return base


def _planted_kuramoto(
    *,
    omega: tuple[float, ...],
    coupling: float,
    steps: int,
    dt: float,
    noise_scale: float = 0.0,
) -> np.ndarray:
    """Integrate an all-to-all Kuramoto system into a phase table.

    This plants a genuine trajectory of a known model; it is a simulation, not
    fabricated measurement data. Optional Gaussian phase noise (seeded, hence
    deterministic) degrades the recoverability so a fitted-but-weak case can be
    exercised honestly.
    """
    node_count = len(omega)
    natural = np.asarray(omega, dtype=np.float64)
    phases = np.zeros((steps, node_count), dtype=np.float64)
    state = np.linspace(0.0, 0.3, node_count, dtype=np.float64)
    for index in range(steps):
        phases[index] = state
        differences = state[np.newaxis, :] - state[:, np.newaxis]
        drift = natural + coupling * np.sin(differences).sum(axis=1) / node_count
        state = state + dt * drift
    if noise_scale > 0.0:
        rng = np.random.default_rng(20260716)
        phases = phases + rng.normal(0.0, noise_scale, size=phases.shape)
    return phases


def test_refused_when_no_fit_was_performed() -> None:
    verdict = classify_phase_sindy_confidence(
        status="requires_at_least_three_samples",
        r_squared=None,
        sample_count=0,
        node_count=0,
        active_terms=0,
        total_terms=0,
        sparsity=1.0,
    )

    assert verdict.posture == POSTURE_REFUSED
    assert verdict.tier == VALIDATION_TIER_SCAFFOLD
    assert verdict.r_squared is None
    assert verdict.samples_per_parameter is None
    assert verdict.reasons


@pytest.mark.parametrize(
    "status",
    [
        "requires_at_least_two_phase_columns",
        "requires_at_least_three_samples",
        "requires_at_least_one_derivative_sample_per_feature",
        "skipped_non_phase_like",
    ],
)
def test_every_skip_status_is_refused(status: str) -> None:
    verdict = classify_phase_sindy_confidence(
        status=status,
        r_squared=None,
        sample_count=0,
        node_count=0,
        active_terms=0,
        total_terms=0,
        sparsity=1.0,
    )

    assert verdict.posture == POSTURE_REFUSED
    assert verdict.tier == VALIDATION_TIER_SCAFFOLD


def test_discovered_on_strong_well_determined_fit() -> None:
    verdict = classify_phase_sindy_confidence(**_strong_fit_kwargs())  # type: ignore[arg-type]

    assert verdict.posture == POSTURE_DISCOVERED
    assert verdict.tier == VALIDATION_TIER_PARTIAL
    assert verdict.samples_per_parameter == pytest.approx(100.0 / 3.0)
    assert any("self-consistent" in reason for reason in verdict.reasons)


def test_insufficient_on_low_r_squared() -> None:
    verdict = classify_phase_sindy_confidence(
        **_strong_fit_kwargs(r_squared=0.4)  # type: ignore[arg-type]
    )

    assert verdict.posture == POSTURE_INSUFFICIENT_EVIDENCE
    assert verdict.tier == VALIDATION_TIER_SCAFFOLD
    assert any("R²" in reason for reason in verdict.reasons)


def test_insufficient_on_under_determined_fit() -> None:
    verdict = classify_phase_sindy_confidence(
        **_strong_fit_kwargs(sample_count=6, node_count=3)  # type: ignore[arg-type]
    )

    assert verdict.posture == POSTURE_INSUFFICIENT_EVIDENCE
    assert verdict.tier == VALIDATION_TIER_SCAFFOLD
    assert any("under-determined" in reason for reason in verdict.reasons)


def test_insufficient_when_no_active_terms_survived() -> None:
    verdict = classify_phase_sindy_confidence(
        **_strong_fit_kwargs(active_terms=0)  # type: ignore[arg-type]
    )

    assert verdict.posture == POSTURE_INSUFFICIENT_EVIDENCE
    assert any("no active terms" in reason for reason in verdict.reasons)


def test_insufficient_when_r_squared_missing_on_a_fitted_block() -> None:
    verdict = classify_phase_sindy_confidence(
        **_strong_fit_kwargs(r_squared=None)  # type: ignore[arg-type]
    )

    assert verdict.posture == POSTURE_INSUFFICIENT_EVIDENCE
    assert verdict.r_squared is None
    assert any("no R²" in reason for reason in verdict.reasons)


def test_samples_per_parameter_unknown_when_node_count_zero() -> None:
    verdict = classify_phase_sindy_confidence(
        **_strong_fit_kwargs(node_count=0)  # type: ignore[arg-type]
    )

    assert verdict.posture == POSTURE_INSUFFICIENT_EVIDENCE
    assert verdict.samples_per_parameter is None
    assert any("parameter count is unknown" in reason for reason in verdict.reasons)


def test_custom_policy_can_admit_a_looser_fit() -> None:
    loose = SindyConfidencePolicy(min_r_squared=0.3, min_samples_per_parameter=1.0)

    verdict = classify_phase_sindy_confidence(
        **_strong_fit_kwargs(r_squared=0.4, sample_count=6, node_count=3),  # type: ignore[arg-type]
        policy=loose,
    )

    assert verdict.posture == POSTURE_DISCOVERED
    assert verdict.tier == VALIDATION_TIER_PARTIAL


def test_policy_exposes_conservative_defaults() -> None:
    assert DEFAULT_SINDY_CONFIDENCE_POLICY.min_r_squared == 0.9
    assert DEFAULT_SINDY_CONFIDENCE_POLICY.min_samples_per_parameter == 5.0


def test_to_audit_record_is_json_serialisable() -> None:
    verdict = classify_phase_sindy_confidence(**_strong_fit_kwargs())  # type: ignore[arg-type]

    record = verdict.to_audit_record()
    round_tripped = json.loads(json.dumps(record))

    assert round_tripped["tier"] == VALIDATION_TIER_PARTIAL
    assert round_tripped["posture"] == POSTURE_DISCOVERED
    assert isinstance(round_tripped["reasons"], list)
    assert round_tripped["samples_per_parameter"] == pytest.approx(100.0 / 3.0)


@given(
    status=st.sampled_from(
        [
            "fitted",
            "skipped_non_phase_like",
            "requires_at_least_three_samples",
        ]
    ),
    r_squared=st.one_of(
        st.none(),
        st.floats(min_value=-5.0, max_value=1.0, allow_nan=False),
    ),
    sample_count=st.integers(min_value=0, max_value=5000),
    node_count=st.integers(min_value=0, max_value=64),
    active_terms=st.integers(min_value=0, max_value=64),
    total_terms=st.integers(min_value=0, max_value=64),
    sparsity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_classifier_never_awards_external_validation(
    status: str,
    r_squared: float | None,
    sample_count: int,
    node_count: int,
    active_terms: int,
    total_terms: int,
    sparsity: float,
) -> None:
    verdict = classify_phase_sindy_confidence(
        status=status,
        r_squared=r_squared,
        sample_count=sample_count,
        node_count=node_count,
        active_terms=active_terms,
        total_terms=total_terms,
        sparsity=sparsity,
    )

    assert verdict.tier != VALIDATION_TIER_EXTERNALLY_VALIDATED
    assert verdict.tier in VALID_VALIDATION_TIERS
    assert verdict.posture in {
        POSTURE_DISCOVERED,
        POSTURE_INSUFFICIENT_EVIDENCE,
        POSTURE_REFUSED,
    }
    # 'partial' is granted exactly when — and only when — the posture is
    # 'discovered'; every other verdict stays at the honest 'scaffold' floor.
    assert (verdict.tier == VALIDATION_TIER_PARTIAL) == (
        verdict.posture == POSTURE_DISCOVERED
    )


def test_block_adapter_refuses_a_skipped_block() -> None:
    samples = np.asarray(
        [[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0]],
        dtype=np.float64,
    )
    report = discover_time_series_structure(
        samples,
        columns=("temperature", "pressure"),
        sample_period_s=1.0,
    )

    verdict = classify_phase_sindy_block(report.phase_sindy)

    assert verdict.posture == POSTURE_REFUSED
    assert verdict.tier == VALIDATION_TIER_SCAFFOLD


def test_block_adapter_discovers_a_clean_planted_kuramoto() -> None:
    phases = _planted_kuramoto(omega=(1.0, 1.2), coupling=0.8, steps=240, dt=0.02)
    report = discover_time_series_structure(
        phases,
        columns=("theta_0", "theta_1"),
        sample_period_s=0.02,
    )
    assert report.phase_sindy["status"] == "fitted"

    verdict = classify_phase_sindy_block(report.phase_sindy)

    assert verdict.posture == POSTURE_DISCOVERED
    assert verdict.tier == VALIDATION_TIER_PARTIAL
    assert verdict.r_squared is not None
    assert verdict.r_squared >= DEFAULT_SINDY_CONFIDENCE_POLICY.min_r_squared


def test_block_adapter_downgrades_a_noise_dominated_planted_kuramoto() -> None:
    phases = _planted_kuramoto(
        omega=(1.0, 1.2),
        coupling=0.8,
        steps=240,
        dt=0.02,
        noise_scale=1.5,
    )
    report = discover_time_series_structure(
        phases,
        columns=("theta_0", "theta_1"),
        sample_period_s=0.02,
    )

    verdict = classify_phase_sindy_block(report.phase_sindy)

    # Noise this large destroys the sine-difference structure; the honest
    # posture must not be 'discovered'.
    assert verdict.posture != POSTURE_DISCOVERED
    assert verdict.tier == VALIDATION_TIER_SCAFFOLD


def test_confidence_dataclass_is_frozen() -> None:
    verdict = SindyConfidence(
        tier=VALIDATION_TIER_SCAFFOLD,
        posture=POSTURE_REFUSED,
        r_squared=None,
        samples_per_parameter=None,
    )

    with pytest.raises(AttributeError):
        verdict.tier = VALIDATION_TIER_PARTIAL  # type: ignore[misc]
