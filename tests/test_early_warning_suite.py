# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — domain-adaptable early-warning suite tests

"""Tests for the domain-adaptable early-warning suite.

The suite is the domain-neutral runner: given a :class:`SuiteObservables`
bundle it runs the three detector members and the weighted fusion under one
alarm contract, with no knowledge of the domain that produced the observables.
The tests exercise the bundle's cross-field validation surface, the phase
derivation that lets an adapter supply only phases, the neutral run on a genuine
coherence rise and on an incoherent null, the missing-threshold contract, and
the :class:`DomainObservableAdapter` protocol on a conforming and a
non-conforming class.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    CriticalSlowingDownWarning,
)
from scpn_phase_orchestrator.monitor.early_warning_suite import (
    CRITICAL_SLOWING_DOWN,
    ENSEMBLE_WEIGHTED,
    SUITE_DETECTORS,
    SYNCHRONISATION,
    TRANSITION_ENTROPY,
    DomainObservableAdapter,
    SuiteObservables,
    SuiteWarnings,
    observables_from_phases,
    run_early_warning_suite,
)
from scpn_phase_orchestrator.monitor.ensemble_warning import EnsembleWarning
from scpn_phase_orchestrator.monitor.explosive_sync import ExplosiveSyncWarning
from scpn_phase_orchestrator.monitor.synchronisation import SynchronisationWarning

_THRESHOLDS = {
    CRITICAL_SLOWING_DOWN: 3.0,
    SYNCHRONISATION: 3.0,
    TRANSITION_ENTROPY: 3.0,
    ENSEMBLE_WEIGHTED: 1.0,
}


def _rising_coherence_phases(
    n_nodes: int = 8, length: int = 2000, seed: int = 0
) -> np.ndarray:
    """Return phases that drift incoherently then lock to a shared phase.

    A ``lock`` ramp over the second half blends each node's independent drift
    into a common phase, so the Kuramoto order parameter climbs toward one — the
    synchronisation precursor the suite must catch.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    lock = np.clip((t - length // 2) / (length // 2), 0.0, 1.0)
    common = 0.01 * t
    phases = np.empty((n_nodes, length), dtype=np.float64)
    for node in range(n_nodes):
        individual = (
            rng.uniform(0.0, 2.0 * np.pi) + 0.02 * t + rng.standard_normal(length) * 0.3
        )
        phases[node] = (1.0 - lock) * individual + lock * common
    return phases


def _incoherent_phases(
    n_nodes: int = 8, length: int = 2000, seed: int = 1
) -> np.ndarray:
    """Return phases that never lock — a no-transition null."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return rng.uniform(0.0, 2.0 * np.pi, (n_nodes, length)) + 0.02 * t


def _valid_bundle_fields() -> dict[str, object]:
    """Return keyword fields for a minimal valid :class:`SuiteObservables`."""
    phases = np.array([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]], dtype=np.float64)
    return {
        "phases": phases,
        "phase_field": np.sin(phases),
        "order_parameter": np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        "sampling_rate_hz": 32.0,
    }


# --------------------------------------------------------------------------- #
# SuiteObservables construction and cross-field validation
# --------------------------------------------------------------------------- #


def test_bundle_accepts_consistent_observables() -> None:
    bundle = SuiteObservables(**_valid_bundle_fields())
    assert bundle.n_nodes == 2
    assert bundle.n_samples == 4
    assert bundle.sampling_rate_hz == 32.0
    assert bundle.phases.dtype == np.float64
    assert bundle.phase_field.flags["C_CONTIGUOUS"]


def test_bundle_rejects_single_node_phases() -> None:
    fields = _valid_bundle_fields()
    fields["phases"] = np.array([[0.0, 0.1, 0.2, 0.3]], dtype=np.float64)
    fields["phase_field"] = np.array([[0.0, 0.1, 0.2, 0.3]], dtype=np.float64)
    with pytest.raises(ValueError, match="at least two nodes"):
        SuiteObservables(**fields)


def test_bundle_rejects_mismatched_phase_field_shape() -> None:
    fields = _valid_bundle_fields()
    fields["phase_field"] = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="phase_field must share the shape"):
        SuiteObservables(**fields)


def test_bundle_rejects_order_parameter_length_mismatch() -> None:
    fields = _valid_bundle_fields()
    fields["order_parameter"] = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    with pytest.raises(ValueError, match="order_parameter length must match"):
        SuiteObservables(**fields)


def test_bundle_rejects_order_parameter_below_zero() -> None:
    fields = _valid_bundle_fields()
    fields["order_parameter"] = np.array([-0.01, 0.3, 0.4, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match=r"order_parameter must lie in \[0, 1\]"):
        SuiteObservables(**fields)


def test_bundle_rejects_order_parameter_above_one() -> None:
    fields = _valid_bundle_fields()
    fields["order_parameter"] = np.array([0.2, 0.3, 0.4, 1.5], dtype=np.float64)
    with pytest.raises(ValueError, match=r"order_parameter must lie in \[0, 1\]"):
        SuiteObservables(**fields)


def test_bundle_accepts_order_parameter_at_unit_bounds() -> None:
    fields = _valid_bundle_fields()
    fields["order_parameter"] = np.array([0.0, 0.5, 1.0, 0.5], dtype=np.float64)
    bundle = SuiteObservables(**fields)
    assert float(bundle.order_parameter[2]) == 1.0


# --------------------------------------------------------------------------- #
# _validate_field surface, reached through the phases argument
# --------------------------------------------------------------------------- #


def test_bundle_rejects_boolean_phases() -> None:
    fields = _valid_bundle_fields()
    fields["phases"] = np.array(
        [[True, False, True, False], [False, True, False, True]]
    )
    with pytest.raises(ValueError, match="must not contain boolean"):
        SuiteObservables(**fields)


def test_bundle_rejects_complex_phases() -> None:
    fields = _valid_bundle_fields()
    fields["phases"] = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    with pytest.raises(ValueError, match="phases must be real-valued"):
        SuiteObservables(**fields)


def test_bundle_rejects_non_castable_phases() -> None:
    fields = _valid_bundle_fields()
    fields["phases"] = np.array([["a", "b"], ["c", "d"]])
    with pytest.raises(ValueError, match="phases must be a real float array"):
        SuiteObservables(**fields)


def test_bundle_rejects_one_dimensional_phases() -> None:
    fields = _valid_bundle_fields()
    fields["phases"] = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="must be two-dimensional"):
        SuiteObservables(**fields)


def test_bundle_rejects_zero_length_phases() -> None:
    fields = _valid_bundle_fields()
    fields["phases"] = np.zeros((2, 0), dtype=np.float64)
    with pytest.raises(ValueError, match="at least one sample"):
        SuiteObservables(**fields)


def test_bundle_rejects_non_finite_phases() -> None:
    fields = _valid_bundle_fields()
    fields["phases"] = np.array([[0.0, np.inf, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]])
    with pytest.raises(ValueError, match="only finite values"):
        SuiteObservables(**fields)


# --------------------------------------------------------------------------- #
# _validate_series surface, reached through the order_parameter argument
# --------------------------------------------------------------------------- #


def test_bundle_rejects_boolean_order_parameter() -> None:
    fields = _valid_bundle_fields()
    fields["order_parameter"] = np.array([True, False, True, False])
    with pytest.raises(ValueError, match="must not contain boolean"):
        SuiteObservables(**fields)


def test_bundle_rejects_complex_order_parameter() -> None:
    fields = _valid_bundle_fields()
    fields["order_parameter"] = np.array([0.2 + 1j, 0.3, 0.4, 0.5])
    with pytest.raises(ValueError, match="order_parameter must be real-valued"):
        SuiteObservables(**fields)


def test_bundle_rejects_non_castable_order_parameter() -> None:
    fields = _valid_bundle_fields()
    fields["order_parameter"] = np.array(["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="order_parameter must be a real float array"):
        SuiteObservables(**fields)


def test_bundle_rejects_two_dimensional_order_parameter() -> None:
    fields = _valid_bundle_fields()
    fields["order_parameter"] = np.zeros((2, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="must be one-dimensional"):
        SuiteObservables(**fields)


def test_bundle_rejects_zero_length_order_parameter() -> None:
    fields = _valid_bundle_fields()
    fields["order_parameter"] = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="at least one sample"):
        SuiteObservables(**fields)


def test_bundle_rejects_non_finite_order_parameter() -> None:
    fields = _valid_bundle_fields()
    fields["order_parameter"] = np.array([0.2, np.nan, 0.4, 0.5])
    with pytest.raises(ValueError, match="only finite values"):
        SuiteObservables(**fields)


# --------------------------------------------------------------------------- #
# _positive_real surface, reached through sampling_rate_hz
# --------------------------------------------------------------------------- #


def test_bundle_rejects_boolean_sampling_rate() -> None:
    fields = _valid_bundle_fields()
    fields["sampling_rate_hz"] = True
    with pytest.raises(ValueError, match="sampling_rate_hz must be a positive real"):
        SuiteObservables(**fields)


def test_bundle_rejects_non_real_sampling_rate() -> None:
    fields = _valid_bundle_fields()
    fields["sampling_rate_hz"] = "32"
    with pytest.raises(ValueError, match="sampling_rate_hz must be a positive real"):
        SuiteObservables(**fields)


def test_bundle_rejects_non_finite_sampling_rate() -> None:
    fields = _valid_bundle_fields()
    fields["sampling_rate_hz"] = np.inf
    with pytest.raises(ValueError, match="finite and positive"):
        SuiteObservables(**fields)


def test_bundle_rejects_non_positive_sampling_rate() -> None:
    fields = _valid_bundle_fields()
    fields["sampling_rate_hz"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        SuiteObservables(**fields)


# --------------------------------------------------------------------------- #
# observables_from_phases
# --------------------------------------------------------------------------- #


def test_observables_from_phases_derives_field_and_order() -> None:
    phases = _rising_coherence_phases(n_nodes=4, length=64)
    bundle = observables_from_phases(phases, sampling_rate_hz=32.0)
    assert isinstance(bundle, SuiteObservables)
    assert bundle.n_nodes == 4
    assert bundle.n_samples == 64
    assert bundle.sampling_rate_hz == 32.0
    np.testing.assert_allclose(bundle.phase_field, np.sin(phases))
    expected_order = np.abs(np.mean(np.exp(1j * phases), axis=0))
    np.testing.assert_allclose(bundle.order_parameter, expected_order)
    assert np.all(bundle.order_parameter >= 0.0)
    assert np.all(bundle.order_parameter <= 1.0 + 1e-9)


def test_observables_from_phases_rejects_single_node() -> None:
    with pytest.raises(ValueError, match="at least two nodes"):
        observables_from_phases(
            np.array([[0.0, 0.1, 0.2, 0.3]], dtype=np.float64), sampling_rate_hz=32.0
        )


def test_observables_from_phases_rejects_malformed_phases() -> None:
    with pytest.raises(ValueError, match="must be two-dimensional"):
        observables_from_phases(
            np.array([0.0, 0.1, 0.2], dtype=np.float64), sampling_rate_hz=32.0
        )


# --------------------------------------------------------------------------- #
# run_early_warning_suite
# --------------------------------------------------------------------------- #


def test_suite_runs_all_members_and_fusion() -> None:
    bundle = observables_from_phases(_rising_coherence_phases(), sampling_rate_hz=32.0)
    warnings = run_early_warning_suite(bundle, thresholds=_THRESHOLDS)
    assert isinstance(warnings, SuiteWarnings)
    assert isinstance(warnings.critical_slowing_down, CriticalSlowingDownWarning)
    assert isinstance(warnings.synchronisation, SynchronisationWarning)
    assert isinstance(warnings.transition_entropy, ExplosiveSyncWarning)
    assert isinstance(warnings.ensemble, EnsembleWarning)
    verdicts = warnings.triggered()
    assert set(verdicts) == set(SUITE_DETECTORS)
    assert all(isinstance(flag, bool) for flag in verdicts.values())


def test_suite_alarms_synchronisation_on_rising_coherence() -> None:
    bundle = observables_from_phases(_rising_coherence_phases(), sampling_rate_hz=32.0)
    warnings = run_early_warning_suite(bundle, thresholds=_THRESHOLDS)
    assert warnings.synchronisation.warning_triggered is True
    assert warnings.triggered()[SYNCHRONISATION] is True
    assert warnings.synchronisation.warning_sample is not None
    # A member and the fusion share the same window grid.
    assert np.array_equal(
        warnings.synchronisation.window_starts, warnings.ensemble.window_starts
    )


def test_suite_silent_on_incoherent_null() -> None:
    bundle = observables_from_phases(_incoherent_phases(), sampling_rate_hz=32.0)
    # The fused threshold in `_THRESHOLDS` is a permissive 1.0, which a weighted
    # mean of noisy z-scores can cross; a null run must be judged at a
    # conservative operating point, as the matched-FA harness calibrates it.
    conservative = {**_THRESHOLDS, ENSEMBLE_WEIGHTED: 6.0}
    warnings = run_early_warning_suite(bundle, thresholds=conservative)
    assert warnings.synchronisation.warning_triggered is False
    assert warnings.ensemble.warning_triggered is False


def test_suite_honours_analysis_controls() -> None:
    bundle = observables_from_phases(
        _rising_coherence_phases(length=512), sampling_rate_hz=32.0
    )
    warnings = run_early_warning_suite(
        bundle,
        thresholds=_THRESHOLDS,
        relative_gate=0.02,
        window=64,
        step=8,
        baseline_fraction=0.3,
        persistence=3,
    )
    assert warnings.synchronisation.window == 64
    assert warnings.synchronisation.step == 8
    assert warnings.synchronisation.persistence == 3
    assert warnings.ensemble.persistence == 3


def test_suite_raises_on_missing_threshold() -> None:
    bundle = SuiteObservables(**_valid_bundle_fields())
    incomplete = {
        SYNCHRONISATION: 3.0,
        TRANSITION_ENTROPY: 3.0,
        ENSEMBLE_WEIGHTED: 1.0,
    }
    with pytest.raises(KeyError):
        run_early_warning_suite(bundle, thresholds=incomplete, window=4, step=1)


# --------------------------------------------------------------------------- #
# DomainObservableAdapter protocol
# --------------------------------------------------------------------------- #


class _PhaseAdapter:
    """A conforming adapter that turns raw phases into the neutral bundle."""

    def __init__(self, sampling_rate_hz: float) -> None:
        self._rate = sampling_rate_hz

    @property
    def domain(self) -> str:
        return "unit_test_phase"

    def observables(self, raw: np.ndarray) -> SuiteObservables:
        return observables_from_phases(raw, sampling_rate_hz=self._rate)


class _NotAnAdapter:
    """A class missing the ``observables`` method — not a valid adapter."""

    @property
    def domain(self) -> str:
        return "incomplete"


def test_conforming_adapter_satisfies_protocol() -> None:
    adapter = _PhaseAdapter(sampling_rate_hz=32.0)
    assert isinstance(adapter, DomainObservableAdapter)
    assert adapter.domain == "unit_test_phase"
    bundle = adapter.observables(_rising_coherence_phases(n_nodes=4, length=64))
    assert isinstance(bundle, SuiteObservables)
    assert bundle.n_nodes == 4


def test_incomplete_class_is_not_an_adapter() -> None:
    assert not isinstance(_NotAnAdapter(), DomainObservableAdapter)
