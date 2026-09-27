# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Early warning measurement source types

"""Exercise original measurement types across the real warning pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    CriticalSlowingDownWarning,
    critical_slowing_down_warning,
    surrogate_score_threshold,
)
from scpn_phase_orchestrator.monitor.early_warning_suite import (
    SuiteWarnings,
    observables_from_phases,
    run_early_warning_suite,
)
from scpn_phase_orchestrator.monitor.ensemble_warning import (
    ensemble_warning,
    member_from_synchronisation,
)
from scpn_phase_orchestrator.monitor.explosive_sync import (
    ExplosiveSyncWarning,
    explosive_sync_warning,
)
from scpn_phase_orchestrator.monitor.synchronisation import (
    SynchronisationWarning,
    synchronisation_warning,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _phases() -> FloatArray:
    """Return two evolving channels with nonconstant phase separation."""
    time = np.arange(256, dtype=np.float64) / 20
    return np.vstack([time, 1.2 * time + 0.2 * np.sin(time)])


def _run(detector: str, values: FloatArray) -> object:
    """Call a public detector or the complete observable-to-ensemble pipeline."""
    if detector == "synchronisation":
        return synchronisation_warning(values, window=32, step=16)
    if detector == "critical":
        return critical_slowing_down_warning(values, window=32, step=16)
    if detector == "explosive":
        return explosive_sync_warning(values, window=32, step=16)
    observables = observables_from_phases(values, sampling_rate_hz=20)
    return run_early_warning_suite(
        observables,
        thresholds={
            "critical_slowing_down": 3.0,
            "synchronisation": 3.0,
            "transition_entropy": 3.0,
            "ensemble_weighted": 3.0,
        },
        window=32,
        step=16,
    )


@pytest.mark.parametrize(
    "detector", ["synchronisation", "critical", "explosive", "suite"]
)
@pytest.mark.parametrize(
    "dtype", ["U16", "bool", "complex128", "timedelta64[ns]", "object-time"]
)
def test_warning_pipeline_rejects_measurement_aliases(
    detector: str, dtype: str
) -> None:
    """Temporal object values remain temporal until the public refusal."""
    source = _phases()
    values = (
        np.array(
            [np.timedelta64(i, "ns") for i in range(source.size)], dtype=object
        ).reshape(source.shape)
        if dtype == "object-time"
        else source.astype(dtype)
    )
    with pytest.raises(ValueError):
        _run(detector, cast(FloatArray, values))


@pytest.mark.parametrize(
    "detector", ["synchronisation", "critical", "explosive", "suite"]
)
def test_warning_pipeline_preserves_real_object_samples(detector: str) -> None:
    """Numeric objects produce the same sealed public result as float storage."""
    expected = _run(detector, _phases())
    actual = _run(detector, cast(FloatArray, _phases().astype(object)))
    if isinstance(expected, SynchronisationWarning):
        assert isinstance(actual, SynchronisationWarning)
        np.testing.assert_array_equal(actual.synchrony_index, expected.synchrony_index)
        assert actual.summary() == expected.summary()
    elif isinstance(expected, CriticalSlowingDownWarning):
        assert isinstance(actual, CriticalSlowingDownWarning)
        np.testing.assert_array_equal(actual.combined_z, expected.combined_z)
        assert actual.summary() == expected.summary()
    elif isinstance(expected, ExplosiveSyncWarning):
        assert isinstance(actual, ExplosiveSyncWarning)
        np.testing.assert_array_equal(actual.entropy_index, expected.entropy_index)
        assert actual.summary() == expected.summary()
    else:
        assert isinstance(expected, SuiteWarnings)
        assert isinstance(actual, SuiteWarnings)
        np.testing.assert_array_equal(
            actual.ensemble.fused_score, expected.ensemble.fused_score
        )
        assert actual.triggered() == expected.triggered()


def test_ensemble_rejects_temporal_member_scores() -> None:
    """A real detector record cannot be resealed with unit-bearing scores."""
    warning = synchronisation_warning(_phases(), window=32, step=16)
    member = member_from_synchronisation(warning)
    values = np.array([np.timedelta64(1, "ns")] * member.oriented_z.size, dtype=object)
    with pytest.raises(ValueError):
        replace(member, oriented_z=cast(FloatArray, values))


def test_ensemble_rejects_temporal_weights() -> None:
    """Temporal fusion weights cannot become real confidence coefficients."""
    member = member_from_synchronisation(synchronisation_warning(_phases(), window=32))
    with pytest.raises(ValueError):
        ensemble_warning([member], weights=[cast(float, np.timedelta64(1, "ms"))])


def test_surrogate_threshold_rejects_temporal_seed() -> None:
    """A duration cannot select a bootstrap random stream."""
    with pytest.raises(ValueError):
        surrogate_score_threshold(_phases(), rng=cast(int, np.timedelta64(1, "ns")))


@pytest.mark.parametrize("detector", ["synchronisation", "critical", "explosive"])
@pytest.mark.parametrize(
    "field", ["window", "step", "z_threshold", "baseline_fraction"]
)
def test_warning_rejects_temporal_controls(detector: str, field: str) -> None:
    """A duration is neither a window count nor a dimensionless alarm gate."""
    value = np.timedelta64(2, "ns")
    window = cast(int, value) if field == "window" else 32
    step = cast(int, value) if field == "step" else 16
    threshold = cast(float, value) if field == "z_threshold" else 3.0
    fraction = cast(float, value) if field == "baseline_fraction" else 0.25
    with pytest.raises(ValueError):
        if detector == "synchronisation":
            synchronisation_warning(
                _phases(),
                window=window,
                step=step,
                z_threshold=threshold,
                baseline_fraction=fraction,
            )
        elif detector == "critical":
            critical_slowing_down_warning(
                _phases(),
                window=window,
                step=step,
                z_threshold=threshold,
                baseline_fraction=fraction,
            )
        else:
            explosive_sync_warning(
                _phases(),
                window=window,
                step=step,
                z_threshold=threshold,
                baseline_fraction=fraction,
            )


def test_ensemble_rejects_temporal_baseline_count() -> None:
    """A timedelta cannot become the baseline boundary of a sealed member."""
    member = member_from_synchronisation(synchronisation_warning(_phases(), window=32))
    with pytest.raises(ValueError):
        replace(member, n_baseline_windows=cast(int, np.timedelta64(1, "ns")))


@pytest.mark.parametrize(
    "detector", ["synchronisation", "critical", "explosive", "suite"]
)
def test_warning_rejects_boolean_inside_mixed_numeric_sequence(detector: str) -> None:
    """Sequence promotion cannot hide a Boolean among real measurements."""
    values: list[list[float | bool]] = [list(row) for row in _phases()]
    values[0][0] = True
    with pytest.raises(ValueError):
        _run(detector, cast(FloatArray, values))


def test_observable_bundle_rejects_temporal_sampling_rate() -> None:
    """A duration cannot masquerade as a phase sampling frequency."""
    with pytest.raises(ValueError):
        observables_from_phases(
            _phases(), sampling_rate_hz=cast(float, np.timedelta64(20, "ms"))
        )
