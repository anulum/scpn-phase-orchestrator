# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Supervisor measurement ingress

"""Exercise measurement source types through real supervisor public APIs."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.supervisor.causal import learn_causal_graph
from scpn_phase_orchestrator.supervisor.information_geometry import (
    propose_information_geometry_control,
)
from scpn_phase_orchestrator.supervisor.predictive import (
    FEPPredictiveSupervisor,
    PredictiveSupervisor,
)


@pytest.mark.parametrize("dtype", ["U8", "S8", "timedelta64[ms]", "datetime64[ms]"])
def test_causal_trace_rejects_measurement_aliases(dtype: str) -> None:
    """Text and temporal trace values cannot become causal influence edges."""
    source = np.array([0, 1, 0, 1, 0, 1]).astype(dtype).tolist()
    trace = {
        "source": cast("list[float]", source),
        "target": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    }
    with pytest.raises(ValueError, match="trace signal"):
        learn_causal_graph(trace)


@pytest.mark.parametrize("argument", ["phases", "omegas", "knm", "alpha"])
@pytest.mark.parametrize(
    "dtype", ["U8", "S8", "timedelta64[ms]", "datetime64[ms]", "object_time"]
)
def test_prediction_rejects_measurement_aliases(argument: str, dtype: str) -> None:
    """Every forward-model state array validates original measurement types."""
    phases = np.array([0.0, 1.0])
    omegas = np.ones(2)
    knm = np.zeros((2, 2))
    alpha = np.zeros((2, 2))
    raw = {"phases": phases, "omegas": omegas, "knm": knm, "alpha": alpha}[argument]
    invalid = raw.astype(object if dtype == "object_time" else dtype)
    if dtype == "object_time":
        invalid.flat[0] = np.timedelta64(1, "ms")
    with pytest.raises(ValueError, match=argument):
        PredictiveSupervisor(2, 0.01).predict(
            invalid if argument == "phases" else phases,
            invalid if argument == "omegas" else omegas,
            invalid if argument == "knm" else knm,
            invalid if argument == "alpha" else alpha,
        )


@pytest.mark.parametrize(
    "argument", ["current_distribution", "target_distribution", "coupling_gradient"]
)
@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize(
    "dtype", ["U8", "S8", "timedelta64[ms]", "datetime64[ms]", "object_time"]
)
def test_geometry_rejects_measurement_aliases_before_backend_dispatch(
    argument: str, backend: str, dtype: str
) -> None:
    """NumPy and JAX proposals share the same fail-closed ingress contract."""
    current = np.array([0.4, 0.6])
    target = np.array([0.5, 0.5])
    gradient = np.array([0.1, -0.1])
    raw = {
        "current_distribution": current,
        "target_distribution": target,
        "coupling_gradient": gradient,
    }[argument]
    invalid = raw.astype(object if dtype == "object_time" else dtype)
    if dtype == "object_time":
        invalid.flat[0] = np.timedelta64(1, "ms")
    with pytest.raises(ValueError, match=argument):
        propose_information_geometry_control(
            invalid if argument == "current_distribution" else current,
            invalid if argument == "target_distribution" else target,
            invalid if argument == "coupling_gradient" else gradient,
            max_step=0.1,
            backend=backend,
        )


def test_supervisors_preserve_real_numeric_object_results() -> None:
    """Compatible numerical objects preserve prediction and proposal hashes."""
    phases = np.array([0.0, 1.0])
    omegas = np.ones(2)
    matrix = np.zeros((2, 2))
    expected = PredictiveSupervisor(2, 0.01).predict(phases, omegas, matrix, matrix)
    actual = PredictiveSupervisor(2, 0.01).predict(
        phases.astype(object),
        omegas.astype(object),
        matrix.astype(object),
        matrix.astype(object),
    )
    assert actual == expected
    current = np.array([0.4, 0.6])
    target = np.array([0.5, 0.5])
    reference = propose_information_geometry_control(current, target, max_step=0.1)
    result = propose_information_geometry_control(
        current.astype(object), target.astype(object), max_step=0.1
    )
    assert result.proposal_hash == reference.proposal_hash


@pytest.mark.parametrize("argument", ["phases", "omegas"])
@pytest.mark.parametrize("dtype", ["U8", "timedelta64[ms]", "object_time"])
def test_fep_invalid_measurements_do_not_publish_assessment(
    argument: str, dtype: str
) -> None:
    """Failed ingress cannot mutate the FEP supervisor's last assessment."""
    phases = np.array([0.0, 1.0])
    omegas = np.ones(2)
    raw = phases if argument == "phases" else omegas
    invalid = raw.astype(object if dtype == "object_time" else dtype)
    if dtype == "object_time":
        invalid.flat[0] = np.timedelta64(1, "ms")
    supervisor = FEPPredictiveSupervisor(2, 0.01)
    assert supervisor.last_assessment is None
    with pytest.raises(ValueError, match=argument):
        supervisor.assess(
            invalid if argument == "phases" else phases,
            invalid if argument == "omegas" else omegas,
        )
    assert supervisor.last_assessment is None
    assert np.isfinite(supervisor.assess(phases, omegas).free_energy)
