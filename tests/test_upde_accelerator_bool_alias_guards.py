# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE accelerator boolean-alias guard tests
"""Boolean-alias guard tests for UPDE accelerator validation boundaries."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _basin_stability_validation,
    _engine_validation,
    _envelope_validation,
    _geometric_validation,
    _hypergraph_validation,
    _inertial_validation,
    _market_validation,
    _pac_validation,
    _simplicial_validation,
    _splitting_validation,
    _swarmalator_validation,
    _validation_common,
)

FloatArray = NDArray[np.float64]
ValidationCall = Callable[[], object]


def test__validation_common_helper_is_directly_linked_to_bool_alias_tests() -> None:
    """Keep the shared UPDE validation helper visible to linkage guards."""
    assert _validation_common.contains_boolean_alias([np.bool_(True)])


def _vector(values: list[float]) -> FloatArray:
    """Return a finite float64 vector for valid comparison inputs."""
    return np.asarray(values, dtype=np.float64)


def _flat_zero_matrix(size: int) -> FloatArray:
    """Return a flattened zero matrix with ``size`` rows and columns."""
    return np.zeros(size * size, dtype=np.float64)


def _bool_alias_vector() -> list[float | np.bool_]:
    """Return a raw Python vector containing a NumPy boolean alias."""
    return [0.25, np.bool_(True)]


def _engine_case() -> object:
    """Validate UPDE engine inputs with a boolean alias in ``phases``."""
    return _engine_validation.validate_upde_backend_inputs(
        phases=_bool_alias_vector(),
        omegas=_vector([0.0, 0.0]),
        knm=_flat_zero_matrix(2),
        alpha=_flat_zero_matrix(2),
        zeta=0.0,
        psi=0.0,
        dt=0.01,
        n_steps=1,
        method="euler",
        n_substeps=1,
        atol=1e-9,
        rtol=1e-9,
    )


def _engine_scalar_case() -> object:
    """Validate UPDE engine controls with a NumPy boolean scalar alias."""
    return _engine_validation.validate_upde_backend_inputs(
        phases=_vector([0.25, 0.5]),
        omegas=_vector([0.0, 0.0]),
        knm=_flat_zero_matrix(2),
        alpha=_flat_zero_matrix(2),
        zeta=0.0,
        psi=0.0,
        dt=np.bool_(True),
        n_steps=1,
        method="euler",
        n_substeps=1,
        atol=1e-9,
        rtol=1e-9,
    )


def _basin_case() -> object:
    """Validate basin-stability inputs with a boolean alias in initial phases."""
    return _basin_stability_validation.validate_basin_stability_inputs(
        phases_init=_bool_alias_vector(),
        omegas=_vector([0.0, 0.0]),
        knm_flat=_flat_zero_matrix(2),
        alpha_flat=_flat_zero_matrix(2),
        n=2,
        k_scale=1.0,
        dt=0.01,
        n_transient=0,
        n_measure=1,
    )


def _envelope_input_case() -> object:
    """Validate envelope extraction inputs with a boolean alias in amplitudes."""
    return _envelope_validation.validate_extract_envelope_input(
        amps=_bool_alias_vector(),
        window=1,
    )


def _envelope_output_case() -> object:
    """Validate envelope output with a boolean alias in the returned envelope."""
    return _envelope_validation.validate_extract_envelope_output(
        value=_bool_alias_vector(),
        n=2,
    )


def _pac_input_case() -> object:
    """Validate phase-amplitude coupling inputs with a boolean alias in phases."""
    return _pac_validation.validate_modulation_index_inputs(
        theta_low=_bool_alias_vector(),
        amp_high=_vector([1.0, 1.0]),
        n_bins=2,
    )


def _pac_output_case() -> object:
    """Validate modulation-index output with a boolean alias in the bins."""
    return _pac_validation.validate_pac_matrix_output(
        value=[0.0, np.bool_(True), 0.0, 1.0],
        n=2,
    )


def _geometric_case() -> object:
    """Validate torus-control inputs with a boolean alias in phases."""
    return _geometric_validation.validate_torus_inputs(
        phases=_bool_alias_vector(),
        omegas=_vector([0.0, 0.0]),
        knm_flat=_flat_zero_matrix(2),
        alpha_flat=_flat_zero_matrix(2),
        n=2,
        zeta=0.0,
        psi=0.0,
        dt=0.01,
        n_steps=1,
    )


def _market_case() -> object:
    """Validate market-order inputs with a boolean alias in phase history."""
    return _market_validation.validate_market_order_inputs(
        phases_flat=_bool_alias_vector(),
        t=1,
        n=2,
    )


def _inertial_case() -> object:
    """Validate inertial inputs with a boolean alias in positions."""
    return _inertial_validation.validate_inertial_inputs(
        theta=_bool_alias_vector(),
        omega_dot=_vector([0.0, 0.0]),
        power=_vector([0.0, 0.0]),
        knm_flat=_flat_zero_matrix(2),
        inertia=_vector([1.0, 1.0]),
        damping=_vector([0.1, 0.1]),
        n=2,
        dt=0.01,
    )


def _hypergraph_phase_case() -> object:
    """Validate hypergraph inputs with a boolean alias in phases."""
    return _hypergraph_validation.validate_hypergraph_inputs(
        phases=_bool_alias_vector(),
        omegas=_vector([0.0, 0.0]),
        n=2,
        edge_nodes=np.asarray([0, 1], dtype=np.int64),
        edge_offsets=np.asarray([0], dtype=np.int64),
        edge_strengths=_vector([0.5]),
        knm_flat=_flat_zero_matrix(2),
        alpha_flat=_flat_zero_matrix(2),
        zeta=0.0,
        psi=0.0,
        dt=0.01,
        n_steps=1,
    )


def _hypergraph_edge_case() -> object:
    """Validate hypergraph inputs with a boolean alias in edge nodes."""
    return _hypergraph_validation.validate_hypergraph_inputs(
        phases=_vector([0.0, 0.1]),
        omegas=_vector([0.0, 0.0]),
        n=2,
        edge_nodes=[0, np.bool_(True)],
        edge_offsets=np.asarray([0], dtype=np.int64),
        edge_strengths=_vector([0.5]),
        knm_flat=_flat_zero_matrix(2),
        alpha_flat=_flat_zero_matrix(2),
        zeta=0.0,
        psi=0.0,
        dt=0.01,
        n_steps=1,
    )


def _simplicial_case() -> object:
    """Validate simplicial inputs with a boolean alias in phases."""
    return _simplicial_validation.validate_simplicial_inputs(
        phases=_bool_alias_vector(),
        omegas=_vector([0.0, 0.0]),
        knm_flat=_flat_zero_matrix(2),
        alpha_flat=_flat_zero_matrix(2),
        n=2,
        zeta=0.0,
        psi=0.0,
        sigma2=0.0,
        dt=0.01,
        n_steps=1,
    )


def _splitting_case() -> object:
    """Validate Strang-splitting inputs with a boolean alias in phases."""
    return _splitting_validation.validate_splitting_inputs(
        phases=_bool_alias_vector(),
        omegas=_vector([0.0, 0.0]),
        knm_flat=_flat_zero_matrix(2),
        alpha_flat=_flat_zero_matrix(2),
        n=2,
        zeta=0.0,
        psi=0.0,
        dt=0.01,
        n_steps=1,
    )


def _swarmalator_case() -> object:
    """Validate swarmalator inputs with a boolean alias in positions."""
    return _swarmalator_validation.validate_swarmalator_inputs(
        pos=[0.0, np.bool_(True), 0.2, 0.3],
        phases=_vector([0.0, 0.1]),
        omegas=_vector([0.0, 0.0]),
        n=2,
        dim=2,
        a=0.1,
        b=0.2,
        j=0.3,
        k=0.4,
        dt=0.01,
    )


BOOL_ALIAS_CASES: tuple[tuple[str, ValidationCall], ...] = (
    ("engine phases", _engine_case),
    ("engine scalar", _engine_scalar_case),
    ("basin phases", _basin_case),
    ("envelope input", _envelope_input_case),
    ("envelope output", _envelope_output_case),
    ("pac input", _pac_input_case),
    ("pac output", _pac_output_case),
    ("geometric phases", _geometric_case),
    ("market phases", _market_case),
    ("inertial theta", _inertial_case),
    ("hypergraph phases", _hypergraph_phase_case),
    ("hypergraph edge nodes", _hypergraph_edge_case),
    ("simplicial phases", _simplicial_case),
    ("splitting phases", _splitting_case),
    ("swarmalator positions", _swarmalator_case),
)


@pytest.mark.parametrize(("case_name", "call"), BOOL_ALIAS_CASES)
def test_upde_accelerator_validators_reject_raw_numpy_bool_aliases(
    case_name: str,
    call: ValidationCall,
) -> None:
    """Require raw-list NumPy boolean aliases to fail before float coercion."""
    with pytest.raises((TypeError, ValueError), match="boolean"):
        call()
