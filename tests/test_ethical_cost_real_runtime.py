# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public ethical-cost mathematical regressions

"""Exercise actual default ethical-cost owners with independent graph identities."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf.ethical import FloatArray, compute_ethical_cost

INSTALLED_PROBE = r"""
import cProfile, hashlib, importlib, importlib.util, inspect, json, sys
from pathlib import Path
import numpy as np
from scpn_phase_orchestrator.ssgf import ethical
from scpn_phase_orchestrator.coupling import spectral
data = json.loads(sys.argv[1])
owner = data['owner']
assert owner in {'python', 'rust'}
assert ethical._HAS_RUST == (owner == 'rust')
if owner == 'python':
    assert importlib.util.find_spec('spo_kernel') is None
    assert spectral.ACTIVE_BACKEND == 'python'
else:
    import spo_kernel
    assert inspect.isbuiltin(spo_kernel.compute_ethical_cost_rust)
p = np.asarray(data['phases'], dtype=np.float64)
k = np.asarray(data['knm'], dtype=np.float64).reshape(len(p), len(p))
profile = cProfile.Profile()
profile.enable()
error = None
try:
    cost = ethical.compute_ethical_cost(p, k, **data['parameters'])
except ValueError as exc:
    error = str(exc)
    cost = None
finally:
    profile.disable()
source_hash = hashlib.sha256(Path(ethical.__file__).read_bytes()).hexdigest()
native_calls = sum(e.callcount for e in profile.getstats()
                   if isinstance(e.code, str)
                   and 'compute_ethical_cost_rust' in e.code)
values = ([cost.J_sec, cost.phi_ethics, cost.c15_sec,
           cost.constraints_violated] if cost is not None else None)
record = dict(owner=owner, module=ethical.__file__, prefix=sys.prefix,
              source_sha256=source_hash, cost=values, error=error,
              native_calls=native_calls)
if owner == 'rust':
    function = spo_kernel.compute_ethical_cost_rust
    binary = Path(importlib.import_module(function.__module__).__file__)
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    record.update(binary=str(binary), binary_sha256=binary_hash)
print(json.dumps(record))
"""


def installed_cost(
    owner: str, phases: FloatArray, knm: FloatArray, **parameters: float
) -> tuple[float, float, float, int]:
    """Call the original public function in a qualified isolated installation.

    Parameters
    ----------
    owner : str
        ``python`` requires an actually kernel-absent installation; ``rust``
        requires its original compiled builtin. Neither modifies resolution.
    phases : FloatArray
        Finite phase vector serialized unchanged across the process boundary.
    knm : FloatArray
        Matching square coupling matrix for the real installed consumer.
    **parameters : float
        Original public cost weights and thresholds.

    Returns
    -------
    tuple[float, float, float, int]
        Actual installed score, weighted penalty, total and violation count.

    Raises
    ------
    ValueError
        If the original installed public function refuses the numerical inputs.
    """
    assert owner in {"python", "rust"}
    raw = os.environ.get("SPO_ETHICAL_" + owner.upper() + "_PROFILE")
    assert raw is not None, "Qualified installed ethical-cost profile is required"
    executable = Path(raw)
    assert executable.is_file()
    payload = json.dumps(
        {
            "owner": owner,
            "phases": phases.tolist(),
            "knm": knm.tolist(),
            "parameters": parameters,
        }
    )
    result = subprocess.run(
        [str(executable), "-I", "-B", "-c", INSTALLED_PROBE, payload],
        cwd=executable.parent.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = cast("dict[str, object]", json.loads(result.stdout))
    assert Path(str(record["module"])).is_relative_to(Path(str(record["prefix"])))
    assert (
        record["source_sha256"]
        == hashlib.sha256(
            Path(compute_ethical_cost.__code__.co_filename).read_bytes()
        ).hexdigest()
    )
    if record["error"] is not None:
        raise ValueError(str(record["error"]))
    assert record["native_calls"] == (1 if owner == "rust" and phases.size else 0)
    values = cast("list[float]", record["cost"])
    return values[0], values[1], values[2], int(values[3])


@pytest.mark.parametrize("weight", [np.nextafter(0.0, 1.0), 1e-300, 1e-16, 1.0])
def test_density_counts_all_exactly_nonzero_signed_entries(weight: float) -> None:
    """Diagonal and negative tiny entries count without a numerical edge cutoff."""
    phases = np.zeros(2)
    knm = np.full((2, 2), -weight)
    before = knm.copy()
    cost = compute_ethical_cost(
        phases,
        knm,
        alpha_R=0.0,
        beta_K=0.0,
        gamma_Q=1.0,
        nu_S=0.0,
        kappa=0.0,
        R_min=0.0,
        connectivity_min=0.0,
    )
    assert cost.J_sec == 2.0
    assert cost.phi_ethics == 0.0
    assert cost.c15_sec == -1.0
    assert cost.constraints_violated == 0
    np.testing.assert_array_equal(knm, before)
    np.testing.assert_array_equal(phases, [0.0, 0.0])


@pytest.mark.parametrize("weight", [1e-300, 1e-16, 1.0, 1e100])
def test_two_node_connectivity_retains_scale(weight: float) -> None:
    """A two-node graph has lambda2=2w and normalized connectivity w."""
    cost = compute_ethical_cost(
        np.zeros(2),
        np.array([[0.0, -weight], [-weight, 0.0]]),
        alpha_R=0.0,
        beta_K=1.0,
        gamma_Q=0.0,
        nu_S=0.0,
        kappa=0.0,
        R_min=0.0,
        connectivity_min=0.0,
    )
    assert cost.J_sec / weight == pytest.approx(1.0, rel=1e-13, abs=0.0)
    assert cost.phi_ethics == 0.0
    assert cost.c15_sec == 1.0 - cost.J_sec
    assert cost.constraints_violated == 0


def test_population_dispersion_uses_raw_phase_not_wrapped_phase() -> None:
    """Phases 0 and 2pi have population standard deviation pi."""
    cost = compute_ethical_cost(
        np.array([0.0, 2.0 * np.pi]),
        np.zeros((2, 2)),
        alpha_R=0.0,
        beta_K=0.0,
        gamma_Q=0.0,
        nu_S=1.0,
        kappa=0.0,
        R_min=0.0,
        connectivity_min=0.0,
    )
    assert cost.J_sec == pytest.approx(-1.0, abs=1e-15)
    assert cost.c15_sec == pytest.approx(2.0, abs=1e-15)
    assert cost.phi_ethics == 0.0 and cost.constraints_violated == 0


def test_finite_signed_parameters_weight_penalty_once() -> None:
    """Negative kappa preserves finite signed diagnostics and violation counts."""
    cost = compute_ethical_cost(
        np.zeros(1),
        np.zeros((1, 1)),
        alpha_R=-0.4,
        beta_K=0.0,
        gamma_Q=0.0,
        nu_S=0.0,
        kappa=-2.0,
        R_min=1.5,
        connectivity_min=0.25,
        max_coupling=-10.0,
    )
    assert cost.J_sec == -0.4
    assert cost.phi_ethics == -2.0 * (0.5**2 + 0.25**2)
    assert cost.c15_sec == 1.0 - cost.J_sec + cost.phi_ethics
    assert cost.constraints_violated == 2


@pytest.mark.parametrize(
    "name",
    [
        "alpha_R",
        "beta_K",
        "gamma_Q",
        "nu_S",
        "kappa",
        "R_min",
        "connectivity_min",
        "max_coupling",
    ],
)
@pytest.mark.parametrize(
    "alias",
    [
        True,
        np.bool_(False),
        "0.5",
        1j,
        np.timedelta64(1, "s"),
        np.array([0.5]),
        10**400,
    ],
)
def test_scalar_source_aliases_are_refused_before_empty_return(
    name: str, alias: object
) -> None:
    """Empty identity still enforces original scalar types before owner selection."""
    with pytest.raises(ValueError, match="finite real number"):
        compute_ethical_cost(
            np.zeros(0), np.zeros((0, 0)), **{name: cast("float", alias)}
        )


@pytest.mark.parametrize(
    "alias",
    [
        np.array([True, False]),
        np.array(["0", "1"]),
        np.array([0j, 1j]),
        np.array([0, 1], dtype="timedelta64[s]"),
        np.array([0.0, True], dtype=object),
    ],
)
def test_phase_measurement_aliases_are_refused(alias: object) -> None:
    """Source types cannot be laundered into finite float64 phase measurements."""
    with pytest.raises(ValueError, match="plain real numbers"):
        compute_ethical_cost(cast("FloatArray", alias), np.zeros((2, 2)))


@pytest.mark.parametrize(
    "alias",
    [
        np.ones((2, 2), dtype=bool),
        np.full((2, 2), "1"),
        np.ones((2, 2), dtype=complex),
        np.ones((2, 2), dtype="timedelta64[s]"),
        np.array([[0.0, True], [1.0, 0.0]], dtype=object),
    ],
)
def test_coupling_measurement_aliases_are_refused(alias: object) -> None:
    """Original matrix source types are rejected before any owner gets a float copy."""
    with pytest.raises(ValueError, match="plain real numbers"):
        compute_ethical_cost(np.zeros(2), cast("FloatArray", alias))


def test_plain_real_object_measurements_keep_signed_graph_semantics() -> None:
    """Supported real objects retain exact density, symmetry and input storage."""
    phases = np.array([0, np.pi], dtype=object)
    matrix = np.array([[1, -0.5], [np.int64(1), 0.0]], dtype=object)
    phase_before, matrix_before = phases.copy(), matrix.copy()
    cost = compute_ethical_cost(cast("FloatArray", phases), cast("FloatArray", matrix))
    assert cost.J_sec == pytest.approx(0.475, abs=1e-14)
    assert cost.phi_ethics == pytest.approx(0.04, abs=1e-14)
    assert cost.c15_sec == pytest.approx(0.565, abs=1e-14)
    assert cost.constraints_violated == 1
    np.testing.assert_array_equal(phases, phase_before)
    np.testing.assert_array_equal(matrix, matrix_before)


def test_noncontiguous_readonly_measurements_preserve_field_order_and_storage() -> None:
    """Typed read-only views are copied for FFI without changing their producers."""
    phase_store = np.array([0.0, 99.0, np.pi, 98.0])
    matrix_store = np.array([[1.0, 99.0, -0.5, 98.0], [1.0, 97.0, 0.0, 96.0]])
    phases, matrix = phase_store[::2], matrix_store[:, ::2]
    phases.flags.writeable = matrix.flags.writeable = False
    phase_before, matrix_before = phase_store.copy(), matrix_store.copy()
    cost = compute_ethical_cost(phases, matrix)
    assert cost.J_sec == pytest.approx(0.475, abs=1e-14)
    assert cost.phi_ethics == pytest.approx(0.04, abs=1e-14)
    assert cost.c15_sec == pytest.approx(0.565, abs=1e-14)
    assert cost.constraints_violated == 1
    np.testing.assert_array_equal(phase_store, phase_before)
    np.testing.assert_array_equal(matrix_store, matrix_before)


@pytest.mark.parametrize("parameter", ["beta_K", "kappa"])
def test_finite_parameter_can_still_produce_unrepresentable_cost(
    parameter: str,
) -> None:
    """Finite arithmetic inputs never justify returning an infinite score or penalty."""
    matrix = np.array([[0.0, 1e100], [1e100, 0.0]])
    params = {parameter: 1e308}
    with pytest.raises(ValueError, match="arithmetic must remain finite"):
        compute_ethical_cost(np.zeros(2), matrix, **params)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_negative_finite_coupling_refuses_laplacian_overflow(n: int) -> None:
    """Both owners refuse overflowing reciprocal magnitudes without NaN or warnings."""
    matrix = np.full((n, n), -1e308)
    np.fill_diagonal(matrix, 0.0)
    with pytest.raises(ValueError, match="arithmetic must remain finite"):
        compute_ethical_cost(np.zeros(n), matrix)


def test_finite_raw_phases_refuse_unrepresentable_dispersion() -> None:
    """A phase-dispersion overflow remains a public ValueError on either owner."""
    with pytest.raises(ValueError, match="arithmetic must remain finite"):
        compute_ethical_cost(np.array([1e200, -1e200]), np.zeros((2, 2)))


def test_finite_positive_coupling_refuses_unrepresentable_residual_square() -> None:
    """The public boundary rejects an overflowing penalty even with zero multiplier."""
    phases = np.zeros(2)
    matrix = np.array([[0.0, 1e200], [1e200, 0.0]])
    original = matrix.copy()
    with pytest.raises(ValueError, match="arithmetic must remain finite"):
        compute_ethical_cost(phases, matrix, kappa=0.0)
    np.testing.assert_array_equal(matrix, original)
    healthy = compute_ethical_cost(phases, np.zeros((2, 2)))
    assert healthy.J_sec == pytest.approx(0.4, abs=1e-14)
    assert healthy.c15_sec == pytest.approx(0.61, abs=1e-14)


@pytest.mark.parametrize("name", ["R_min", "connectivity_min"])
def test_finite_target_refuses_unrepresentable_residual_square(name: str) -> None:
    """A finite target can overflow its squared residual without corrupting inputs."""
    phases, matrix = np.zeros(1), np.zeros((1, 1))
    with pytest.raises(ValueError, match="arithmetic must remain finite"):
        compute_ethical_cost(phases, matrix, **{name: 1e200})
    np.testing.assert_array_equal(phases, [0.0])
    np.testing.assert_array_equal(matrix, [[0.0]])
    recovered = compute_ethical_cost(phases, matrix, R_min=0.0, connectivity_min=0.0)
    assert recovered.J_sec == pytest.approx(0.4, abs=1e-14)
    assert recovered.phi_ethics == 0.0 and recovered.constraints_violated == 0
