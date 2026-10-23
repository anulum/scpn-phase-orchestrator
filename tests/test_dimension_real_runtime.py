# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual dimension runtime contracts

"""Dimension regressions through required public runtimes without substitution."""

from __future__ import annotations

import math
import subprocess
from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.monitor._dimension_mojo import (
    _EXE_PATH,
)
from scpn_phase_orchestrator.monitor.dimension import (
    correlation_dimension,
    correlation_integral,
    kaplan_yorke_dimension,
)

pytestmark = pytest.mark.native_runtime

OWNERS = ("python", "rust", "go", "julia", "mojo")


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
@pytest.mark.parametrize("spectrum", [(), (0.0,), (-0.0, 0.0), (0.0,) * 7])
def test_zero_spectrum_returns_formula_dimension(
    backend: str, spectrum: tuple[float, ...]
) -> None:
    """The zero-spectrum formula returns its length without changing signed zeros."""
    values = np.array(spectrum, dtype=np.float64)
    before = values.copy()
    assert kaplan_yorke_dimension(values, backend=backend) == float(len(spectrum))
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(np.signbit(values), np.signbit(before))


@pytest.mark.parametrize("backend", OWNERS)
@pytest.mark.parametrize(
    ("point", "radius"),
    [
        (np.finfo(np.float64).max / 2, np.finfo(np.float64).max),
        (1e-200, 5e-201),
        (1.0, 1.0),
        (1.0, math.nextafter(1.0, math.inf)),
        (0.0, 0.0),
        (0.0, math.ulp(0.0)),
    ],
)
def test_integral_extreme_distance_and_strict_threshold(
    backend: str, point: float, radius: float
) -> None:
    """Require each actual owner to count a scalar distance with strict radius."""
    trajectory = np.array([[0.0], [point]], dtype=np.float64)
    before = trajectory.copy()
    expected = float(math.dist([0.0], [point]) < radius)
    actual = correlation_integral(trajectory, [radius], backend=backend)
    np.testing.assert_array_equal(actual, [expected])
    np.testing.assert_array_equal(trajectory, before)


@pytest.mark.parametrize("backend", OWNERS)
def test_integral_unrepresentable_distance_is_outside_finite_radius(
    backend: str,
) -> None:
    """Treat an overflowing true distance as outside every finite radius."""
    largest = np.finfo(np.float64).max
    result = correlation_integral([[-largest], [largest]], [largest], backend=backend)
    np.testing.assert_array_equal(result, [0.0])


@pytest.mark.parametrize("backend", OWNERS)
def test_kaplan_yorke_finite_prefix_cancellation(backend: str) -> None:
    """Compare overflowing binary64 prefixes against an exact decimal sum."""
    largest = np.finfo(np.float64).max
    spectrum = np.array([largest, largest, -largest, -largest, -largest])
    before = spectrum.copy()
    with localcontext() as context:
        context.prec = 700
        exact = [Decimal.from_float(float(value)) for value in spectrum]
        assert sum(exact[:4]) == 0
        assert sum(exact) < 0
    assert kaplan_yorke_dimension(spectrum, backend=backend) == 4.0
    np.testing.assert_array_equal(spectrum, before)


@pytest.mark.parametrize("backend", OWNERS)
@pytest.mark.parametrize("scale", [1.0, 1e-305, 1e305])
def test_kaplan_yorke_positive_scale_invariance(backend: str, scale: float) -> None:
    """Retain a fractional dimension even when its denominator is tiny."""
    spectrum = np.array([-2.0, 0.0, 1.0]) * scale
    assert kaplan_yorke_dimension(spectrum, backend=backend) == pytest.approx(2.5)


@pytest.mark.parametrize("backend", OWNERS)
def test_integral_sorted_radii_and_fresh_result(backend: str) -> None:
    """Verify sorting, pair normalization and output ownership on an exact line."""
    radii = np.array([4.0, 0.0, 2.0, 1.0])
    result = correlation_integral([0.0, 1.0, 3.0], radii, backend=backend)
    np.testing.assert_allclose(result, [0.0, 0.0, 1.0 / 3.0, 1.0], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(radii, [4.0, 0.0, 2.0, 1.0])
    result[:] = -1
    repeated = correlation_integral([0.0, 1.0, 3.0], radii, backend=backend)
    assert np.all(repeated >= 0)


@pytest.mark.parametrize("backend", OWNERS)
def test_dimension_refuses_unrepresentable_radius_grid(backend: str) -> None:
    """Refuse a log grid outside finite positive float64 radii explicitly."""
    largest = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="finite positive radii"):
        correlation_dimension([[-largest], [largest]], backend=backend)


@pytest.mark.parametrize("operation", ["integral", "dimension", "kaplan_yorke"])
def test_unknown_backend_refused_even_for_degenerate_input(operation: str) -> None:
    """Require explicit owner validation before any degenerate early return."""
    with pytest.raises(ValueError, match="unknown dimension backend"):
        if operation == "integral":
            correlation_integral([], [], backend="unknown")
        elif operation == "dimension":
            correlation_dimension([], backend="unknown")
        else:
            kaplan_yorke_dimension([], backend="unknown")


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
@pytest.mark.parametrize("samples", [0, 1, 3])
def test_degenerate_dimension_reports_actual_pair_fraction(
    backend: str, samples: int
) -> None:
    """A zero-diameter result does not invent pairs for empty/singleton data."""
    trajectory = np.zeros((samples, 2))
    result = correlation_dimension(trajectory, max_pairs=100, backend=backend)
    expected = 0.0 if samples < 2 else 1.0
    assert result.D2 == 0.0
    np.testing.assert_array_equal(result.C_eps, [expected])
    np.testing.assert_array_equal(
        result.C_eps,
        correlation_integral(
            trajectory, result.epsilons, max_pairs=100, backend=backend
        ),
    )


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", ["python", "go", "julia", "mojo"])
def test_constant_dimension_preserves_empty_seeded_sample(backend: str) -> None:
    """An actual discarded-only draw remains zero in the D2 result curve."""
    result = correlation_dimension(
        np.zeros((3, 2)), max_pairs=1, seed=1, backend=backend
    )
    assert result.D2 == 0.0
    np.testing.assert_array_equal(result.C_eps, [0.0])


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
@pytest.mark.parametrize("operation", ["integral", "dimension"])
def test_zero_coordinate_trajectory_is_refused(backend: str, operation: str) -> None:
    """The public geometry domain requires at least one phase-space coordinate."""
    with pytest.raises(ValueError, match="positive coordinate dimension"):
        if operation == "integral":
            correlation_integral(np.empty((2, 0)), [1.0], backend=backend)
        else:
            correlation_dimension(np.empty((2, 0)), backend=backend)


def _scalar_integral(
    trajectory: NDArray[np.float64],
    radii: NDArray[np.float64],
    pairs: list[tuple[int, int]],
) -> list[float]:
    """Count scalar Euclidean distances independently of every numerical owner."""
    distances = [math.dist(trajectory[i], trajectory[j]) for i, j in pairs]
    return [
        sum(distance < radius for distance in distances) / len(distances)
        if distances
        else 0.0
        for radius in sorted(radii)
    ]


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
@pytest.mark.parametrize("power", [-700, 0, 700])
@pytest.mark.parametrize("transform", [False, True])
def test_multidimensional_distance_affine_contract(
    backend: str, power: int, transform: bool
) -> None:
    """Scaled coordinate permutations/translations preserve full-pair geometry."""
    scale = math.ldexp(1.0, power)
    trajectory = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
                [6.0, 8.0, 0.0],
                [6.0, 0.0, 0.0],
                [0.0, 8.0, 0.0],
            ]
        )
        * scale
    )
    if transform:
        trajectory = trajectory[:, ::-1] + 16.0 * scale
    before = trajectory.copy()
    radii = np.array([11.0, 0.0, 1.0, 5.5, 9.5]) * scale
    pairs = [
        (i, j) for i in range(len(trajectory)) for j in range(i + 1, len(trajectory))
    ]
    expected = _scalar_integral(trajectory, radii, pairs)
    actual = correlation_integral(trajectory, radii, max_pairs=100, backend=backend)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(trajectory, before)


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", ["python", "go", "julia", "mojo"])
@pytest.mark.parametrize("seed", [0, 1, 42, 2**64 - 1])
@pytest.mark.parametrize("quota", [1, 7, 17])
def test_seeded_prepared_pairs_match_scalar_reference(
    backend: str, seed: int, quota: int
) -> None:
    """Real non-Rust owners retain the specified draw/filter/normalization law."""
    trajectory = np.array([[float(i), float(3 * (i % 3))] for i in range(12)])
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(trajectory), quota)
    second = rng.integers(0, len(trajectory), quota)
    pairs = [(int(i), int(j)) for i, j in zip(first, second, strict=True) if i != j]
    radii = np.array([12.5, 0.0, 2.5, 5.5])
    actual = correlation_integral(
        trajectory, radii, max_pairs=quota, seed=seed, backend=backend
    )
    np.testing.assert_allclose(
        actual, _scalar_integral(trajectory, radii, pairs), rtol=0.0, atol=1e-15
    )


@pytest.mark.native_runtime
@pytest.mark.parametrize(
    "payload",
    [
        "",
        "OTHER 1",
        "KY",
        "KY -1",
        "KY 1.0 1",
        "KY 2 1",
        "KY 1 1 2",
        "KY 1 NaN",
        "KY 1 inf",
        "KY 1 1e+",
        "KY 1 1e9999",
        "KY 9223372036854775808",
        "CI",
        "CI 2 0 0 0",
        "CI -1 1 0 0",
        "CI 2x 1 0 0",
        "CI 2.0 1 0 0",
        "CI 2 1 2 1 0 1 1 1 1 0 1",
        "CI 2 1 1 1 -1 1 2 0 1",
        "CI 2 1 1 1 0 2 2 0 1",
        "CI 2 1 1 1 0 0 2 0 1",
        "CI 2 1 1 1 0.0 1 2 0 1",
        "CI 2 1 1 1 0 1 -1 0 1",
        "CI 2 1 1 1 0 1 NaN 0 1",
        "CI 2 1 1 2 0 1 2 1 0 1",
        "CI 2 1 1 1 0 1 2 0 inf",
        "CI 2 1 1 1 0 1 2 0 1e+-1",
        "CI 2 1 1 1 0 1 2 0",
        "CI 2 1 1 1 0 1 2 0 1 9",
    ],
)
def test_actual_mojo_cli_refuses_malformed_requests(payload: str) -> None:
    """The real executable refuses malformed metadata/numbers without partial stdout."""
    completed = subprocess.run(
        [str(_EXE_PATH)],
        input=payload + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert not completed.stdout.strip()
    assert completed.stderr.strip()


@pytest.mark.native_runtime
@pytest.mark.parametrize(
    "payload,expected",
    [
        ("KY +3 +1e0 .0 -2.", [2.5]),
        ("CI +2 +1 +1 +3 +0 +1 +.0 1.e0 2E0 -0.0 +1e0", [0.0, 0.0, 1.0]),
    ],
)
def test_actual_mojo_cli_accepts_complete_decimal_tokens(
    payload: str, expected: list[float]
) -> None:
    """The actual parser accepts signed counts and complete decimal exponents."""
    completed = subprocess.run(
        [str(_EXE_PATH)],
        input=payload + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    actual = [float(line) for line in completed.stdout.splitlines()]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
def test_dimension_curve_plateau_and_result_ownership(backend: str) -> None:
    """Scalar pair counts/log slopes qualify the selected plateau and fresh arrays."""
    trajectory = np.array([[float(i), float((i % 3) / 4)] for i in range(24)])
    before = trajectory.copy()
    result = correlation_dimension(
        trajectory, n_epsilons=23, max_pairs=1000, backend=backend
    )
    pairs = [
        (i, j) for i in range(len(trajectory)) for j in range(i + 1, len(trajectory))
    ]
    expected = _scalar_integral(trajectory, result.epsilons, pairs)
    np.testing.assert_allclose(result.C_eps, expected, rtol=0.0, atol=1e-15)
    valid = [i for i, count in enumerate(expected) if count > 0.0]
    assert len(valid) >= 3
    local = [
        (math.log(expected[j]) - math.log(expected[i]))
        / (math.log(float(result.epsilons[j])) - math.log(float(result.epsilons[i])))
        for i, j in zip(valid[:-1], valid[1:], strict=True)
    ]
    aligned = [0.0] * (len(result.epsilons) - 1)
    aligned[valid[0] : valid[0] + len(local)] = local
    np.testing.assert_allclose(result.slope, aligned, rtol=0.0, atol=1e-12)
    lower, upper = result.scaling_range
    begin = next(
        i for i, index in enumerate(valid) if float(result.epsilons[index]) == lower
    )
    end = next(
        i for i, index in enumerate(valid) if float(result.epsilons[index]) == upper
    )
    window = min(5, len(local))
    assert end - begin == window

    def variance(values: list[float]) -> float:
        """Use scalar compensated sums for the finite local-slope population."""
        mean = math.fsum(values) / len(values)
        return math.fsum((value - mean) ** 2 for value in values) / len(values)

    selected = local[begin:end]
    assert (
        variance(selected)
        <= min(variance(local[i : i + window]) for i in range(len(local) - window + 1))
        + 1e-12
    )
    assert (
        pytest.approx(max(0.0, math.fsum(selected) / len(selected)), abs=1e-12)
        == result.D2
    )
    np.testing.assert_array_equal(trajectory, before)
    original_radii = result.epsilons.copy()
    for array in (result.epsilons, result.C_eps, result.slope):
        assert not np.shares_memory(array, trajectory)
        array[:] = -99.0
    repeated = correlation_dimension(
        trajectory, n_epsilons=23, max_pairs=1000, backend=backend
    )
    np.testing.assert_array_equal(repeated.epsilons, original_radii)
    np.testing.assert_allclose(repeated.C_eps, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(repeated.slope, aligned, rtol=0.0, atol=1e-12)


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
@pytest.mark.parametrize("operation", ["integral", "dimension"])
@pytest.mark.parametrize(
    "control,value",
    [
        ("seed", 2**64),
        ("max_pairs", int(np.iinfo(np.uintp).max) + 1),
    ],
)
def test_public_sampling_controls_refuse_native_range_overflow(
    backend: str, operation: str, control: str, value: int
) -> None:
    """Public sampling controls share finite native ranges across every owner."""
    controls = {"max_pairs": 100, "seed": 42}
    controls[control] = value
    with pytest.raises(ValueError, match=control):
        if operation == "integral":
            correlation_integral([0.0, 1.0, 3.0], [2.0], backend=backend, **controls)
        else:
            correlation_dimension([0.0, 1.0, 3.0], backend=backend, **controls)


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
def test_public_maximal_valid_sampling_controls_need_no_quota_allocation(
    backend: str,
) -> None:
    """Native-width maximum controls still use only three existing full pairs."""
    result = correlation_integral(
        [0.0, 1.0, 3.0],
        [2.0],
        max_pairs=int(np.iinfo(np.uintp).max),
        seed=2**64 - 1,
        backend=backend,
    )
    np.testing.assert_allclose(result, [1.0 / 3.0], rtol=0.0, atol=1e-15)


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
@pytest.mark.parametrize(
    "count",
    [
        int(np.iinfo(np.intp).max) + 1,
        int(np.iinfo(np.intp).max) // np.dtype(np.float64).itemsize + 1,
    ],
)
def test_radius_count_refused_before_degenerate_shortcut(
    backend: str, count: int
) -> None:
    """Even zero-diameter data cannot conceal an unrepresentable array count."""
    with pytest.raises(ValueError, match="n_epsilons"):
        correlation_dimension([0.0], n_epsilons=count, backend=backend)
