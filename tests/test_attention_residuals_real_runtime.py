# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual phase attention owners and UPDE consumers

"""Qualify real optional owners without replacing loaders, flags or results."""

from __future__ import annotations

import cProfile
import os
from typing import cast

import numpy as np
import pytest

from benchmarks.attnres_reference import phase_attention_oracle
from scpn_phase_orchestrator.coupling.attention_residuals import (
    AVAILABLE_BACKENDS,
    attnres_modulate,
    default_projections,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine

BACKENDS = tuple(AVAILABLE_BACKENDS)


def test_required_runtime_profile_is_present() -> None:
    """A requested qualification profile cannot pass with a missing owner."""
    requested = os.environ.get("SPO_ATTNRES_REQUIRED_BACKENDS", "python").split(",")
    assert set(requested).issubset(BACKENDS), (requested, BACKENDS)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "width,heads", [(2, 1), (4, 2), (6, 3), (8, 4), (12, 4), (16, 8)]
)
@pytest.mark.parametrize("radius", [None, 1, 7])
def test_projection_dimensions_and_masks_match_independent_oracle(
    backend: str, width: int, heads: int, radius: int | None
) -> None:
    """Every admitted owner preserves signed, absent, isolated and banded edges."""
    coupling = np.array(
        [
            [0.0, -0.3, 0.0, 0.2, 0.0],
            [-0.3, 0.0, 0.7, 0.0, 0.0],
            [0.0, 0.7, 0.0, -0.2, 0.0],
            [0.2, 0.0, -0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    phases = np.array([0.1, 0.7, 1.8, 2.4, 3.1])
    weights = default_projections(n_heads=heads, d_model=width, seed=6)
    expected = phase_attention_oracle(coupling, phases, weights, radius=radius)
    actual = attnres_modulate(
        coupling,
        phases,
        w_q=weights[0],
        w_k=weights[1],
        w_v=weights[2],
        w_o=weights[3],
        n_heads=heads,
        block_size=radius,
        backend=backend,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)
    np.testing.assert_array_equal(actual[coupling == 0.0], 0.0)
    assert np.max(np.abs(actual - coupling)) > 1e-4
    np.testing.assert_allclose(actual, actual.T, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_owner_and_analytic_two_node_readout(backend: str) -> None:
    """Original compiled or Python owners compute the analytic half-strength case."""
    projection = np.eye(2).reshape(1, 2, 2)
    coupling = np.array([[0.0, -0.3], [-0.3, 0.0]])
    phases = np.array([0.0, np.pi / 2.0])
    with cProfile.Profile() as profile:
        output = attnres_modulate(
            coupling,
            phases,
            w_q=projection,
            w_k=projection,
            w_v=projection,
            w_o=np.eye(2),
            n_heads=1,
            backend=backend,
        )
    profile.create_stats()
    calls = {entry[2] for entry in profile.stats}
    expected_owner = {
        "rust": "<built-in method spo_kernel.spo_kernel.attnres_modulate_rust>",
        "go": "attnres_modulate_go",
        "julia": "attnres_modulate_julia",
        "mojo": "attnres_modulate_mojo",
        "python": "_python_fallback",
    }[backend]
    assert expected_owner in calls, (backend, sorted(calls))
    np.testing.assert_allclose(
        output, [[0.0, -0.375], [-0.375, 0.0]], rtol=0.0, atol=1e-12
    )
    assert "_python_fallback" not in calls or backend == "python"


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_output_readout_is_half_strength_and_identity_is_a_copy(
    backend: str,
) -> None:
    """Zero projections still modulate existing edges; lambda zero alone is identity."""
    coupling = np.array([[0.0, 0.3], [0.3, 0.0]])
    phases = np.array([0.2, 1.0])
    projection = np.zeros((1, 4, 4))
    out = attnres_modulate(
        coupling,
        phases,
        w_q=projection,
        w_k=projection,
        w_v=projection,
        w_o=np.zeros((4, 4)),
        n_heads=1,
        backend=backend,
    )
    np.testing.assert_allclose(out, coupling * 1.25, rtol=0.0, atol=1e-15)
    identity = attnres_modulate(coupling, phases, lambda_=0.0, backend=backend)
    np.testing.assert_array_equal(identity, coupling)
    assert not np.shares_memory(identity, coupling)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("edge", [9e-13, -9e-13, 2e-12])
@pytest.mark.parametrize("gain", [0.5, 5.0])
def test_tiny_nonzero_edges_remain_edges_during_modulation(
    backend: str, edge: float, gain: float
) -> None:
    """The topology mask preserves exact zeros without erasing weak real edges."""
    coupling = np.array([[0.0, edge], [edge, 0.0]])
    zero = np.zeros((1, 2, 2))
    output = attnres_modulate(
        coupling,
        np.array([0.1, 0.7]),
        w_q=zero,
        w_k=zero,
        w_v=zero,
        w_o=np.zeros((2, 2)),
        n_heads=1,
        lambda_=gain,
        backend=backend,
    )
    np.testing.assert_allclose(
        output, coupling * (1.0 + gain / 2.0), rtol=2e-15, atol=0.0
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_minimum_finite_logit_is_an_allowed_attention_score(backend: str) -> None:
    """A valid extreme score cannot collide with a native masked-entry marker."""
    query = np.zeros((2, 2, 1))
    key = query.copy()
    value = query.copy()
    query[0, 0, 0] = np.finfo(np.float64).max
    key[0, 0, 0] = -1.0
    value[0, 0, 0] = value[1, 1, 0] = 1.0
    output = attnres_modulate(
        np.array([[0.0, 0.3], [0.3, 0.0]]),
        np.zeros(2),
        w_q=query,
        w_k=key,
        w_v=value,
        w_o=np.eye(2),
        n_heads=2,
        backend=backend,
    )
    np.testing.assert_allclose(output, [[0.0, 0.45], [0.45, 0.0]], rtol=0.0, atol=2e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fault", ["scale", "projection", "logits", "norm", "coupling"])
def test_unrepresentable_intermediates_are_refused(backend: str, fault: str) -> None:
    """Finite measurements cannot publish an overflowed or fabricated iterate."""
    coupling = np.array([[0.0, 0.3], [0.3, 0.0]])
    phases = np.array([0.4, 0.5])
    weights = [a.copy() for a in default_projections(n_heads=1, d_model=2)]
    temperature = np.nextafter(0.0, 1.0) if fault == "scale" else 1.0
    if fault == "projection":
        weights[0].fill(np.finfo(np.float64).max)
    elif fault == "logits":
        weights[0].fill(1e155)
        weights[1].fill(1e155)
    elif fault == "norm":
        weights[2].fill(1e155)
    elif fault == "coupling":
        coupling[0, 1] = coupling[1, 0] = np.finfo(np.float64).max
        for array in weights:
            array.fill(0.0)
    with pytest.raises(ValueError):
        attnres_modulate(
            coupling,
            phases,
            w_q=weights[0],
            w_k=weights[1],
            w_v=weights[2],
            w_o=weights[3],
            n_heads=1,
            temperature=float(temperature),
            backend=backend,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_large_representable_coupling_survives_symmetrisation(backend: str) -> None:
    """A finite 1.05e308 result is not lost by adding both rows before halving."""
    coupling = np.array([[0.0, 1e308], [1e308, 0.0]])
    projection = np.zeros((1, 2, 2))
    out = attnres_modulate(
        coupling,
        np.array([0.0, 0.4]),
        w_q=projection,
        w_k=projection,
        w_v=projection,
        w_o=np.zeros((2, 2)),
        n_heads=1,
        lambda_=0.1,
        backend=backend,
    )
    np.testing.assert_allclose(out[0, 1] / 1e308, 1.05, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("antipodal", [True, False])
@pytest.mark.parametrize("edge", [0.3, -0.3])
def test_large_gain_preserves_cosine_endpoint_bounds_and_edge_signs(
    backend: str, antipodal: bool, edge: float
) -> None:
    """Rounding outside the cosine range cannot reverse a signed physical edge."""
    phase = 0.1972727272727273
    phases = np.array([phase, phase + np.pi if antipodal else phase])
    coupling = np.array([[0.0, edge], [edge, 0.0]])
    projection = np.eye(2).reshape(1, 2, 2)
    gain = 1e16
    actual = attnres_modulate(
        coupling,
        phases,
        w_q=projection,
        w_k=projection,
        w_v=projection,
        w_o=np.eye(2) * 1e120,
        n_heads=1,
        lambda_=gain,
        backend=backend,
    )
    expected = edge if antipodal else edge * (1.0 + gain)
    np.testing.assert_allclose(actual[0, 1], expected, rtol=2e-15, atol=1e-12)
    assert np.signbit(actual[0, 1]) == np.signbit(edge)
    assert abs(edge) <= abs(actual[0, 1]) <= abs(edge) * (1.0 + gain)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("temperature,scale", [(1e308, 1e153), (3e-309, 1e-154)])
def test_extreme_representable_temperature_preserves_scaled_logits(
    backend: str, temperature: float, scale: float
) -> None:
    """Finite scaled logits survive overflowing products and reciprocal temperatures."""
    coupling = np.array([[0.0, 0.3, 0.5], [0.3, 0.0, 0.7], [0.5, 0.7, 0.0]])
    phases = np.array([0.1, 0.4, 0.9])
    projection = np.eye(8).reshape(1, 8, 8)
    weights = (projection * scale, projection * scale, projection, np.eye(8))
    expected = phase_attention_oracle(
        coupling, phases, weights, temperature=temperature
    )
    actual = attnres_modulate(
        coupling,
        phases,
        w_q=weights[0],
        w_k=weights[1],
        w_v=weights[2],
        w_o=weights[3],
        n_heads=1,
        temperature=temperature,
        backend=backend,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unused_logits_do_not_reject_a_completely_masked_graph(backend: str) -> None:
    """An index band excludes overflowing scores while preserving existing edges."""
    coupling = np.array([[0.0, 0.0, 0.3], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    projection = np.eye(2).reshape(1, 2, 2)
    actual = attnres_modulate(
        coupling,
        np.array([0.1, 0.4, 0.9]),
        w_q=projection * 1e155,
        w_k=projection * 1e155,
        w_v=projection,
        w_o=np.eye(2),
        n_heads=1,
        block_size=1,
        backend=backend,
    )
    np.testing.assert_array_equal(actual, coupling)


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_upde_consumer_preserves_equation_signs_and_units(backend: str) -> None:
    """The modulated matrix drives the unnormalised rad/s Sakaguchi Euler law."""
    coupling = np.array([[0.0, 0.3, -0.1], [0.3, 0.0, 0.2], [-0.1, 0.2, 0.0]])
    phases = np.array([0.1, 1.4, 2.7])
    omegas = np.array([0.7, -0.2, 1.1])
    alpha = np.array([[0.0, 0.2, -0.1], [0.1, 0.0, 0.4], [-0.3, 0.2, 0.0]])
    weights = default_projections(n_heads=2, d_model=4, seed=12)
    independent_k = phase_attention_oracle(coupling, phases, weights, strength=0.1)
    actual_k = attnres_modulate(
        coupling,
        phases,
        w_q=weights[0],
        w_k=weights[1],
        w_v=weights[2],
        w_o=weights[3],
        n_heads=2,
        lambda_=0.1,
        backend=backend,
    )
    expected = phases.copy()
    for target in range(3):
        derivative = (
            omegas[target]
            + sum(
                independent_k[target, source]
                * np.sin(phases[source] - phases[target] - alpha[target, source])
                for source in range(3)
            )
            + 0.2 * np.sin(0.5 - phases[target])
        )
        expected[target] = (phases[target] + 0.01 * derivative) % (2.0 * np.pi)
    actual = UPDEEngine(n_oscillators=3, dt=0.01, method="euler").step(
        phases, omegas, actual_k, 0.2, 0.5, alpha
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)


def test_invalid_explicit_owner_is_rejected_before_computation() -> None:
    """Misspelled, empty and numeric backend requests cannot silently fall back."""
    coupling = np.array([[0.0, 0.3], [0.3, 0.0]])
    phases = np.array([0.2, 0.8])
    for name in ("", "RUST", "unknown", cast(str, 1)):
        with pytest.raises(ValueError, match="unknown attention backend"):
            attnres_modulate(coupling, phases, backend=name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_constructor_validates_projection_contracts(backend: str) -> None:
    """Reject malformed projections even when no oscillator needs a runtime."""
    valid = np.zeros((1, 4, 4))
    with pytest.raises(ValueError, match="w_o shape"):
        attnres_modulate(
            np.empty((0, 0)),
            np.empty(0),
            w_q=valid,
            w_k=valid,
            w_v=valid,
            w_o=np.zeros((3, 4)),
            n_heads=1,
            backend=backend,
        )
    with pytest.raises(ValueError, match="not divisible"):
        attnres_modulate(np.empty((0, 0)), np.empty(0), n_heads=3, backend=backend)
    empty = attnres_modulate(np.empty((0, 0)), np.empty(0), backend=backend)
    assert empty.shape == (0, 0)
