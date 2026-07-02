# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spatial coupling modulator tests

"""Behavioural tests for ``coupling.spatial_modulator``."""

from __future__ import annotations

import ctypes
import importlib
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias, cast, get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from benchmarks.spatial_modulator_benchmark import (
    benchmark_spatial_modulator_polyglot_parity_gate,
)
from scpn_phase_orchestrator.coupling import SpatialCouplingModulator
from scpn_phase_orchestrator.coupling import (
    _spatial_modulator_go as public_spatial_go,
)
from scpn_phase_orchestrator.coupling import (
    _spatial_modulator_julia as public_spatial_julia,
)
from scpn_phase_orchestrator.coupling import (
    _spatial_modulator_mojo as public_spatial_mojo,
)
from scpn_phase_orchestrator.coupling import spatial_modulator as sm_mod
from scpn_phase_orchestrator.coupling.spatial_modulator import spatial_modulate
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _spatial_modulator_go,
    _spatial_modulator_julia,
    _spatial_modulator_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _spatial_modulator_go as direct_spatial_go,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _spatial_modulator_julia as direct_spatial_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _spatial_modulator_mojo as direct_spatial_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _spatial_modulator_validation as spatial_validation,
)
from scpn_phase_orchestrator.upde import swarmalator as sw_mod
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine
from tests.typing_contracts import assert_precise_ndarray_hint

TWO_PI = 2.0 * np.pi
FloatArray: TypeAlias = NDArray[np.float64]
DirectArgs: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    int,
    int,
    float,
    int,
    float,
    float,
    float,
]


def _base() -> FloatArray:
    return np.array(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 0.5],
            [1.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )


def _positions() -> FloatArray:
    return np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float64)


def test_inverse_plus_one_matches_analytical_values() -> None:
    modulator = SpatialCouplingModulator(K_base=0.5)
    got = modulator.modulate(_base(), _positions())

    expected = np.array(
        [
            [0.0, 2.0 * 0.5 / 6.0, 1.0 * 0.5 / 11.0],
            [2.0 * 0.5 / 6.0, 0.0, 0.5 * 0.5 / 6.0],
            [1.0 * 0.5 / 11.0, 0.5 * 0.5 / 6.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(got, expected, atol=1.0e-12)
    np.testing.assert_allclose(np.diag(got), 0.0, atol=1.0e-12)


def test_exponential_power_law_and_inverse_distance_contracts() -> None:
    positions = np.array([[0.0], [2.0]], dtype=np.float64)
    base = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    exp_mod = SpatialCouplingModulator(
        K_base=3.0,
        decay_form="exponential",
        decay_length_scale=2.0,
    ).modulate(base, positions)
    power_mod = SpatialCouplingModulator(
        K_base=3.0,
        decay_form="power_law",
        decay_exponent=2.0,
        decay_length_scale=2.0,
    ).modulate(base, positions)
    inv_mod = SpatialCouplingModulator(
        K_base=3.0,
        decay_form="inverse_distance",
        epsilon=1.0e-6,
    ).modulate(base, positions)

    assert exp_mod[0, 1] == pytest.approx(3.0 * np.exp(-1.0), abs=1.0e-12)
    assert power_mod[0, 1] == pytest.approx(3.0 / 4.0, abs=1.0e-12)
    assert inv_mod[0, 1] == pytest.approx(3.0 / np.sqrt(4.0 + 1.0e-6), abs=1.0e-12)


def test_jacobian_positions_matches_jax_reference() -> None:
    from jax import config

    cast(Callable[[str, bool], None], config.update)("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp

    positions = np.array([[0.0, 0.1], [1.2, -0.4], [2.0, 0.7]], dtype=np.float64)
    modulator = SpatialCouplingModulator(K_base=0.7)

    def reference(flat: Any) -> Any:
        x = flat.reshape((3, 2))
        diff = x[:, None, :] - x[None, :, :]
        squared = jnp.sum(diff * diff, axis=2)
        safe_squared = jnp.where(jnp.eye(3, dtype=bool), 1.0, squared)
        distances = jnp.sqrt(safe_squared)
        weights = 0.7 / (1.0 + distances)
        weights = weights.at[jnp.diag_indices(3)].set(0.0)
        return weights.reshape(-1)

    expected = np.asarray(jax.jacobian(reference)(jnp.asarray(positions.ravel())))
    expected = expected.reshape(3, 3, 3, 2)
    got = modulator.jacobian_positions(positions)
    diagonal = np.eye(3, dtype=bool)
    np.testing.assert_allclose(got[diagonal], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(got[~diagonal], expected[~diagonal], atol=1.0e-9)


def test_public_function_and_lazy_export() -> None:
    got = spatial_modulate(_base(), _positions(), K_base=0.5)
    via_export = SpatialCouplingModulator(K_base=0.5).modulate(_base(), _positions())
    np.testing.assert_allclose(got, via_export, atol=1.0e-12)


def test_invalid_boundaries_fail_closed() -> None:
    modulator = SpatialCouplingModulator(K_base=1.0)
    invalid_cases = (
        (
            _base().astype(object),
            np.array([[True], [False], [True]], dtype=object),
            "positions",
        ),
        (np.eye(3), _positions(), "diagonal"),
        (np.array([[0.0, 1.0j], [1.0, 0.0]]), np.array([[0.0], [1.0]]), "real"),
        (_base(), np.array([[0.0], [np.inf], [1.0]]), "finite"),
    )
    for knm, positions, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            modulator.modulate(knm, positions)
    with pytest.raises(ValueError, match="decay_form"):
        SpatialCouplingModulator(K_base=1.0, decay_form="unknown")
    with pytest.raises(ValueError, match="K_base"):
        SpatialCouplingModulator(K_base=-1.0)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=16, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_symmetric_base_preserves_symmetry_and_permutation_equivariance(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = 5
    raw = rng.uniform(0.0, 1.0, size=(n, n))
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    positions = rng.normal(size=(n, 3))
    modulator = SpatialCouplingModulator(K_base=0.4)
    got = modulator.modulate(knm, positions)
    np.testing.assert_allclose(got, got.T, atol=1.0e-12)
    perm = rng.permutation(n)
    permuted = modulator.modulate(knm[np.ix_(perm, perm)], positions[perm])
    np.testing.assert_allclose(permuted, got[np.ix_(perm, perm)], atol=1.0e-12)


def test_swarmalator_python_reference_uses_bit_true_inverse_distance_kernel() -> None:
    previous = sw_mod.ACTIVE_BACKEND
    sw_mod.ACTIVE_BACKEND = "python"
    try:
        rng = np.random.default_rng(7)
        n = 4
        dim = 2
        dt = 0.01
        pos = rng.normal(size=(n, dim))
        phases = rng.uniform(0.0, TWO_PI, size=n)
        omegas = rng.normal(size=n)
        engine = SwarmalatorEngine(n, dim=dim, dt=dt)
        got_pos, got_phases = engine.step(
            pos, phases, omegas, a=1.0, b=1.0, j=0.3, k=0.8
        )
    finally:
        sw_mod.ACTIVE_BACKEND = previous

    expected_pos = pos.copy()
    expected_phases = phases.copy()
    eps = 1.0e-6
    for i in range(n):
        diff = pos - pos[i]
        d2 = np.sum(diff**2, axis=1)
        dist = np.sqrt(d2 + eps)
        cos_diff = np.cos(phases - phases[i])
        sin_diff = np.sin(phases - phases[i])
        attract = (1.0 + 0.3 * cos_diff) / dist
        # Canonical OHS inverse-distance core: b / (|x_j - x_i|^2 + eps),
        # the scalar b/(d2+eps) multiplying the separation vector (b = 1.0).
        repulse = 1.0 / (d2 + eps)
        expected_pos[i] = (
            pos[i] + dt * np.sum(diff * (attract - repulse)[:, None], axis=0) / n
        )
        expected_phases[i] = (
            phases[i] + dt * (omegas[i] + 0.8 * float(np.mean(sin_diff / dist)))
        ) % TWO_PI
    np.testing.assert_allclose(got_pos, expected_pos, atol=1.0e-12)
    np.testing.assert_allclose(got_phases, expected_phases, atol=1.0e-12)


class _FakeGoSpatialLib:
    def __init__(self, values: tuple[float, ...], rc: int = 0) -> None:
        self.values = values
        self.rc = rc

    def SpatialModulate(self, *_args: object) -> int:
        out_ref = ctypes.cast(
            cast(Any, _args[-1]),
            ctypes.POINTER(ctypes.c_double),
        )
        for index, value in enumerate(self.values):
            out_ref[index] = value
        return self.rc


class _FakeJuliaSpatialModule:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values

    def spatial_modulate(self, *_args: object) -> FloatArray:
        return np.array(self.values, dtype=np.float64)


class _ArrayAliasRejector:
    def __array__(self, dtype: object | None = None) -> object:
        raise TypeError("array alias rejected")


class _ComplexAliasRejector:
    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, dtype: object | None = None) -> object:
        self.calls += 1
        if self.calls == 1:
            return np.array([object()], dtype=object)
        raise TypeError("complex alias rejected")


class _FirstArrayCallRejector:
    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, dtype: object | None = None) -> object:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("first array conversion rejected")
        return np.array(["not-a-number"], dtype=object)


def _direct_args() -> DirectArgs:
    return (
        np.array([0.0, 0.2, 0.2, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        2,
        1,
        1.0,
        0,
        1.0,
        1.0,
        1.0e-12,
    )


def test_alias_probe_exceptions_are_treated_as_absent_aliases() -> None:
    assert sm_mod._contains_boolean_alias(_ArrayAliasRejector()) is False
    assert sm_mod._contains_complex_alias(_ComplexAliasRejector()) is False


def test_one_dimensional_positions_and_custom_distance_modulate() -> None:
    modulator = SpatialCouplingModulator(
        K_base=1.0,
        distance_fn=lambda _left, _right: np.array(
            [[0.0, 2.0], [2.0, 0.0]],
            dtype=np.float64,
        ),
    )

    field = modulator.modulation_matrix(np.array([0.0, 1.0], dtype=np.float64))
    got = modulator.modulate(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    )

    expected = np.array([[0.0, 1.0 / 3.0], [1.0 / 3.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(field, expected)
    np.testing.assert_allclose(got, expected)


def test_public_backend_output_rejects_first_array_conversion_failure() -> None:
    with pytest.raises(ValueError, match="finite real-valued"):
        sm_mod._validate_backend_output(_FirstArrayCallRejector(), n=2)


def test_backend_loader_success_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm_mod, "require_juliacall_main", lambda: object())
    monkeypatch.setattr(_spatial_modulator_go, "_load_lib", lambda: object())

    assert sm_mod._load_julia_fn() is _spatial_modulator_julia.spatial_modulate_julia
    assert sm_mod._load_go_fn() is _spatial_modulator_go.spatial_modulate_go


def test_dispatch_skips_failed_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_backend(_name: str) -> object:
        raise ImportError("backend unavailable")

    monkeypatch.setattr(sm_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(sm_mod, "AVAILABLE_BACKENDS", [])
    monkeypatch.setattr(sm_mod, "_load_backend", unavailable_backend)

    assert sm_mod._dispatch() is None


def test_direct_julia_ensure_cache_and_file_guards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cached = object()
    monkeypatch.setattr(_spatial_modulator_julia, "_JULIA_MODULE", cached)
    assert _spatial_modulator_julia._ensure() is cached

    class _FakeJuliaMain:
        def __init__(self, module: object) -> None:
            self.SpatialModulatorJL = module
            self.includes: list[str] = []

        def include(self, path: str) -> None:
            self.includes.append(path)

    module = object()
    side_file = tmp_path / "spatial_modulator.jl"
    side_file.write_text("module SpatialModulatorJL end\n", encoding="utf-8")
    fake_main = _FakeJuliaMain(module)
    monkeypatch.setattr(_spatial_modulator_julia, "_JULIA_MODULE", None)
    monkeypatch.setattr(_spatial_modulator_julia, "_JULIA_FILE", side_file)
    monkeypatch.setattr(
        _spatial_modulator_julia,
        "require_julia_main",
        lambda: fake_main,
    )

    assert _spatial_modulator_julia._ensure() is module
    assert fake_main.includes == [str(side_file)]

    monkeypatch.setattr(_spatial_modulator_julia, "_JULIA_MODULE", None)
    monkeypatch.setattr(
        _spatial_modulator_julia,
        "_JULIA_FILE",
        tmp_path / "missing.jl",
    )
    with pytest.raises(ImportError, match="julia side-file not found"):
        _spatial_modulator_julia._ensure()


@pytest.mark.parametrize(
    ("module", "fn_name", "loader_name"),
    [
        (_spatial_modulator_go, "spatial_modulate_go", "_load_lib"),
        (_spatial_modulator_julia, "spatial_modulate_julia", "_ensure"),
        (_spatial_modulator_mojo, "spatial_modulate_mojo", "_ensure_exe"),
    ],
)
def test_direct_backend_rejects_invalid_inputs_before_runtime_loading(
    monkeypatch: pytest.MonkeyPatch,
    module: types.ModuleType,
    fn_name: str,
    loader_name: str,
) -> None:
    def forbidden_loader() -> None:
        raise AssertionError("runtime loader must not be called")

    monkeypatch.setattr(module, loader_name, forbidden_loader)
    args = list(_direct_args())
    args[0] = np.array([1.0, 0.2, 0.2, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="diagonal"):
        getattr(module, fn_name)(*args)


def test_direct_go_julia_mojo_outputs_are_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _spatial_modulator_go,
        "_load_lib",
        lambda: _FakeGoSpatialLib((0.0, 0.1, 0.1, 0.0)),
    )
    np.testing.assert_allclose(
        _spatial_modulator_go.spatial_modulate_go(*_direct_args()),
        np.array([0.0, 0.1, 0.1, 0.0], dtype=np.float64),
    )

    monkeypatch.setattr(
        _spatial_modulator_julia,
        "_ensure",
        lambda: _FakeJuliaSpatialModule((0.0, 0.2, 0.2, 0.0)),
    )
    np.testing.assert_allclose(
        _spatial_modulator_julia.spatial_modulate_julia(*_direct_args()),
        np.array([0.0, 0.2, 0.2, 0.0], dtype=np.float64),
    )

    monkeypatch.setattr(
        _spatial_modulator_mojo, "_ensure_exe", lambda: "spatial_modulator"
    )
    mojo_subprocess = cast(Any, _spatial_modulator_mojo.__dict__["subprocess"])
    monkeypatch.setattr(
        mojo_subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout="0.0\n0.3\n0.3\n0.0\n", stderr=""
        ),
    )
    np.testing.assert_allclose(
        _spatial_modulator_mojo.spatial_modulate_mojo(*_direct_args()),
        np.array([0.0, 0.3, 0.3, 0.0], dtype=np.float64),
    )


def test_direct_julia_rejects_boolean_output_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BooleanJuliaSpatialModule:
        def spatial_modulate(self, *_args: object) -> object:
            return np.array([0.0, True, 0.5, 0.0], dtype=object)

    monkeypatch.setattr(_spatial_modulator_julia, "_ensure", _BooleanJuliaSpatialModule)

    with pytest.raises(ValueError, match="finite real-valued"):
        _spatial_modulator_julia.spatial_modulate_julia(*_direct_args())


def test_public_spatial_accelerator_wrappers_forward_to_direct_modules() -> None:
    assert (
        importlib.import_module(
            "scpn_phase_orchestrator.coupling._spatial_modulator_go"
        )
        is public_spatial_go
    )
    assert (
        importlib.import_module(
            "scpn_phase_orchestrator.coupling._spatial_modulator_julia"
        )
        is public_spatial_julia
    )
    assert (
        importlib.import_module(
            "scpn_phase_orchestrator.coupling._spatial_modulator_mojo"
        )
        is public_spatial_mojo
    )
    assert (
        importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.coupling._spatial_modulator_go"
        )
        is direct_spatial_go
    )
    assert (
        importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.coupling._spatial_modulator_julia"
        )
        is direct_spatial_julia
    )
    assert (
        importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.coupling._spatial_modulator_mojo"
        )
        is direct_spatial_mojo
    )
    assert (
        importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.coupling._spatial_modulator_validation"
        )
        is spatial_validation
    )
    assert (
        public_spatial_go.spatial_modulate_go is direct_spatial_go.spatial_modulate_go
    )
    assert public_spatial_go._load_lib is direct_spatial_go._load_lib
    assert (
        public_spatial_julia.spatial_modulate_julia
        is direct_spatial_julia.spatial_modulate_julia
    )
    assert (
        public_spatial_mojo.spatial_modulate_mojo
        is direct_spatial_mojo.spatial_modulate_mojo
    )
    assert public_spatial_mojo._ensure_exe is direct_spatial_mojo._ensure_exe


def test_spatial_backend_validation_rejects_bad_output_invariants() -> None:
    with pytest.raises(ValueError, match="output length"):
        spatial_validation.validate_spatial_modulator_output([0.0, 1.0], n=2)
    with pytest.raises(ValueError, match="diagonal"):
        spatial_validation.validate_spatial_modulator_output(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            n=2,
        )
    with pytest.raises(ValueError, match="finite"):
        spatial_validation.validate_spatial_modulator_inputs(
            np.zeros(4, dtype=np.float64),
            np.array([0.0, np.inf], dtype=np.float64),
            2,
            1,
            1.0,
            0,
            1.0,
            1.0,
            1.0e-12,
        )


def test_rust_loader_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_spo = types.ModuleType("spo_kernel")

    def spatial_modulate_rust(*_args: object) -> list[float]:
        return [0.0, 0.4, 0.4, 0.0]

    fake_spo.__dict__["spatial_modulate_rust"] = spatial_modulate_rust
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)
    fn = sm_mod._load_rust_fn()
    np.testing.assert_allclose(
        fn(*_direct_args()),
        np.array([0.0, 0.4, 0.4, 0.0], dtype=np.float64),
    )


def test_rust_loader_rejects_boolean_output_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_spo = types.ModuleType("spo_kernel")

    def spatial_modulate_rust(*_args: object) -> object:
        return np.array([0.0, True, 0.5, 0.0], dtype=object)

    fake_spo.__dict__["spatial_modulate_rust"] = spatial_modulate_rust
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)
    fn = sm_mod._load_rust_fn()

    with pytest.raises(ValueError, match="finite real-valued"):
        fn(*_direct_args())


def test_backend_annotations_use_float64_ndarray() -> None:
    for fn in (
        _spatial_modulator_go.spatial_modulate_go,
        _spatial_modulator_julia.spatial_modulate_julia,
        _spatial_modulator_mojo.spatial_modulate_mojo,
    ):
        hints = get_type_hints(fn)
        for name in ("k_nm_flat", "positions_flat", "return"):
            assert_precise_ndarray_hint(hints[name])
            assert "float64" in str(hints[name])


def test_spatial_modulator_benchmark_gate_reports_contracts() -> None:
    out = benchmark_spatial_modulator_polyglot_parity_gate(
        n=6, dim=2, calls=1, seed=2026
    )
    records = cast(
        list[dict[str, object]],
        json.loads(str(out["backend_records_json"])),
    )
    thresholds = cast(
        dict[str, object],
        json.loads(str(out["acceptance_thresholds_json"])),
    )

    assert out["suite"] == "spatial_modulator_polyglot_parity_gate"
    assert out["backend_count"] == 5
    assert out["python_reference_present"] == 1
    assert out["acceptance_passed"] == 1
    assert float(cast(float, out["manual_formula_abs_error"])) <= 1.0e-12
    assert float(cast(float, out["translation_abs_error"])) <= 1.0e-12
    assert float(cast(float, out["permutation_abs_error"])) <= 1.0e-12
    assert float(cast(float, out["symmetry_abs_error"])) <= 1.0e-12
    assert float(cast(float, out["zero_diagonal_abs_error"])) <= 1.0e-12
    assert out["nearer_pair_weight_exceeds_far_pair"] == 1
    assert thresholds["require_inverse_plus_one_formula"] is True
    assert thresholds["require_translation_invariance"] is True
    assert thresholds["production_timing_claim"] is False
    for record in records:
        if record["status"] == "available":
            assert record["parity_passed"] is True
            assert record["matrix_sha256"] is not None
        else:
            assert record["unavailable_reason"]
