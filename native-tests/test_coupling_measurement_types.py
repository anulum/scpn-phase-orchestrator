# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native coupling measurement types

"""Verify original input types at the installed Rust coupling boundaries."""

from __future__ import annotations

import numpy as np
import pytest
import spo_kernel


@pytest.mark.parametrize("argument", ["knm", "positions"])
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), "1.0", np.timedelta64(1, "ms"), np.datetime64("2026-01-01")],
)
def test_native_spatial_vectors_reject_aliases(argument: str, value: object) -> None:
    """Raw Rust vector extraction cannot erase boolean or temporal aliases."""
    matrix = [0.0, 1.0, 1.0, 0.0]
    positions = [0.0, 1.0]
    invalid: list[object] = list(matrix if argument == "knm" else positions)
    invalid[0] = value
    with pytest.raises(ValueError, match=argument):
        spo_kernel.spatial_modulate_rust(
            invalid if argument == "knm" else matrix,
            invalid if argument == "positions" else positions,
            2,
            1,
            1.0,
            0,
            1.0,
            1.0,
            1e-12,
        )


@pytest.mark.parametrize(
    "argument",
    [
        "n",
        "dim",
        "k_base",
        "decay_form_code",
        "decay_exponent",
        "decay_length_scale",
        "epsilon",
    ],
)
@pytest.mark.parametrize("value", [True, np.bool_(True), "1", np.timedelta64(1, "ms")])
def test_native_spatial_controls_reject_aliases(argument: str, value: object) -> None:
    """Counts, form codes and real controls check types before conversion."""
    controls: dict[str, object] = {
        "n": 1,
        "dim": 1,
        "k_base": 1.0,
        "decay_form_code": 0,
        "decay_exponent": 1.0,
        "decay_length_scale": 1.0,
        "epsilon": 1e-12,
    }
    controls[argument] = value
    with pytest.raises(ValueError):
        spo_kernel.spatial_modulate_rust([0.0], [0.0], **controls)


@pytest.mark.parametrize("dtype", ["timedelta64[ms]", "datetime64[ms]", "bool", "U8"])
def test_native_spectral_array_abi_rejects_aliases(dtype: str) -> None:
    """The typed ndarray ABI rejects aliases without a Rust algorithm change."""
    matrix = np.array([0.0, 1.0, 1.0, 0.0]).astype(dtype)
    with pytest.raises(TypeError):
        spo_kernel.fiedler_value_rust(matrix, 2)


def test_native_spatial_real_objects_and_controls_preserve_result() -> None:
    """Genuine NumPy and Python scalar numbers retain the spatial formula."""
    actual = spo_kernel.spatial_modulate_rust(
        np.array([0.0, 1.0, 1.0, 0.0], dtype=object),
        np.array([0.0, 1.0], dtype=object),
        np.int64(2),
        np.int64(1),
        np.float64(1.0),
        np.int32(0),
        np.float64(1.0),
        1.0,
        1e-12,
    )
    np.testing.assert_allclose(actual, [0.0, 0.5, 0.5, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "argument", ["n", "n_heads", "block_size", "temperature", "lambda_"]
)
@pytest.mark.parametrize("value", [True, np.bool_(True), "1", np.timedelta64(1, "ms")])
def test_native_attnres_controls_reject_aliases(argument: str, value: object) -> None:
    """The real Rust attention boundary preserves original scalar source types."""
    from scpn_phase_orchestrator.coupling.attention_residuals import default_projections

    q, key, v, o = default_projections()
    controls: dict[str, object] = {
        "n": 2,
        "n_heads": 4,
        "block_size": -1,
        "temperature": 1.0,
        "lambda_": 0.5,
    }
    controls[argument] = value
    with pytest.raises(ValueError):
        spo_kernel.attnres_modulate_rust(
            np.array([0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 1.0]),
            q.ravel(),
            key.ravel(),
            v.ravel(),
            o.ravel(),
            **controls,
        )


@pytest.mark.parametrize("argument", range(6))
@pytest.mark.parametrize("dtype", ["timedelta64[ms]", "datetime64[ms]", "bool", "U8"])
def test_native_attnres_array_abi_rejects_aliases(argument: int, dtype: str) -> None:
    """The native ndarray ABI refuses non-float measurement buffers."""
    from scpn_phase_orchestrator.coupling.attention_residuals import default_projections

    q, key, v, o = default_projections()
    arrays = [
        np.array([0.0, 1.0, 1.0, 0.0]),
        np.array([0.0, 1.0]),
        q.ravel(),
        key.ravel(),
        v.ravel(),
        o.ravel(),
    ]
    arrays[argument] = arrays[argument].astype(dtype)
    with pytest.raises(TypeError):
        spo_kernel.attnres_modulate_rust(*arrays, 2, 4, -1, 1.0, 0.5)


@pytest.mark.parametrize("argument", ["n", "target_ratio"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "1", np.timedelta64(1, "ms")])
def test_native_ei_controls_reject_aliases(argument: str, value: object) -> None:
    """Native E/I controls reject aliases before extraction to Rust values."""
    k = np.array([0.0, 1.0, 1.0, 0.0])
    e = np.array([0], dtype=np.int64)
    i = np.array([1], dtype=np.int64)
    controls: dict[str, object] = {"n": 2, "target_ratio": 1.0}
    controls[argument] = value
    with pytest.raises(ValueError):
        spo_kernel.adjust_ei_ratio_rust(
            k, excitatory_indices=e, inhibitory_indices=i, **controls
        )
    if argument == "n":
        with pytest.raises(ValueError):
            spo_kernel.compute_ei_balance_rust(k, value, e, i)


@pytest.mark.parametrize("dtype", ["timedelta64[ms]", "datetime64[ms]", "bool", "U8"])
def test_native_ei_array_abi_rejects_aliases(dtype: str) -> None:
    """Real typed Rust E/I arrays refuse non-float measurement buffers."""
    k = np.array([0.0, 1.0, 1.0, 0.0]).astype(dtype)
    e = np.array([0], dtype=np.int64)
    i = np.array([1], dtype=np.int64)
    with pytest.raises(TypeError):
        spo_kernel.compute_ei_balance_rust(k, 2, e, i)
    with pytest.raises(TypeError):
        spo_kernel.adjust_ei_ratio_rust(k, 2, e, i)


@pytest.mark.parametrize("argument", ["n_regions", "seed"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "2", np.timedelta64(2, "ms")])
def test_native_connectome_metadata_rejects_aliases(
    argument: str, value: object
) -> None:
    """Rust generator metadata rejects aliases before RNG or allocation."""
    with pytest.raises(ValueError):
        spo_kernel.load_hcp_connectome_rust(
            value if argument == "n_regions" else 4,
            value if argument == "seed" else 42,
        )


@pytest.mark.parametrize("n_regions", [0, 1])
def test_native_connectome_requires_two_regions(n_regions: int) -> None:
    """The native generator shares the public minimum region count."""
    with pytest.raises(ValueError, match="n_regions"):
        spo_kernel.load_hcp_connectome_rust(n_regions, 42)


@pytest.mark.parametrize("seed", [0, 2**64 - 1, np.uint64(42)])
def test_native_connectome_preserves_unsigned_seed_domain(seed: object) -> None:
    """Unsigned integer metadata retains its full supported seed domain."""
    matrix = spo_kernel.load_hcp_connectome_rust(4, seed).reshape(4, 4)
    assert np.all(np.isfinite(matrix))
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(4))
