# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Connectome matrix-validation and FFI guard contracts

"""Validation-boundary contracts for the connectome coupling loaders.

These exercise the rejection paths the synthetic and optional-Rust loaders use to
refuse malformed connectome output: the float-coercion failure, the shape and
symmetry checks in ``_coerce_connectome_matrix``, the native-dtype and
object-array guards on the optional Rust FFI result, and the module-load
fallback that marks the Rust kernel absent when ``spo_kernel`` cannot import.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling import connectome


def test_coerce_rejects_non_float_convertible_object_matrix() -> None:
    """A symmetric object matrix whose entries are not numbers fails coercion."""
    value = np.array([[0.0, "x"], ["x", 0.0]], dtype=object)

    with pytest.raises(ValueError, match="must be a float matrix"):
        connectome._validate_connectome_matrix(value, n_regions=2, source="probe")


def test_coerce_rejects_wrong_shape_matrix() -> None:
    """A well-typed float matrix of the wrong shape is rejected with its shape."""
    value = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=r"shape \(3, 3\), got \(2, 2\)"):
        connectome._coerce_connectome_matrix(value, n_regions=3, source="probe")


def test_coerce_rejects_asymmetric_matrix() -> None:
    """A finite non-negative matrix that is not symmetric is rejected."""
    value = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="must be symmetric"):
        connectome._coerce_connectome_matrix(value, n_regions=2, source="probe")


def test_coerce_accepts_clean_symmetric_matrix() -> None:
    """The happy path returns a contiguous float64 copy with the zero diagonal."""
    value = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64)

    out = connectome._validate_connectome_matrix(value, n_regions=2, source="probe")

    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(out, value)


@pytest.mark.parametrize(
    ("dtype", "message"),
    [
        (bool, "must not contain boolean values"),
        (complex, "must contain real-valued weights"),
    ],
)
def test_rust_loader_rejects_native_bool_or_complex_dtype(
    monkeypatch: pytest.MonkeyPatch, dtype: type, message: str
) -> None:
    """A native bool/complex Rust array is refused before any float coercion."""
    connectome._load_hcp_connectome_cached.cache_clear()

    def fake_rust_load_hcp(n_regions: int, seed: int) -> np.ndarray:
        return np.ones((n_regions, n_regions), dtype=dtype).ravel()

    monkeypatch.setattr(connectome, "_HAS_RUST", True)
    monkeypatch.setattr(connectome, "_rust_load_hcp", fake_rust_load_hcp, raising=False)

    with pytest.raises(ValueError, match=message):
        connectome.load_hcp_connectome(2, seed=4242)


def test_rust_loader_rejects_object_array_without_float_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An object Rust array of non-numbers fails after the bool/complex guards."""
    connectome._load_hcp_connectome_cached.cache_clear()

    def fake_rust_load_hcp(n_regions: int, seed: int) -> np.ndarray:
        return np.array([0.0, "x", "y", 0.0], dtype=object)

    monkeypatch.setattr(connectome, "_HAS_RUST", True)
    monkeypatch.setattr(connectome, "_rust_load_hcp", fake_rust_load_hcp, raising=False)

    with pytest.raises(ValueError, match="must contain real-valued weights"):
        connectome.load_hcp_connectome(2, seed=4343)


def test_module_marks_rust_absent_when_kernel_cannot_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-importing without ``spo_kernel`` drives the ImportError fallback branch."""
    original = connectome._HAS_RUST
    monkeypatch.setitem(sys.modules, "spo_kernel", None)

    reloaded = importlib.reload(connectome)
    try:
        assert reloaded._HAS_RUST is False
    finally:
        monkeypatch.undo()
        importlib.reload(connectome)

    assert connectome._HAS_RUST is original
