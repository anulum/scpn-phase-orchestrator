# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy dispatch contracts

"""Strict transfer-entropy dispatch and validation contract tests."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from typing import TypeAlias, cast

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import transfer_entropy as te_mod

FloatArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
PhaseBackend: TypeAlias = Callable[[FloatArray, FloatArray, int], object]
MatrixBackend: TypeAlias = Callable[[FloatArray, int, int, int], object]


class _UncoercibleArray:
    """Array-like probe that refuses NumPy object-array coercion."""

    def __array__(
        self,
        _dtype: object | None = None,
    ) -> np.ndarray[tuple[()], np.dtype[np.object_]]:
        """Raise the same conversion failure shape as an invalid array provider."""
        raise ValueError("cannot expose array payload")


class _ExplodingReal:
    """Virtual real number that raises during float conversion."""

    def __float__(self) -> float:
        """Fail as a malformed scalar backend payload would fail."""
        raise TypeError("backend scalar is not convertible")


Real.register(_ExplodingReal)


def _phase_series() -> FloatArray:
    """Return a deterministic two-channel phase series."""
    source = np.array([0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1, 4.0])
    target = np.array([0.1, 0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1])
    return cast("FloatArray", np.vstack([source, target]).astype(np.float64))


def test_dispatch_skips_missing_backend_function_and_exhausts_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing backend entries are skipped before the dispatcher falls back."""
    previous_backend = te_mod.ACTIVE_BACKEND
    previous_available = list(te_mod.AVAILABLE_BACKENDS)
    previous_loader = te_mod._LOADERS["rust"]
    te_mod.ACTIVE_BACKEND = "rust"
    te_mod.AVAILABLE_BACKENDS = ["rust"]
    te_mod._BACKEND_CACHE.clear()
    monkeypatch.setitem(te_mod._LOADERS, "rust", lambda: {"other": object()})

    try:
        resolved = te_mod._dispatch("phase_te")
    finally:
        te_mod.ACTIVE_BACKEND = previous_backend
        te_mod.AVAILABLE_BACKENDS = previous_available
        monkeypatch.setitem(te_mod._LOADERS, "rust", previous_loader)
        te_mod._BACKEND_CACHE.clear()

    assert resolved is None


def test_boolean_alias_probe_fails_closed_for_uncoercible_array() -> None:
    """Boolean alias probing must not reject objects that cannot be coerced."""
    assert te_mod._contains_boolean_alias(_UncoercibleArray()) is False


def test_phase_transfer_entropy_rejects_uncoercible_source_vector() -> None:
    """Public pairwise TE validation rejects nonnumeric vector payloads."""
    with pytest.raises(ValueError, match="source must be a finite 1-D phase vector"):
        te_mod.phase_transfer_entropy(
            np.array(["not-a-phase", "still-not-a-phase", "bad"], dtype=object),
            np.array([0.0, 0.1, 0.2], dtype=np.float64),
        )


def test_transfer_entropy_matrix_rejects_uncoercible_phase_series() -> None:
    """Public matrix TE validation rejects nonnumeric phase-series payloads."""
    with pytest.raises(
        ValueError,
        match="phase_series must be a finite 2-D phase series",
    ):
        te_mod.transfer_entropy_matrix(
            np.array([["bad", "phase", "payload"]], dtype=object),
        )


def test_public_phase_transfer_entropy_rejects_nonreal_backend_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend scalars must be finite real numbers."""

    def invalid_phase_te(
        _source: FloatArray,
        _target: FloatArray,
        _n_bins: int,
    ) -> object:
        return object()

    monkeypatch.setattr(te_mod, "_dispatch", lambda _name: invalid_phase_te)

    series = _phase_series()
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        te_mod.phase_transfer_entropy(series[0], series[1], n_bins=2)


def test_public_phase_transfer_entropy_rejects_unfloatable_real_backend_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Virtual real backend scalars must still support float conversion."""

    def invalid_phase_te(
        _source: FloatArray,
        _target: FloatArray,
        _n_bins: int,
    ) -> _ExplodingReal:
        return _ExplodingReal()

    monkeypatch.setattr(te_mod, "_dispatch", lambda _name: invalid_phase_te)

    series = _phase_series()
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        te_mod.phase_transfer_entropy(series[0], series[1], n_bins=2)


def test_public_transfer_entropy_matrix_rejects_nonnumeric_backend_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend matrices must be coercible to finite float64 values."""

    def invalid_matrix(
        _phase_series: FloatArray,
        _n_osc: int,
        _n_time: int,
        _n_bins: int,
    ) -> object:
        return object()

    monkeypatch.setattr(te_mod, "_dispatch", lambda _name: invalid_matrix)

    with pytest.raises(ValueError, match="matrix must be numeric"):
        te_mod.transfer_entropy_matrix(_phase_series(), n_bins=2)


def test_reference_pairwise_transfer_entropy_returns_zero_for_short_series() -> None:
    """The reference estimator returns zero for underspecified histories."""
    source = np.array([0.0, 0.5], dtype=np.float64)
    target = np.array([0.25, 0.75], dtype=np.float64)

    assert te_mod._phase_te_reference(source, target, 4) == 0.0
