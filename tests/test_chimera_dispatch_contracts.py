# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera dispatch contract guards

"""Module-specific contracts for Chimera backend dispatch and validation."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _chimera_go as chimera_go_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _chimera_julia as chimera_julia_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _chimera_mojo as chimera_mojo_mod,
)
from scpn_phase_orchestrator.monitor import chimera as chimera_mod

FloatArray: TypeAlias = NDArray[np.float64]
ChimeraBackend: TypeAlias = Callable[[FloatArray, FloatArray, int], FloatArray]


class _ObjectProbe:
    """Array-like value that rejects object-dtype probing only."""

    def __array__(self, dtype: object = None) -> FloatArray:
        """Return numeric data unless the caller asks for object dtype."""
        if dtype is object or dtype == np.dtype(object):
            raise TypeError("object probing disabled")
        return np.array([0.0, 0.1], dtype=np.float64)


class _UncoercibleProbe:
    """Array-like value that rejects every NumPy coercion attempt."""

    def __array__(self, dtype: object = None) -> FloatArray:
        """Raise to exercise validation probes that must tolerate coercion failure."""
        raise TypeError(f"cannot coerce with dtype {dtype!r}")


def _matrix() -> FloatArray:
    """Return a valid two-node zero-diagonal coupling matrix."""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def _local_backend(
    _phases: FloatArray,
    _knm_flat: FloatArray,
    n_oscillators: int,
) -> FloatArray:
    """Return a deterministic valid local-order vector."""
    return np.full(n_oscillators, 0.5, dtype=np.float64)


def test_optional_backend_loader_returns_mojo_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Mojo loader returns the local-order backend after executable probing."""

    def _fake_ensure_exe() -> str:
        return "chimera_mojo"

    monkeypatch.setattr(chimera_mojo_mod, "_ensure_exe", _fake_ensure_exe)
    monkeypatch.setattr(
        chimera_mojo_mod,
        "local_order_parameter_mojo",
        _local_backend,
    )

    assert chimera_mod._load_mojo_fn() is _local_backend


def test_optional_backend_loader_returns_julia_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Julia loader returns the local-order backend when juliacall imports."""
    fake_juliacall = ModuleType("juliacall")
    fake_juliacall.Main = object()
    monkeypatch.setitem(sys.modules, "juliacall", fake_juliacall)
    monkeypatch.setattr(
        chimera_julia_mod,
        "local_order_parameter_julia",
        _local_backend,
    )

    assert chimera_mod._load_julia_fn() is _local_backend


def test_optional_backend_loader_returns_go_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Go loader returns the local-order backend after library probing."""

    def _fake_load_lib() -> object:
        return object()

    monkeypatch.setattr(chimera_go_mod, "_load_lib", _fake_load_lib)
    monkeypatch.setattr(chimera_go_mod, "local_order_parameter_go", _local_backend)

    assert chimera_mod._load_go_fn() is _local_backend


def test_resolve_backends_keeps_python_when_optional_loaders_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend resolution ignores unavailable optional loaders."""
    previous_loaders = dict(chimera_mod._LOADERS)

    def _missing_backend() -> ChimeraBackend:
        raise ImportError("optional backend unavailable")

    for backend_name in previous_loaders:
        monkeypatch.setitem(chimera_mod._LOADERS, backend_name, _missing_backend)
    try:
        active_backend, available_backends = chimera_mod._resolve_backends()
    finally:
        chimera_mod._LOADERS.update(previous_loaders)
        chimera_mod._BACKEND_CACHE.clear()

    assert active_backend == "python"
    assert available_backends == ["python"]


def test_dispatch_returns_none_when_no_backend_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch returns ``None`` if every configured backend fails."""
    previous_backend = chimera_mod.ACTIVE_BACKEND
    previous_available = list(chimera_mod.AVAILABLE_BACKENDS)
    previous_loader = chimera_mod._LOADERS["go"]

    def _missing_backend() -> ChimeraBackend:
        raise ImportError("go backend unavailable")

    chimera_mod.ACTIVE_BACKEND = "go"
    chimera_mod.AVAILABLE_BACKENDS = ["go"]
    chimera_mod._BACKEND_CACHE.clear()
    monkeypatch.setitem(chimera_mod._LOADERS, "go", _missing_backend)
    try:
        backend = chimera_mod._dispatch()
    finally:
        chimera_mod.ACTIVE_BACKEND = previous_backend
        chimera_mod.AVAILABLE_BACKENDS = previous_available
        monkeypatch.setitem(chimera_mod._LOADERS, "go", previous_loader)
        chimera_mod._BACKEND_CACHE.clear()

    assert backend is None


def test_chimera_state_rejects_string_index_sequence() -> None:
    """Index lists reject string-like payloads instead of iterating characters."""
    with pytest.raises(ValueError, match="coherent_indices"):
        chimera_mod.ChimeraState(coherent_indices=cast("list[int]", "01"))


def test_alias_probe_helpers_treat_unprobeable_payload_as_absent() -> None:
    """Boolean and complex alias probes fail closed to normal numeric coercion."""
    probe = _ObjectProbe()

    assert chimera_mod._contains_boolean_alias(probe) is False
    assert chimera_mod._contains_complex_alias(probe) is False
    assert chimera_mod._has_complex_payload(probe) is False


def test_complex_payload_probe_handles_uncoercible_values() -> None:
    """Complex-payload probing tolerates values NumPy cannot coerce."""
    assert chimera_mod._has_complex_payload(_UncoercibleProbe()) is False


def test_local_order_parameter_accepts_empty_public_input() -> None:
    """The public local-order API preserves empty-system semantics."""
    local_order = chimera_mod.local_order_parameter(
        np.array([], dtype=np.float64),
        np.zeros((0, 0), dtype=np.float64),
    )

    assert local_order.shape == (0,)


def test_local_order_parameter_rejects_nonnumeric_phase_payload() -> None:
    """The public local-order API rejects nonnumeric phases."""
    with pytest.raises(ValueError, match="phases must be a finite"):
        chimera_mod.local_order_parameter(
            np.array(["bad", "payload"], dtype=object),
            _matrix(),
        )


def test_local_order_parameter_rejects_nonnumeric_coupling_payload() -> None:
    """The public local-order API rejects nonnumeric coupling matrices."""
    with pytest.raises(ValueError, match="knm must be a finite square"):
        chimera_mod.local_order_parameter(
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([["bad", "payload"], ["data", "0.0"]], dtype=object),
        )


def test_local_order_parameter_rejects_nonnumeric_backend_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend local-order output must be numeric before range checks."""

    def _bad_backend(
        _phases: FloatArray,
        _knm_flat: FloatArray,
        _n_oscillators: int,
    ) -> object:
        return ["bad", "payload"]

    monkeypatch.setattr(chimera_mod, "_dispatch", lambda: _bad_backend)

    with pytest.raises(ValueError, match="output must be numeric"):
        chimera_mod.local_order_parameter(
            np.array([0.0, 0.1], dtype=np.float64),
            _matrix(),
        )
