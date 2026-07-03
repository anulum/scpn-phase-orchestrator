# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Winding dispatch fail-closed contracts

"""Dispatch and defensive-validation contracts for winding numbers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import winding as winding_module


class _ArrayRaises:
    """Array-like sentinel that fails NumPy coercion."""

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        """Raise during NumPy array coercion."""
        raise ValueError("array coercion refused")


def _raising_loader() -> Callable[..., np.ndarray]:
    """Raise as an unavailable optional winding backend loader."""
    raise RuntimeError("optional backend unavailable")


def test_resolve_backends_keeps_python_when_optional_probes_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import-time backend resolution keeps Python after optional probe failures."""
    for backend in ("rust", "mojo", "julia", "go"):
        monkeypatch.setitem(winding_module._LOADERS, backend, _raising_loader)

    active, available = winding_module._resolve_backends()

    assert active == "python"
    assert available == ["python"]


def test_dispatch_returns_python_when_non_python_loaders_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime dispatch returns ``None`` when no configured backend can load."""
    previous_backend = winding_module.ACTIVE_BACKEND
    previous_available = list(winding_module.AVAILABLE_BACKENDS)
    winding_module.ACTIVE_BACKEND = "go"
    winding_module.AVAILABLE_BACKENDS = ["go", "julia"]
    winding_module._BACKEND_CACHE.clear()
    monkeypatch.setitem(winding_module._LOADERS, "go", _raising_loader)
    monkeypatch.setitem(winding_module._LOADERS, "julia", _raising_loader)

    try:
        backend = winding_module._dispatch()
    finally:
        winding_module.ACTIVE_BACKEND = previous_backend
        winding_module.AVAILABLE_BACKENDS = previous_available
        winding_module._BACKEND_CACHE.clear()

    assert backend is None


def test_alias_detectors_treat_uncoercible_payloads_as_non_aliases() -> None:
    """Alias probes fail open so public validation can raise the typed error."""
    payload = _ArrayRaises()

    assert winding_module._contains_boolean_alias(payload) is False
    assert winding_module._contains_complex_alias(payload) is False
    assert winding_module._has_complex_payload(payload) is False
    assert winding_module._is_numeric_string_alias(1.0) is False
    assert winding_module._is_numeric_string_alias("not-a-number") is False
    assert winding_module._contains_numeric_string_alias(payload) is False
    assert (
        winding_module._contains_numeric_string_alias(
            np.array([1.0, "not-a-number"], dtype=object)
        )
        is False
    )
    assert (
        winding_module._contains_numeric_string_alias(
            np.array([1.0, "2.0"], dtype=object)
        )
        is True
    )


def test_julia_backend_loader_returns_winding_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Julia backend loader must expose the winding-number callable."""
    monkeypatch.setattr(winding_module, "require_juliacall_main", lambda: None)

    loaded = winding_module._load_julia_fn()

    assert callable(loaded)


def test_backend_winding_rejects_uncoercible_array_like_output() -> None:
    """Backend output must be array-like before shape and numeric checks."""
    with pytest.raises(ValueError, match="array-like"):
        winding_module._validate_backend_winding(_ArrayRaises(), n=1, t=2)


def test_backend_winding_rejects_non_numeric_object_output() -> None:
    """Backend output object arrays must still coerce to numeric windings."""
    with pytest.raises(ValueError, match="numeric"):
        winding_module._validate_backend_winding(
            np.array([object()], dtype=object),
            n=1,
            t=2,
        )


def test_backend_winding_rejects_expected_shape_mismatch() -> None:
    """Expected reference shape is part of the exact backend contract."""
    with pytest.raises(ValueError, match="reference shape"):
        winding_module._validate_backend_winding(
            np.array([0], dtype=np.int64),
            n=1,
            t=2,
            expected=np.array([0, 0], dtype=np.int64),
        )
