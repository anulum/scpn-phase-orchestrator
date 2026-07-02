# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - market public backend output contracts

"""Public dispatcher contracts for market accelerator backend outputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import market as market_mod
from scpn_phase_orchestrator.upde.market import market_order_parameter, market_plv

FloatArray: TypeAlias = NDArray[np.float64]
MarketOrderFn: TypeAlias = Callable[[FloatArray, int, int], FloatArray]
MarketPLVFn: TypeAlias = Callable[[FloatArray, int, int, int], FloatArray]


def _install_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    order_output: FloatArray,
    plv_output: FloatArray,
) -> None:
    """Install a deterministic optional market backend for public API tests."""

    def order_backend(
        _phases_flat: FloatArray,
        _t: int,
        _n: int,
    ) -> FloatArray:
        """Return the configured order-parameter payload."""
        return np.ascontiguousarray(order_output, dtype=np.float64)

    def plv_backend(
        _phases_flat: FloatArray,
        _t: int,
        _n: int,
        _window: int,
    ) -> FloatArray:
        """Return the configured flattened PLV payload."""
        return np.ascontiguousarray(plv_output, dtype=np.float64)

    def load_backend() -> tuple[MarketOrderFn, MarketPLVFn]:
        """Return the configured optional backend callables."""
        return order_backend, plv_backend

    monkeypatch.setattr(market_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(market_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(market_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setattr(market_mod, "_LOADERS", {"rust": load_backend})


def test_public_order_parameter_rejects_malformed_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject non-finite optional backend order parameters at the public API."""
    _install_optional_backend(
        monkeypatch,
        order_output=np.array([0.25, np.nan, 0.75], dtype=np.float64),
        plv_output=np.array([1.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="order parameter"):
        market_order_parameter(np.zeros((3, 2), dtype=np.float64))


def test_public_plv_rejects_nonphysical_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject asymmetric optional backend PLV matrices at the public API."""
    _install_optional_backend(
        monkeypatch,
        order_output=np.ones(4, dtype=np.float64),
        plv_output=np.array([1.0, 0.2, 0.4, 1.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="symmetric"):
        market_plv(np.zeros((4, 2), dtype=np.float64), window=4)


def test_public_dispatch_falls_back_to_python_after_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall through to the Python floor when optional backend loading fails."""

    def fail_loader() -> tuple[MarketOrderFn, MarketPLVFn]:
        """Raise the optional-backend unavailability signal."""
        raise ImportError("compiled market backend unavailable")

    monkeypatch.setattr(market_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(market_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(market_mod, "AVAILABLE_BACKENDS", ["python"])
    monkeypatch.setattr(market_mod, "_LOADERS", {"rust": fail_loader})

    result = market_order_parameter(np.zeros((2, 2), dtype=np.float64))

    np.testing.assert_allclose(result, np.ones(2, dtype=np.float64))
