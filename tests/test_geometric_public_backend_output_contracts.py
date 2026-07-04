# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - geometric public backend output contracts

"""Public dispatcher contracts for geometric accelerator backend outputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import geometric as geometric_mod
from scpn_phase_orchestrator.upde.geometric import TorusEngine

FloatArray: TypeAlias = NDArray[np.float64]
TorusBackendFn: TypeAlias = Callable[
    [FloatArray, FloatArray, FloatArray, FloatArray, int, float, float, float, int],
    FloatArray,
]


def _install_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output: object,
) -> None:
    """Install a deterministic optional torus backend for public API tests."""

    def torus_backend(
        _phases: FloatArray,
        _omegas: FloatArray,
        _knm_flat: FloatArray,
        _alpha_flat: FloatArray,
        _n: int,
        _zeta: float,
        _psi: float,
        _dt: float,
        _n_steps: int,
    ) -> FloatArray:
        """Return the configured torus phase payload."""
        return cast(FloatArray, output)

    def load_backend() -> TorusBackendFn:
        """Return the configured optional backend callable."""
        return torus_backend

    monkeypatch.setattr(geometric_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(geometric_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(geometric_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setattr(geometric_mod, "_LOADERS", {"rust": load_backend})


def _run_public_engine() -> FloatArray:
    """Run a small public torus integration through the dispatcher."""
    engine = TorusEngine(n_oscillators=3, dt=0.01)
    return engine.run(
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        np.array([1.0, 0.5, -0.25], dtype=np.float64),
        np.zeros((3, 3), dtype=np.float64),
        0.0,
        0.0,
        np.zeros((3, 3), dtype=np.float64),
        n_steps=2,
    )


def test_public_torus_rejects_wrong_length_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject cardinality drift from optional torus backends."""
    _install_optional_backend(
        monkeypatch,
        output=np.array([0.1, 0.2], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="result must contain 3 values"):
        _run_public_engine()


def test_public_torus_rejects_non_finite_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject non-finite optional torus phases at the public API."""
    _install_optional_backend(
        monkeypatch,
        output=np.array([0.1, np.nan, 0.3], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="result must contain only finite values"):
        _run_public_engine()


def test_public_torus_rejects_out_of_domain_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject optional backend phases outside the torus domain."""
    _install_optional_backend(
        monkeypatch,
        output=np.array([0.1, geometric_mod.TWO_PI + 0.01, 0.3], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="result phases must lie in"):
        _run_public_engine()


def test_public_torus_rejects_numeric_string_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject optional torus backend output numeric-string aliases."""
    _install_optional_backend(
        monkeypatch,
        output=np.array(["0.4", "0.5", "0.6"], dtype=object),
    )

    with pytest.raises(ValueError, match="result.*numeric-string"):
        _run_public_engine()


def test_public_torus_accepts_valid_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return validated optional torus phases when the payload is physical."""
    expected = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    _install_optional_backend(monkeypatch, output=expected)

    np.testing.assert_allclose(_run_public_engine(), expected)


def test_public_torus_uses_python_floor_after_stale_backend_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the Python floor when stale optional backend metadata cannot load."""

    def fail_loader() -> TorusBackendFn:
        """Raise the optional-backend unavailability signal."""
        raise ImportError("compiled torus backend unavailable")

    monkeypatch.setattr(geometric_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(geometric_mod, "ACTIVE_BACKEND", "go")
    monkeypatch.setattr(geometric_mod, "AVAILABLE_BACKENDS", ["go"])
    monkeypatch.setattr(geometric_mod, "_LOADERS", {"go": fail_loader})

    np.testing.assert_allclose(
        _run_public_engine(),
        np.array([0.12, 0.21, 0.295], dtype=np.float64),
    )
