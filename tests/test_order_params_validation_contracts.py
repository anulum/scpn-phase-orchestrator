# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order-parameter validation contracts

"""Order-parameter backend validation and resolver contracts."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _order_params_julia as order_params_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _order_params_validation as order_params_validation,
)
from scpn_phase_orchestrator.upde import order_params


class _ArrayRaises:
    """Array-like sentinel that refuses NumPy coercion."""

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        """Raise during NumPy array coercion."""
        raise ValueError("array coercion refused")


def test_order_params_julia_resolver_rejects_runtime_without_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher must not advertise Julia when ``juliacall.Main`` is absent."""
    stub = ModuleType("juliacall")
    monkeypatch.setitem(sys.modules, "juliacall", stub)
    order_params._BACKEND_CACHE.clear()

    with pytest.raises(ImportError, match="juliacall.Main unavailable"):
        order_params._load_julia_fns()


def test_direct_julia_loader_rejects_runtime_without_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct Julia bridge loading must report an unavailable runtime cleanly."""
    stub = ModuleType("juliacall")
    monkeypatch.setitem(sys.modules, "juliacall", stub)
    monkeypatch.setattr(order_params_julia, "_JULIA_MODULE", None)

    with pytest.raises(ImportError, match="juliacall.Main unavailable"):
        order_params_julia._ensure_julia_loaded()


def test_boolean_alias_detector_treats_uncoercible_payload_as_non_alias() -> None:
    """Alias detection must fail open so typed validation errors stay precise."""
    assert order_params_validation._contains_boolean_alias(_ArrayRaises()) is False


def test_phase_vector_rejects_non_numeric_object_array() -> None:
    """Phase vectors must coerce to real numeric samples."""
    with pytest.raises(ValueError, match="phases must be numeric"):
        order_params_validation.validate_order_parameter_inputs(
            np.array([object()], dtype=object)
        )


def test_phase_vector_rejects_uncoercible_array_like_payload() -> None:
    """Phase vector validation must raise a typed error for bad array-likes."""
    with pytest.raises(ValueError, match="phases must be array-like"):
        order_params_validation.validate_order_parameter_inputs(_ArrayRaises())


def test_index_vector_rejects_non_vector_indices() -> None:
    """Layer indices must be one-dimensional before integer validation."""
    with pytest.raises(ValueError, match="indices must be a one-dimensional"):
        order_params_validation.validate_layer_coherence_inputs(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([[0]], dtype=np.int64),
        )


def test_finite_scalar_rejects_uncoercible_unit_interval_output() -> None:
    """Backend scalar outputs must be finite real numbers."""
    with pytest.raises(ValueError, match="PLV must be a finite real scalar"):
        order_params_validation.validate_unit_interval_output(object(), name="PLV")
