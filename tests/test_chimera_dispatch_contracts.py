# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera dispatch contract guards

"""Module-specific contracts for Chimera backend dispatch and validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import chimera as chimera_mod
from tests.test_chimera_real_runtime import installed_absent_probe

FloatArray: TypeAlias = NDArray[np.float64]
ChimeraBackend: TypeAlias = Callable[[FloatArray, FloatArray, int], FloatArray]


def _matrix() -> FloatArray:
    """Return a valid two-node zero-diagonal coupling matrix."""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


@pytest.mark.native_runtime
def test_optional_backend_loader_returns_mojo_callable() -> None:
    """Exercise the original mojo owner through the named public API."""
    phases = np.array([0.0, 0.0, np.pi])
    coupling = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    actual = chimera_mod.local_order_parameter(phases, coupling, backend="mojo")
    np.testing.assert_allclose(actual, [0.0, 0.0, 1.0], atol=1e-9, rtol=0)


@pytest.mark.native_runtime
def test_optional_backend_loader_returns_julia_callable() -> None:
    """Exercise the original julia owner through the named public API."""
    phases = np.array([0.0, 0.0, np.pi])
    coupling = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    actual = chimera_mod.local_order_parameter(phases, coupling, backend="julia")
    np.testing.assert_allclose(actual, [0.0, 0.0, 1.0], atol=1e-9, rtol=0)


@pytest.mark.native_runtime
def test_optional_backend_loader_returns_go_callable() -> None:
    """Exercise the original go owner through the named public API."""
    phases = np.array([0.0, 0.0, np.pi])
    coupling = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    actual = chimera_mod.local_order_parameter(phases, coupling, backend="go")
    np.testing.assert_allclose(actual, [0.0, 0.0, 1.0], atol=1e-9, rtol=0)


@pytest.mark.native_runtime
def test_resolve_backends_keeps_python_when_optional_loaders_fail() -> None:
    """A genuine installed runtime without native artifacts resolves Python."""
    record = installed_absent_probe()
    assert record["active"] == "python"
    assert record["available"] == ["python"]
    assert record["local"] == [0.0, 0.0]


@pytest.mark.native_runtime
def test_dispatch_returns_none_when_no_backend_survives() -> None:
    """Real absence preserves automatic output and strictly refuses named owners."""
    record = installed_absent_probe()
    assert record["incoherent"] == [0, 1]
    assert record["missing"] == ["rust", "mojo", "julia", "go"]


def test_chimera_state_rejects_string_index_sequence() -> None:
    """Index lists reject string-like payloads instead of iterating characters."""
    with pytest.raises(ValueError, match="coherent_indices"):
        chimera_mod.ChimeraState(coherent_indices=cast("list[int]", "01"))


def test_alias_probe_helpers_treat_unprobeable_payload_as_absent() -> None:
    """Public admission preserves real object values and refuses typed aliases."""
    p = np.array([0.0, 0.1], dtype=object)
    values = chimera_mod.local_order_parameter(
        cast(FloatArray, p), _matrix(), backend="python"
    )
    np.testing.assert_allclose(values, [1.0, 1.0], atol=1e-12, rtol=0)
    for alias in (True, 0.0 + 0.0j, "0.0"):
        with pytest.raises(ValueError):
            chimera_mod.local_order_parameter(
                cast(FloatArray, np.array([alias, 0.1], dtype=object)), _matrix()
            )


def test_complex_payload_probe_handles_uncoercible_values() -> None:
    """The real public API refuses ragged measurement sequences before dispatch."""
    ragged = cast(FloatArray, [[0.0], [0.0, 0.1]])
    with pytest.raises(ValueError):
        chimera_mod.local_order_parameter(ragged, _matrix())


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
        """Return deliberately invalid backend output for refusal qualification."""
        return ["bad", "payload"]

    monkeypatch.setattr(chimera_mod, "_dispatch", lambda backend=None: _bad_backend)

    with pytest.raises(ValueError, match="output must be numeric"):
        chimera_mod.local_order_parameter(
            np.array([0.0, 0.1], dtype=np.float64),
            _matrix(),
        )
