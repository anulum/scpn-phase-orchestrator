# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — stateless UPDE dispatcher validation contracts

"""Fail-closed contracts for public stateless UPDE dispatch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_phase_orchestrator.upde._engine_validation as engine_validation
from scpn_phase_orchestrator.upde import _run as run_mod

FloatArray = NDArray[np.float64]


def _payload() -> list[Any]:
    """Return a valid public ``upde_run`` argument payload."""
    return [
        np.array([0.1, 0.2], dtype=np.float64),
        np.array([1.0, 1.2], dtype=np.float64),
        np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        0.0,
        0.0,
        0.01,
        1,
        "euler",
        1,
        1e-6,
        1e-3,
    ]


def _identity_backend(phases: FloatArray, *_args: object) -> FloatArray:
    """Return a copy of the supplied phases for dispatch-boundary tests."""
    return phases.copy()


def test_core_engine_validator_is_directly_linked() -> None:
    """Keep the core-owned validator visible to module-linkage checks."""
    assert callable(engine_validation.validate_upde_backend_inputs)
    assert callable(engine_validation.validate_upde_backend_output)
    assert callable(engine_validation.validate_upde_schedule_backend_inputs)


@pytest.mark.parametrize(
    ("index", "replacement", "match"),
    [
        (0, np.array(["0.1", "0.2"]), "phases"),
        (0, np.array([], dtype=np.float64), "phases"),
        (1, np.array(["1.0", "1.2"]), "omegas"),
        (2, np.array([["0", "0.3"], ["0.3", "0"]]), "knm"),
        (2, np.zeros(3, dtype=np.float64), "knm"),
        (3, np.array([["0", "0"], ["0", "0"]]), "alpha"),
        (3, np.zeros((1, 2, 2), dtype=np.float64), "alpha"),
        (4, "0.0", "zeta"),
        (5, True, "psi"),
        (6, "0.01", "dt"),
        (7, "1", "n_steps"),
        (8, 1, "method"),
        (9, True, "n_substeps"),
        (9, 1.5, "n_substeps"),
        (10, "1e-6", "atol"),
        (11, "1e-3", "rtol"),
    ],
)
def test_public_run_rejects_aliases_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    index: int,
    replacement: object,
    match: str,
) -> None:
    """Reject coercible aliases before an optional backend sees them."""
    dispatched = False

    def _backend(*_args: object) -> FloatArray:
        nonlocal dispatched
        dispatched = True
        return np.zeros(2, dtype=np.float64)

    payload = _payload()
    payload[index] = replacement
    monkeypatch.setattr(run_mod, "_dispatch", lambda: _backend)

    with pytest.raises((TypeError, ValueError), match=match):
        run_mod.upde_run(*payload)

    assert not dispatched


def test_public_schedule_rejects_numeric_string_aliases_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a coercible frequency schedule before backend selection."""
    payload = _payload()
    dispatched = False

    def _backend(*_args: object) -> FloatArray:
        nonlocal dispatched
        dispatched = True
        return np.zeros(2, dtype=np.float64)

    monkeypatch.setattr(run_mod, "_dispatch_schedule", lambda: _backend)

    with pytest.raises(TypeError, match="omega_schedule"):
        run_mod.upde_run_omega_schedule(
            payload[0],
            np.array([["1.0", "1.2"]]),
            payload[2],
            payload[3],
            payload[4],
            payload[5],
            payload[6],
            payload[8],
            payload[9],
            payload[10],
            payload[11],
        )

    assert not dispatched


@pytest.mark.parametrize(
    "backend_output",
    [
        np.array(["0.1", "0.2"]),
        np.array([True, False]),
        np.array([0.1, np.bool_(True)], dtype=object),
        np.array([0.1, np.inf]),
        np.array([0.1]),
        np.array([[0.1, 0.2]]),
    ],
)
def test_public_run_rejects_malformed_backend_outputs(
    monkeypatch: pytest.MonkeyPatch,
    backend_output: NDArray[Any],
) -> None:
    """Do not publish malformed optional-backend phase evidence."""
    monkeypatch.setattr(
        run_mod,
        "_dispatch",
        lambda: lambda *_args: backend_output,
    )

    with pytest.raises((TypeError, ValueError)):
        run_mod.upde_run(*_payload())


def test_public_schedule_rejects_numeric_string_backend_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the output contract to the schedule dispatcher too."""
    payload = _payload()
    monkeypatch.setattr(
        run_mod,
        "_dispatch_schedule",
        lambda: lambda *_args: np.array(["0.1", "0.2"]),
    )

    with pytest.raises(TypeError, match="result"):
        run_mod.upde_run_omega_schedule(
            payload[0],
            np.array([[1.0, 1.2]], dtype=np.float64),
            payload[2],
            payload[3],
            payload[4],
            payload[5],
            payload[6],
            payload[8],
            payload[9],
            payload[10],
            payload[11],
        )


def test_public_run_normalises_valid_array_likes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep ordinary real-valued array-like inputs supported."""
    payload = _payload()
    payload[0] = payload[0].tolist()
    payload[1] = payload[1].tolist()
    payload[2] = payload[2].ravel().tolist()
    payload[3] = payload[3].ravel().tolist()
    monkeypatch.setattr(run_mod, "_dispatch", lambda: _identity_backend)

    result = run_mod.upde_run(*payload)

    np.testing.assert_allclose(result, np.array([0.1, 0.2]))
    assert result.dtype == np.float64
