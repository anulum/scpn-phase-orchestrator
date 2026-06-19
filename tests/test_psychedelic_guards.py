# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Psychedelic monitor validation and fallback guards

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import psychedelic as psychedelic_mod
from scpn_phase_orchestrator.monitor.psychedelic import (
    _validate_entropy_value,
    _validate_reduced_coupling,
    entropy_from_phases,
    reduce_coupling,
    simulate_psychedelic_trajectory,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine

_PHASES = np.array([0.1, 0.2], dtype=np.float64)
_OMEGAS = np.zeros(2, dtype=np.float64)
_KNM = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
_ALPHA = np.zeros((2, 2), dtype=np.float64)


class TestReduceCoupling:
    def test_rejects_non_coercible_matrix(self) -> None:
        knm = np.array([["a", "b"], ["c", "d"]], dtype=object)
        with pytest.raises(ValueError, match="finite 2-D matrix"):
            reduce_coupling(knm, 0.5)

    def test_rejects_non_square_matrix(self) -> None:
        with pytest.raises(ValueError, match="must be square"):
            reduce_coupling(np.zeros((2, 3)), 0.5)

    def test_numpy_fallback_scales_matrix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(psychedelic_mod, "_HAS_RUST_REDUCE", False)

        reduced = reduce_coupling(_KNM, 0.25)

        np.testing.assert_allclose(reduced, _KNM * 0.75)


class TestEntropyFromPhases:
    def test_rejects_non_coercible_phases(self) -> None:
        phases = np.array(["a", "b"], dtype=object)
        with pytest.raises(ValueError, match="finite 1-D phase vector"):
            entropy_from_phases(phases)

    def test_falls_back_to_numpy_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _failing(_phases: object, _bins: int) -> float:
            raise RuntimeError("simulated backend runtime failure")

        monkeypatch.setattr(psychedelic_mod, "_dispatch", lambda: _failing)

        entropy = entropy_from_phases(np.linspace(0.0, 6.0, 12))

        assert np.isfinite(entropy)
        assert entropy >= 0.0


class TestSimulateTrajectoryGuards:
    def _engine(self) -> UPDEEngine:
        return UPDEEngine(2, dt=0.01)

    def test_rejects_knm_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
            simulate_psychedelic_trajectory(
                self._engine(),
                _PHASES,
                _OMEGAS,
                np.zeros((3, 3)),
                _ALPHA,
                [0.0],
            )

    def test_rejects_boolean_reduction_schedule(self) -> None:
        with pytest.raises(TypeError, match="reduction_schedule must be a 1-D"):
            simulate_psychedelic_trajectory(
                self._engine(),
                _PHASES,
                _OMEGAS,
                _KNM,
                _ALPHA,
                True,  # type: ignore[arg-type]  # invalid on purpose
            )

    def test_rejects_negative_step_count(self) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            simulate_psychedelic_trajectory(
                self._engine(),
                _PHASES,
                _OMEGAS,
                _KNM,
                _ALPHA,
                [0.0],
                n_steps_per_level=-1,
            )


class TestOutputContracts:
    """Direct validation of untrusted reduced-coupling and entropy output."""

    def test_reduced_coupling_rejects_non_coercible_output(self) -> None:
        bad = np.array(["a", "b", "c", "d"], dtype=object)
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_reduced_coupling(bad, expected_shape=(2, 2))

    def test_entropy_value_rejects_non_coercible_output(self) -> None:
        with pytest.raises(ValueError, match="entropy output must be numeric"):
            _validate_entropy_value("not-a-number", n_bins=36)
