# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NaN/Inf edge case tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.binding.types import BoundaryDef
from scpn_phase_orchestrator.coupling.geometry_constraints import validate_knm
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import (
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

# -- UPDEEngine.step() --


class TestUPDENanInf:
    def _make_engine(self, n: int = 4) -> UPDEEngine:
        return UPDEEngine(n_oscillators=n, dt=0.01)

    def test_nan_phases_rejected(self) -> None:
        """NaN phases rejected by Rust FFI, or propagate through Python path."""
        eng = self._make_engine()
        phases = np.array([0.0, np.nan, 1.0, 2.0])
        omegas = np.ones(4)
        knm = np.zeros((4, 4))
        alpha = np.zeros((4, 4))
        try:
            result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            assert np.any(np.isnan(result))
        except ValueError:
            pass  # Rust FFI rejects NaN at boundary

    def test_inf_phases_rejected(self) -> None:
        """Inf phases rejected by Rust FFI, or propagate through Python path."""
        eng = self._make_engine()
        phases = np.array([0.0, np.inf, 1.0, 2.0])
        omegas = np.ones(4)
        knm = np.zeros((4, 4))
        alpha = np.zeros((4, 4))
        try:
            result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            assert not np.all(np.isfinite(result))
        except ValueError:
            pass  # Rust FFI rejects Inf at boundary

    def test_nan_zeta_raises(self) -> None:
        eng = self._make_engine()
        phases = np.linspace(0, 3, 4)
        omegas = np.ones(4)
        knm = np.zeros((4, 4))
        alpha = np.zeros((4, 4))
        with pytest.raises(ValueError, match="finite"):
            eng.step(phases, omegas, knm, np.nan, 0.0, alpha)

    def test_inf_psi_raises(self) -> None:
        eng = self._make_engine()
        phases = np.linspace(0, 3, 4)
        omegas = np.ones(4)
        knm = np.zeros((4, 4))
        alpha = np.zeros((4, 4))
        with pytest.raises(ValueError, match="finite"):
            eng.step(phases, omegas, knm, 0.0, np.inf, alpha)

    def test_nan_omegas_rejected(self) -> None:
        """NaN omegas rejected by Rust FFI, or propagate through Python path."""
        eng = self._make_engine()
        phases = np.linspace(0, 3, 4)
        omegas = np.array([1.0, np.nan, 1.0, 1.0])
        knm = np.zeros((4, 4))
        alpha = np.zeros((4, 4))
        try:
            result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            assert np.isnan(result[1])
        except ValueError:
            pass  # Rust FFI rejects NaN at boundary

    def test_nan_knm_rejected(self) -> None:
        """NaN in Knm rejected by Rust FFI, or propagate through Python path."""
        eng = self._make_engine()
        phases = np.linspace(0, 3, 4)
        omegas = np.ones(4)
        knm = np.zeros((4, 4))
        knm[0, 1] = np.nan
        alpha = np.zeros((4, 4))
        try:
            result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            assert np.any(np.isnan(result))
        except ValueError:
            pass  # Rust FFI rejects NaN at boundary


# -- order_params --


class TestOrderParamsNanInf:
    def test_nan_phases(self) -> None:
        phases = np.array([0.0, np.nan, 1.0])
        r, psi = compute_order_parameter(phases)
        assert np.isnan(r) or np.isnan(psi), "NaN input should produce NaN output"

    def test_inf_phases(self) -> None:
        phases = np.array([0.0, np.inf, 1.0])
        r, psi = compute_order_parameter(phases)
        assert np.isnan(r) or np.isnan(psi), "Inf input should produce NaN output"

    def test_plv_nan_input(self) -> None:
        a = np.array([0.0, np.nan, 1.0])
        b = np.array([0.0, 1.0, 2.0])
        plv = compute_plv(a, b)
        assert np.isnan(plv), "NaN in PLV input should produce NaN"

    def test_plv_inf_input(self) -> None:
        a = np.array([0.0, np.inf, 1.0])
        b = np.array([0.0, 1.0, 2.0])
        plv = compute_plv(a, b)
        assert np.isnan(plv), "Inf in PLV input should produce NaN"

    def test_layer_coherence_nan(self) -> None:
        phases = np.array([0.0, np.nan, 1.0, 2.0])
        mask = np.array([True, True, False, False])
        r = compute_layer_coherence(phases, mask)
        assert np.isnan(r), "NaN in masked phases should produce NaN coherence"


# -- BoundaryObserver --


class TestBoundaryNanInf:
    def _make_observer(self) -> BoundaryObserver:
        defs = [
            BoundaryDef(
                name="temp",
                variable="T",
                lower=0.0,
                upper=100.0,
                severity="hard",
            ),
        ]
        return BoundaryObserver(defs)

    def test_nan_value_no_violation(self) -> None:
        """NaN comparison returns False, so NaN values should not trigger violation."""
        obs = self._make_observer()
        state = obs.observe({"T": float("nan")})
        # NaN < 0.0 → False, NaN > 100.0 → False → no violation
        assert len(state.violations) == 0

    def test_inf_value_triggers_violation(self) -> None:
        obs = self._make_observer()
        state = obs.observe({"T": float("inf")})
        # inf > 100.0 → True
        assert len(state.violations) == 1
        assert len(state.hard_violations) == 1

    def test_neg_inf_triggers_violation(self) -> None:
        obs = self._make_observer()
        state = obs.observe({"T": float("-inf")})
        # -inf < 0.0 → True
        assert len(state.violations) == 1

    def test_missing_variable_ignored(self) -> None:
        obs = self._make_observer()
        state = obs.observe({"other": 42.0})
        assert len(state.violations) == 0


# -- CouplingBuilder --


class TestCouplingNanInf:
    def test_nan_base_strength_rejected(self) -> None:
        """Rust FFI rejects NaN base_strength; Python path propagates NaN."""
        builder = CouplingBuilder()
        try:
            cs = builder.build(n_layers=3, base_strength=float("nan"), decay_alpha=0.3)
            assert np.all(np.isnan(cs.knm) | (cs.knm == 0.0))
        except ValueError:
            pass  # Rust FFI rejects NaN

    def test_inf_base_strength_rejected(self) -> None:
        """Rust FFI rejects Inf base_strength; Python path propagates Inf."""
        builder = CouplingBuilder()
        try:
            cs = builder.build(n_layers=3, base_strength=float("inf"), decay_alpha=0.3)
            assert np.any(np.isinf(cs.knm))
        except ValueError:
            pass  # Rust FFI rejects Inf

    def test_nan_decay_alpha_rejected(self) -> None:
        """Rust FFI rejects NaN decay_alpha; Python path propagates NaN."""
        builder = CouplingBuilder()
        try:
            cs = builder.build(n_layers=3, base_strength=1.0, decay_alpha=float("nan"))
            assert np.any(np.isnan(cs.knm))
        except ValueError:
            pass  # Rust FFI rejects NaN


# -- ImprintModel --


class TestImprintNanInf:
    def test_nan_exposure(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=5.0)
        state = ImprintState(m_k=np.zeros(3), last_update=0.0)
        exposure = np.array([1.0, np.nan, 1.0])
        new = model.update(state, exposure, dt=1.0)
        assert np.isnan(new.m_k[1])

    def test_inf_exposure(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=5.0)
        state = ImprintState(m_k=np.zeros(3), last_update=0.0)
        exposure = np.array([1.0, np.inf, 1.0])
        new = model.update(state, exposure, dt=1.0)
        # clip to saturation
        assert new.m_k[1] == pytest.approx(5.0)

    def test_nan_dt(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=5.0)
        state = ImprintState(m_k=np.ones(3), last_update=0.0)
        exposure = np.ones(3)
        new = model.update(state, exposure, dt=float("nan"))
        assert np.any(np.isnan(new.m_k))

    def test_negative_decay_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ImprintModel(decay_rate=-0.1, saturation=5.0)

    def test_zero_saturation_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ImprintModel(decay_rate=0.1, saturation=0.0)


# -- validate_knm --


class TestValidateKnm:
    def test_valid_knm(self) -> None:
        knm = np.array([[0.0, 0.5], [0.5, 0.0]])
        validate_knm(knm)

    def test_not_square(self) -> None:
        with pytest.raises(ValueError, match="square"):
            validate_knm(np.zeros((2, 3)))

    def test_not_symmetric(self) -> None:
        knm = np.array([[0.0, 0.5], [0.3, 0.0]])
        with pytest.raises(ValueError, match="symmetric"):
            validate_knm(knm)

    def test_negative_entries(self) -> None:
        knm = np.array([[0.0, -0.5], [-0.5, 0.0]])
        with pytest.raises(ValueError, match="negative"):
            validate_knm(knm)

    def test_nonzero_diagonal(self) -> None:
        knm = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="diagonal"):
            validate_knm(knm)


# -- BoundaryDef validation --


class TestBoundaryDefValidation:
    def test_lower_exceeds_upper_rejected(self) -> None:
        with pytest.raises(ValueError, match="lower.*upper"):
            BoundaryDef(
                name="bad", variable="x", lower=10.0, upper=5.0, severity="hard"
            )

    def test_equal_lower_upper_rejected(self) -> None:
        with pytest.raises(ValueError, match="lower.*upper"):
            BoundaryDef(name="eq", variable="x", lower=5.0, upper=5.0, severity="soft")

    def test_none_bounds_accepted(self) -> None:
        bd = BoundaryDef(
            name="open", variable="x", lower=None, upper=None, severity="soft"
        )
        assert bd.name == "open"

    def test_valid_bounds_accepted(self) -> None:
        bd = BoundaryDef(
            name="ok", variable="x", lower=0.0, upper=100.0, severity="hard"
        )
        assert bd.lower == 0.0 and bd.upper == 100.0
