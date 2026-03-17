# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coverage gap tests

"""Tests covering remaining coverage gaps across multiple modules."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator.adapters.quantum_control_bridge import (
    QuantumControlBridge,
)
from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec
from scpn_phase_orchestrator.binding.types import (
    AmplitudeSpec,
    ImprintSpec,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.supervisor.petri_net import Guard, Marking, PetriNet, Place
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyRule,
    _extract_metric,
    load_policy_rules,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

TWO_PI = 2.0 * np.pi


def _make_upde(r_values, **kwargs):
    layers = [LayerState(R=r, psi=0.0, **kwargs) for r in r_values]
    n = len(r_values)
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(n) if n else np.empty((0, 0)),
        stability_proxy=float(np.mean(r_values)) if r_values else 0.0,
        regime_id="NOMINAL",
    )


# ──────────────────────────────────────────────────────────────────────
# engine.py: force Python fallback paths (Rust is available on this
# machine, so we must set engine._rust = None to test Python code)
# ──────────────────────────────────────────────────────────────────────


def _py_engine(n, dt=0.01, method="euler", **kw):
    """Create a UPDEEngine forced to use Python backend."""
    engine = UPDEEngine(n, dt=dt, method=method, **kw)
    engine._rust = None
    return engine


class TestUPDEEnginePythonPath:
    def test_euler_step(self):
        n = 4
        engine = _py_engine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (n,)
        assert np.all(result >= 0.0)
        assert np.all(result < TWO_PI)

    def test_euler_with_zeta(self):
        n = 4
        engine = _py_engine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = engine.step(phases, omegas, knm, 1.0, np.pi / 2, alpha)
        assert np.all(result > 0.0)

    def test_rk4_step(self):
        n = 4
        engine = _py_engine(n, dt=0.01, method="rk4")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 2.0, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))

    def test_rk45_step(self):
        n = 4
        engine = _py_engine(n, dt=0.01, method="rk45")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 2.0, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))
        assert engine.last_dt > 0.0

    def test_rk45_with_zeta(self):
        n = 4
        engine = _py_engine(n, dt=0.01, method="rk45")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 2.0, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = engine.step(phases, omegas, knm, 0.5, 1.0, alpha)
        assert np.all(np.isfinite(result))

    def test_run_euler(self):
        n = 4
        engine = _py_engine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, 10)
        assert result.shape == (n,)
        assert np.all(result >= 0.0)
        assert np.all(result < TWO_PI)

    def test_run_rk4(self):
        n = 4
        engine = _py_engine(n, dt=0.01, method="rk4")
        phases = np.zeros(n)
        omegas = np.ones(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, 5)
        assert result.shape == (n,)

    def test_run_rk45(self):
        n = 4
        engine = _py_engine(n, dt=0.01, method="rk45")
        phases = np.zeros(n)
        omegas = np.ones(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, 5)
        assert result.shape == (n,)


class TestUPDEEngineValidation:
    def test_nan_zeta_raises(self):
        engine = _py_engine(4, dt=0.01)
        with pytest.raises(ValueError, match="finite"):
            engine.step(
                np.zeros(4),
                np.ones(4),
                np.zeros((4, 4)),
                float("nan"),
                0.0,
                np.zeros((4, 4)),
            )

    def test_inf_psi_raises(self):
        engine = _py_engine(4, dt=0.01)
        with pytest.raises(ValueError, match="finite"):
            engine.step(
                np.zeros(4),
                np.ones(4),
                np.zeros((4, 4)),
                0.0,
                float("inf"),
                np.zeros((4, 4)),
            )


# ──────────────────────────────────────────────────────────────────────
# physical.py: force Python fallback path for extract()
# ──────────────────────────────────────────────────────────────────────


class TestPhysicalExtractor:
    def test_invalid_signal_shape(self):
        ext = PhysicalExtractor()
        with pytest.raises(ValueError, match="1-D"):
            ext.extract(np.zeros((2, 3)), 1000.0)

    def test_single_sample_raises(self):
        ext = PhysicalExtractor()
        with pytest.raises(ValueError, match=">= 2"):
            ext.extract(np.array([1.0]), 1000.0)

    def test_zero_envelope_quality(self):
        from scipy.signal import hilbert

        signal = np.zeros(100)
        quality = PhysicalExtractor._envelope_quality(signal, hilbert(signal))
        assert quality == 0.0

    def test_extract_python_path(self, monkeypatch):
        """Force Python fallback by patching _HAS_RUST to False."""
        import scpn_phase_orchestrator.oscillators.physical as phys_mod

        monkeypatch.setattr(phys_mod, "_HAS_RUST", False)
        ext = PhysicalExtractor(node_id="py_test")
        t = np.arange(0, 0.5, 1.0 / 1000)
        signal = np.sin(TWO_PI * 10.0 * t)
        states = ext.extract(signal, 1000.0)
        assert len(states) == 1
        assert 0.0 <= states[0].theta < TWO_PI
        assert states[0].channel == "P"
        assert states[0].quality > 0.5


# ──────────────────────────────────────────────────────────────────────
# pac.py: force Python fallback for modulation_index, pac_matrix, pac_gate
# ──────────────────────────────────────────────────────────────────────


class TestPACPythonPath:
    def test_modulation_index_python(self, monkeypatch):
        """Block spo_kernel import so Python KL-divergence path runs."""
        import scpn_phase_orchestrator.upde.pac as pac_mod

        _has_attr = hasattr(__builtins__, "__import__")
        original_import = __builtins__.__import__ if _has_attr else __import__

        def _block_spo(name, *args, **kwargs):
            if name == "spo_kernel":
                raise ImportError("blocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _block_spo)

        rng = np.random.default_rng(42)
        theta = rng.uniform(0, TWO_PI, 200)
        amp = 1.0 + 0.5 * np.cos(theta)
        mi = pac_mod.modulation_index(theta, amp)
        assert 0.0 <= mi <= 1.0
        assert mi > 0.0  # correlated signal → nonzero MI

    def test_modulation_index_empty(self, monkeypatch):
        import scpn_phase_orchestrator.upde.pac as pac_mod

        _has_attr = hasattr(__builtins__, "__import__")
        original_import = __builtins__.__import__ if _has_attr else __import__

        def _block_spo(name, *args, **kwargs):
            if name == "spo_kernel":
                raise ImportError("blocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _block_spo)
        assert pac_mod.modulation_index(np.array([]), np.array([])) == 0.0

    def test_modulation_index_zero_amplitude(self, monkeypatch):
        import scpn_phase_orchestrator.upde.pac as pac_mod

        _has_attr = hasattr(__builtins__, "__import__")
        original_import = __builtins__.__import__ if _has_attr else __import__

        def _block_spo(name, *args, **kwargs):
            if name == "spo_kernel":
                raise ImportError("blocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _block_spo)
        theta = np.linspace(0, TWO_PI, 100)
        amp = np.zeros(100)
        assert pac_mod.modulation_index(theta, amp) == 0.0

    def test_pac_matrix_python(self, monkeypatch):
        import scpn_phase_orchestrator.upde.pac as pac_mod

        _has_attr = hasattr(__builtins__, "__import__")
        original_import = __builtins__.__import__ if _has_attr else __import__

        def _block_spo(name, *args, **kwargs):
            if name == "spo_kernel":
                raise ImportError("blocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _block_spo)

        rng = np.random.default_rng(7)
        n, T = 3, 50
        phases = rng.uniform(0, TWO_PI, (T, n))
        amps = np.abs(rng.standard_normal((T, n))) + 0.1
        mat = pac_mod.pac_matrix(phases, amps)
        assert mat.shape == (n, n)
        assert np.all(mat >= 0.0)

    def test_pac_matrix_dim_mismatch(self, monkeypatch):
        import scpn_phase_orchestrator.upde.pac as pac_mod

        _has_attr = hasattr(__builtins__, "__import__")
        original_import = __builtins__.__import__ if _has_attr else __import__

        def _block_spo(name, *args, **kwargs):
            if name == "spo_kernel":
                raise ImportError("blocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _block_spo)

        with pytest.raises(ValueError, match="2-D"):
            pac_mod.pac_matrix(np.zeros(10), np.zeros(10))

    def test_pac_gate(self):
        from scpn_phase_orchestrator.upde.pac import pac_gate

        assert pac_gate(0.5) is True
        assert pac_gate(0.2) is False
        assert pac_gate(0.3, threshold=0.3) is True


# ──────────────────────────────────────────────────────────────────────
# order_params.py: force Python fallback for compute_order_parameter, compute_plv
# ──────────────────────────────────────────────────────────────────────


class TestOrderParamsPythonPath:
    def test_compute_order_parameter_python(self, monkeypatch):
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "_HAS_RUST", False)
        phases = np.zeros(8)
        r, psi = op_mod.compute_order_parameter(phases)
        np.testing.assert_allclose(r, 1.0, atol=1e-10)

    def test_compute_plv_python(self, monkeypatch):
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "_HAS_RUST", False)
        a = np.zeros(10)
        b = np.zeros(10)
        plv = op_mod.compute_plv(a, b)
        np.testing.assert_allclose(plv, 1.0, atol=1e-10)

    def test_compute_plv_size_mismatch(self, monkeypatch):
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "_HAS_RUST", False)
        with pytest.raises(ValueError, match="equal-length"):
            op_mod.compute_plv(np.zeros(5), np.zeros(3))


# ──────────────────────────────────────────────────────────────────────
# knm.py: force Python fallback for build()
# ──────────────────────────────────────────────────────────────────────


class TestCouplingBuilderPythonPath:
    def test_build_python(self, monkeypatch):
        import scpn_phase_orchestrator.coupling.knm as knm_mod

        monkeypatch.setattr(knm_mod, "_HAS_RUST", False)
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder

        builder = CouplingBuilder()
        state = builder.build(4, 0.5, 0.3)
        assert state.knm.shape == (4, 4)
        assert np.all(np.diag(state.knm) == 0.0)
        assert state.active_template == "default"

    def test_build_with_amplitude_python(self, monkeypatch):
        import scpn_phase_orchestrator.coupling.knm as knm_mod

        monkeypatch.setattr(knm_mod, "_HAS_RUST", False)
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder

        builder = CouplingBuilder()
        state = builder.build_with_amplitude(4, 0.5, 0.3, 0.2, 0.1)
        assert state.knm_r is not None
        assert state.knm_r.shape == (4, 4)


# ──────────────────────────────────────────────────────────────────────
# cli.py: non-research safety tier, no-oscillators, amplitude mode,
#          Psi action, policy_rules, queuewaves check, scaffold bad name
# ──────────────────────────────────────────────────────────────────────


class TestCLIRun:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_run_non_research_safety_tier(self, runner, tmp_path):
        spec = {
            "name": "tier-test",
            "version": "1.0.0",
            "safety_tier": "clinical",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "5"])
        assert result.exit_code == 0
        assert "WARNING" in result.output

    def test_run_no_oscillators_exits_1(self, runner, tmp_path):
        spec = {
            "name": "empty-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": []},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "5"])
        assert result.exit_code != 0
        assert "no oscillators" in result.output

    def test_run_amplitude_mode(self, runner, tmp_path):
        spec = {
            "name": "sl-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
                {"name": "L2", "index": 1, "oscillator_ids": ["o2", "o3"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": [1]},
            "boundaries": [],
            "actuators": [],
            "amplitude": {
                "mu": 1.0,
                "epsilon": 1.0,
                "amp_coupling_strength": 0.1,
                "amp_coupling_decay": 0.3,
            },
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "25"])
        assert result.exit_code == 0
        assert "mean_amplitude" in result.output

    def test_run_with_protocol_net(self, runner, tmp_path):
        spec = {
            "name": "petri-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.01,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
            "protocol_net": {
                "places": ["warmup", "nominal"],
                "initial": {"warmup": 1},
                "place_regime": {"warmup": "NOMINAL", "nominal": "NOMINAL"},
                "transitions": [
                    {
                        "name": "start",
                        "inputs": [{"place": "warmup"}],
                        "outputs": [{"place": "nominal"}],
                        "guard": "stability_proxy > 0.0",
                    },
                ],
            },
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "5"])
        assert result.exit_code == 0

    def test_run_with_imprint_and_geometry(self, runner, tmp_path):
        spec = {
            "name": "imprint-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
            "imprint_model": {"decay_rate": 0.01, "saturation": 5.0},
            "geometry_prior": {"constraint_type": "symmetric_non_negative"},
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "5"])
        assert result.exit_code == 0

    def test_run_with_physical_driver(self, runner, tmp_path):
        spec = {
            "name": "driver-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {
                "physical": {"frequency": 10.0, "amplitude": 1.0, "zeta": 0.1},
                "informational": {},
                "symbolic": {},
            },
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "5"])
        assert result.exit_code == 0

    def test_run_with_informational_driver(self, runner, tmp_path):
        spec = {
            "name": "info-driver-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {
                "physical": {},
                "informational": {"cadence_hz": 5.0, "zeta": 0.1},
                "symbolic": {},
            },
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "5"])
        assert result.exit_code == 0

    def test_run_with_symbolic_driver(self, runner, tmp_path):
        spec = {
            "name": "sym-driver-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {
                "physical": {},
                "informational": {},
                "symbolic": {"sequence": [0.0, 1.57, 3.14], "zeta": 0.1},
            },
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "5"])
        assert result.exit_code == 0

    def test_run_with_policy_rules(self, runner, tmp_path):
        spec = {
            "name": "policy-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.01,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
                {"name": "L2", "index": 1, "oscillator_ids": ["o2", "o3"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": [1]},
            "boundaries": [
                {"name": "R_low", "variable": "R", "lower": 0.99, "severity": "hard"},
            ],
            "actuators": [],
        }
        spec_path = tmp_path / "binding_spec.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        policy = {
            "rules": [
                {
                    "name": "boost_on_degraded",
                    "regime": ["DEGRADED", "CRITICAL"],
                    "condition": {
                        "metric": "stability_proxy",
                        "op": "<",
                        "threshold": 0.9,
                    },
                    "action": {
                        "knob": "K",
                        "scope": "global",
                        "value": 0.05,
                        "ttl_s": 5.0,
                    },
                },
            ],
        }
        policy_path = tmp_path / "policy.yaml"
        policy_path.write_text(yaml.dump(policy), encoding="utf-8")

        result = runner.invoke(main, ["run", str(spec_path), "--steps", "20"])
        assert result.exit_code == 0

    def test_run_with_audit_log(self, runner, tmp_path):
        spec = {
            "name": "audit-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
        }
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")
        audit_path = str(tmp_path / "audit.jsonl")
        result = runner.invoke(
            main,
            ["run", str(spec_path), "--steps", "5", "--audit", audit_path],
        )
        assert result.exit_code == 0

    def test_scaffold_bad_name(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(main, ["scaffold", "bad name!"])
        assert result.exit_code != 0

    def test_run_with_zeta_action_ttl_expiry(self, runner, tmp_path):
        """Exercise zeta knob with TTL → zeta_ttl counts down → zeta = 0.0.

        Policy rules must live in a sibling policy.yaml file, not inline.
        ttl_s=0.03 with dt=0.01 → zeta_ttl=3 steps → expires on step 3.
        """
        spec = {
            "name": "zeta-ttl",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.01,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
                {"name": "L2", "index": 1, "oscillator_ids": ["o2", "o3"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": [1]},
            "boundaries": [],
            "actuators": [
                {
                    "name": "zeta_drive",
                    "knob": "zeta",
                    "scope": "global",
                    "limits": [0.0, 0.5],
                }
            ],
        }
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        policy = {
            "rules": [
                {
                    "name": "inject_zeta",
                    "regime": ["NOMINAL"],
                    "condition": {
                        "metric": "stability_proxy",
                        "op": "<",
                        "threshold": 2.0,
                    },
                    "action": {
                        "knob": "zeta",
                        "scope": "global",
                        "value": 0.1,
                        "ttl_s": 0.03,
                    },
                    "max_fires": 1,
                }
            ],
        }
        policy_path = tmp_path / "policy.yaml"
        policy_path.write_text(yaml.dump(policy), encoding="utf-8")

        result = runner.invoke(main, ["run", str(spec_path), "--steps", "20"])
        assert result.exit_code == 0

    def test_run_with_psi_action(self, runner, tmp_path):
        """Exercise the Psi knob action path.

        Policy rules must live in a sibling policy.yaml file.
        """
        spec = {
            "name": "psi-knob",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.01,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
                {"name": "L2", "index": 1, "oscillator_ids": ["o2", "o3"]},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": [1]},
            "boundaries": [],
            "actuators": [
                {
                    "name": "psi_drive",
                    "knob": "Psi",
                    "scope": "global",
                    "limits": [0.0, 6.28],
                }
            ],
        }
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        policy = {
            "rules": [
                {
                    "name": "set_psi",
                    "regime": ["NOMINAL"],
                    "condition": {
                        "metric": "stability_proxy",
                        "op": "<",
                        "threshold": 2.0,
                    },
                    "action": {
                        "knob": "Psi",
                        "scope": "global",
                        "value": 1.57,
                        "ttl_s": 5.0,
                    },
                    "max_fires": 1,
                }
            ],
        }
        policy_path = tmp_path / "policy.yaml"
        policy_path.write_text(yaml.dump(policy), encoding="utf-8")

        result = runner.invoke(main, ["run", str(spec_path), "--steps", "20"])
        assert result.exit_code == 0

    def test_run_empty_layer_oscillators(self, runner, tmp_path):
        """Layer with no oscillator_ids hits the else: r, psi_l = 0.0, 0.0 branch."""
        spec = {
            "name": "empty-layer",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
                {"name": "L2", "index": 1, "oscillator_ids": []},
            ],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": [1]},
            "boundaries": [],
            "actuators": [],
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec), encoding="utf-8")
        result = runner.invoke(main, ["run", str(path), "--steps", "5"])
        assert result.exit_code == 0

    def test_report_empty_log_exits_1(self, runner, tmp_path):
        log = tmp_path / "empty.jsonl"
        log.write_text(json.dumps({"event": "x"}) + "\n", encoding="utf-8")
        result = runner.invoke(main, ["report", str(log)])
        assert result.exit_code != 0

    def test_report_with_actions(self, runner, tmp_path):
        entries = []
        for i in range(5):
            e = {
                "step": i,
                "regime": "DEGRADED",
                "stability": 0.4,
                "layers": [{"R": 0.4, "psi": 0.0}],
                "actions": [{"knob": "K", "scope": "global", "value": 0.05}],
            }
            entries.append(e)
        log = tmp_path / "actions.jsonl"
        log.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["report", str(log)])
        assert result.exit_code == 0
        assert "Actions fired:" in result.output


class TestCLIQueuewavesCheck:
    def test_queuewaves_check_with_anomalies(self, tmp_path):
        cfg = {
            "prometheus_url": "http://localhost:9090",
            "services": [
                {"name": "svc-a", "promql": "up", "layer": "micro"},
                {"name": "svc-b", "promql": "up", "layer": "macro"},
            ],
            "scrape_interval_s": 1.0,
            "buffer_length": 16,
        }
        path = tmp_path / "qw.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["queuewaves", "check", "--config", str(path)])
        assert result.exit_code in (0, 1)
        assert "R_good" in result.output

    def test_queuewaves_check_no_anomalies(self, tmp_path):
        """Permissive thresholds ensure no anomalies → line 622 covered."""
        cfg = {
            "prometheus_url": "http://localhost:9090",
            "services": [
                {"name": "svc-a", "promql": "up", "layer": "macro"},
            ],
            "scrape_interval_s": 1.0,
            "buffer_length": 16,
            "thresholds": {
                "r_bad_warn": 99.0,
                "r_bad_critical": 99.0,
                "plv_cascade": 99.0,
                "imprint_chronic": 99.0,
            },
        }
        path = tmp_path / "qw.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["queuewaves", "check", "--config", str(path)])
        assert result.exit_code == 0
        assert "No anomalies detected" in result.output


class TestCLIQueuewavesServe:
    def test_queuewaves_serve_imports(self, tmp_path, monkeypatch):
        """Verify the serve command imports run_server without actually running."""
        cfg = {
            "prometheus_url": "http://localhost:9090",
            "services": [
                {"name": "svc-a", "promql": "up", "layer": "micro"},
            ],
            "scrape_interval_s": 1.0,
            "buffer_length": 16,
        }
        path = tmp_path / "qw.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")

        mock_run = MagicMock()
        monkeypatch.setattr(
            "scpn_phase_orchestrator.apps.queuewaves.server.run_server",
            mock_run,
        )
        runner = CliRunner()
        result = runner.invoke(main, ["queuewaves", "serve", "--config", str(path)])
        assert result.exit_code == 0
        mock_run.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# cli.py replay: SL chained replay + failed determinism
# ──────────────────────────────────────────────────────────────────────


class TestCLIReplay:
    def test_replay_sl_chained(self, tmp_path):
        n = 2
        engine = StuartLandauEngine(n, dt=0.01)
        engine._use_rust = False
        state = np.array([0.1, 0.2, 0.8, 0.9])
        omegas = np.ones(n)
        mu = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        knm_r = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm_r, 0.0)
        alpha = np.zeros((n, n))

        entries = [
            {
                "header": True,
                "n_oscillators": n,
                "dt": 0.01,
                "seed": 0,
                "amplitude_mode": True,
            }
        ]
        for _ in range(5):
            entries.append(
                {
                    "step": len(entries) - 1,
                    "phases": state.tolist(),
                    "omegas": omegas.tolist(),
                    "knm": knm.tolist(),
                    "alpha": alpha.tolist(),
                    "mu": mu.tolist(),
                    "knm_r": knm_r.tolist(),
                    "zeta": 0.0,
                    "psi_drive": 0.0,
                }
            )
            state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

        log = tmp_path / "sl_replay.jsonl"
        log.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(main, ["replay", str(log), "--verify"])
        assert result.exit_code == 0
        assert "verified" in result.output.lower()

    def test_replay_failed_determinism(self, tmp_path):
        n = 2
        engine = StuartLandauEngine(n, dt=0.01)
        engine._use_rust = False
        state = np.array([0.1, 0.2, 0.8, 0.9])
        omegas = np.ones(n)
        mu = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        knm_r = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm_r, 0.0)
        alpha = np.zeros((n, n))

        entries = [
            {
                "header": True,
                "n_oscillators": n,
                "dt": 0.01,
                "seed": 0,
                "amplitude_mode": True,
            }
        ]
        for _ in range(5):
            entries.append(
                {
                    "step": len(entries) - 1,
                    "phases": state.tolist(),
                    "omegas": omegas.tolist(),
                    "knm": knm.tolist(),
                    "alpha": alpha.tolist(),
                    "mu": mu.tolist(),
                    "knm_r": knm_r.tolist(),
                    "zeta": 0.0,
                    "psi_drive": 0.0,
                }
            )
            state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

        # Tamper with entry to fail verification
        entries[3]["phases"][0] += 10.0
        log = tmp_path / "sl_tampered.jsonl"
        log.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(main, ["replay", str(log), "--verify"])
        assert result.exit_code == 1
        assert "FAILED" in result.output


# ──────────────────────────────────────────────────────────────────────
# policy_rules.py: compound conditions, R_bad/R_good, amplitude metrics
# ──────────────────────────────────────────────────────────────────────


class TestPolicyRules:
    def test_extract_r_bad(self):
        cond = PolicyCondition(metric="R_bad", layer=0, op=">", threshold=0.5)
        state = _make_upde([0.8, 0.6])
        val = _extract_metric(cond, state, [0], [1])
        assert val == pytest.approx(0.6)

    def test_extract_r_bad_none_when_out_of_range(self):
        cond = PolicyCondition(metric="R_bad", layer=5, op=">", threshold=0.5)
        state = _make_upde([0.8])
        val = _extract_metric(cond, state, [0], [])
        assert val is None

    def test_extract_r_good_none_when_no_layer(self):
        cond = PolicyCondition(metric="R_good", layer=None, op=">", threshold=0.5)
        state = _make_upde([0.8])
        val = _extract_metric(cond, state, [0], [])
        assert val is None

    def test_extract_pac_max(self):
        cond = PolicyCondition(metric="pac_max", layer=None, op=">", threshold=0.1)
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.5,
            regime_id="NOMINAL",
            pac_max=0.3,
        )
        val = _extract_metric(cond, state, [0], [])
        assert val == pytest.approx(0.3)

    def test_extract_mean_amplitude(self):
        cond = PolicyCondition(
            metric="mean_amplitude", layer=None, op=">", threshold=0.1
        )
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.5,
            regime_id="NOMINAL",
            mean_amplitude=0.7,
        )
        val = _extract_metric(cond, state, [], [])
        assert val == pytest.approx(0.7)

    def test_extract_subcritical_fraction(self):
        cond = PolicyCondition(
            metric="subcritical_fraction", layer=None, op="<", threshold=0.5
        )
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.5,
            regime_id="NOMINAL",
            subcritical_fraction=0.2,
        )
        val = _extract_metric(cond, state, [], [])
        assert val == pytest.approx(0.2)

    def test_extract_amplitude_spread(self):
        cond = PolicyCondition(
            metric="amplitude_spread", layer=0, op="<", threshold=1.0
        )
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0, amplitude_spread=0.3)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.5,
            regime_id="NOMINAL",
        )
        val = _extract_metric(cond, state, [], [])
        assert val == pytest.approx(0.3)

    def test_extract_mean_amplitude_layer(self):
        cond = PolicyCondition(
            metric="mean_amplitude_layer", layer=0, op=">", threshold=0.0
        )
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0, mean_amplitude=0.6)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.5,
            regime_id="NOMINAL",
        )
        val = _extract_metric(cond, state, [], [])
        assert val == pytest.approx(0.6)

    def test_extract_unknown_metric(self):
        cond = PolicyCondition(metric="bogus", layer=None, op=">", threshold=0.0)
        state = _make_upde([0.5])
        assert _extract_metric(cond, state, [], []) is None

    def test_eval_single_unknown_op(self):
        cond = PolicyCondition(
            metric="stability_proxy",
            layer=None,
            op="!=",
            threshold=0.5,
        )
        state = _make_upde([0.5])
        assert PolicyEngine._eval_single(cond, state, [], []) is False

    def test_compound_or_condition(self):
        cc = CompoundCondition(
            conditions=[
                PolicyCondition(
                    metric="stability_proxy",
                    layer=None,
                    op=">",
                    threshold=0.9,
                ),
                PolicyCondition(
                    metric="stability_proxy",
                    layer=None,
                    op="<",
                    threshold=0.1,
                ),
            ],
            logic="OR",
        )
        state = _make_upde([0.05])
        engine = PolicyEngine([])
        assert engine._check_condition(cc, state, [], []) is True

    def test_compound_and_condition_fails(self):
        cc = CompoundCondition(
            conditions=[
                PolicyCondition(
                    metric="stability_proxy",
                    layer=None,
                    op=">",
                    threshold=0.0,
                ),
                PolicyCondition(
                    metric="stability_proxy",
                    layer=None,
                    op=">",
                    threshold=0.9,
                ),
            ],
            logic="AND",
        )
        state = _make_upde([0.5])
        engine = PolicyEngine([])
        assert engine._check_condition(cc, state, [], []) is False

    def test_cooldown_prevents_repeated_fire(self):
        rule = PolicyRule(
            name="test",
            regimes=["NOMINAL"],
            condition=PolicyCondition(
                metric="stability_proxy", layer=None, op=">", threshold=0.0
            ),
            actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
            cooldown_s=100.0,
        )
        engine = PolicyEngine([rule])
        state = _make_upde([0.5])
        a1 = engine.evaluate(Regime.NOMINAL, state, [0], [])
        assert len(a1) == 1
        a2 = engine.evaluate(Regime.NOMINAL, state, [0], [])
        assert len(a2) == 0

    def test_max_fires_limit(self):
        rule = PolicyRule(
            name="limited",
            regimes=["NOMINAL"],
            condition=PolicyCondition(
                metric="stability_proxy", layer=None, op=">", threshold=0.0
            ),
            actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
            max_fires=2,
        )
        engine = PolicyEngine([rule])
        state = _make_upde([0.5])
        engine.evaluate(Regime.NOMINAL, state, [], [])
        engine.evaluate(Regime.NOMINAL, state, [], [])
        a3 = engine.evaluate(Regime.NOMINAL, state, [], [])
        assert len(a3) == 0

    def test_load_compound_rules(self, tmp_path):
        data = {
            "rules": [
                {
                    "name": "compound_test",
                    "regime": ["NOMINAL"],
                    "conditions": [
                        {"metric": "stability_proxy", "op": ">", "threshold": 0.0},
                        {"metric": "stability_proxy", "op": "<", "threshold": 1.0},
                    ],
                    "logic": "AND",
                    "actions": [
                        {"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 5.0},
                    ],
                },
            ],
        }
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.dump(data), encoding="utf-8")
        rules = load_policy_rules(path)
        assert len(rules) == 1
        assert isinstance(rules[0].condition, CompoundCondition)

    def test_load_empty_rules(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("not_rules: true\n", encoding="utf-8")
        rules = load_policy_rules(path)
        assert rules == []

    def test_extract_r_layer_out_of_bounds(self):
        cond = PolicyCondition(metric="R", layer=99, op=">", threshold=0.0)
        state = _make_upde([0.5])
        assert _extract_metric(cond, state, [], []) is None

    def test_extract_amplitude_spread_out_of_bounds(self):
        cond = PolicyCondition(
            metric="amplitude_spread", layer=99, op="<", threshold=1.0
        )
        state = _make_upde([0.5])
        assert _extract_metric(cond, state, [], []) is None

    def test_extract_mean_amplitude_layer_out_of_bounds(self):
        cond = PolicyCondition(
            metric="mean_amplitude_layer", layer=99, op=">", threshold=0.0
        )
        state = _make_upde([0.5])
        assert _extract_metric(cond, state, [], []) is None


# ──────────────────────────────────────────────────────────────────────
# supervisor/policy.py: petri_adapter path
# ──────────────────────────────────────────────────────────────────────


class TestSupervisorPolicyPetriPath:
    def test_petri_adapter_path(self):
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter
        from scpn_phase_orchestrator.supervisor.petri_net import (
            Arc,
            Transition,
        )

        places = [Place("a"), Place("b")]
        trans = [
            Transition("t", inputs=[Arc("a")], outputs=[Arc("b")]),
        ]
        net = PetriNet(places, trans)
        adapter = PetriNetAdapter(
            net, Marking(tokens={"a": 1}), {"a": "NOMINAL", "b": "NOMINAL"}
        )
        mgr = RegimeManager(cooldown_steps=0)
        policy = SupervisorPolicy(mgr, petri_adapter=adapter)
        state = _make_upde([0.9])
        actions = policy.decide(state, BoundaryState(), petri_ctx={"R": 0.9})
        assert isinstance(actions, list)


# ──────────────────────────────────────────────────────────────────────
# petri_net.py: Guard with missing metric and unknown op
# ──────────────────────────────────────────────────────────────────────


class TestPetriNetGuard:
    def test_guard_missing_metric(self):
        g = Guard(metric="x", op=">", threshold=0.5)
        assert g.evaluate({}) is False

    def test_guard_unknown_op(self):
        g = Guard(metric="x", op="!=", threshold=0.5)
        assert g.evaluate({"x": 0.3}) is False

    def test_place_names_property(self):
        from scpn_phase_orchestrator.supervisor.petri_net import Arc, Transition

        places = [Place("a"), Place("b")]
        trans = [Transition("t", inputs=[Arc("a")], outputs=[Arc("b")])]
        net = PetriNet(places, trans)
        assert net.place_names == frozenset({"a", "b"})


# ──────────────────────────────────────────────────────────────────────
# quantum_control_bridge.py: circuit and statevector imports fail
# ──────────────────────────────────────────────────────────────────────


class TestQuantumControlBridge:
    def test_build_quantum_circuit_import_error(self):
        bridge = QuantumControlBridge(4)
        with pytest.raises(ImportError, match="scpn-quantum-control"):
            bridge.build_quantum_circuit(np.eye(4), np.ones(4), 1.0)

    def test_extract_phases_import_error(self):
        bridge = QuantumControlBridge(4)
        with pytest.raises(ImportError, match="scpn-quantum-control"):
            bridge.extract_phases_from_statevector(None)


# ──────────────────────────────────────────────────────────────────────
# audit/logger.py: phases without omegas raises ValueError
# ──────────────────────────────────────────────────────────────────────


class TestAuditLogger:
    def test_log_step_phases_without_omegas_raises(self, tmp_path):
        log = tmp_path / "bad.jsonl"
        logger = AuditLogger(log)
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.5,
            regime_id="NOMINAL",
        )
        with pytest.raises(RuntimeError, match="omegas"):
            logger.log_step(0, state, [], phases=np.zeros(4))
        logger.close()


# ──────────────────────────────────────────────────────────────────────
# audit/replay.py: SL chained with single entry, verify_determinism_sl divergence
# ──────────────────────────────────────────────────────────────────────


class TestReplayEdgeCases:
    def test_sl_chained_single_entry(self, tmp_path):
        engine = StuartLandauEngine(2, dt=0.01)
        entries = [
            {"header": True, "n_oscillators": 2, "dt": 0.01, "amplitude_mode": True},
            {
                "step": 0,
                "phases": [0.1, 0.2, 0.8, 0.9],
                "omegas": [1.0, 1.0],
                "knm": [[0, 0.3], [0.3, 0]],
                "alpha": [[0, 0], [0, 0]],
            },
        ]
        log = tmp_path / "sl_single.jsonl"
        log.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )
        re = ReplayEngine(log)
        loaded = re.load()
        passed, n = re.verify_determinism_sl_chained(engine, loaded)
        assert passed
        assert n == 0

    def test_sl_chained_detects_divergence(self, tmp_path):
        n = 2
        engine = StuartLandauEngine(n, dt=0.01)
        state = np.array([0.1, 0.2, 0.8, 0.9])
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        mu = np.ones(n)
        knm_r = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm_r, 0.0)

        entries = [
            {"header": True, "n_oscillators": n, "dt": 0.01, "amplitude_mode": True}
        ]
        for i in range(3):
            entries.append(
                {
                    "step": i,
                    "phases": state.tolist(),
                    "omegas": omegas.tolist(),
                    "knm": knm.tolist(),
                    "alpha": alpha.tolist(),
                    "mu": mu.tolist(),
                    "knm_r": knm_r.tolist(),
                    "zeta": 0.0,
                    "psi_drive": 0.0,
                }
            )
            state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

        # Tamper
        entries[2]["phases"][0] += 5.0
        log = tmp_path / "sl_tamper.jsonl"
        log.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )
        re = ReplayEngine(log)
        loaded = re.load()
        passed, n_v = re.verify_determinism_sl_chained(engine, loaded)
        assert not passed


# ──────────────────────────────────────────────────────────────────────
# binding/loader.py: protocol_net parsing, JSON parse error, YAML error
# ──────────────────────────────────────────────────────────────────────


class TestBindingLoader:
    def test_json_parse_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="JSON parse"):
            load_binding_spec(p)

    def test_yaml_parse_error(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(":\n  - :\n    -:", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="YAML parse"):
            load_binding_spec(p)

    def test_non_dict_top_level(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1,2,3]", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="expected mapping"):
            load_binding_spec(p)

    def test_protocol_net_loading(self, tmp_path):
        data = {
            "name": "pnet",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [{"name": "L1", "index": 0, "oscillator_ids": ["a"]}],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.5, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": []},
            "protocol_net": {
                "places": ["warmup", "nominal"],
                "initial": {"warmup": 1},
                "place_regime": {"warmup": "NOMINAL"},
                "transitions": [
                    {
                        "name": "start",
                        "inputs": [{"place": "warmup", "weight": 1}],
                        "outputs": [{"place": "nominal"}],
                        "guard": "R > 0.5",
                    },
                ],
            },
        }
        p = tmp_path / "spec.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        spec = load_binding_spec(p)
        assert spec.protocol_net is not None
        assert len(spec.protocol_net.transitions) == 1

    def test_amplitude_loading(self, tmp_path):
        data = {
            "name": "amp",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [{"name": "L1", "index": 0, "oscillator_ids": ["a"]}],
            "oscillator_families": {
                "p": {"channel": "P", "extractor_type": "hilbert"},
            },
            "coupling": {"base_strength": 0.5, "decay_alpha": 0.3},
            "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
            "objectives": {"good_layers": [0], "bad_layers": []},
            "amplitude": {"mu": 1.0, "epsilon": 0.5},
        }
        p = tmp_path / "spec.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        spec = load_binding_spec(p)
        assert spec.amplitude is not None
        assert spec.amplitude.mu == 1.0


# ──────────────────────────────────────────────────────────────────────
# binding/validator.py: boundary lower > upper, actuator scope mismatch,
#                        imprint validation, amplitude validation
# ──────────────────────────────────────────────────────────────────────


class TestBindingValidatorExtras:
    def test_boundary_lower_gt_upper(self, sample_binding_spec):
        from scpn_phase_orchestrator.binding.types import BoundaryDef

        with pytest.raises(ValueError, match="lower.*upper"):
            BoundaryDef(name="inv", variable="R", lower=0.9, upper=0.1, severity="hard")

    def test_actuator_scope_unknown(self, sample_binding_spec):
        from scpn_phase_orchestrator.binding.types import ActuatorMapping

        bad_act = [
            ActuatorMapping(
                name="bad_scope", knob="K", scope="layer_99", limits=(0.0, 1.0)
            ),
        ]
        bad = replace(sample_binding_spec, actuators=bad_act)
        errors = validate_binding_spec(bad)
        assert any("scope" in e for e in errors)

    def test_invalid_extractor_type(self, sample_binding_spec):
        from scpn_phase_orchestrator.binding.types import OscillatorFamily

        bad_fam = {
            "x": OscillatorFamily(
                channel="P",
                extractor_type="quantum",
                config={},
            ),
        }
        bad = replace(sample_binding_spec, oscillator_families=bad_fam)
        errors = validate_binding_spec(bad)
        assert any("extractor_type" in e for e in errors)

    def test_imprint_negative_decay(self, sample_binding_spec):
        imprint = ImprintSpec(decay_rate=-0.1, saturation=1.0, modulates=["K"])
        bad = replace(sample_binding_spec, imprint_model=imprint)
        errors = validate_binding_spec(bad)
        assert any("decay_rate" in e for e in errors)

    def test_imprint_zero_saturation(self, sample_binding_spec):
        imprint = ImprintSpec(decay_rate=0.1, saturation=0.0, modulates=["K"])
        bad = replace(sample_binding_spec, imprint_model=imprint)
        errors = validate_binding_spec(bad)
        assert any("saturation" in e for e in errors)

    def test_amplitude_non_finite_mu(self, sample_binding_spec):
        amp = AmplitudeSpec(mu=float("inf"), epsilon=1.0)
        bad = replace(sample_binding_spec, amplitude=amp)
        errors = validate_binding_spec(bad)
        assert any("mu" in e for e in errors)

    def test_amplitude_negative_epsilon(self, sample_binding_spec):
        amp = AmplitudeSpec(mu=1.0, epsilon=-0.1)
        bad = replace(sample_binding_spec, amplitude=amp)
        errors = validate_binding_spec(bad)
        assert any("epsilon" in e for e in errors)


# ──────────────────────────────────────────────────────────────────────
# stuart_landau.py: rk45 path
# ──────────────────────────────────────────────────────────────────────


def _py_sl_engine(n, dt=0.01, method="euler", **kw):
    """Create a StuartLandauEngine forced to use Python backend."""
    engine = StuartLandauEngine(n, dt=dt, method=method, **kw)
    engine._use_rust = False
    return engine


class TestStuartLandauPythonPath:
    def _base_params(self, n=4):
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        amps = np.ones(n) * 0.8
        state = np.concatenate([phases, amps])
        omegas = np.ones(n)
        mu = np.full(n, 1.0)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        knm_r = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm_r, 0.0)
        alpha = np.zeros((n, n))
        return state, omegas, mu, knm, knm_r, alpha

    def test_sl_euler_python(self):
        n = 4
        engine = _py_sl_engine(n, dt=0.01)
        state, omegas, mu, knm, knm_r, alpha = self._base_params(n)
        state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(state))

    def test_sl_rk4_python(self):
        n = 4
        engine = _py_sl_engine(n, dt=0.01, method="rk4")
        state, omegas, mu, knm, knm_r, alpha = self._base_params(n)
        state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(state))

    def test_sl_rk45_python(self):
        n = 4
        engine = _py_sl_engine(n, dt=0.01, method="rk45")
        state, omegas, mu, knm, knm_r, alpha = self._base_params(n)
        for _ in range(50):
            state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(state))
        assert np.all(state[:n] >= 0.0)
        assert np.all(state[:n] < TWO_PI)
        assert np.all(state[n:] >= 0.0)

    def test_sl_knm_r_shape_mismatch(self):
        n = 4
        engine = _py_sl_engine(n, dt=0.01)
        state, omegas, mu, knm, _, alpha = self._base_params(n)
        with pytest.raises(ValueError, match="knm_r.shape"):
            engine.step(state, omegas, mu, knm, np.zeros((3, 3)), 0.0, 0.0, alpha)


# ──────────────────────────────────────────────────────────────────────
# reporting/plots.py: _require_matplotlib guard
# ──────────────────────────────────────────────────────────────────────


class TestPlotsRequireGuard:
    def test_require_matplotlib_raises_when_absent(self, monkeypatch):
        import scpn_phase_orchestrator.reporting.plots as plots_mod

        monkeypatch.setattr(plots_mod, "_HAS_MPL", False)
        with pytest.raises(ImportError, match="matplotlib"):
            plots_mod._require_matplotlib()
