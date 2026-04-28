# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau binding tests

"""Tests for AmplitudeSpec in binding types, loader, and validator."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from scpn_phase_orchestrator.binding.loader import (
    BindingLoadError,
    load_binding_spec,
)
from scpn_phase_orchestrator.binding.types import AmplitudeSpec, BindingSpec
from scpn_phase_orchestrator.binding.validator import validate_binding_spec

_MINIMAL_SPEC = {
    "name": "test",
    "version": "0.1.0",
    "safety_tier": "research",
    "sample_period_s": 0.01,
    "control_period_s": 0.1,
    "layers": [{"name": "L0", "index": 0, "oscillator_ids": ["a"]}],
    "oscillator_families": {
        "fam": {"channel": "P", "extractor_type": "hilbert"},
    },
    "coupling": {"base_strength": 0.5, "decay_alpha": 0.3},
    "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
    "objectives": {"good_layers": [0], "bad_layers": []},
}


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


class TestAmplitudeSpecDataclass:
    def test_defaults(self) -> None:
        a = AmplitudeSpec(mu=1.0, epsilon=0.5)
        assert a.amp_coupling_strength == 0.0
        assert a.amp_coupling_decay == 0.3

    def test_all_fields(self) -> None:
        a = AmplitudeSpec(
            mu=2.0,
            epsilon=1.0,
            amp_coupling_strength=0.3,
            amp_coupling_decay=0.5,
        )
        assert a.mu == 2.0
        assert a.epsilon == 1.0

    def test_frozen(self) -> None:
        a = AmplitudeSpec(mu=1.0, epsilon=0.5)
        with pytest.raises(AttributeError):
            a.mu = 2.0  # type: ignore[misc]


class TestBindingSpecAmplitudeField:
    def test_none_by_default(self) -> None:
        spec = BindingSpec(
            name="test",
            version="0.1.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[],
            oscillator_families={},
            coupling=None,  # type: ignore[arg-type]
            drivers=None,  # type: ignore[arg-type]
            objectives=None,  # type: ignore[arg-type]
            boundaries=[],
            actuators=[],
        )
        assert spec.amplitude is None


class TestLoaderAmplitude:
    def test_no_amplitude_block(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, _MINIMAL_SPEC)
        spec = load_binding_spec(p)
        assert spec.amplitude is None

    def test_amplitude_block_parsed(self, tmp_path: Path) -> None:
        data = {
            **_MINIMAL_SPEC,
            "amplitude": {
                "mu": 1.5,
                "epsilon": 0.5,
                "amp_coupling_strength": 0.3,
                "amp_coupling_decay": 0.2,
            },
        }
        p = _write_yaml(tmp_path, data)
        spec = load_binding_spec(p)
        assert spec.amplitude is not None
        assert spec.amplitude.mu == 1.5
        assert spec.amplitude.epsilon == 0.5
        assert spec.amplitude.amp_coupling_strength == 0.3
        assert spec.amplitude.amp_coupling_decay == 0.2

    def test_amplitude_defaults(self, tmp_path: Path) -> None:
        data = {
            **_MINIMAL_SPEC,
            "amplitude": {
                "mu": 1.0,
                "epsilon": 0.5,
            },
        }
        p = _write_yaml(tmp_path, data)
        spec = load_binding_spec(p)
        assert spec.amplitude is not None
        assert spec.amplitude.amp_coupling_strength == 0.0
        assert spec.amplitude.amp_coupling_decay == 0.3

    def test_missing_mu_raises(self, tmp_path: Path) -> None:
        data = {**_MINIMAL_SPEC, "amplitude": {"epsilon": 0.5}}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(BindingLoadError, match="mu"):
            load_binding_spec(p)

    def test_missing_epsilon_raises(self, tmp_path: Path) -> None:
        data = {**_MINIMAL_SPEC, "amplitude": {"mu": 1.0}}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(BindingLoadError, match="epsilon"):
            load_binding_spec(p)


class TestValidatorAmplitude:
    def _make_spec(self, **amp_kwargs: object) -> BindingSpec:
        from scpn_phase_orchestrator.binding.types import (
            CouplingSpec,
            DriverSpec,
            HierarchyLayer,
            ObjectivePartition,
            OscillatorFamily,
        )

        return BindingSpec(
            name="test",
            version="0.1.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[HierarchyLayer("L0", 0, ["a"])],
            oscillator_families={
                "f": OscillatorFamily("P", "hilbert", {}),
            },
            coupling=CouplingSpec(0.5, 0.3, {}),
            drivers=DriverSpec({}, {}, {}),
            objectives=ObjectivePartition([0], []),
            boundaries=[],
            actuators=[],
            amplitude=AmplitudeSpec(**amp_kwargs),  # type: ignore[arg-type]
        )

    def test_valid_amplitude(self) -> None:
        spec = self._make_spec(mu=1.0, epsilon=0.5)
        errors = validate_binding_spec(spec)
        assert errors == []

    def test_negative_epsilon(self) -> None:
        spec = self._make_spec(mu=1.0, epsilon=-0.1)
        errors = validate_binding_spec(spec)
        assert any("epsilon" in e for e in errors)

    def test_inf_mu(self) -> None:
        spec = self._make_spec(mu=math.inf, epsilon=0.5)
        errors = validate_binding_spec(spec)
        assert any("mu" in e for e in errors)

    def test_nan_mu(self) -> None:
        spec = self._make_spec(mu=math.nan, epsilon=0.5)
        errors = validate_binding_spec(spec)
        assert any("mu" in e for e in errors)

    def test_negative_mu_valid(self) -> None:
        spec = self._make_spec(mu=-1.0, epsilon=0.5)
        errors = validate_binding_spec(spec)
        assert errors == []


class TestSLBindingPipelineWiring:
    """Pipeline: load SL domainpack → StuartLandauEngine → amplitudes."""

    def test_cardiac_domainpack_sl_pipeline(self):
        """Load cardiac_rhythm → build SL engine → run → amplitudes≥0."""
        import numpy as np

        from scpn_phase_orchestrator.coupling import CouplingBuilder
        from scpn_phase_orchestrator.upde.stuart_landau import (
            StuartLandauEngine,
        )

        spec = load_binding_spec(
            Path(__file__).parent.parent
            / "domainpacks"
            / "cardiac_rhythm"
            / "binding_spec.yaml"
        )
        assert spec.amplitude is not None
        n = sum(len(ly.oscillator_ids) for ly in spec.layers)
        cs = CouplingBuilder().build(
            n, spec.coupling.base_strength, spec.coupling.decay_alpha
        )
        eng = StuartLandauEngine(n, dt=spec.sample_period_s)
        rng = np.random.default_rng(0)
        state = np.concatenate(
            [
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0.5, 1.5, n),
            ]
        )
        mu = np.full(n, spec.amplitude.mu)
        for _ in range(200):
            state = eng.step(
                state,
                np.ones(n),
                mu,
                cs.knm,
                cs.knm * spec.amplitude.amp_coupling_strength,
                0.0,
                0.0,
                cs.alpha,
                epsilon=spec.amplitude.epsilon,
            )
        assert np.all(state[n:] >= 0.0), "SL amplitudes must be≥0"
        assert np.all(np.isfinite(state))
