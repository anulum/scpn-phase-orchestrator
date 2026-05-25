# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public Python API facade

"""High-level Python facade for binding-spec simulations.

This module gives application code the same basic value as ``spo run`` without
requiring Click command invocation. The facade intentionally returns immutable,
typed result records and keeps execution local: no hardware actuation, no
network IO, and no policy side effects are performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

FloatArray = NDArray[np.float64]

__all__ = ["Orchestrator", "OrchestratorState"]


@dataclass(frozen=True)
class OrchestratorState:
    """Final state returned by :meth:`Orchestrator.run`."""

    spec_name: str
    steps: int
    phases: FloatArray
    omegas: FloatArray
    knm: FloatArray
    alpha: FloatArray
    order_parameter: float
    mean_phase: float
    sample_period_s: float

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serialisable summary without large matrices."""
        return {
            "spec_name": self.spec_name,
            "steps": self.steps,
            "oscillator_count": int(self.phases.size),
            "order_parameter": self.order_parameter,
            "mean_phase": self.mean_phase,
            "sample_period_s": self.sample_period_s,
        }


class Orchestrator:
    """High-level Python entry point for local binding-spec simulations."""

    def __init__(self, spec: BindingSpec):
        if not isinstance(spec, BindingSpec):
            raise TypeError(f"spec must be BindingSpec, got {spec!r}")
        self._validate_executable_spec(spec)
        self.spec = spec

    @classmethod
    def from_yaml(cls, path: str | Path) -> Orchestrator:
        """Load, validate, and construct an orchestrator from a YAML spec."""
        spec = load_binding_spec(Path(path))
        return cls(spec)

    def run(self, *, steps: int = 100, seed: int = 42) -> OrchestratorState:
        """Run a deterministic local Kuramoto simulation.

        The method mirrors the non-actuating local simulation core used by the
        CLI: coupling is derived from the binding spec, phases are seeded from
        ``seed``, and the returned state contains the final Kuramoto order
        parameter. Supervisor policy actions and live actuator writes are not
        executed by this facade.
        """
        steps = _nonnegative_int(steps, name="steps")
        seed = _nonnegative_int(seed, name="seed")
        n_osc = _oscillator_count(self.spec)
        coupling = CouplingBuilder().build(
            n_osc,
            self.spec.coupling.base_strength,
            self.spec.coupling.decay_alpha,
        )
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, 2.0 * np.pi, n_osc).astype(np.float64)
        omegas = np.asarray(self.spec.get_omegas(), dtype=np.float64)
        zeta = _initial_zeta(self.spec)
        psi = float(self.spec.drivers.physical.get("psi", 0.0))
        engine = UPDEEngine(n_osc, dt=self.spec.sample_period_s)
        final_phases = engine.run(
            phases,
            omegas,
            coupling.knm,
            zeta,
            psi,
            coupling.alpha,
            n_steps=steps,
        )
        order_parameter, mean_phase = compute_order_parameter(final_phases)
        return OrchestratorState(
            spec_name=self.spec.name,
            steps=steps,
            phases=final_phases,
            omegas=omegas,
            knm=coupling.knm.copy(),
            alpha=coupling.alpha.copy(),
            order_parameter=order_parameter,
            mean_phase=mean_phase,
            sample_period_s=self.spec.sample_period_s,
        )

    @staticmethod
    def _validate_executable_spec(spec: BindingSpec) -> None:
        errors = validate_binding_spec(spec)
        if errors:
            joined = "; ".join(str(error) for error in errors)
            raise ValueError(f"binding spec validation failed: {joined}")
        if spec.safety_tier != "research":
            raise ValueError(
                f"safety_tier={spec.safety_tier!r} is not executable by the "
                "local Python facade"
            )
        if spec.amplitude is not None:
            raise ValueError(
                "Orchestrator.run currently supports Kuramoto binding specs; "
                "use StuartLandauEngine directly for amplitude-mode specs"
            )
        _oscillator_count(spec)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    coerced = int(value)
    if coerced < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return coerced


def _oscillator_count(spec: BindingSpec) -> int:
    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    if n_osc < 1:
        raise ValueError("binding spec must define at least one oscillator")
    return n_osc


def _initial_zeta(spec: BindingSpec) -> float:
    return max(
        (
            float(config.get("zeta", 0.0))
            for config in spec.drivers.all_channel_configs().values()
        ),
        default=0.0,
    )
