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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

if TYPE_CHECKING:
    from scpn_phase_orchestrator.runtime.simulation import SimulationResult

FloatArray = NDArray[np.float64]

__all__ = ["Orchestrator", "OrchestratorState", "evaluate_binding_spec"]


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


def evaluate_binding_spec(
    spec: BindingSpec | str | Path,
    *,
    steps: int = 100,
    seed: int = 42,
    policy_enabled: bool = True,
) -> SimulationResult:
    """Non-actuating evaluation of any binding spec, open or closed loop.

    Unlike :meth:`Orchestrator.run` (a Kuramoto, research-tier convenience
    facade), this evaluates the full-fidelity simulation core that backs
    ``spo run`` for *any* spec: Kuramoto or Stuart-Landau (amplitude), and any
    safety tier. It is review/simulation only — no hardware actuation, no network
    IO — so the safety-tier gate that blocks live ``spo run`` does not apply here.

    Args:
        spec: A validated :class:`BindingSpec`, or a path to a spec YAML. When a
            path is given, an adjacent ``policy.yaml`` is loaded for the
            closed-loop domainpack policy.
        steps: Number of integration steps.
        seed: RNG seed for the initial phases.
        policy_enabled: Closed-loop supervisor + policy control feedback on
            (``True``) or open-loop baseline (``False``). Running both on the
            same seed isolates the orchestration uplift.

    Returns:
        A ``SimulationResult`` (from ``runtime.simulation``) with the final
        per-objective order parameters, their separation, the regime, and the
        per-step coherence histories.

    Raises:
        ValueError: If the spec fails validation or defines no oscillators.
    """

    from scpn_phase_orchestrator.runtime.simulation import simulate

    spec_path: Path | None = None
    if isinstance(spec, (str, Path)):
        spec_path = Path(spec)
        spec = load_binding_spec(spec_path)
    elif not isinstance(spec, BindingSpec):
        raise TypeError(f"spec must be BindingSpec or path, got {spec!r}")

    errors = validate_binding_spec(spec)
    if errors:
        joined = "; ".join(str(error) for error in errors)
        raise ValueError(f"binding spec validation failed: {joined}")

    steps = _nonnegative_int(steps, name="steps")
    seed = _nonnegative_int(seed, name="seed")
    return simulate(
        spec,
        steps=steps,
        seed=seed,
        policy_enabled=policy_enabled,
        binding_spec_path=spec_path,
    )


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
