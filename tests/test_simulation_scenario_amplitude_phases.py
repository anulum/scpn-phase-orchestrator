# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — scenario phase edits reach the Stuart-Landau state

"""A scenario hook's phase edit must reach the amplitude-mode integrator.

``simulate`` integrates ``sl_state`` (phases and amplitudes) in amplitude
mode, built once before the loop, so a hook that rewrote ``context.phases``
was silently ignored: chaos ``sensor_noise`` of 2 rad had no effect on 32 of
the 36 shipped domainpacks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.runtime.chaos import (
    ChaosFault,
    ChaosSchedule,
    run_resilience_experiment,
)
from scpn_phase_orchestrator.runtime.server import SimulationState
from scpn_phase_orchestrator.runtime.simulation import (
    SimulationScenarioContext,
    simulate,
)

_SPEC = (
    Path(__file__).resolve().parents[1]
    / "domainpacks"
    / "minimal_domain"
    / "binding_spec.yaml"
)


def test_minimal_domain_runs_in_amplitude_mode() -> None:
    """Keep the regression fixture on the Stuart-Landau integration path."""
    assert load_binding_spec(_SPEC).amplitude is not None


def test_hook_phase_edit_is_integrated_like_an_initial_state() -> None:
    """Integrate a public hook's replacement phases in the live solver state."""
    spec = load_binding_spec(_SPEC)
    reference = SimulationState(spec)
    start = reference.phases.copy()
    seen: dict[int, NDArray[np.float64]] = {}

    def hook(context: SimulationScenarioContext) -> None:
        if context.step == 0:
            context.phases = start.copy()
        seen[context.step] = context.phases.copy()

    simulate(spec, steps=2, seed=0, policy_enabled=False, scenario_hook=hook)
    reference.step()  # the same driven Stuart-Landau step from the same phases
    np.testing.assert_allclose(seen[1], reference.phases, atol=1e-12)


def test_sensor_noise_perturbs_an_amplitude_mode_run() -> None:
    """Observe scheduled sensor noise through the public resilience workflow."""
    result = run_resilience_experiment(
        load_binding_spec(_SPEC),
        ChaosSchedule((ChaosFault("sensor_noise", 5, 20, 2.0),)),
        steps=60,
        seed=1,
    )
    assert result.metrics.max_coherence_drop > 0.05
    assert result.perturbed_final_r != pytest.approx(result.nominal_final_r)


@pytest.mark.parametrize("domain", ["minimal_domain", "digital_twin_nchannel"])
@pytest.mark.parametrize(
    ("field", "mutation", "value"),
    [
        (field, mutation, value)
        for field in ("phases", "omegas")
        for mutation, value in (
            ("short", 0.0),
            ("nested", 0.0),
            ("nonfinite", float("nan")),
            ("nonfinite", float("inf")),
            ("nonfinite", float("-inf")),
        )
    ]
    + [
        (field, "scalar", value)
        for field in ("zeta", "psi_target")
        for value in (float("nan"), float("inf"), float("-inf"), True, "0.1")
    ]
    + [("coupling", "identity", None)],
)
def test_hook_refusal_preserves_inputs_and_allows_fresh_run(
    domain: str, field: str, mutation: str, value: object
) -> None:
    """Reject bad hook updates before another step and recover deterministically."""
    spec_path = _SPEC.parents[1] / domain / "binding_spec.yaml"
    spec = load_binding_spec(spec_path)
    frequencies = np.asarray(spec.get_omegas(), dtype=np.float64)
    n_osc = frequencies.size
    invalid: object
    if mutation == "short":
        invalid = np.zeros(n_osc - 1, dtype=np.float64)
        message = f"scenario {field} must have shape"
    elif mutation == "nested":
        invalid = np.zeros((1, n_osc), dtype=np.float64)
        message = f"scenario {field} must have shape"
    elif mutation == "nonfinite":
        assert isinstance(value, float)
        invalid = np.full(n_osc, value, dtype=np.float64)
        message = f"scenario {field} must contain only finite values"
    elif mutation == "scalar":
        invalid = value
        message = f"scenario {field} must be a finite real scalar"
    else:
        invalid = value
        message = "scenario coupling must be a CouplingState"
    supplied_array = invalid.copy() if isinstance(invalid, np.ndarray) else None
    rejected_steps: list[int] = []
    rejected_phases: dict[int, NDArray[np.float64]] = {}

    def reject(context: SimulationScenarioContext) -> None:
        rejected_steps.append(context.step)
        rejected_phases[context.step] = context.phases.copy()
        if context.step == 1:
            setattr(context, field, invalid)

    baseline = simulate(spec, steps=3, seed=13, policy_enabled=False)
    with pytest.raises(ValueError, match=message):
        simulate(spec, steps=3, seed=13, policy_enabled=False, scenario_hook=reject)
    assert rejected_steps == [0, 1]
    np.testing.assert_array_equal(spec.get_omegas(), frequencies)
    if supplied_array is not None:
        np.testing.assert_array_equal(invalid, supplied_array)

    recovered_phases: dict[int, NDArray[np.float64]] = {}

    def record(context: SimulationScenarioContext) -> None:
        recovered_phases[context.step] = context.phases.copy()

    recovered = simulate(
        spec, steps=3, seed=13, policy_enabled=False, scenario_hook=record
    )
    assert list(recovered_phases) == [0, 1, 2]
    for step, phases in rejected_phases.items():
        np.testing.assert_array_equal(phases, recovered_phases[step])
    assert recovered.to_record() == baseline.to_record()
    assert recovered.r_good_history == baseline.r_good_history
    assert recovered.r_bad_history == baseline.r_bad_history
    np.testing.assert_array_equal(recovered.final_phases, baseline.final_phases)
    if baseline.final_amplitudes is not None:
        np.testing.assert_array_equal(
            recovered.final_amplitudes, baseline.final_amplitudes
        )
    else:
        assert recovered.final_amplitudes is None
