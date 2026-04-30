# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital twin N-channel example

"""Run the digital-twin N-channel domainpack."""

from __future__ import annotations

from pathlib import Path

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.server import SimulationState

STEPS = 50
SPEC_PATH = Path(__file__).parent / "binding_spec.yaml"


def main() -> None:
    """Run a short deterministic digital-twin simulation."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise RuntimeError(f"Invalid spec: {errors}")

    sim = SimulationState(spec)
    print("=== Digital Twin N-Channel ===")
    print(f"{'step':>5}  {'R_global':>8}  {'regime':>10}")
    print("-" * 30)
    for _ in range(STEPS):
        state = sim.step()
        if state["step"] % 10 == 0 or state["step"] == 1:
            print(
                f"{state['step']:5d}  {state['R_global']:8.4f}  {state['regime']:>10}"
            )


if __name__ == "__main__":
    main()
