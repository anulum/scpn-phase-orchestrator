#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Cross-Domain Universality
#
# The same 4 lines of code run 5 completely different domains.
# This demonstrates SPO's domain-agnostic design: the binding spec
# is the only thing that changes between plasma physics and traffic.
#
# Usage: python examples/cross_domain_universality.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

from pathlib import Path

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.server import SimulationState

DOMAINS = [
    "plasma_control",
    "cardiac_rhythm",
    "power_grid",
    "traffic_flow",
    "neuroscience_eeg",
]


def main() -> None:
    domainpack_dir = Path(__file__).parent.parent / "domainpacks"

    print("Cross-Domain Universality: Same Code, Different Physics")
    print("=" * 60)
    print(f"\n{'Domain':<22s} {'Osc':>4s} {'Lay':>4s} {'R@50':>6s} {'Regime':<10s}")
    print("-" * 52)

    for domain in DOMAINS:
        spec_path = domainpack_dir / domain / "binding_spec.yaml"
        if not spec_path.exists():
            print(f"{domain:<22s}  NOT FOUND")
            continue

        # THE UNIVERSAL PATTERN: 4 lines
        spec = load_binding_spec(spec_path)  # 1. load
        sim = SimulationState(spec)  # 2. build
        for _ in range(50):  # 3. run
            state = sim.step()
        R = state["R_global"]  # 4. read
        regime = state["regime"]

        n_osc = sum(len(ly.oscillator_ids) for ly in spec.layers)
        n_lay = len(spec.layers)
        print(f"{domain:<22s} {n_osc:>4d} {n_lay:>4d} {R:>6.3f} {regime:<10s}")

    print("\nSame engine. Same API. Same supervisor.")
    print("Only the binding_spec.yaml changes between domains.")


if __name__ == "__main__":
    main()
