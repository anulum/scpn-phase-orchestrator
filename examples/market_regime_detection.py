#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Financial Market Regime Detection
#
# Generates synthetic market data with a synchronization event
# (simulating a crash precursor) and detects the regime transition.
#
# Usage: python examples/market_regime_detection.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.market import (
    detect_regimes,
    extract_phase,
    market_order_parameter,
    sync_warning,
)


def main() -> None:
    rng = np.random.default_rng(42)
    T = 500
    N_ASSETS = 10

    # Phase 1 (t=0..199): normal market — independent random walks
    normal = rng.standard_normal((200, N_ASSETS))

    # Phase 2 (t=200..349): herding — assets begin to correlate
    base_signal = np.sin(np.linspace(0, 8 * np.pi, 150))
    herding = np.column_stack([
        base_signal + 0.3 * rng.standard_normal(150) for _ in range(N_ASSETS)
    ])

    # Phase 3 (t=350..499): crash aftermath — back to noise
    aftermath = rng.standard_normal((150, N_ASSETS))

    returns = np.vstack([normal, herding, aftermath])
    print(f"Synthetic market: {T} timesteps, {N_ASSETS} assets")
    print(f"  t=0-199: normal (independent)")
    print(f"  t=200-349: herding (correlated)")
    print(f"  t=350-499: aftermath (independent)")
    print()

    # Extract phases and compute order parameter
    phases = extract_phase(returns)
    R = market_order_parameter(phases)

    # Detect regimes
    regimes = detect_regimes(R, sync_threshold=0.7, desync_threshold=0.3)
    warnings = sync_warning(R, threshold=0.6, lookback=10)

    # Report
    regime_names = {0: "DESYNC", 1: "TRANSITION", 2: "SYNCHRONIZED"}
    for label in [0, 1, 2]:
        count = np.sum(regimes == label)
        pct = 100 * count / len(regimes)
        print(f"  {regime_names[label]:>14s}: {count:3d} timesteps ({pct:.1f}%)")

    n_warnings = np.sum(warnings)
    print(f"\n  Sync warnings: {n_warnings}")

    # R statistics by phase
    print(f"\n  Mean R (normal):   {np.mean(R[:200]):.3f}")
    print(f"  Mean R (herding):  {np.mean(R[200:350]):.3f}")
    print(f"  Mean R (aftermath):{np.mean(R[350:]):.3f}")

    if np.mean(R[200:350]) > np.mean(R[:200]):
        print("\n  Herding phase detected: R increased during correlation period")


if __name__ == "__main__":
    main()
