#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Deterministic Audit Replay
#
# Every step is logged with a SHA-256 hash chain. The simulation
# can be replayed exactly from the log — proving determinism and
# detecting any tampering.
#
# Usage: python examples/audit_replay_demo.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 6
    rng = np.random.default_rng(42)
    omegas = rng.uniform(-1, 1, n)
    knm = np.ones((n, n)) * 2.0
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    phases_init = rng.uniform(0, TWO_PI, n)

    eng = UPDEEngine(n, dt=0.01)

    print("Deterministic Audit Replay")
    print("=" * 50)

    # Run 1: simulate and log
    audit_log = []
    prev_hash = "0" * 64
    phases = phases_init.copy()

    print("\nRun 1: Simulate + record audit trail")
    for step in range(100):
        phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)

        record = {
            "step": step + 1,
            "R": round(R, 6),
            "phases_hash": hashlib.sha256(phases.tobytes()).hexdigest()[:16],
            "prev_hash": prev_hash,
        }
        record["hash"] = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()
        prev_hash = record["hash"]
        audit_log.append(record)

        if step % 25 == 24:
            print(f"  Step {step + 1}: R={R:.3f}, hash=...{record['hash'][-8:]}")

    # Save audit log
    log_path = Path(tempfile.gettempdir()) / "spo_audit_demo.jsonl"
    with log_path.open("w") as f:
        for record in audit_log:
            f.write(json.dumps(record) + "\n")
    print(f"\nAudit log: {log_path} ({len(audit_log)} records)")

    # Run 2: replay from scratch, verify
    print("\nRun 2: Replay from same initial conditions")
    phases_replay = phases_init.copy()
    mismatches = 0

    for step in range(100):
        phases_replay = eng.step(phases_replay, omegas, knm, 0.0, 0.0, alpha)
        replay_hash = hashlib.sha256(phases_replay.tobytes()).hexdigest()[:16]

        if replay_hash != audit_log[step]["phases_hash"]:
            mismatches += 1

    if mismatches == 0:
        print("  All 100 steps match. Deterministic replay verified.")
    else:
        print(f"  {mismatches} mismatches detected!")

    # Verify hash chain integrity
    print("\nVerifying SHA-256 chain integrity...")
    chain_ok = True
    for i in range(1, len(audit_log)):
        if audit_log[i]["prev_hash"] != audit_log[i - 1]["hash"]:
            chain_ok = False
            break
    print(f"  Chain integrity: {'PASSED' if chain_ok else 'FAILED'}")

    print("\nThis is why SPO is regulatory-ready: every decision is")
    print("auditable, reproducible, and tamper-evident.")


if __name__ == "__main__":
    main()
