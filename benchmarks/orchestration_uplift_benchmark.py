# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Orchestration uplift benchmark

"""Measured open-loop vs closed-loop coherence for the beachhead lead packs.

For each beachhead domainpack this runs the non-actuating simulation core twice
on the same seed — once with the supervisor/policy control feedback disabled
(open loop, the baseline) and once with it enabled (closed loop) — and reports
the change in the coherence objective ``separation = R_good - R_bad``.

The reported ``uplift`` is the measured closed-minus-open separation delta. It is
not asserted to be positive: a domainpack whose policy is not tuned for the
generic (perturbation-free) scenario can show zero or negative uplift, and that
is recorded honestly. The numbers are reproducible because ``evaluate_binding_spec``
is deterministic for a fixed seed and step count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scpn_phase_orchestrator.api import evaluate_binding_spec

# Order parameters are accurate well beyond this, but parallel float reductions
# differ in the last ~1e-16 ULP between runs; rounding to 9 decimals makes the
# committed JSON bit-reproducible while losing no meaningful precision.
_DECIMALS = 9


def _r(value: float) -> float:
    return round(float(value), _DECIMALS)


# Beachhead lead packs across the three verticals (industrial, infrastructure,
# biosignal). All run through the non-actuating eval core regardless of tier.
BEACHHEAD_PACKS: tuple[tuple[str, str], ...] = (
    ("industrial", "rotating_machinery"),
    ("industrial", "manufacturing_spc"),
    ("industrial", "chemical_reactor"),
    ("infrastructure", "power_grid"),
    ("infrastructure", "queuewaves"),
    ("biosignal", "cardiac_rhythm"),
    ("biosignal", "neuroscience_eeg"),
    ("biosignal", "sleep_architecture"),
)

DEFAULT_STEPS = 200
DEFAULT_SEED = 7
DEFAULT_OUTPUT = Path("benchmarks/results/orchestration_uplift.json")


def measure_pack(
    pack: str, *, steps: int, seed: int, domainpacks_root: Path
) -> dict[str, Any]:
    """Return the open/closed-loop coherence record for one domainpack."""

    spec_path = domainpacks_root / pack / "binding_spec.yaml"
    open_loop = evaluate_binding_spec(
        spec_path, steps=steps, seed=seed, policy_enabled=False
    )
    closed = evaluate_binding_spec(
        spec_path, steps=steps, seed=seed, policy_enabled=True
    )
    return {
        "pack": pack,
        "amplitude_mode": closed.amplitude_mode,
        "open_loop": {
            "r_good": _r(open_loop.r_good),
            "r_bad": _r(open_loop.r_bad),
            "separation": _r(open_loop.separation),
            "regime": open_loop.final_regime,
        },
        "closed_loop": {
            "r_good": _r(closed.r_good),
            "r_bad": _r(closed.r_bad),
            "separation": _r(closed.separation),
            "regime": closed.final_regime,
            "action_total": closed.action_total,
        },
        "uplift": _r(closed.separation - open_loop.separation),
    }


def run_benchmark(*, steps: int, seed: int, domainpacks_root: Path) -> dict[str, Any]:
    """Measure every beachhead pack and return the full benchmark record."""

    records = []
    for vertical, pack in BEACHHEAD_PACKS:
        record = measure_pack(
            pack, steps=steps, seed=seed, domainpacks_root=domainpacks_root
        )
        record["vertical"] = vertical
        records.append(record)
    return {
        "benchmark": "orchestration-uplift",
        "version": "1.0.0",
        "steps": steps,
        "seed": seed,
        "metric": "separation = R_good - R_bad; uplift = closed - open",
        "packs": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--domainpacks-root", type=Path, default=Path("domainpacks"))
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = run_benchmark(
        steps=args.steps, seed=args.seed, domainpacks_root=args.domainpacks_root
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for record in report["packs"]:
        print(
            f"{record['vertical']:14s} {record['pack']:20s} "
            f"open={record['open_loop']['separation']:+.4f} "
            f"closed={record['closed_loop']['separation']:+.4f} "
            f"uplift={record['uplift']:+.4f} "
            f"actions={record['closed_loop']['action_total']}"
        )
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
