# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Orchestration uplift benchmark

"""Measured scenario-driven open-loop vs closed-loop coherence.

For each beachhead domainpack this runs the non-actuating simulation core twice
on the same seed and the same deterministic perturbation schedule — once with
the supervisor/policy control feedback disabled (open loop, the baseline) and
once with it enabled (closed loop) — and reports the change in the coherence
objective ``separation = R_good - R_bad``.

The reported ``uplift`` is the measured closed-minus-open separation delta. It is
not asserted to be positive: a domainpack whose policy is not tuned for the
scenario schedule can show zero or negative uplift, and that is recorded
honestly. The numbers are reproducible because ``simulate`` is deterministic
for a fixed seed, step count, and scenario hook.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.runtime.simulation import (
    SimulationScenarioContext,
    simulate,
)

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

DEFAULT_STEPS = 250
DEFAULT_SEED = 7
DEFAULT_OUTPUT = Path("benchmarks/results/orchestration_uplift.json")
ScenarioHook = Callable[[SimulationScenarioContext], None]


def _layer_ids(ctx: SimulationScenarioContext, layer: int) -> list[int]:
    return ctx.layer_osc_ranges.get(layer, [])


def _scenario_hook(pack: str, base_omegas: np.ndarray) -> ScenarioHook:
    base = np.asarray(base_omegas, dtype=np.float64)

    def hook(ctx: SimulationScenarioContext) -> None:
        if pack == "rotating_machinery":
            if ctx.step < 40:
                ctx.omegas[: min(3, ctx.omegas.size)] = base[:3] * (
                    (ctx.step + 1) / 40.0
                )
            if 40 <= ctx.step < 80:
                ids = _layer_ids(ctx, 3)
                ctx.omegas[ids] = base[ids] * 0.95
            if ctx.step == 80:
                ctx.omegas[:] = base
            if 120 <= ctx.step < 160:
                ids = _layer_ids(ctx, 2)
                ctx.omegas[ids] += 0.005 * ctx.rng.standard_normal(len(ids))
            if ctx.step == 160:
                ids = _layer_ids(ctx, 1)
                if len(ids) > 1:
                    ctx.omegas[ids[1]] = 0.8
        elif pack == "manufacturing_spc":
            if 50 <= ctx.step < 100:
                ctx.omegas[_layer_ids(ctx, 0)] += 0.003
            if ctx.step == 100:
                ids = _layer_ids(ctx, 1)
                if ids:
                    ctx.omegas[ids[0]] = 0.01
            if ctx.step == 130:
                ctx.omegas[:] = base
                ctx.zeta = 0.3
                ctx.psi_target = 0.0
            if ctx.step == 170:
                ctx.zeta = 0.0
        elif pack == "chemical_reactor":
            if 50 <= ctx.step < 100:
                ctx.omegas[_layer_ids(ctx, 0)] *= 1.01
            if ctx.step == 100:
                heat_ids = _layer_ids(ctx, 1)
                flow_ids = _layer_ids(ctx, 3)
                if len(heat_ids) > 2:
                    ctx.omegas[heat_ids[2]] = 0.01
                if len(flow_ids) > 1:
                    ctx.omegas[flow_ids[1]] = 0.05
            if ctx.step == 150:
                ctx.omegas[:] = base
                ctx.zeta = 0.4
            if ctx.step == 200:
                ctx.zeta = 0.0
        elif pack == "power_grid":
            if ctx.step == 50:
                ctx.omegas[_layer_ids(ctx, 3)] *= 2.5
            if 100 <= ctx.step < 150:
                ids = _layer_ids(ctx, 4)
                ctx.omegas[ids] += 0.01 * ctx.rng.standard_normal(len(ids))
            if ctx.step == 150:
                ids = _layer_ids(ctx, 0)
                if ids:
                    tripped = ids[0]
                    ctx.omegas[tripped] = 0.0
                    knm = ctx.coupling.knm.copy()
                    knm[tripped, :] *= 0.1
                    knm[:, tripped] *= 0.1
                    ctx.coupling = CouplingState(
                        knm=knm,
                        alpha=ctx.coupling.alpha,
                        active_template=ctx.coupling.active_template,
                        knm_r=ctx.coupling.knm_r,
                    )
            if ctx.step == 175:
                ids = _layer_ids(ctx, 3)
                ctx.omegas[ids] = base[ids]
                ctx.zeta = 0.2
        elif pack == "queuewaves":
            if 40 <= ctx.step < 80:
                ids = _layer_ids(ctx, 0)
                ctx.omegas[ids] = np.array([2.0, 2.2, 1.8], dtype=np.float64)[
                    : len(ids)
                ]
            if 80 <= ctx.step < 120:
                ids = _layer_ids(ctx, 0)
                if ids:
                    ctx.phases[ids] = float(np.mean(ctx.phases[ids]))
                    if ctx.step % 5 == 0:
                        ctx.phases[ids] += ctx.rng.uniform(-0.05, 0.05, len(ids))
            if ctx.step == 120:
                ctx.omegas[:] = base
                ctx.zeta = 0.3
            if ctx.step == 160:
                ctx.zeta = 0.1
        elif pack == "cardiac_rhythm":
            if 50 <= ctx.step < 100 and ctx.step % 8 == 0:
                ids = _layer_ids(ctx, 2)
                ctx.phases[ids] += ctx.rng.uniform(0.5, 1.5, len(ids))
            if 100 <= ctx.step < 150:
                ctx.omegas[_layer_ids(ctx, 2)] = 2.5
            if ctx.step == 150:
                ids = _layer_ids(ctx, 2)
                ctx.omegas[ids] = base[ids]
                ctx.zeta = 0.3
            if ctx.step == 200:
                ctx.zeta = 0.5
                ctx.psi_target = 2.0 * np.pi * 1.17 * ctx.step * ctx.sample_period_s
        elif pack == "neuroscience_eeg":
            if 75 <= ctx.step < 150:
                ids = _layer_ids(ctx, 0)
                if ids:
                    ctx.phases[ids] = ctx.phases[ids] * 0.95 + 0.05 * ctx.phases[ids[0]]
            if 150 <= ctx.step < 225:
                ctx.zeta = 0.3
                ctx.psi_target = 2.0 * np.pi * 10.0 * ctx.step * ctx.sample_period_s
            if ctx.step == 225:
                ctx.zeta = 0.0
                ctx.psi_target = 0.0
        elif pack == "sleep_architecture":
            if 50 <= ctx.step < 100:
                ctx.omegas[_layer_ids(ctx, 0)] *= 0.7
                ctx.omegas[_layer_ids(ctx, 2)] *= 1.4
            if 100 <= ctx.step < 150:
                ids = _layer_ids(ctx, 2)
                if ids:
                    ctx.phases[ids] += ctx.rng.uniform(-0.2, 0.2, len(ids))
            if ctx.step == 150:
                ctx.omegas[:] = base
                ctx.zeta = 0.15
            if ctx.step == 200:
                ctx.zeta = 0.0

    return hook


def measure_pack(
    pack: str, *, steps: int, seed: int, domainpacks_root: Path
) -> dict[str, Any]:
    """Return the open/closed-loop coherence record for one domainpack."""

    spec_path = domainpacks_root / pack / "binding_spec.yaml"
    spec = load_binding_spec(spec_path)
    hook = _scenario_hook(pack, np.asarray(spec.get_omegas(), dtype=np.float64))
    open_loop = simulate(
        spec,
        steps=steps,
        seed=seed,
        policy_enabled=False,
        binding_spec_path=spec_path,
        scenario_hook=hook,
    )
    closed = simulate(
        spec,
        steps=steps,
        seed=seed,
        policy_enabled=True,
        binding_spec_path=spec_path,
        scenario_hook=hook,
    )
    return {
        "pack": pack,
        "amplitude_mode": closed.amplitude_mode,
        "scenario_profile": "domainpack_demo_perturbations",
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
        "version": "1.1.0",
        "steps": steps,
        "seed": seed,
        "metric": (
            "scenario-driven separation = R_good - R_bad; uplift = closed - open"
        ),
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
