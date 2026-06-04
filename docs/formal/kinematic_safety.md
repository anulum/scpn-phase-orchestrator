# Formal Kinematic Safety

`SPOFormal.Kinematic` is the PHA-C.6 Lean 4 proof lane for finite-horizon
kinematic merge-window safety. It gives downstream repositories a reusable
proof template for reasoning about fixed-point relative-distance budgets before
a moving-frame merge is accepted.

## What it proves

The Lean module proves three production contracts:

- A discrete Gronwall-style budget bounds a metric-distance sequence whenever
  each step is bounded by a Lipschitz multiplier plus an additive drive term.
- In the zero-gain case, a single scalar inequality
  `initialTolerance + horizonSteps * driveBound <= mergeWindowTolerance`
  certifies the whole finite horizon.
- A Boolean merge-window mirror is true exactly when both phase dispersion and
  spatial distance are within their fixed-point tolerances.

The proof surface is dependency-light. It uses Lean core plus `Std`, has no
`axiom`, `sorry`, `admit`, `unsafe`, or linter suppression, and is checked by
`tools/check_lean_proofs.sh`.

## What it does not prove

The module does not claim continuous-time Kuramoto stability, arbitrary-topology
Lyapunov safety, hardware timing safety, or site-specific actuator safety. Those
require separate model assumptions and physical evidence. The intended pipeline
is:

1. Runtime or simulation converts physical distances and phase dispersions into
   reviewed fixed-point units.
2. Domain code supplies `KinematicBounds` and proves or checks that the step
   recurrence assumptions hold.
3. Lean proves that those finite-step assumptions imply the merge-window budget
   and Boolean lock predicates over the reviewed horizon.

## Files

| File | Purpose |
|---|---|
| `formal/lean/SPOFormal/Kinematic.lean` | Generic finite-horizon kinematic lemmas. |
| `formal/lean/test/KinematicTest.lean` | Concrete smoke instantiation for a merge-window horizon. |
| `formal/lean/SPOFormal.lean` | Re-exports the proof module. |
| `tools/check_lean_proofs.sh` | Placeholder/linter-suppression gate plus direct Lean checks. |

## Core API

```lean
structure KinematicBounds where
  initialTolerance : Nat
  lipschitzStepGain : Nat
  relativeVelocityStepBound : Nat
  couplingResidualStepBound : Nat
  mergeWindowTolerance : Nat
  horizonSteps : Nat
```

The main theorem is:

```lean
theorem distance_bound_under_lipschitz_coupling
    (distance : Nat -> Nat)
    (cfg : KinematicBounds)
    (hInitial : distance 0 <= cfg.initialTolerance)
    (hStep : forall k,
      distance (k + 1) <=
        distance k + cfg.lipschitzStepGain * distance k + cfg.driveBound) :
    forall k, distance k <= cfg.budget k
```

For the common reviewed zero-gain horizon, downstream code can use:

```lean
theorem merge_window_invariant_zero_gain
    (distance : Nat -> Nat)
    (cfg : KinematicBounds)
    (hGain : cfg.lipschitzStepGain = 0)
    (hWindow : cfg.initialTolerance + cfg.horizonSteps * cfg.driveBound <=
      cfg.mergeWindowTolerance)
    (hInitial : distance 0 <= cfg.initialTolerance)
    (hStep : forall k, distance (k + 1) <= distance k + cfg.driveBound) :
    forall k, k <= cfg.horizonSteps -> distance k <= cfg.mergeWindowTolerance
```

## Validation command

```bash
./tools/check_lean_proofs.sh
```

That command rejects placeholders, rejects disabled proof-source linters, checks
each proof module with warnings treated as errors, builds the Lake library, and
checks the smoke instantiation.
