# Knm Semantics

## Matrix Contract

The coupling matrix `K_ij` (Knm) satisfies:

1. **Symmetric:** `K_ij = K_ji`. Coupling is bidirectional.
2. **Non-negative:** `K_ij >= 0`. Negative coupling is handled via the alpha lag term.
3. **Zero diagonal:** `K_ii = 0`. No self-coupling.

`CouplingBuilder.build()` enforces all three invariants.

## Default Construction

```
K_ij = base_strength * exp(-decay_alpha * |i - j|)
K_ii = 0
```

Parameters from the binding spec `coupling` section:

- `base_strength`: peak coupling magnitude (default 0.45)
- `decay_alpha`: exponential decay rate with layer distance (default 0.3)

## Source-Target Interpretation

`K_ij` is the strength with which oscillator j influences oscillator i. In the UPDE derivative:

```
sum_j K_ij sin(theta_j - theta_i - alpha_ij)
```

Row i receives coupling from all columns j. Increasing row i scales how receptive oscillator i is.

## Template System

Multiple Knm matrices can be pre-computed and stored as named templates. The binding spec `coupling.templates` maps names to template identifiers:

```yaml
coupling:
  base_strength: 0.45
  decay_alpha: 0.3
  templates:
    storm: storm_decoupled
    recovery: recovery_boosted
```

`CouplingBuilder.switch_template(state, template_name, templates)` swaps the active matrix.

## Regime Switching

The supervisor can switch Knm templates based on regime:

| Regime | Template | Rationale |
|--------|----------|-----------|
| NOMINAL | default | Standard coupling |
| DEGRADED | default | Same matrix, but K boosted via ControlAction |
| CRITICAL | storm_decoupled | Reduced inter-layer coupling to isolate fault |
| RECOVERY | recovery_boosted | Gradual coupling restoration |

Template switching is atomic: one matrix replaces another. The alpha matrix is preserved across switches unless explicitly changed.

## Imprint Modulation

When the imprint model is active, effective Knm is:

```
K_ij_effective = K_ij * (1 + M_i)
```

Row-wise scaling. High imprint on oscillator i increases its receptivity to all neighbours.

## References

- **[kuramoto1975]** Y. Kuramoto (1975). Self-entrainment of a population of coupled non-linear oscillators. *Lecture Notes in Physics* 39, 420–422. — Coupling matrix formulation.
- **[acebron2005]** J. A. Acebrón et al. (2005). The Kuramoto model: a simple paradigm for synchronization phenomena. *Rev. Mod. Phys.* 77, 137–185. — Coupling strength and synchronisation thresholds.

## Why this matters in real runs

- The direction convention for `K_ij` prevents a common control bug where row/column
  roles are accidentally flipped in downstream policy rules.
- Template switching gives deterministic regime behavior and makes recovery/critical
  coupling changes auditable.
- The imprint scaling term is the operational bridge between memory and interaction
  strength without rewriting the supervisor policy.

## Deployment interpretation

The `K_ij` convention and row-wise semantics are the primary anti-footgun
guardrails when teams migrate from toy scripts to operator runs.

Two practical effects in production are:
- deterministic controller behavior during template transitions, and
- explainable coupling changes after imprint or recovery actions.

Because `K` changes materially affect stability, this spec should be treated as a
control contract, not just a simulation constant table.

## Practical validation sequence

Before a run, validate these contract points:

- symmetry enforcement and zero diagonal in the constructed matrix,
- active template selection against runtime regime state,
- imprint scaling assumptions for every row.

These checks should be included in pre-run summaries because a coupling change is
one of the highest-impact configuration moves and must be explicit in evidence logs.
