# Build Knm Templates

How to construct coupling matrices for a new domain.

## Step 1: Start from Default

The default `CouplingBuilder.build()` produces:

```
K_ij = base_strength * exp(-decay_alpha * |i - j|)
K_ii = 0
```

This gives nearest-neighbour-dominant coupling with exponential falloff. Good starting point when you have no domain-specific coupling data.

Typical values: `base_strength = 0.45`, `decay_alpha = 0.3`.

## Step 2: Domain Knowledge Overrides

If you know specific coupling strengths from physics, data, or domain expertise:

1. Generate the default matrix.
2. Override specific entries.
3. Re-symmetrise: `K = (K + K.T) / 2`.
4. Zero diagonal: `np.fill_diagonal(K, 0)`.

Example: oscillators 1 and 2 have a known strong coupling of 0.8:

```python
K[1, 2] = 0.8
K[2, 1] = 0.8
```

## Step 3: Calibrate from Data

If you have phase time-series data:

1. Compute pairwise PLV between all oscillator pairs.
2. Use PLV as a proxy for coupling strength: `K_ij = scale * PLV_ij`.
3. Enforce the three invariants (symmetric, non-negative, zero diagonal).

## Step 4: Template Switching

Define multiple Knm matrices for different regimes:

```yaml
coupling:
  base_strength: 0.45
  decay_alpha: 0.3
  templates:
    default: default
    storm: storm_decoupled
    recovery: recovery_boosted
```

- **storm_decoupled:** Reduce inter-layer coupling to isolate faults. Set cross-layer entries to 0.1x default.
- **recovery_boosted:** Increase intra-layer coupling to accelerate re-synchronisation. Set intra-layer entries to 1.5x default.

Switch templates via `CouplingBuilder.switch_template()`.

## Step 5: Validate

Run the domain simulation and check:

1. R_good converges to > 0.7 under default template.
2. R_bad stays below 0.3.
3. Template switch during CRITICAL regime reduces R_bad.
4. Recovery template restores R_good within 50 steps.

Iterate on coupling values until these criteria are met.

## Common Pitfalls

- **Too strong coupling:** R_good saturates at 1.0 instantly, but R_bad also goes to 1.0. Reduce base_strength.
- **Too weak coupling:** R_good stays near 0. Increase base_strength or reduce decay_alpha.
- **Asymmetric override without re-symmetrisation:** Violates the Knm contract. Always enforce `K = (K + K.T) / 2`.

## References

- **[acebron2005]** J. A. Acebrón et al. (2005). The Kuramoto model: a simple paradigm for synchronization phenomena. *Rev. Mod. Phys.* 77, 137–185. — Coupling strength ranges and their effect on synchronisation.
- **[lachaux1999]** J.-P. Lachaux et al. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping* 8, 194–208. — PLV as coupling-strength proxy (Step 3).
