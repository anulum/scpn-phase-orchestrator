# Knm Calibration

## Overview

The coupling matrix K_nm encodes interaction strengths between
oscillators. It is the central parameter of the UPDE — it determines
which oscillators influence each other, how strongly, and through what
topology. Calibration is the process of determining K_nm values from
domain data, prior knowledge, or optimisation targets.

A well-calibrated K_nm produces:
- R_good converging to the target coherence level.
- R_bad remaining suppressed.
- Stable dynamics without divergence or collapse.
- PLV matrix matching observed pairwise correlations.

A poorly calibrated K_nm produces:
- Either no synchronisation (K too low) or pathological lock-in (K too high).
- R oscillations that the supervisor cannot stabilise.
- Divergence in adaptive-step integrators.

## Construction Pipeline

```
CouplingBuilder.build(n_osc, base_strength, decay_alpha)
    -> distance-decay initialisation
    -> template overlay (optional)
    -> geometry projection (optional)
    -> E/I balance (optional)
    -> spectral validation (optional)
    -> imprint modulation (runtime)
```

### Step 1: Distance-Decay Initialisation

Default construction uses exponential decay with oscillator index
distance:

```
K_nm[i,j] = base_strength * exp(-decay_alpha * |i - j|)
K_nm[i,i] = 0  (zero diagonal — no self-coupling)
```

Parameters `base_strength` and `decay_alpha` are specified in
`binding_spec.coupling`:

```yaml
coupling:
  base_strength: 1.5
  decay_alpha: 0.3
```

For layer-structured systems, distance is computed between layer
indices, not oscillator indices. The `SCPN_LAYER_TIMESCALES` constants
provide empirically validated decay rates for the 15+1 SCPN layers.

### Step 2: Template Overlay

Coupling templates (named K_nm patterns) replace or modulate the
distance-decay matrix. Templates encode domain-specific topologies:

```yaml
coupling:
  templates:
    nearest_neighbour:
      pattern: band
      bandwidth: 1
      strength: 2.0
    small_world:
      pattern: small_world
      rewire_prob: 0.1
      strength: 1.0
    hierarchical:
      pattern: block_diagonal
      block_sizes: [4, 4, 8]
      strengths: [2.0, 1.5, 1.0]
```

Templates are managed by `KnmTemplate` and `KnmTemplateSet`. The
supervisor can switch the active template at runtime via
`CouplingState.active_template`, enabling topology-dependent control
strategies.

### Step 3: Geometry Projection

`GeometryConstraint` subclasses enforce structural properties:

- **SymmetryConstraint**: `K_nm = (K_nm + K_nm.T) / 2`. Ensures
  undirected coupling.
- **NonNegativeConstraint**: `K_nm = max(K_nm, 0)`. No negative
  coupling strengths.
- **Custom**: a `GeometryConstraint` subclass implementing `project(K_nm) -> K_nm` and preserving finite real square shape.

Projection is applied via `project_knm()`, which chains all active
constraints:

```python
from scpn_phase_orchestrator.coupling import (
    CouplingBuilder, NonNegativeConstraint, SymmetryConstraint,
    project_knm, validate_knm,
)

state = CouplingBuilder().build(4, base_strength=0.45, decay_alpha=0.3)
K = project_knm(state.knm, [SymmetryConstraint(), NonNegativeConstraint()])
validate_knm(K)
```

The symmetry mean is evaluated without overflowing a finite pair or erasing
representable subnormals. Projection returns an independent matrix, applies
constraints in the supplied order and zeros its diagonal last. The empty
matrix is accepted; invalid original numeric types or custom constraint output
refuse before being used downstream. The
[geometry contract](geometry_constraints.md) explains the direct Rust equivalent
and the measured runtime boundary. Finite structural validation alone does not
establish stable integration at arbitrarily large coupling strengths.

### Step 4: E/I Balance (Optional)

`compute_ei_balance` returns an `EIBalance` summary of signed outgoing
coupling means. `adjust_ei_ratio` returns an independent matrix with each
unique inhibitory source row scaled once. Supply the source groups from the
binding; duplicate indices count once. This two-oscillator example uses
disjoint groups:

```python
import numpy as np
from scpn_phase_orchestrator.coupling import adjust_ei_ratio, compute_ei_balance

K = np.array([[0.0, 2.0], [1.0, 0.0]])
summary = compute_ei_balance(K, excitatory_indices=[0], inhibitory_indices=[1])
ratio = summary.ratio  # 2.0
balanced_K = adjust_ei_ratio(K, [0], [1], target_ratio=1.0)
```

The default adjustment target is `1.0`. This numerical target and the summary's
balance interval do not establish a physiological regime or a universal neural
ratio. Signed inputs remain supported; silent groups return an unchanged copy,
and non-finite adjustments or a scale rounded to zero raise. Target attainment
requires disjoint groups, adequate float64 precision and an adjusted inhibitory
mean with magnitude at least `1e-15`; smaller means use the silent convention.
See the [E/I API contract](../reference/api/coupling.md#ei-balance) for the
complete validation, empty-group and overlap semantics.

### Step 5: Spectral Validation (Optional)

The spectral properties of K_nm determine synchronisation behaviour:

- **Fiedler value** (`fiedler_value(K)`): second-smallest eigenvalue
  of the graph Laplacian. Must be positive for connected coupling
  (all oscillators can influence each other through chains).
- **Critical coupling** (`critical_coupling(K, omegas)`): minimum
  global coupling strength for synchronisation onset.
- **Spectral gap** (`spectral_gap(K)`): ratio of Fiedler value to
  largest eigenvalue. Larger gap = faster convergence to sync.

```python
from scpn_phase_orchestrator.coupling import fiedler_value, critical_coupling

fv = fiedler_value(K)
assert fv > 0, "coupling graph is disconnected"

Kc = critical_coupling(K, omegas)
assert base_strength > Kc, f"below critical coupling: {base_strength} < {Kc}"
```

---

## Calibration Methods

### Manual Calibration

Set `base_strength` and `decay_alpha` from domain knowledge.
Appropriate for:
- Small oscillator counts (N < 20).
- Well-characterised systems with known coupling strengths.
- Initial exploration before data-driven refinement.

Guidelines:
- Start with `base_strength = 2 * gamma` where gamma is the
  half-width of the natural frequency distribution.
- Set `decay_alpha = 1.0 / characteristic_length` where
  characteristic_length is the typical interaction range
  (in oscillator-index units).

### Empirical Calibration

Estimate K_nm from observed phase-locking:

1. Record phase trajectories from domain data (minimum 100 cycles
   per oscillator pair for statistical reliability).
2. Compute pairwise PLV matrix:
   ```python
   from scpn_phase_orchestrator.upde import compute_plv
   plv_matrix = compute_plv(phase_trajectory)
   ```
3. Threshold: `K_nm[i,j] = plv_matrix[i,j] if plv > threshold else 0`.
4. Scale: multiply by a global factor to match target R.

The threshold determines sparsity. Higher threshold = sparser K_nm
(fewer connections, faster integration). Lower threshold = denser
K_nm (more connections, potentially richer dynamics but slower).

### Bayesian Calibration

Fit K_nm posteriors given observed R trajectories and known omega_n:

1. Define prior: `K_nm ~ Exponential(rate=1.0)` (non-negative, sparse).
2. Likelihood: simulate UPDE with candidate K_nm, compare R trajectory
   to observed R trajectory using MSE.
3. Sample: MCMC (emcee, PyMC) or optimise (scipy.optimize.minimize).
4. Posterior: MAP estimate or posterior mean as point estimate.

This is computationally expensive (each likelihood evaluation requires
a full UPDE simulation) but produces uncertainty estimates on K_nm
entries, which can inform confidence-weighted coupling.

### SINDy-Based Calibration

The `autotune.sindy` module uses Sparse Identification of Nonlinear
Dynamics to discover K_nm from phase trajectory data:

```python
from scpn_phase_orchestrator.autotune import PhaseSINDy

sindy = PhaseSINDy()
K_estimated = sindy.fit(phase_trajectory, omegas, dt=0.01)
```

SINDy fits a sparse linear combination of candidate coupling functions
to the observed phase derivatives, directly recovering K_nm entries.
This is faster than Bayesian calibration but assumes the Kuramoto
coupling form is correct.

### Auto-Tune Pipeline

The full auto-tune pipeline chains extraction, frequency identification,
coupling estimation, and binding spec generation:

```python
from scpn_phase_orchestrator.autotune import (
    extract_phases, identify_frequencies, estimate_coupling,
    identify_binding_spec,
)

phases = extract_phases(raw_signals, sample_rate=256.0)
freqs = identify_frequencies(phases)
K_est = estimate_coupling(phases, freqs)
spec = identify_binding_spec(phases, freqs, K_est)
```

---

## SCPN Calibration Anchors

For the 15+1 SCPN layer model, calibration anchors are empirically
validated coupling strengths stored in `SCPN_CALIBRATION_ANCHORS`:

```python
from scpn_phase_orchestrator.coupling import SCPN_CALIBRATION_ANCHORS

# anchors[layer_i][layer_j] = validated coupling strength
print(SCPN_CALIBRATION_ANCHORS)
```

These anchors were derived from:
- Inter-layer correlation analysis across 12 domain datasets.
- Consistency with Ott-Antonsen mean-field predictions.
- Stability analysis (all eigenvalues of the linearised system have
  negative real part at the sync fixed point).

Anchors are NOT hard constraints — they are starting points. Domain-
specific calibration should adjust from these baselines.

---

## Validation

After calibration, verify the K_nm produces correct dynamics:

1. **Structural**: `validate_knm(K)` checks symmetry, non-negativity,
   zero diagonal, finite entries.
2. **Spectral**: `fiedler_value(K) > 0` (connected graph).
3. **Dynamic**: simulate 1000 steps with `spo run`. Check:
   - R_good converges to target (within 10%).
   - R_bad stays below threshold.
   - No NaN/Inf in phase trajectory.
4. **Empirical**: compare simulated PLV matrix against observed PLV.
   Frobenius norm of difference should decrease with calibration
   iterations.
5. **Boundary**: verify no boundary violations during simulation.

```python
# Quick validation script
engine = UPDEEngine(n=spec.n_oscillators)
phases, omegas = extract_initial_conditions(spec, data)
K = builder.build(spec)

for step in range(1000):
    engine.step(phases, omegas, K, zeta=0, psi=0, alpha=alpha)

R_final = compute_order_parameter(phases)[0]
assert R_final > 0.5, f"R={R_final:.3f} below target"
```

---

## Coupling Matrix Properties

| Property | Requirement | Check |
|----------|------------|-------|
| Symmetry | `K_nm = K_nm.T` (undirected) | `validate_knm()` |
| Non-negative | `K_nm >= 0` | `validate_knm()` |
| Zero diagonal | `K_nm[i,i] = 0` | `validate_knm()` |
| Finite | No NaN/Inf | `validate_knm()` |
| Connected | Fiedler value > 0 | `fiedler_value(K)` |
| Sparse (optional) | `nnz / N^2 < 0.3` | Manual check |
| Stable | Re(eigenvalues) < 0 at sync | `check_stability()` |

---

## References

- `src/scpn_phase_orchestrator/coupling/knm.py` — `CouplingBuilder`, `CouplingState`.
- `src/scpn_phase_orchestrator/coupling/spectral.py` — spectral analysis utilities.
- `src/scpn_phase_orchestrator/autotune/` — automated calibration pipeline.
- `docs/specs/knm_semantics.md` — semantic interpretation of K_nm entries.
- `docs/specs/eval_protocol.md` — evaluation metrics for calibration quality.
- **[acebron2005]** J. A. Acebron et al. (2005). The Kuramoto model. *Rev. Mod. Phys.* 77, 137-185. — Critical coupling derivation.
- **[brunton2016]** S. L. Brunton et al. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS* 113, 3932-3937. — SINDy method.
