# Synthetic HCP-Inspired Connectome

## 1. Mathematical Formalism

### Structural Connectivity Matrix

The `load_hcp_connectome` function generates a synthetic coupling
matrix $W \in \mathbb{R}^{N \times N}$ that mimics the macroscale
structural connectivity of the human brain. The matrix is constructed
from three architectural components:

### Component 1: Intra-Hemispheric Coupling

For each hemisphere (left: indices $[0, N/2)$, right: $[N/2, N)$),
the coupling between regions $i$ and $j$ within the same hemisphere
follows exponential distance decay:

$$W_{ij}^{\text{intra}} = K_{\text{intra}} \cdot e^{-\beta \cdot |i - j|} + \epsilon_{ij}$$

where:
- $K_{\text{intra}} = 0.5$ — intra-hemispheric base strength
- $\beta = 0.3$ — spatial decay rate
- $\epsilon_{ij} \sim \mathcal{N}(0, 0.02)$ — biological noise,
  clamped to $\geq 0$

The exponential decay models the well-established principle that
structural connectivity decreases with geodesic cortical distance
(Hagmann et al. 2008; Ercsey-Ravasz et al. 2013).

### Component 2: Inter-Hemispheric (Callosal) Connections

The corpus callosum connects homotopic regions (same functional
area in opposite hemispheres) with the strongest fibres:

$$W_{ij}^{\text{callosal}} = K_{\text{inter}} \cdot e^{-0.5 \cdot |\text{offset}|}$$

where:
- $K_{\text{inter}} = 0.15$ — inter-hemispheric base strength
- offset is the displacement from the homotopic position
- spread = $\min(3, N/2)$ — maximum callosal range

Homotopic connections (offset = 0) have full strength;
non-homotopic callosal fibres decay with offset distance.

### Component 3: Default Mode Network Hubs

The default mode network (DMN) consists of interconnected hub
regions that are more strongly coupled than their cortical-distance
neighbours would predict:

$$W_{ij}^{\text{DMN}} = \begin{cases} W_{ij} + K_{\text{hub}} & \text{if } i, j \in \text{DMN} \text{ and } i \neq j \\ W_{ij} & \text{otherwise} \end{cases}$$

where $K_{\text{hub}} = 0.3$ and the DMN nodes are placed at
cortical fractions $\{0.15, 0.45, 0.65, 0.85\}$ of each hemisphere,
representing approximately:
- 0.15: medial prefrontal cortex (mPFC)
- 0.45: posterior cingulate cortex / precuneus (PCC)
- 0.65: lateral parietal cortex
- 0.85: medial temporal lobe (MTL)

### Final Assembly

The raw matrix is symmetrised and cleaned:

$$W = \frac{W + W^T}{2}, \quad W_{ii} = 0, \quad W_{ij} \geq 0$$

This ensures undirected, non-negative coupling with no self-loops.

### Spectral Properties

The resulting matrix has characteristic spectral structure:
- $\lambda_2(L(W)) > 0$ (connected graph)
- Clear spectral gap between $\lambda_2$ and $\lambda_3$ when
  hemispheric structure is present
- DMN hubs appear as high-degree nodes in the degree distribution

---

## 2. Theoretical Context

### Why Synthetic Rather Than Real Data?

Real HCP data (Van Essen et al. 2013) requires:
1. Large download (~10 GB per subject for diffusion MRI)
2. Preprocessing pipeline (FreeSurfer + MRtrix3 tractography)
3. Parcellation-dependent matrix dimensions
4. Data use agreements

The synthetic generator provides:
- **Zero external dependencies** — no downloads, no licenses
- **Arbitrary size** — any $N$, not locked to parcellation atlas
- **Deterministic** — same seed → same matrix (reproducible tests)
- **Structurally realistic** — preserves the key architectural
  principles that affect synchronisation dynamics

### Neuroanatomical Basis

The three-component architecture is grounded in:

1. **Exponential distance decay** — Ercsey-Ravasz et al. (2013)
   showed that macaque cortical connectivity decays exponentially
   with inter-areal distance. This has been confirmed in human DTI
   tractography (Hagmann et al. 2008).

2. **Callosal homotopic bias** — Jarbo et al. (2012) demonstrated
   that corpus callosum fibres preferentially connect mirror
   regions across hemispheres. This homotopic bias is the strongest
   inter-hemispheric connectivity pattern.

3. **DMN hub structure** — The default mode network (Raichle et al.
   2001; Buckner et al. 2008) consists of hub regions with
   disproportionately strong interconnections. These hubs are critical
   for understanding resting-state dynamics and consciousness
   theories (Tononi & Koch 2015).

### Historical Context

- **Hagmann, P. et al. (2008):** First comprehensive mapping of the
  human structural connectome using diffusion spectrum imaging. 
  Identified exponential distance-decay as the primary connectivity
  principle.
- **Van Essen, D. C. et al. (2013):** The Human Connectome Project —
  1200 subjects, multimodal imaging, publicly available.
- **Cakan, C. & Obermayer, K. (2021):** neurolib — Python framework
  for whole-brain simulation with real HCP connectivity.
  Used the 80-region Desikan-Killiany parcellation.
- **Ercsey-Ravasz, M. et al. (2013):** Quantified exponential
  distance rule in macaque cortex.
- **Raichle, M. E. et al. (2001):** Discovery of the default mode
  network as a coherent resting-state network.

### Comparison: Synthetic vs Real HCP

| Property | Synthetic | Real HCP (neurolib) |
|----------|-----------|---------------------|
| Size | Any $N \geq 2$ | Fixed 80 regions |
| Dependencies | None | neurolib + data |
| Deterministic | Yes (seeded) | Fixed (subject average) |
| Hemispheric structure | Parametric | Anatomical |
| DMN hubs | At fixed fractions | At parcellation regions |
| Biological noise | Gaussian | Subject variability |
| Suitable for | Algorithm testing | Realistic simulation |

The `load_neurolib_hcp` function provides access to the real HCP
data (80 regions, Desikan-Killiany atlas) when neurolib is installed.

---

## 3. Pipeline Position

```
 Domain: neuroscience (EEG, fMRI, MEG)
                      │
                      ↓
 ┌── load_hcp_connectome(n_regions, seed) ────────┐
 │                                                 │
 │  Generates: symmetric (N, N) coupling matrix   │
 │  Components: intra-hemi + callosal + DMN hubs  │
 │  Rust path: deterministic native generation    │
 │                                                 │
 └──────────────────┬──────────────────────────────┘
                    │
                    ↓
         W = coupling matrix
                    │
                    ↓
         UPDEEngine.step(phases, omegas, W, ζ, ψ, α)
                    │
                    ↓
         compute_order_parameter(phases) → R, ψ
                    │
                    ↓
         Monitor / Supervisor / SSGF geometry control
```

### Alternative: Real HCP Data

```
 load_neurolib_hcp(n_regions=80)
            │
            ↓
 Real structural connectivity (80 × 80)
            │
            ↓
 Same downstream pipeline
```

### Input Contracts

**load_hcp_connectome:**

| Parameter | Type | Range | Default | Meaning |
|-----------|------|-------|---------|---------|
| `n_regions` | `int` | $\geq 2$ | — | Number of cortical regions |
| `seed` | `int` | any | 42 | Random seed for noise |

**load_neurolib_hcp:**

| Parameter | Type | Range | Default | Meaning |
|-----------|------|-------|---------|---------|
| `n_regions` | `int` | $[2, 80]$ | 80 | Regions to return |

### Output Contract

| Field | Type | Shape | Constraints |
|-------|------|-------|-------------|
| (return) | `NDArray[float64]` | `(N, N)` | Symmetric, $\geq 0$, diagonal = 0 |

---

## 4. Features

- **Three-component architecture** — intra-hemispheric decay, callosal
  connections, DMN hub boosting
- **Arbitrary size** — any $N \geq 2$, not locked to atlas parcellation
- **Deterministic** — same seed produces identical matrix
- **Symmetric** — $(W + W^T)/2$ ensures undirected coupling
- **Non-negative** — all entries clamped to $\geq 0$
- **Zero diagonal** — no self-coupling
- **Biological noise** — Gaussian perturbation for realism
- **Rust FFI acceleration** — 13-50x speedup over Python
- **Real data option** — `load_neurolib_hcp` for genuine HCP connectivity
- **Hemispheric structure** — enables study of inter-hemispheric
  synchronisation and callosal function
- **DMN modelling** — hub-and-spoke structure for consciousness research

---

## 5. Usage Examples

### Basic: Generate Connectome

```python
from scpn_phase_orchestrator.coupling.connectome import load_hcp_connectome

W = load_hcp_connectome(n_regions=80, seed=42)
print(f"Shape: {W.shape}")           # (80, 80)
print(f"Symmetric: {(W == W.T).all()}")  # True
print(f"Max coupling: {W.max():.4f}")
print(f"Min off-diag: {W[W > 0].min():.4f}")
```

### Whole-Brain Simulation

```python
import numpy as np
from scpn_phase_orchestrator.coupling.connectome import load_hcp_connectome
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 80
W = load_hcp_connectome(N)
eng = UPDEEngine(N, dt=0.01)
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
# Alpha band oscillators: 8-12 Hz
omegas = rng.uniform(8, 12, N) * 2 * np.pi
alpha = np.zeros((N, N))

for step in range(2000):
    phases = eng.step(phases, omegas, W, 0.0, 0.0, alpha)

R, psi = compute_order_parameter(phases)
print(f"R = {R:.4f}")
```

### Hemispheric Coherence Analysis

```python
import numpy as np
from scpn_phase_orchestrator.coupling.connectome import load_hcp_connectome
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 80
half = N // 2
W = load_hcp_connectome(N)

# Check hemispheric coupling structure
intra_left = W[:half, :half].sum()
intra_right = W[half:, half:].sum()
inter = W[:half, half:].sum()
print(f"Intra-left: {intra_left:.1f}")
print(f"Intra-right: {intra_right:.1f}")
print(f"Inter-hemi: {inter:.1f}")
print(f"Ratio intra/inter: {(intra_left + intra_right) / (2 * inter):.1f}")
```

### Compare Synthetic vs Real HCP

```python
import numpy as np
from scpn_phase_orchestrator.coupling.connectome import (
    load_hcp_connectome,
    load_neurolib_hcp,
)

W_synth = load_hcp_connectome(80)
try:
    W_real = load_neurolib_hcp(80)
    # Compare spectral properties
    L_synth = np.diag(W_synth.sum(axis=1)) - W_synth
    L_real = np.diag(W_real.sum(axis=1)) - W_real
    eig_synth = np.sort(np.linalg.eigvalsh(L_synth))
    eig_real = np.sort(np.linalg.eigvalsh(L_real))
    print(f"λ₂ (synth): {eig_synth[1]:.4f}")
    print(f"λ₂ (real):  {eig_real[1]:.4f}")
except ImportError:
    print("neurolib not installed — only synthetic available")
```

---

## 6. Technical Reference

### Function: load_hcp_connectome

::: scpn_phase_orchestrator.coupling.connectome

### Architectural Constants

| Constant | Value | Meaning | Source |
|----------|-------|---------|--------|
| `INTRA_HEMI_STRENGTH` | 0.5 | Base intra-hemispheric coupling | Hagmann et al. 2008 |
| `INTER_HEMI_STRENGTH` | 0.15 | Base callosal coupling | Jarbo et al. 2012 |
| `DMN_HUB_BOOST` | 0.3 | Additional DMN hub coupling | Buckner et al. 2008 |
| Decay rate (intra) | 0.3 | Exponential distance decay | Ercsey-Ravasz et al. 2013 |
| Callosal spread | 3 | Max offset for non-homotopic fibres | — |
| DMN fractions | [0.15, 0.45, 0.65, 0.85] | Hub positions | Raichle et al. 2001 |

### Rust Implementation

The Rust path (`connectome.rs`) decomposes the matrix generation
into four pure functions:

```rust
pub fn load_hcp_connectome(n_regions: usize, seed: u64) -> Vec<f64>
fn build_intra_hemi(knm: &mut [f64], n: usize, half: usize, seed: u64)
fn build_inter_hemi(knm: &mut [f64], n: usize, half: usize)
fn add_dmn_hubs(knm: &mut [f64], n: usize, half: usize)
fn symmetrise(knm: &mut [f64], n: usize)
```

The noise is generated using a 64-bit LCG (linear congruential
generator) for determinism without external dependencies. The LCG
constants are from Knuth's TAOCP.

### Python Implementation

The Python path uses `numpy.random.default_rng(seed)` for noise
generation. Due to different PRNG algorithms (NumPy uses PCG64,
Rust uses LCG), the Python and Rust paths produce **different**
matrices for the same seed. The structural properties (symmetric,
hemispheric, DMN hubs) are identical; only the noise differs.

### Auto-Select Logic

```python
try:
    from spo_kernel import load_hcp_connectome_rust as _rust_load_hcp
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
```

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
Median of 50-100 iterations.

| N | Python (µs) | Rust (µs) | Speedup |
|---|-------------|-----------|---------|
| 20 | 228.5 | 4.5 | **50.4x** |
| 80 | 843.7 | 47.8 | **17.6x** |
| 200 | 4045.3 | 301.9 | **13.4x** |

### Why 13-50x Speedup?

The Python path creates multiple intermediate NumPy arrays
(distance matrix, exponential, noise, clip) with associated memory
allocation overhead. The Rust path operates in-place on a single
`Vec<f64>`, avoiding all temporary allocations.

The speedup decreases with $N$ because the $O(N^2)$ computation
begins to dominate the fixed Python overhead.

### Memory Usage

- Matrix: $N^2$ floats (51.2 KB for $N = 80$)
- Python: ~4 temporary $N^2$ arrays (204.8 KB for $N = 80$)
- Rust: 1 allocation ($N^2$ output)

### Test Coverage

- **Rust tests:** 8 (connectome module in spo-engine)
  - Output size, diagonal zero, symmetric, non-negative,
    intra > inter, small N, deterministic, different seed
- **Python tests:** 21 (`tests/test_connectome.py`)
  - Shape, symmetry, non-negative, diagonal zero, hemispheric
    structure, DMN hubs, spectral properties, pipeline wiring,
    neurolib interface (optional), determinism, edge cases
- **Source lines:** 179 (Rust) + 144 (Python) = 323 total

---

## 8. Citations

1. **Hagmann, P., Cammoun, L., Gigandet, X., Meuli, R., Honey, C. J.,
   Wedeen, V. J., & Sporns, O.** (2008).
   "Mapping the structural core of human cerebral cortex."
   *PLoS Biology* 6(7):e159.
   DOI: [10.1371/journal.pbio.0060159](https://doi.org/10.1371/journal.pbio.0060159)

2. **Van Essen, D. C., Smith, S. M., Barch, D. M., Behrens, T. E. J.,
   Yacoub, E., & Ugurbil, K.** (2013).
   "The WU-Minn Human Connectome Project: An overview."
   *NeuroImage* 80:62-79.
   DOI: [10.1016/j.neuroimage.2013.05.041](https://doi.org/10.1016/j.neuroimage.2013.05.041)

3. **Cakan, C. & Obermayer, K.** (2021).
   "neurolib: A simulation framework for whole-brain neural mass
   modeling."
   *NeuroImage* 227:117474.
   DOI: [10.1016/j.neuroimage.2020.117474](https://doi.org/10.1016/j.neuroimage.2020.117474)

4. **Ercsey-Ravasz, M., Markov, N. T., Lamy, C., Van Essen, D. C.,
   Knoblauch, K., Toroczkai, Z., & Kennedy, H.** (2013).
   "A predictive network model of cerebral cortical connectivity
   based on a distance rule."
   *Neuron* 80(1):184-197.
   DOI: [10.1016/j.neuron.2013.07.036](https://doi.org/10.1016/j.neuron.2013.07.036)

5. **Raichle, M. E., MacLeod, A. M., Snyder, A. Z., Powers, W. J.,
   Gusnard, D. A., & Shulman, G. L.** (2001).
   "A default mode of brain function."
   *PNAS* 98(2):676-682.
   DOI: [10.1073/pnas.98.2.676](https://doi.org/10.1073/pnas.98.2.676)

6. **Buckner, R. L., Andrews-Hanna, J. R., & Schacter, D. L.** (2008).
   "The brain's default network: Anatomy, function, and relevance
   to disease."
   *Annals of the New York Academy of Sciences* 1124(1):1-38.
   DOI: [10.1196/annals.1440.011](https://doi.org/10.1196/annals.1440.011)

7. **Jarbo, K., Verstynen, T., & Schneider, W.** (2012).
   "In vivo quantification of global connectivity in the human
   corpus callosum."
   *NeuroImage* 59(3):1988-1996.
   DOI: [10.1016/j.neuroimage.2011.09.056](https://doi.org/10.1016/j.neuroimage.2011.09.056)

8. **Tononi, G. & Koch, C.** (2015).
   "Consciousness: Here, there and everywhere?"
   *Philosophical Transactions of the Royal Society B* 370(1668):20140167.
   DOI: [10.1098/rstb.2014.0167](https://doi.org/10.1098/rstb.2014.0167)

---

## Edge Cases and Limitations

### n_regions = 2

With only 2 regions, there is 1 region per hemisphere. The matrix
is 2×2 with a single off-diagonal entry representing the callosal
connection. No DMN structure is possible.

### Odd n_regions

The hemispheric split is $\lfloor N/2 \rfloor$ left, remainder right.
For odd $N$, the right hemisphere has one more region. The callosal
connections bridge the two unequal halves.

### Very Large N (> 1000)

The matrix grows as $O(N^2)$. For $N = 1000$, the matrix is 8 MB.
The generation time remains sub-millisecond in Rust but reaches
~4 ms in Python.

### PRNG Difference Between Python and Rust

The Python path uses PCG64 (NumPy default) and the Rust path uses
a 64-bit LCG for noise generation. This means **the exact matrices
differ** between backends for the same seed. The structural properties
(symmetry, hemispheric structure, DMN hubs) are preserved; only
the noise pattern differs.

For reproducible cross-backend comparisons, use `seed=42` and
compare statistical properties (mean, std, spectral gap) rather
than element-wise values.

### Not Real Brain Data

This generator captures qualitative architectural principles, not
quantitative fibre counts. For realistic whole-brain simulation,
use `load_neurolib_hcp` with real HCP data. The synthetic generator
is intended for algorithm development and testing.

---

## Troubleshooting

### Issue: neurolib Import Fails

**Symptom:** `ImportError: neurolib is required for real HCP data`

**Diagnosis:** The `load_neurolib_hcp` function requires the
`neurolib` package. It is not installed by default because it brings
heavy dependencies (numba, tqdm, xarray).

**Solution:** `pip install neurolib`. Only needed for real HCP data;
the synthetic generator works without any external dependencies.

### Issue: Matrices Differ Between Python and Rust

**Diagnosis:** This is expected — the PRNG algorithms differ
(PCG64 vs LCG). See §Edge Cases above.

**Solution:** Compare structural properties (spectral gap, degree
distribution, hemispheric ratio) rather than element-wise values.
For deterministic cross-backend comparison, implement the same PRNG
in both backends (future work).

### Issue: Spectral Gap Too Small

**Diagnosis:** For large $N$ with default parameters, the long-range
connections become very weak ($e^{-0.3 \cdot N/2}$), leading to
near-disconnection between distant regions.

**Solution:** Increase `INTRA_HEMI_STRENGTH` or decrease the decay
rate. Alternatively, use a custom coupling matrix with the
`CouplingBuilder`.

### Issue: DMN Hubs Not Visible in Degree Distribution

**Diagnosis:** The DMN boost ($K_{\text{hub}} = 0.3$) adds to
all DMN-to-DMN connections but not DMN-to-non-DMN. For small $N$,
the degree boost may be masked by the intra-hemispheric baseline.

**Solution:** For $N < 16$, increase the DMN fraction or boost
value. The effect is clearest for $N \geq 40$ where the DMN nodes
are well-separated in index space.

### Issue: Inter-Hemispheric Synchronisation Too Weak

**Diagnosis:** The callosal strength (0.15) is 3.3x weaker than
intra-hemispheric strength (0.5). This mimics the biological
ratio but may be too weak for applications requiring strong
bilateral coupling.

**Solution:** Scale the inter-hemispheric connections post-hoc:
```python
W = load_hcp_connectome(80)
half = 40
W[:half, half:] *= 2.0  # Double callosal strength
W[half:, :half] *= 2.0
```

---

## Integration with Other SPO Modules

### With OttAntonsenReduction

The OA reduction assumes all-to-all coupling. The HCP connectome
is sparse and heterogeneous. For OA validation, use a reduced
"mean-field effective coupling" derived from the connectome:

$$K_{\text{eff}} = \frac{1}{N} \sum_{i,j} W_{ij}$$

### With SSGF Geometry Control

The SSGF engine can use the HCP connectome as the initial geometry:

```python
W = load_hcp_connectome(80)
# SSGF will adapt W to optimise synchronisation
# while respecting the connectome's structural constraints
```

### With Sleep Staging Monitor

Brain-state-dependent connectivity changes can be modelled by
scaling the connectome:

```python
# Wake: full connectivity
W_wake = load_hcp_connectome(80)
# NREM: reduced long-range connectivity
W_nrem = W_wake.copy()
W_nrem[:40, 40:] *= 0.5  # Reduced callosal
W_nrem[40:, :40] *= 0.5
```
