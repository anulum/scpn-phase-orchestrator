# Synthetic and optional HCP connectomes

`load_hcp_connectome(n_regions, seed=42)` produces synthetic structural weights
for algorithm development. `load_neurolib_hcp(n_regions=80)` reads the optional
neurolib HCP dataset. Both return an independent writable C-contiguous
`float64` square matrix with finite non-negative symmetric weights and exact
zero diagonal. Supply that matrix explicitly to an integrator or monitor.

The synthetic constants and region-index distances are heuristics. They are
not measured fibre counts, geodesic cortical distances or calibrated anatomical
coordinates. The background references below explain the architectural motifs;
they do not establish these numerical parameters or biological validity.

## Public contracts

| Loader | Region count | Seed | Data and dependencies |
|---|---|---|---|
| `load_hcp_connectome` | Genuine non-boolean integer, at least 2; dense storage must be addressable | Genuine non-boolean integer in `0..2**64-1`, default 42 | NumPy; original Rust builtin when available |
| `load_neurolib_hcp` | Genuine non-boolean integer in `2..80`, default 80 | No seed | Original neurolib dataset and its dependencies |

NumPy integer scalars are admitted. Boolean, text, float, complex and temporal
metadata aliases are refused before allocation or optional dataset I/O. Count
or seed type errors raise `TypeError`; invalid ranges, dense-storage overflow,
allocation refusal and invalid structural weights raise `ValueError`. Missing
neurolib raises `ImportError`. Invalid native output is refused without changing
the data source.

For synthetic generation, `8 * n_regions**2` must fit the platform's signed
addressable byte range. This check does not certify available RAM. Python
allocation failures and Rust reservation failures become a public `ValueError`.
Validation and intermediate arrays consume additional memory. Up to 128
synthetic matrices are cached across actual generator/count/seed keys; every
public call returns a separate copy. Warm-cache copies and cold generation
have different costs.

Producer types are checked before float conversion: boolean, complex,
numeric-string and temporal aliases are refused. Finite real numeric object
storage remains compatible. Shape, finiteness, non-negativity and symmetry
(`atol=1e-12`, `rtol=0`) are checked. Synthetic output must already have an
exact zero diagonal, including refusal of subnormal self-edges. The HCP loader
admits the provider matrix first, copies its requested slice, then clears the
provider diagonal before final structural validation.

## Synthetic construction

Let `h = floor(N/2)`. The left half contains `h` nodes; the right half contains
`N-h`. Region indices are synthetic coordinates.

Within each half, each directed off-diagonal weight starts as

$$A_{ij} = \max(0,\;0.5e^{-0.3|i-j|}+\epsilon_{ij}).$$

Python draws PCG64 Gaussian perturbations with standard deviation `0.02` for
each complete block, including diagonal draws that are subsequently cleared.
Rust advances a wrapping 64-bit LCG for off-diagonal entries only and maps its
upper 31 bits to a uniform perturbation in `[-0.02, 0.02)`. Each owner is seeded
and repeatable. Their noise distributions and draw schedules differ, so the
same seed does not imply elementwise Python/Rust equality.

For left index `a` and right-local index `b < h`, callosal weights are

$$A_{a,h+b}=A_{h+b,a}=0.15e^{-0.5|a-b|},$$

when `|a-b| <= min(3,h)`; remaining cross-half entries start at zero. For odd
counts the extra right node has intra-half edges rather than a homotopic pair.

Hub fractions are `[0.15, 0.45, 0.65, 0.85]`. Their left indices are
`floor(f*h)` and their right indices add `h`. Repeated indices retain their
historical multiplicity. If `m_i` counts occurrences of index `i`, each
unequal hub pair receives `0.3*m_i*m_j`. At `N=2`, each node occurs four times,
so both off-diagonal entries are exactly `0.15 + 16*0.3 = 4.95`, independently
of the seed. These fractions are synthetic placements rather than atlas labels.

Finally, average the two directions, clear the diagonal and clamp weights to
zero. No spectral-gap, connectivity, synchronisation-regime or brain-state
outcome is guaranteed for arbitrary sizes and seeds.

## Original neurolib HCP ingress

The qualified neurolib 0.6.2 `Dataset("hcp")` provider reads its packaged HCP
subject matrices. Its source uses the AAL2/LRLR ordering, removes designated
subcortical indices from the original 94-region arrays, normalises each subject
by its own maximum and averages the seven subject structural matrices to
80 cortical regions. `load_neurolib_hcp(k)` returns the top-left `k`-region
slice in that existing ordering; it does not select or fit another parcellation.

The synthetic left/right block ordering and the real provider ordering differ.
Do not interpret a top-left slice or a synthetic half as an anatomical selection
in the other dataset. No clinical or physiological calibration follows from
loading either matrix. Review the upstream dataset terms before redistributing
its assets; this package provides the loader, not a copied dataset.

## Explicit downstream use

```python
import numpy as np
from scpn_phase_orchestrator.coupling.connectome import load_hcp_connectome
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

n = 20
weights = load_hcp_connectome(n, seed=42)
phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
engine = UPDEEngine(n, dt=0.01, method="rk4")
for _ in range(300):
    phases = engine.step(phases, np.ones(n), weights, 0.0, 0.0, np.zeros_like(weights))
coherence, mean_phase = compute_order_parameter(phases)
assert np.isfinite(coherence) and np.isfinite(mean_phase)
```

To use original HCP weights, install neurolib and replace the matrix source:

```python
from scpn_phase_orchestrator.coupling.connectome import load_neurolib_hcp

weights = load_neurolib_hcp(20)
assert weights.shape == (20, 20)
```

The loaders do not attach a supervisor, change SSGF geometry or infer a sleep
stage. Those consumers require explicit configuration. Independent scalar
Euler/RK4 trajectory, torus, coherence and regime oracles exercise original
synthetic and HCP downstream consumers in the owning tests.

## Native boundary

The registered original `spo_kernel.load_hcp_connectome_rust(n_regions, seed)`
builtin preserves its two-argument ABI and flat row-major `float64` output of
length `n_regions**2`. It independently refuses metadata aliases, counts below
two, square/byte overflow and allocation refusal. The Python wrapper reshapes
and validates that original result.

Rust core callers can use
`try_load_hcp_connectome(usize, u64) -> Result<Vec<f64>, &'static str>`.
The retained `load_hcp_connectome(usize, u64) -> Vec<f64>` core API preserves its
historical all-zero outputs at counts zero/one and panics on storage or
allocation failure. Foreign callers use the checked API, so those failures
become `ValueError` rather than an indexing panic.

## Current measurements

The [current local comparison](../data/connectome_comparison.local.json) records
real installed Python and original Rust owners using the same NumPy 2.5.3,
current production source hashes and binary identity. Each case has 20 batches:
100 unique-seed cold calls or 2,000 already-cached fixed-seed copies. Full
owner-specific independent matrix oracles pass before timing admission.

The complete Rust release production source prefix remains byte-identical
after subsequent test-only additions; its identity is recorded with the
original measured file and current candidate hashes.

These are local diagnostics on a shared, unreserved workstation. A separate owning-test cohort ran concurrently; its profile-mutation
negatives restore the original bytes and all comparison source/matrix/call
checks passed. The source-reference comment correction leaves the complete
Python AST unchanged, as bound in the numeric artifact. Cold public calls include
original generation, foreign marshalling where applicable, structural validation
and a copy; they are not Rust-core timings or a controlled speed-up claim.

| Regions | Mode | Public Python (µs) | Public Rust (µs) |
|---:|---|---:|---:|
| 16 | cold | 979.94 | 1585.98 |
| 16 | warm | 8.94 | 7.78 |
| 64 | cold | 11851.53 | 18299.30 |
| 64 | warm | 9.50 | 9.27 |
| 256 | cold | 124676.69 | 213088.56 |
| 256 | warm | 44.13 | 35.62 |

| Regions | Uncached Rust core median sample (µs) |
|---:|---:|
| 16 | 1.93 |
| 64 | 19.39 |
| 256 | 445.47 |

For reproduction with trusted operator-owned installations of the same current
source and Python binary:

```bash
python -m benchmarks.connectome_benchmark --python /path/to/python-profile/bin/python --rust /path/to/rust-profile/bin/python
```

The Python profile must actually lack `spo_kernel`; the Rust profile must expose
the original compiled builtin. The runner records source/binary identities and
observed native calls, refuses mismatched installed source, and checks copy
ownership. `make connectome-quality` checks owning source, consumers and
benchmarks. CI separately qualifies actual installed owners and all original
contracts, with a 100% line-and-branch gate for both Python production modules.

::: scpn_phase_orchestrator.coupling.connectome

## Background references

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

3. **Cakan, C., Jajcay, N., & Obermayer, K.** (2023; online 2021).
   "neurolib: A Simulation Framework for Whole-Brain Neural Mass Modeling."
   *Cognitive Computation* 15:1132–1152.
   DOI: [10.1007/s12559-021-09931-9](https://doi.org/10.1007/s12559-021-09931-9)

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
