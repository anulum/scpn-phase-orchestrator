# Findings from sc-neurocore for scpn-phase-orchestrator
**Source:** sc-neurocore v3.14.0, 29 notebooks, 2928+ tests, 340 new session tests
**Date:** 2026-03-29
**Purpose:** What the SNN engine has validated that extends or constrains SPO

---

## 1. WHAT SC-NEUROCORE HAS VALIDATED

### Network Topology Generators — Verified
6 generators tested (26 tests, all passing):
- `random_connectivity(n, n, p, w)`: Erdos-Renyi, correct density
- `small_world(n, k, p_rewire, w)`: Watts-Strogatz, verified symmetric
- `scale_free(n, m, w)`: Barabasi-Albert, hub degree > 3× mean
- `ring_topology(n, k, w)`: exact degree 2k, wrap-around
- `grid_topology(r, c, radius, w)`: corner=3, centre=8, no self-loops
- `all_to_all(n_src, n_tgt, w)`: full matrix

**Relevance:** Your NB36 (quantum-control) found small-world gets the LARGEST
FIM boost (+0.31). Our topology generators can produce these structures for
the SNN. SPO should test FIM on the same topologies using our generators
as reference implementations.

### Equation-to-Verilog Compiler
We compile arbitrary ODE neuron equations to Q8.8 Verilog:
- LIF: `dv/dt = -(v - E_L)/tau_m + I/C`
- FitzHugh-Nagumo: 2 coupled ODEs
- Izhikevich: v² term → chained multiply-shift

**Relevance:** SPO's Kuramoto ODE `dθ/dt = ω + Σ K sin(Δθ)` could be
compiled to FPGA using the same pipeline. The sin() term maps to a 16-entry
piecewise-linear LUT (~1-2% accuracy). For SPO's N=512 oscillators on
Artix-7 100T: ~63400 / 50 ≈ 1268 oscillators fit.

### YOUR FINDING #7 CONFIRMED IN SNN CONTEXT
K symmetry breaks after STDP updates. This is the SAME phenomenon as your
gradient training breaking K = K^T after ~30 Adam steps.

**Our measurement:** After 1 second of STDP (1000 steps at dt=1ms), weight
asymmetry |W - W^T| / |W| grows to ~5-15% depending on input statistics.

**Our action:** Implementing W = (W + W^T) / 2 enforcement after each STDP
batch, per your recommendation. We prefer option 1 (enforce symmetry) over
option 2 (directed Kuramoto) because quantum-control showed directed coupling
HURTS sync (K_c +12%).

### YOUR FINDING #6 CONFIRMED: Float32 Phase Drift
sc-neurocore SCPN layers use float32 for phase variables. Over 10K steps,
accumulated drift = 10000 × 1.3e-4 = 1.3 rad ≈ 20% of full circle.

**Our action:** Upgrading phase-sensitive computations (SCPN L2 Kuramoto,
sheaf defect, connection curvature) to float64. Spike computations remain
float32 (phase drift irrelevant for binary spike output).

### SC Arithmetic Convergence — New for SPO
We proved (test_sc_convergence.py, 13 tests):
- AND multiplication converges O(1/√L)
- Sobol quasi-random converges O(1/L) — quadratically faster
- CORDIV division is monotonic
- Decorrelation is critical (correlated inputs produce biased results)
- Hoeffding bound L ≥ ln(2/δ) / (2ε²) verified empirically

**Relevance:** If SPO ever maps to SC hardware, the Sobol encoder should be
used (10× fewer bits for same precision). Bitstream length L=1024 gives ~3%
error — sufficient for Kuramoto coupling computation.

---

## 2. WHAT WE NEED FROM SPO

### Ott-Antonsen Reference Data
You validated R = sqrt(1 - K_c/K) at N=512. Can you provide:
1. Raw (K, R) data points for comparison with SNN R(K) sweep?
2. Finite-size corrections at N=100, N=1000?
3. Time-to-sync vs K near K_c?

### Analytical Inverse for STDP Validation
SPO achieves >0.90 coupling recovery for N≤16. After SNN training with STDP,
we want to extract K_eff using SPO's `analytical_inverse()` and compare.
Can you expose this as a standalone function we can call?

### Stuart-Landau Parameters for LIF
You validated Hopf bifurcation r → sqrt(mu). What is the mapping from
LIF neuron parameters (tau_m, v_threshold, I_drive) to Stuart-Landau (mu, gamma)?
This would let us verify SNN amplitude dynamics against SPO's validated baseline.

### GPU Benchmark Comparison
SPO gets 19.4× JAX speedup at N=2048. sc-neurocore gets 39-202× Rust speedup
over Python (different metric — per-neuron, not per-oscillator). Can you run
SPO on the same hardware (AMD EPYC / A6000) for direct comparison?

---

## 3. THREE-CODEBASE ALIGNMENT TABLE

| Property | quantum-control | phase-orchestrator | sc-neurocore |
|----------|:-:|:-:|:-:|
| Kuramoto ODE | ✓ (classical + quantum) | ✓ (RK4 O(dt⁴)) | ✓ (SCPN L2) |
| K_nm matrix | ✓ (r=0.951 PhysioNet) | ✓ (Kuramoto input) | ✓ (build_knm tested) |
| FIM feedback | ✓ implemented | ✗ planned | ✗ **implementing now** |
| K symmetry | inherent (Pauli) | breaks after 30 steps | breaks after STDP → **fixing** |
| Float64 phases | ✓ | ✓ (JAX default) | ✗ float32 → **upgrading** |
| Topology generators | N/A | N/A | ✓ (6 generators, 26 tests) |
| FPGA backend | N/A | N/A | ✓ (equation-to-Verilog + formal) |
| Universality class | BKT (β→0) | mean-field (β=0.5) | **untested — next experiment** |
| Spike-based ALU | N/A | N/A | ✓ (Turing-complete, 27 tests) |
| SC fault tolerance | N/A | N/A | ✓ (14 tests, TMR) |

---

## 4. KEY NUMBERS FOR SPO

| Metric | Value |
|--------|-------|
| SNN max tested | 50K neurons (benchmark), 200 (notebooks) |
| Rust engine speedup | 39-202× vs Brian2 |
| STDP symmetry drift | 5-15% after 1s |
| FPGA capacity (Artix-7 100T) | ~1200 LIF neurons |
| SC encoding error (L=1024) | ~3% |
| Sobol advantage | ~10× fewer bits vs Bernoulli |
| Phase drift (float32) | 1.3e-4 rad/step |
| Notebook coverage | 29 notebooks, all executable |
| Test coverage | 100% core (2928+ pytest) |
