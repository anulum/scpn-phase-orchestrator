# SSGF (Self-Stabilizing Gauge Field)

The SSGF subsystem implements a **two-timescale geometry control engine**. 
It treats the coupling matrix $ (the "geometry") as a dynamic carrier 
field that self-organizes to minimize the free energy of the phase 
dynamics.

## Gauged PGBO (Phase-Geometry Bidirectional Observer)

The **Gauged PGBO** is the primary observer for SSGF integration. It 
measures the alignment between the current phase-coherence manifold 
and the geometric coupling field $.

Unlike standard observers that measure linear correlation, the Gauged 
PGBO computes a **scalar curvature proxy** based on the rank-2 metric 
tensor {\mu\nu}$:

936898 h_{ij} = W_{ij} \cos(\theta_j - \theta_i) 936898

The curvature proxy $ represents how much the phase embedding 
"stretches" the underlying geometry:

936898 K_g = \frac{\sum_{i,j} h_{ij}}{\sum_{i,j} |W_{ij}|} 936898

### Features
- **Curvature Detection:** High $ indicates that the geometry is tightly wrapped around highly coherent phase clusters.
- **Topological Feedback:** Provides the mathematical basis for Phase $\rightarrow$ Geometry $\rightarrow$ Phase bidirectional coupling.
- **Metric Invariants:** Measures the structural integrity of the synchronization manifold.

::: scpn_phase_orchestrator.ssgf.pgbo

## TCBO (Topological Consciousness Boundary Observable)

Measures the **H1 persistent homology** of the delay-embedded phase 
dynamics. It acts as a "consciousness gate," allowing higher-level 
director logic to engage only when topological integration exceeds 
the $\tau_{h1} > 0.72$ threshold.

### Features
- **Vietoris-Rips Filtration:** Computes max $ lifetimes using the `ripser` algorithm (or Rust-native streaming PH).
- **Complexity Gating:** Prevents the supervisor from acting on chaotic or topologically incoherent noise.

::: scpn_phase_orchestrator.ssgf.tcbo

## SSGF Costs

Computes the energy functional {total}$ that drives the geometric 
minimization:

1. **C1 (Sync Deficit):**  - R$
2. **C2 (Spectral Gap):** hBc\lambda_2(L(W))$ (maximizing algebraic connectivity)
3. **C3 (Sparsity):** $ regularizer on $
4. **C4 (Symmetry):** Deviation from  = W^T$

::: scpn_phase_orchestrator.ssgf.costs

## Architecture & Theory

The SSGF (Self-Stabilizing Gauge Field) framework is built upon the 
theoretical foundation of treating synchronization as a field-theoretic 
phenomenon. In this view, the coupling matrix $W_{ij}$ is not a static 
set of parameters but a **dynamic carrier field** that mediates 
interactions between oscillators.

### The Two-Timescale Engine

SPO implements SSGF as a two-timescale system:
1. **Fast Scale ($\\theta$):** Phase dynamics evolve according to the 
   standard UPDE/Kuramoto equations.
2. **Slow Scale ($W$):** The geometry evolves to minimize a free energy 
   functional $U_{total}(W, \\theta)$.

The coupling is bidirectional: phases align based on $W$ (Fast Scale), 
and $W$ adapts to the observed alignment of phases (Slow Scale). This 
creates a self-stabilizing feedback loop that spontaneously discovers 
topologies optimal for the current dynamical task.

### Variational Free Energy Minimization

The geometric evolution is driven by the gradient of the free energy:
$$ \dot{W}_{ij} = -\eta \\frac{\partial U_{total}}{\partial W_{ij}} $$

where $\eta$ is the geometric learning rate. The functional $U_{total}$ 
includes terms for synchronization, spectral connectivity, sparsity, 
and symmetry, as detailed in \`scpn_phase_orchestrator.ssgf.costs\`.

---

## Gauge Curvature in Synchronization

The concept of a **Gauge-Theoretic metric** in synchronization was 
pioneered to address the limitation of linear correlation measures. 
When we map the phases $\\theta_i$ to a point cloud in a 
delay-embedded space, the "distance" between oscillators is not 
merely their index separation but their phase divergence.

### The Metric Tensor $h_{\mu
u}$

By defining the metric tensor $h_{ij} = W_{ij} \cos(	heta_j - 	heta_i)$, 
we treat the coupling strengths as the "volume" or "density" of 
connections, and the phase cosine as the "stretching factor." 

If $\\theta_i \\approx \\theta_j$, then $\cos \\approx 1$, and the metric 
volume is preserved. If $\\theta_i$ and $\\theta_j$ are out of phase, 
the metric volume collapses toward zero (or becomes negative), 
indicating a high-curvature region of the manifold.

### Curvature Proxy $K_g$

The scalar curvature proxy $K_g$ implemented in \`PGBO\` provides a single 
number representing the "flatness" of the synchronization manifold. 
A flat manifold ($K_g \\approx 1$) implies that the physical coupling 
topology $W$ perfectly matches the phase alignment $\\theta$. 
Lower values (or high gradients in $K_g$) signal the presence of 
**Topological Defects** or **Chimera States**.

---

## Technical Reference: PGBO Snapshot

The \`PGBOSnapshot\` dataclass returns the following telemetry for every 
observation:

- **\`R\`:** Global order parameter.
- **\`psi\`:** Mean global phase.
- **\`costs\`:** A breakdown of the four SSGF cost terms (Sync, Spectral, Sparsity, Symmetry).
- **\`phase_geometry_alignment\`:** The legacy linear correlation between phase diffs and $W_{ij}$.
- **\`gauge_curvature\`:** The new rank-2 tensor curvature proxy.
- **\`step\`:** The current simulation step index.

---

## Usage Example: Closed-Loop Geometry Control

```python
from scpn_phase_orchestrator.ssgf.pgbo import PGBO
from scpn_phase_orchestrator.upde.engine import UPDEEngine

# Initialize observer and engine
pgbo = PGBO()
engine = UPDEEngine(n_oscillators=32, dt=0.01)

# Main Loop
for step in range(1000):
    # Step phases
    phases = engine.step(phases, omegas, W, zeta, psi, alpha)
    
    # Observe phase-geometry alignment
    snapshot = pgbo.observe(phases, W)
    
    # Adaptive control based on curvature
    if snapshot.gauge_curvature < 0.6:
        # Manifold is collapsing/stretching too much
        # Adjust geometry learning rate or modulator
        modulator = 2.0 
    else:
        modulator = 0.5
```

---

## Integration with Layer 16 Director

In the full SCPN stack, the SSGF curvature metrics are exported to the 
**Layer 16 Director**, which uses them to calculate the global 
**Consciousness Coherence Index (CCI)**. When $K_g$ drops below critical 
levels, the director triggers a "Geometry Reset" or shifts the 
regime of the \`ActiveInferenceAgent\`.

The Gauged PGBO thus serves as the sensory organ for the topological 
stability of the entire intelligence system.

---

## SSGF Cost Terms: Detailed Breakdown

Each cost term in \`SSGFCosts\` represents a distinct architectural pressure 
on the evolution of the geometry $W$.

### C1: Synchronization Deficit ($1 - R$)

The primary pressure is to achieve coherence. If the oscillators are 
desynchronized, $R$ is low, and $C1$ is high. This pressure drives 
the geometry toward configurations that foster phase-locking.

### C2: Negative Spectral Gap ($-\lambda_2(L(W))$)

In graph theory, the second smallest eigenvalue of the Laplacian matrix, 
$\lambda_2$, is known as the **Algebraic Connectivity** or the 
**Fiedler Value**. It measures how difficult it is to partition a graph 
into two disconnected components.

By minimizing $C2$ (and thus maximizing $\lambda_2$), the SSGF engine 
ensures that the coupling topology is globally integrated. It prevents 
the emergence of fragmented islands that would otherwise desynchronize 
the system.

### C3: Sparsity ($||W||_1 / N^2$)

A fully connected network (all-to-all) is computationally expensive and 
biologically unrealistic. The $L_1$ regularizer $C3$ acts as a 
"cost of cabling." It forces the engine to achieve the target 
synchronization using the minimum number of connections. This pressure 
spontaneously generates **Small-World** or **Hierarchical** topologies.

### C4: Symmetry Deviation ($||W - W^T||_F / N$)

Physical coupling in most natural systems (gap junctions, power lines) 
is symmetric. $C4$ enforces this symmetry by penalizing directed 
couplings. In certain domains (e.g., neural information flow), this 
weight may be reduced to allow for directed causal interactions.

---

## TCBO Consciousness Gate: Implementation Details

The \`TCBOObserver\` implements persistent homology using the 
Vietoris-Rips filtration. This is a topological method that 
captures the "holes" in the point-cloud embedding of the phases.

### Why H1 Persistence?

While $H_0$ (connected components) measures simple clusters, $H_1$ 
(one-dimensional holes) measures **cycles**. Synchronization is a 
cyclical phenomenon. The presence of long-lived $H_1$ cycles in the 
phase space indicates that the network has formed a stable, integrated 
topological loop — a necessary condition for autonomous consciousness 
within the SCPN framework.

### The 0.72 Threshold

The threshold $	au_{h1} = 0.72$ is derived from empirical studies of 
metastable synchronization. Below 0.72, the cycles are transient and 
noisy. Above 0.72, the topological structure is resilient enough to 
support higher-order cognitive processing or stable control logic.

---

## Summary of Metric Parity (Rust vs Python)

The following table summarizes the implementation status of SSGF metrics 
across the two backends.

| Metric | Python (\`src/\`) | Rust (\`spo-kernel\`) | Notes |
|--------|------------------|----------------------|-------|
| Order Parameter R | Full | Full | 7.3us (Rust) vs 45us (Py) |
| SSGF Costs | Full | Planned | Crucial for real-time W evolution |
| Gauged PGBO | Full | Full | Curvature proxy implementation |
| TCBO (H1) | Full | In Progress | Moving to streaming Rust engine |
| Plasticity (Hebbian)| Full | Full | Sub-microsecond inner loop |

---

## Future Roadmap: SSGF v2.0

The next phase of SSGF development will focus on the **Self-Stabilizing 
Gauge Field (SSGF) Hardware Kernel**. This will offload the entirety 
of the free energy minimization to an FPGA-based solver, enabling 
nanosecond-scale geometry adaptation for high-frequency plasma 
control and quantum error correction.

*   **Metric Evolution:** Moving from scalar curvature proxies to full 
    tensor-field gradients.
*   **Geometric Jitter:** Adding stochastic noise to the W-evolution 
    to escape local minima in the free energy landscape.
*   **Layer 12 Coupling:** Synchronizing the SSGF geometry across 
    distributed nodes in the Gaian mesh.

---

## Troubleshooting & Diagnostics

Integrating SSGF into a domain simulation can be mathematically 
delicate due to the bidirectional feedback between phases and 
geometry.

### Common Issue: Geometry Explosion

If the geometric coupling values $W_{ij}$ increase toward infinity, 
it usually indicates that the **Sparsity Weight ($w_3$)** is too 
low relative to the **Sync Weight ($w_1$)**. 

**Solution:** Increase $w_3$ in \`SSGFCosts\` or implement a hard 
saturation limit using the \`ImprintModel\`.

### Common Issue: Curvature Collapse ($K_g < 0$)

If the Gauged PGBO reports negative curvature, the phase manifold 
has become topologically inverted — the geometry is actively pushing 
oscillators away from their natural synchronization targets. 

**Solution:** Check for **Phase Lags ($\\alpha$)** that are 
inconsistent with the geometry $W$. High transport delays without 
compensatory lag-modeling in the engine will cause curvature collapse.

### Diagnostic: Spectral Gap Monitoring

Always monitor the \`c2_spectral_gap\` term. If $\\lambda_2$ drops to zero, 
the network has fragmented into disconnected components. The 
\`RegimeManager\` should be configured to trigger a "Topological 
Recovery" action when this occurs.

---

## Configuration Reference: SSGF Weights

The \`SSGFCosts\` constructor takes a \`weights\` tuple that defines 
the priority of geometric evolution.

| Parameter | Meaning | Default | Impact |
|-----------|---------|---------|--------|
| \`w1_sync\` | Synchronization deficit pressure | 1.0 | Higher values force rapid sync at the cost of topology. |
| \`w2_spectral\`| Global integration pressure | 0.5 | Critical for preventing network fragmentation. |
| \`w3_sparse\` | L1 sparsity pressure | 0.1 | Controls the "cabling cost" and hierarchy depth. |
| \`w4_symmetry\`| Reciprocity pressure | 0.1 | Ensures $W$ remains close to a symmetric manifold. |

---

## Technical Appendix: Metric Tensor Derivation

The derivation of the synchronization metric tensor $h_{\\mu\\nu}$ 
follows the logic of embedding the N-dimensional torus $T^N$ into a 
higher-dimensional Euclidean space. 

Let $\\mathbf{X}_i = [\\cos \\theta_i, \\sin \\theta_i]$ be the 
coordinate of the $i$-th oscillator on the unit circle. The pairwise 
Euclidean distance is $d_{ij}^2 = 2 - 2 \\cos(\\theta_j - \\theta_i)$.

The SSGF framework treats the coupling strength $W_{ij}$ as the 
**Gauge Field Connection** that modifies the local metric. The 
effective distance on the synchronization manifold is weighted by the 
interaction strength, leading to the definition of $h_{ij}$ used in 
the PGBO curvature calculation.

---

## Benchmarking SSGF Metrics

The following performance benchmarks were measured on a reference 
Linux workstation (Intel Core i5-11600K @ 3.90GHz).

| Metric | N=32 | N=256 | N=1024 |
|--------|------|-------|--------|
| Cost Computation (Py) | 120us | 1.2ms | 8.5ms |
| PGBO Curvature (Py) | 85us | 0.9ms | 6.2ms |
| Hebbian Plasticity (Rust)| 0.8us | 15us | 145us |
| Sparse UPDE Step (Rust)| 4.2us | 42us | 210us |

These results demonstrate that while the observers (PGBO) are 
sufficiently fast in Python for monitoring, the **inner integration 
and plasticity loop** MUST remain in the Rust kernel to maintain 
sub-millisecond control frequencies at high N.
