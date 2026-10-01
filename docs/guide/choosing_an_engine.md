<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Choosing an engine

SCPN Phase Orchestrator ships more than a dozen phase-dynamics engines. Almost
every user needs only one of them. This page is the shortlist: the golden path
first, then when to reach for a specialised engine, and where each is documented.

## Start here — `UPDEEngine` via `simulate()`

The **golden path** is the general `UPDEEngine` (a Kuramoto UPDE integrator with
pre-allocated scratch arrays) driven through
[`simulate()`][scpn_phase_orchestrator.runtime.simulation.simulate]. You do not
construct the engine yourself: you write a binding spec and run it.

```bash
spo quickstart power --steps 200 --seed 7   # validate → run → replay → report
```

```python
from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.runtime.simulation import simulate

spec = load_binding_spec("domainpacks/minimal_domain/binding_spec.yaml")
result = simulate(spec, steps=200, seed=7)
print(result.r_good, result.separation, result.final_regime)
```

This path gives you the supervisor policy loop, the conformal admission gate, the
audit log (which now [fails closed](../guide/production.md) on a broken event
stream), and replayable evidence — the whole assurance envelope. Reach for a
different engine only when your dynamics genuinely differ from first-order
Kuramoto phase coupling.

## Supported specialised modes

These have a dedicated, documented entry point:

| Engine | When to use | Where |
| --- | --- | --- |
| `StuartLandauEngine` | Phase **and** amplitude matter (limit-cycle oscillators). | [Stuart-Landau Amplitude Mode](stuart_landau.md) |
| `SparseUPDEEngine` | Sparse graphs requiring `O(N + E)` storage and derivative work. | [CSR contract](../reference/api/upde.md#sparse-engine) |
| `JaxUPDEEngine` | JIT integration on the configured JAX device, with validated NumPy host readback (needs the `nn` extra). | [JAX contract](../reference/api/upde.md#jax-accelerated-kuramoto-engine) |

For device-array autodiff, use the [dense/masked Kuramoto functional API](../reference/api/nn.md#kuramoto-model).
Both JAX Kuramoto surfaces canonicalise phases in their configured precision;
gradients are defined away from the wrap cut. CPU execution is supported and
does not imply a hardware accelerator. Standard dense and CSR steps retain
their adaptive proposal on numerical refusal and accept subsequent valid input.

## Specialised dynamics engines (advanced)

The remaining engines model specific physics. They are research surfaces — powerful
where they fit, but outside SPO's externally validated niche (see below). Each is
documented in [Advanced Dynamics](advanced_dynamics.md):

| Engine | Dynamics |
| --- | --- |
| `DelayedEngine` | Kuramoto with time-delayed coupling. |
| `DopplerEngine` | Graph-weighted Doppler velocity correction. |
| `HypergraphEngine` | Arbitrary k-body (hypergraph) coupling. |
| `InertialKuramotoEngine` | Second-order swing-equation Kuramoto (power grids). |
| `MovingFrameUPDEEngine` | Chamber-frame axial positions with collision checks (fusion / MIF). |
| `SheafUPDEEngine` | Cellular-sheaf phase vectors and anisotropic block coupling; [full-interval and recovery contract](../reference/api/upde.md#cellular-sheaf-engine). |
| `SimplicialEngine` | Pairwise + simplicial (3-body) coupling. |
| `SplittingEngine` | Strang-split stepper (operator splitting). |
| `SwarmalatorEngine` | Swarmalators (coupled phase + spatial position). |
| `TorusEngine` | Symplectic Euler on the torus `T^N`. |

## Honesty note — validated niche

Choosing a fancier engine does not make a claim validated. SPO's externally
validated result is **grid inter-area modal damping** (IEEE-39 / Kundur vs ANDES
eigenvalues); generic early-warning across the other modalities is honestly
reported at chance on real data. See the README *Evidence status* table. Pick the
engine your physics needs, then rely on the [honest evaluation
auditor](../reference/api/evaluation.md) — not the engine's sophistication — to
tell you whether a detector beats chance on your data.
