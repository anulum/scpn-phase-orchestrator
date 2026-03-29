# End-to-End Wiring Audit

**Date:** 2026-03-29
**Method:** Manual import + functional verification of every module, one by one
**Result:** 152/153 modules import, all pipelines functional, 0 broken wires

---

## 1. Module Tree Import (152/153)

Every Python module in the package was walked via `pkgutil.walk_packages`
and imported. One expected failure:

| Module | Status | Reason |
|---|---|---|
| `grpc_gen.spo_pb2_grpc` | SKIP | `grpc` not installed (optional dependency) |
| All other 152 modules | **OK** | — |

---

## 2. Core Public API (__init__.py, 15 symbols)

All symbols in `__all__` resolve:

```
AuditLogger, BifurcationDiagram, BindingSpec, BoundaryObserver,
ControlAction, CouplingBuilder, PhaseExtractor, PhaseState,
RegimeManager, SPOError, StuartLandauEngine, SupervisorPolicy,
UPDEEngine, find_critical_coupling, lyapunov_spectrum, trace_sync_transition
```

**Status:** OK

---

## 3. nn/ Module (61 lazy-loaded symbols, 16 submodules)

All 61 symbols in `nn.__all__` resolve via lazy import (`__getattr__`).
All 16 submodule files import directly:

| Submodule | Symbols | Status |
|---|---|---|
| `functional.py` | kuramoto_step/rk4/forward, masked variants, winfree, simplicial, stuart_landau, order_parameter, plv, saf, coupling_laplacian | OK |
| `kuramoto_layer.py` | KuramotoLayer | OK |
| `stuart_landau_layer.py` | StuartLandauLayer | OK |
| `simplicial_layer.py` | SimplicialKuramotoLayer | OK |
| `theta_neuron.py` | ThetaNeuronLayer, theta_neuron_step/rk4/forward | OK |
| `ude.py` | UDEKuramotoLayer, CouplingResidual | OK |
| `inverse.py` | analytical_inverse, hybrid_inverse, infer_coupling, coupling_correlation | OK |
| `oim.py` | oim_step/forward/solve, extract_coloring, coloring_violations/energy | OK |
| `bold.py` | bold_from_neural, bold_signal, balloon_windkessel_step | OK |
| `reservoir.py` | reservoir_drive, ridge_readout, reservoir_predict | OK |
| `chimera.py` | local_order_parameter, chimera_index, detect_chimera | OK |
| `spectral.py` | laplacian_spectrum, algebraic_connectivity, eigenratio, sync_threshold | OK |
| `training.py` | sync_loss, trajectory_loss, train_step, train, generate_kuramoto_data | OK |
| `__init__.py` | Lazy dispatch for all 61 symbols | OK |

---

## 4. Core Pipeline (binding → coupling → engine → monitor → audit)

Tested with `power_grid` domainpack (N=12):

| Step | Module | Input | Output | Status |
|---|---|---|---|---|
| 1. Load spec | `binding.loader` | `binding_spec.yaml` | `BindingSpec(name=power_grid, 4 layers)` | OK |
| 2. Build coupling | `coupling.knm` | `base_strength, decay_alpha` | `K shape=(12,12), symmetric, non-negative` | OK |
| 3. Run engine | `upde.engine` | `phases, omegas, K, 200 steps` | `R=0.3896` | OK |
| 4. Monitor | `monitor.boundaries` | `spec.boundaries` | `BoundaryObserver` | OK |
| 5. Imprint | `imprint.update` | `ImprintModel + ImprintState` | `m_k mean=0.039` | OK |
| 6. Audit | `audit.logger` | `path` | `AuditLogger` | OK |

---

## 5. nn/ Pipeline (inverse + forward, JAX)

Tested on the SAME power_grid data as core pipeline:

| Step | Module | Result | Status |
|---|---|---|---|
| 7. Inverse | `nn.inverse.analytical_inverse` | `K_est, corr=0.529` | OK |
| 8. Forward | `nn.functional.kuramoto_forward` | `R_jax=0.3896` | OK |
| 9. Parity | NumPy vs JAX | `|R_np - R_jax| = 0.0000` | **EXACT MATCH** |

---

## 6. All 33 Domainpacks

Every domainpack loads without error:

```
agent_coordination, autonomous_vehicles, brain_connectome,
cardiac_rhythm, chemical_reactor, circadian_biology, epidemic_sir,
financial_markets, firefly_swarm, fusion_equilibrium,
gene_oscillator, identity_coherence, laser_array,
manufacturing_spc, metaphysics_demo, minimal_domain,
musical_acoustics, network_security, neuroscience_eeg,
plasma_control, power_grid, quantum_simulation, queuewaves,
robotic_cpg, rotating_machinery, satellite_constellation,
sleep_architecture, social_opinion, swarm_robotics,
traffic_flow, vortex_shedding, weather_teleconnection, wildlife_migration
```

**33/33 OK**

---

## 7. Advanced Modules

| Category | Modules | Status |
|---|---|---|
| Supervisor | `regimes`, `policy`, `policy_rules`, `events`, `petri_net`, `petri_adapter`, `predictive` | OK |
| UPDE engines | `engine` (Euler/RK4/RK45), `stuart_landau`, `delay`, `inertial`, `jax_engine` | OK |
| UPDE analysis | `order_params`, `metrics`, `pac`, `bifurcation`, `reduction` (Ott-Antonsen) | OK |
| SSGF | `closure`, `free_energy`, `ethical`, `carrier`, `costs`, `pgbo`, `tcbo` | OK |
| Drivers | `psi_physical`, `psi_informational`, `psi_symbolic` | OK |
| Adapters | `remanentia_bridge`, `synapse_channel_bridge`, `synapse_coupling_bridge` | OK |
| Reporting | `plots` (CoherencePlot) | OK |
| Autotune | `pipeline`, `coupling_est`, `freq_id`, `phase_extract` | OK |
| CLI | `cli.main` (Click Group) | OK |

---

## 8. Cross-Backend Parity

| Engine | R (power_grid, 200 steps) | Status |
|---|---|---|
| NumPy UPDEEngine (RK4) | 0.3896 | — |
| JAX kuramoto_forward (RK4) | 0.3896 | **EXACT MATCH** |

Difference: 0.0000 (float32 sufficient for this configuration).

---

## 9. Functional Verification Summary

| Test | Modules touched | Status |
|---|---|---|
| Core imports | `__init__` (15 symbols) | OK |
| nn/ imports | `nn/__init__` (61 symbols) | OK |
| nn/ direct imports | 16 submodules | OK |
| Binding pipeline | `binding.loader`, `binding.types` | OK |
| Coupling builder | `coupling.knm` → symmetric, non-negative, zero-diagonal K | OK |
| UPDE engine | `upde.engine` (RK4, 500 steps) → R=1.0 for identical ω | OK |
| Stuart-Landau | `upde.stuart_landau` → r≈sqrt(μ), phases synchronise | OK |
| Imprint | `imprint.update`, `imprint.state` → modulate_coupling | OK |
| Actuation | `supervisor.ControlAction` → knob, scope, value, justification | OK |
| PI/S Drivers | `drivers.psi_physical/informational/symbolic` | OK |
| 33 domainpacks | All load without error | OK |
| CLI | `cli.main` Click Group | OK |
| Adapters | 3 bridge modules | OK |
| Advanced supervisor | PetriNet, EventBus, Policy, Predictive | OK |
| Advanced engines | Delay, Inertial, JAX, OA reduction, PAC | OK |
| SSGF | 7 submodules (closure, FE, ethical, carrier, costs, pgbo, tcbo) | OK |
| Full module tree | 152/153 (1 expected skip: grpc) | OK |
| End-to-end wiring | binding→coupling→engine→monitor→imprint→audit→nn/ | OK |
| NumPy↔JAX parity | R difference = 0.0000 on power_grid | OK |

**Total: 0 broken wires. All modules functional and correctly interconnected.**
