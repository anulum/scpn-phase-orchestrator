<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Notebooks and Demos Inventory -->

# Notebooks & Demos

This page is the user-facing inventory for runnable learning artefacts:
notebooks, terminal examples, Streamlit tools, CLI demos, and the browser
WASM demo.

Use notebooks to understand a capability, then move production work into a
domainpack, CLI run, Python facade, or audited replay path.

| Learning goal | Notebook or demo | Production path |
|---------------|------------------|-----------------|
| First domainpack run | `02_minimal_domain.ipynb` | `spo validate` and `spo run` on a reviewed binding |
| Queue and retry cascades | `01_queuewaves_retry_storm.ipynb` | QueueWaves guide and production deployment |
| Geometry and topology | `03_geometry_walk.ipynb` | geometry constraints and coupling templates |
| Binding specs | `09_binding_spec.ipynb` | binding API plus raw-source tutorial |
| Audit replay | `08_audit_replay.ipynb` | deterministic replay tutorial and audit API |
| Autotune | `12_autotune_pipeline.ipynb` | autotune API and replay-only learner guide |
| Reference domains | power grid, market, sleep, swarmalator notebooks | domainpack gallery plus domain-specific validation |

## Notebook Inventory

Run notebooks from the repository root:

```bash
pip install -e ".[dev,notebook,plot]"
jupyter lab notebooks/
```

| Notebook | Surface | Purpose |
|----------|---------|---------|
| `01_queuewaves_retry_storm.ipynb` | QueueWaves | Retry-storm recovery and supervisor action trace |
| `02_minimal_domain.ipynb` | Domain authoring | Smallest complete domainpack workflow |
| `03_geometry_walk.ipynb` | Symbolic channel | Graph-walk phases and geometry coupling |
| `04_bio_stub.ipynb` | Biology | Multi-scale biological oscillator mapping |
| `05_manufacturing_spc.ipynb` | Manufacturing | SPC sensors, bad-layer suppression, policy rules |
| `06_stuart_landau_amplitude.ipynb` | UPDE | Phase-amplitude dynamics and PAC |
| `07_policy_petri_net.ipynb` | Supervisor | Policy DSL, regime FSM, Petri net sequencing |
| `08_audit_replay.ipynb` | Audit | SHA256-chained audit trail and deterministic replay |
| `09_binding_spec.ipynb` | Binding | Binding spec schema walkthrough |
| `10_reporting_adapters.ipynb` | Reporting/adapters | Reporting and external bridge patterns |
| `11_identity_coherence.ipynb` | SSGF | Identity coherence, chimera, plasticity |
| `12_autotune_pipeline.ipynb` | Autotune | Frequency identification and coupling estimation |
| `13_ssgf_closure.ipynb` | SSGF | Free-energy closure loop |
| `14_chimera_detection.ipynb` | Monitor | Chimera detection workflow |
| `15_spectral_analysis.ipynb` | Coupling | Spectral alignment analysis |
| `16_sleep_staging.ipynb` | Monitor/domainpack | Sleep-stage phase dynamics |
| `17_power_grid_stability.ipynb` | Power systems | Inertial Kuramoto and generator-trip transient |
| `18_market_regime_detection.ipynb` | Finance | Market phase extraction and regime detection |
| `19_swarmalator_dynamics.ipynb` | Robotics | Spatial + phase swarmalator dynamics |

CI executes the shipped notebook suite on Python 3.12 with `nbconvert`.
See the [Notebook Execution Matrix](notebook_execution_matrix.md) for
per-notebook extras, runtime class, and CI expectation.

## Terminal Examples

Run examples from the repository root with `PYTHONPATH=src` for a source
checkout:

```bash
PYTHONPATH=src python examples/supervisor_advantage.py
PYTHONPATH=src python examples/failure_recovery.py
PYTHONPATH=src python examples/cross_domain_universality.py
```

| Example family | Scripts |
|----------------|---------|
| First run and universality | `cross_domain_universality.py`, `multi_engine_comparison.py`, `scaling_showcase.py` |
| Supervisor and recovery | `supervisor_advantage.py`, `failure_recovery.py`, `petri_policy_demo.py` |
| Domain-specific demos | `cardiac_rhythm.py`, `epidemic_sir.py`, `market_regime_detection.py`, `neuroscience_eeg.py`, `plasma_control.py`, `power_grid_stability.py`, `traffic_flow.py` |
| Analysis methods | `hodge_decomposition.py`, `inverse_coupling_demo.py`, `inverse_kuramoto.py`, `plasticity_learning.py`, `stochastic_resonance.py`, `stuart_landau_bifurcation.py`, `swarmalator_dynamics.py` |
| Integration surfaces | `agent_coordination.py`, `audit_replay_demo.py`, `eeg_file_ingestion.py`, `neurocore_cosimulation.py`, `prometheus_queuewaves.py`, `ssgf_closure_loop.py` |

There are `27` Python example scripts in `examples/`.

## Interactive Demos

| Demo | Command or URL | Notes |
|------|----------------|-------|
| SPO Studio | `streamlit run tools/spo_studio.py` | Browse domainpacks and tune control knobs |
| Binding Spec Studio | `streamlit run tools/binding_spec_studio.py` | Edit and validate binding specs |
| Policy Studio | `streamlit run tools/policy_studio.py` | Build and dry-run policy rules |
| Browser WASM demo | `docs/demo/index.html` or GitHub Pages `/demo/` | Runs the WASM Kuramoto engine in a browser |
| CLI demo | `spo demo --domain minimal_domain --steps 20` | Terminal demo for any packaged domainpack |

## Production Continuation

After a notebook or demo works:

1. Validate the binding spec with `spo validate`.
2. Run a deterministic simulation with `spo run --seed`.
3. Enable audit logging and replay it with `spo replay --verify`.
4. Serve the model with `spo serve` or the gRPC server.
5. Connect Prometheus/OpenTelemetry if the model is production-facing.

See [Notebook to Production](../guide/notebook_to_production.md) for the
full handoff path.
