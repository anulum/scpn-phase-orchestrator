<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Notebook Execution Matrix -->

# Notebook Execution Matrix

This matrix records how each notebook is expected to run, which extras it
needs, and whether the current CI path executes it.

## Execution Policy

Repository CI currently validates notebooks on Python 3.12 with:

```bash
pip install --require-hashes --no-deps -r requirements/dev-lock.txt
pip install --no-deps -e .
jupyter nbconvert --execute --to notebook notebooks/*.ipynb --ExecutePreprocessor.timeout=120
```

For local work, install notebook extras and run from the repository root:

```bash
python -m pip install -e ".[dev,notebook,plot]"
jupyter lab notebooks/
```

## Matrix

| Notebook | Primary surface | Required extras | CI expectation | Runtime class | Notes |
|----------|-----------------|-----------------|----------------|---------------|-------|
| `01_queuewaves_retry_storm.ipynb` | QueueWaves | `dev`, `notebook`, `plot`, `queuewaves` | executed | short | service-queue retry storm and supervisor trace |
| `02_minimal_domain.ipynb` | Domain authoring | `dev`, `notebook`, `plot` | executed | short | smallest complete binding workflow |
| `03_geometry_walk.ipynb` | Symbolic channel | `dev`, `notebook`, `plot` | executed | short | graph-walk phases and geometry coupling |
| `04_bio_stub.ipynb` | Biology | `dev`, `notebook`, `plot` | executed | short | multi-scale biological oscillator mapping |
| `05_manufacturing_spc.ipynb` | Manufacturing | `dev`, `notebook`, `plot` | executed | short | SPC sensor mapping and policy behaviour |
| `06_stuart_landau_amplitude.ipynb` | UPDE | `dev`, `notebook`, `plot` | executed | short | phase-amplitude dynamics and PAC |
| `07_policy_petri_net.ipynb` | Supervisor | `dev`, `notebook`, `plot` | executed | short | policy DSL, regimes, and Petri net sequencing |
| `08_audit_replay.ipynb` | Audit | `dev`, `notebook`, `plot` | executed | short | deterministic replay and hash-chain checks |
| `09_binding_spec.ipynb` | Binding | `dev`, `notebook`, `plot` | executed | short | schema walkthrough and validation |
| `10_reporting_adapters.ipynb` | Reporting/adapters | `dev`, `notebook`, `plot`, `full` when bridge deps are used | executed | medium | external bridge patterns; optional bridge imports may need `full` |
| `11_identity_coherence.ipynb` | SSGF | `dev`, `notebook`, `plot` | executed | medium | identity coherence, chimera, and plasticity |
| `12_autotune_pipeline.ipynb` | Autotune | `dev`, `notebook`, `plot` | executed | medium | frequency identification and coupling estimation |
| `13_ssgf_closure.ipynb` | SSGF | `dev`, `notebook`, `plot` | executed | medium | free-energy closure loop |
| `14_chimera_detection.ipynb` | Monitor | `dev`, `notebook`, `plot` | executed | medium | chimera detection workflow |
| `15_spectral_analysis.ipynb` | Coupling | `dev`, `notebook`, `plot` | executed | medium | spectral alignment analysis |
| `16_sleep_staging.ipynb` | Monitor/domainpack | `dev`, `notebook`, `plot` | executed | medium | sleep-stage phase dynamics |
| `17_power_grid_stability.ipynb` | Power systems | `dev`, `notebook`, `plot` | executed | medium | inertial Kuramoto transient |
| `18_market_regime_detection.ipynb` | Finance | `dev`, `notebook`, `plot` | executed | medium | Hilbert phase extraction and market regimes |
| `19_swarmalator_dynamics.ipynb` | Robotics | `dev`, `notebook`, `plot` | executed | medium | spatial and phase swarmalator dynamics |

Runtime classes:

| Class | Expected local behaviour |
|-------|--------------------------|
| short | should complete within the CI 120-second per-notebook timeout on normal CI hardware |
| medium | may approach the timeout on slower machines; prefer interactive execution when debugging |
| local-only | not currently used by the 19 shipped notebooks; reserve for future GPU, quantum, or external-service notebooks |

## Maintenance Rules

When adding or changing a notebook:

1. Add or update its row in this matrix.
2. State required extras and whether external data or services are needed.
3. Keep default cell counts and step counts suitable for the CI timeout.
4. Prefer deterministic seeds for simulations.
5. Link a terminal example or guide that covers the same workflow.
6. If a notebook cannot run in CI, mark it `local-only` and explain why.

## Local Failure Recovery

Use the [Troubleshooting](../getting-started/troubleshooting.md) notebook
section when imports, optional dependencies, or timeouts fail.
