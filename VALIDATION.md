# Validation

## Test Matrix

| Suite | Count | Scope |
|-------|------:|-------|
| Python unit/integration | 1219 | `pytest tests/` across 70+ files |
| Rust unit/integration | 191 | `cargo test --workspace` across 5 crates |
| FFI parity | 5 | Python vs Rust engine output match (euler, rk4, rk45, order_parameter, stuart_landau) |
| Notebook execution | 5 | `nbclient` runs all `.ipynb` cells under Python 3.12 |
| Domainpack validation | 24 | Each domainpack exercises `binding_spec.yaml → run.py` end-to-end |

CI runs tests on Python 3.10–3.13 and Rust on Linux, macOS, Windows.

## CI Validation Gates

All gates must pass before `ci-gate` allows merge.

| Gate | Tool | What it enforces |
|------|------|------------------|
| lint | `ruff check` + `ruff format` + `check_version_sync.py` | Style, formatting, version parity |
| typecheck | `mypy src/` | Static type safety on public API |
| test | `pytest --cov` (4 Python versions) | Functional correctness + coverage XML |
| coverage-guard | `tools/coverage_guard.py` | Per-module thresholds from `coverage_guard_thresholds.json` |
| module-linkage | `tools/check_test_module_linkage.py` | Every source module has ≥1 test reference |
| security | `bandit -r src/` + `pip-audit` | OWASP + supply-chain vulnerabilities |
| rust-check | `cargo fmt/clippy/doc/test` (3 OS) | Formatting, lints, docs, correctness |
| ffi-test | `maturin develop` + `pytest` (3 OS × 2 Python) | Rust↔Python binding correctness |
| cargo-audit | `cargo audit` | Rust dependency CVEs |
| rust-msrv | Rust 1.75.0 build | Minimum supported Rust version |
| benchmark | `bench/run_benchmarks.py` → `bench/compare_baseline.py` | ≤20% regression vs baseline |

## Phase Dynamics Correctness

### Kuramoto coupling

The UPDE stepper implements `dθ_n/dt = ω_n + Σ_m K_nm sin(θ_m − θ_n)` with
phase-difference coupling (not matrix-vector product of absolute sines). Tests
verify:

- Identical phases remain synchronised under zero coupling
- Strong coupling drives R → 1 (Kuramoto synchronisation)
- Zero coupling preserves natural frequency drift
- Phase wrapping stays in [0, 2π)

### Adaptive integration (RK45)

Dormand–Prince RK45 with embedded error control. Tests verify:

- Phases stay bounded after 500 steps
- Adaptive dt grows for smooth dynamics
- Synchronisation tendency under strong coupling matches RK4
- Python and Rust produce identical trajectories (atol < 1e-6)

### Order parameter

`R = |mean(exp(iθ))|` computed identically in Python (numpy) and Rust. Parity
test asserts `|R_py − R_rust| < 1e-10`.

## Performance Benchmarks

`bench/run_benchmarks.py` measures `μs/step` across a 30-point grid:

- **N** ∈ {8, 16, 64, 256, 1024} oscillators
- **Methods**: euler, rk4, rk45
- **Backends**: Python (numpy), Rust (spo_kernel)

Output includes system fingerprint (Python/numpy/scipy versions, platform) for
reproducibility. `bench/compare_baseline.py` fails CI if any config regresses
>20% vs `bench/baseline.json`.

## Domainpack Integration (24 domains)

Each domainpack provides a `binding_spec.yaml` + `run.py` exercising the full
pipeline: binding validation → oscillator instantiation → UPDE stepping →
coherence monitoring → boundary enforcement.

Domains span: autonomous_vehicles, bio_stub, cardiac_rhythm, chemical_reactor,
circadian_biology, epidemic_sir, firefly_swarm, fusion_equilibrium, geometry_walk,
laser_array, manufacturing_spc, metaphysics_demo, minimal_domain, network_security,
neuroscience_eeg, plasma_control, pll_clock, power_grid, quantum_simulation,
queuewaves, rotating_machinery, satellite_constellation, swarm_robotics,
traffic_flow.

`tests/test_domainpack_validation.py` validates all specs parse and execute
without error.

## Regeneration

```bash
# Full local validation
pytest tests/ -v --tb=short
cargo test --workspace
cargo clippy --workspace -- -D warnings
python bench/run_benchmarks.py --json
python tools/coverage_guard.py
python tools/check_test_module_linkage.py
```
