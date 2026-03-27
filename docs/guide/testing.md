# Testing Guide

SCPN Phase Orchestrator ships with **3,000+ tests** at 99%+ line coverage. The suite includes unit tests, integration tests, and — critically — **hypothesis-driven property-based tests** that prove mathematical invariants hold across hundreds of random inputs per test.

## Running Tests

```bash
# Full suite
py -3.12 -m pytest tests/ -v --tb=short

# Single file
py -3.12 -m pytest tests/test_prop_lyapunov_dimension.py -v

# Only property-based tests
py -3.12 -m pytest tests/test_prop_*.py -v

# With coverage
py -3.12 -m pytest tests/ --cov=scpn_phase_orchestrator --cov-report=term-missing
```

## Hypothesis Profiles

The project defines hypothesis profiles in `pyproject.toml`:

| Profile | `max_examples` | Use case |
|---------|----------------|----------|
| `dev` | 50 | Local development (default) |
| `ci` | 500 | CI pipeline, thorough |

Select a profile:

```bash
py -3.12 -m pytest tests/ --hypothesis-profile=ci
```

## Test Architecture

### Property-Based Tests (`test_prop_*.py`)

These are computational theorem provers. Each `@given` test generates 50-500 random inputs and verifies that a mathematical invariant holds for all of them. If any counterexample is found, hypothesis shrinks it to the minimal failing case.

| File | Tests | What it proves |
|------|-------|----------------|
| `test_prop_lyapunov_dimension.py` | 44 | Lyapunov spectrum: length=N, sorted descending, finite. Kaplan-Yorke D_KY ∈ [0,N]. Correlation integral monotonic in ε. |
| `test_prop_basin_stability.py` | 28 | S_B ∈ [0,1], n_converged ≤ n_samples. Multi-basin threshold monotonicity. Strong coupling → high S_B. |
| `test_prop_entropy_transfer.py` | 25 | EPR ≥ 0, TE ≥ 0, TE diagonal = 0. TE-adaptive coupling preserves zero diagonal and non-negativity. |
| `test_prop_hodge_spectral.py` | 30 | Laplacian: PSD, row sums = 0. Fiedler λ₂ > 0 iff connected. Hodge: gradient + curl + harmonic = total. |
| `test_prop_recurrence_rqa.py` | 26 | Recurrence matrix: symmetric, diagonal = True. RR, DET, LAM ∈ [0,1]. Cross-recurrence shape and bounds. |
| `test_prop_chimera_winding.py` | 19 | Chimera index ∈ [0,1], coherent/incoherent disjoint. Winding numbers integer-valued, reverse ≈ negation. |
| `test_prop_free_energy_boltzmann.py` | 25 | Boltzmann weight ∈ (0,1] for U ≥ 0, monotonic in U and T. SSGF costs: c1 ∈ [0,1], c3 ≥ 0, c4 = 0 for symmetric W. |
| `test_prop_embedding_poincare.py` | 15 | Delay embedding shape = (T-(m-1)τ, m). Optimal delay ≥ 1. Optimal dimension ∈ [1, max_dim]. |
| `test_prop_ei_balance_npe.py` | 18 | Phase distance: symmetric, diagonal = 0, values ∈ [0,π]. NPE ∈ [0,1], sync → 0. EI ratio ≥ 0. |
| `test_prop_simplicial_reduction.py` | 8 | σ₂ = 0 reduces to standard Kuramoto (exact match). σ₂ ≠ 0 differs. |
| `test_prop_swarmalator_inertial.py` | 10 | Swarmalator: J=0 decouples phase from position. Inertial: θ wrapped to [0,2π). |
| `test_prop_plasticity_stochastic.py` | 22 | Eligibility: symmetric, ∈ [-1,1], zero diagonal. StochasticInjector: D=0 → no change, output ∈ [0,2π). |

### Degenerate Edge Cases (`test_degenerate_edges.py`)

98 tests that push every engine to its boundaries: N=1 oscillator, dt=0, zero coupling (free rotation), identical phases, phase wrapping at 0 and 2π, extreme coupling strengths, negative frequencies. Parametrised across all 5 engine types (UPDE, Stuart-Landau, Simplicial, Swarmalator, Inertial).

### Cross-Module Roundtrips (`test_roundtrip_consistency.py`)

86 tests that verify mathematical consistency across module boundaries:

- Synchronised phases → R ≈ 1, PLV ≈ 1, NPE ≈ 0, chimera_index ≈ 0 (four independent measures agree)
- Spectral λ₂ predicts synchronisability → verified by simulation
- Projection roundtrip: `project_knm` always produces valid K_nm
- Simplicial σ₂ = 0 roundtrip: reduces to standard Kuramoto exactly
- Free rotation → analytical winding number matches
- Transfer entropy: directional, correct shape
- NPE vs R anti-correlation across synchronisation spectrum

### Module Tests

Dedicated test files for each subsystem covering unit-level behaviour, input validation, edge cases, and dataclass contracts:

| Subsystem | Files | Modules tested |
|-----------|-------|----------------|
| SSGF | `test_ssgf_modules.py` | GeometryCarrier, CyberneticClosure, EthicalCost |
| UPDE math | `test_upde_math.py` | TorusEngine, IntegrationConfig, check_stability, OttAntonsenReduction |
| Coupling | `test_coupling_modules.py` | LagModel, UniversalPrior, KnmTemplateSet |
| Drivers | `test_drivers_oscillators.py` | PhysicalDriver, PhaseQualityScorer, CoherenceMonitor |
| Supervisor | `test_supervisor_modules.py` | EventBus, RegimeManager, InformationalDriver, SymbolicDriver |
| Imprint | `test_imprint_actuation.py` | ImprintModel, ActionProjector, ActuationMapper |
| Bifurcation | `test_bifurcation.py` | trace_sync_transition, find_critical_coupling |

## Writing New Tests

### Property test pattern

```python
from hypothesis import given, settings
from hypothesis import strategies as st

class TestMyInvariant:
    @given(
        n=st.integers(min_value=2, max_value=12),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_output_bounded(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        result = my_function(phases)
        assert 0.0 <= result <= 1.0
```

Key conventions:

- `deadline=None` for tests that run simulations (Lyapunov, basin stability)
- Small N (2-16) for CPU speed — property tests run 50-500 iterations
- `suppress_health_check=[HealthCheck.too_slow]` for Monte Carlo tests
- Use `_connected_knm(n, seed=seed)` helper for reproducible symmetric coupling matrices
- Tolerances for float comparison: `atol=1e-12` for exact, `atol=1e-10` for simulation

### Degenerate edge test pattern

```python
@pytest.mark.parametrize("n", [2, 4, 8])
def test_zero_coupling_free_rotation(self, n: int) -> None:
    eng = UPDEEngine(n, dt=0.01)
    # ... verify analytical prediction under extreme conditions
```

## CI Integration

CI runs the full suite on Python 3.10 (without Rust kernel) and Python 3.12 (with Rust kernel). The Python fallback uses pure-NumPy integrators; the Rust path uses `spo-kernel` via PyO3. Tests handle both paths — see `test_degenerate_edges.py::TestUPDEZeroDt` for the pattern.

Coverage gate: **95% minimum**, currently at **99.33%**.
