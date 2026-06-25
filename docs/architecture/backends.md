# Backends & Dispatch

How SPO selects a compute backend, which languages exist, and the honest
build-versus-runtime status of each. All claims are against the code as of
2026-06-26.

## 1. Strategy

Each accelerated kernel keeps a **pure-Python reference implementation as the
floor** and optionally dispatches to a faster compiled backend when one is
importable. Selection is **per kernel**, resolved at import time, and falls
through a **fastest-first chain** to Python. Importing the compatibility shim
does not by itself load any native code:

```python
# _compat.py
HAS_RUST = importlib.util.find_spec("spo_kernel") is not None
TWO_PI = 2.0 * np.pi
```

> Importing `_compat` does not load the Rust extension or change backend
> selection; concrete modules decide when to dispatch.

## 2. Dispatch chains

| Lane | Chain (fastest → floor) | Selector |
|------|-------------------------|----------|
| `upde` integrators | Rust → WebGPU → Mojo → Julia → Go → Python | `upde/_run.py` (`ACTIVE_BACKEND`, `AVAILABLE_BACKENDS`) |
| `monitor` kernels | Rust → Mojo → Julia → Go → Python | per-observer `_load_*_fns()` (e.g. `monitor/lyapunov.py`) |
| `coupling` kernels | Rust → (Mojo/Julia/Go) → Python | per-module dispatch (`hodge`, `spectral`, `te_adaptive`, …) |
| `oscillators` extract | Rust → Python | `_HAS_RUST_*` flags (`oscillators/*.py`) |

The per-module Python files named `_<kernel>_<lang>.py` in `coupling/`, `monitor/`,
and `upde/` are thin forwarders to the real implementations under
`experimental/accelerators/<lane>/`. Go is loaded by `ctypes`, Julia by
`juliacall`, Mojo and WebGPU by their bridges, Rust by importing `spo_kernel`.

## 3. Languages and their status

| Backend | Where | Status (2026-06-26) |
|---------|-------|---------------------|
| **Python** | every kernel | Reference floor. Always available; correctness baseline. |
| **Rust** | `spo-kernel/` (6 crates) | Built from source into the selected Python environment with `python tools/install_spo_kernel.py --release` or `make bridge PYTHON=.venv/bin/python`. When `spo_kernel` is absent, `HAS_RUST=False` and Python (or another backend) runs. PyO3/maturin via the `rust` optional extra. |
| **JAX** | `nn/` | Differentiable track; optional `[nn]` extra. Hardware-agnostic JIT (GPU if present). Not part of the default step loop. |
| **Mojo** | `mojo/` (35 files) + `experimental/accelerators/**/_*_mojo.py` | Source + bridges present; opt-in, environment-gated. |
| **Julia** | `julia/` (35 files) + `_*_julia.py` bridges (`juliacall`) | Source + bridges present; opt-in, environment-gated. |
| **Go** | `go/` (35 files, compiled `.so`) + `_*_go.py` bridges (`ctypes`) | Source + compiled libraries + bridges present; opt-in. |
| **WebGPU** | `experimental/accelerators/upde/_engine_webgpu.py` | UPDE engine path only. |
| **WASM** | `spo-kernel/crates/spo-wasm/` | Minimal standalone Kuramoto stepper (`wasm-bindgen`); not linked to `spo-engine`; pedagogical/edge. |
| **FPGA** | `spo-kernel/crates/spo-fpga/src/kuramoto_core.v` | 16-oscillator Kuramoto core, Q16.16, CORDIC pipeline, Zynq-7020 target. Syntactically complete Verilog; **no synthesis report or hardware validation** — aspirational. |

## 4. Rust kernel (`spo-kernel/`)

Six crates (verified against `Cargo.toml` members):

| Crate | Purpose |
|-------|---------|
| `spo-types` | Shared config / state / error types. |
| `spo-engine` | UPDE integrators, coupling, order params, monitors (spectral, Lyapunov, transfer entropy, Hodge, recurrence, Koopman-EDMD, …). The bulk of the kernel. |
| `spo-oscillators` | Phase extraction (physical / informational / symbolic) + quality scoring. |
| `spo-supervisor` | Regime FSM, boundary observer, coherence, Petri net, policy, projector. **Exists but is not imported by the Python supervisor** — the Python `supervisor/` runs on NumPy. |
| `spo-ffi` | PyO3 bindings (`cdylib`); exposes the `_rust` functions; ships as the `spo_kernel` wheel. |
| `spo-wasm` | Browser/edge WASM stepper (standalone). |

> ARCHITECTURE.md (root) lists 5 crates and omits `spo-wasm`; the verified count
> is 6 in the workspace plus a non-Rust Verilog directory.

Install the FFI through the repository helper instead of a bare `maturin`
executable:

```bash
python tools/install_spo_kernel.py --release
python tools/install_spo_kernel.py --check-only
```

The helper invokes `python -m maturin`, so the built extension lands in the
same interpreter used by `spo run`, tests, and notebooks.

## 5. Intentionally Python-only kernels

Two kernels keep Rust **disabled** because LAPACK/FFT-backed NumPy/SciPy
out-performs the current Rust path; the Rust source exists but is not bound:

- `oscillators` phase extraction nuance and `coupling/coupling_est` (least-squares
  coupling estimation) — pure Python/NumPy.
- `coupling/plasticity` — `plasticity.rs` exists in `spo-engine` but is **not**
  dispatched from Python. The native model includes decay and `dt`; the public
  Python API is the validated TCBO-gated three-factor update. Treat the Rust file
  as a native reference/parity candidate until its semantics are intentionally
  aligned and benchmarked.

## 6. Benchmarks

Rust Criterion benches live under `spo-kernel/crates/spo-engine/benches/`
(`upde_bench`, `parallel_bench`, `monitors_bench`, `utility_bench`,
`twin_confidence_bench`) and are real harnesses producing measured histograms.
Python benches live under `bench/` and `benchmarks/`. Measured numbers are not
committed to public docs; treat any quoted figure as environment-specific and
re-measure before relying on it.
