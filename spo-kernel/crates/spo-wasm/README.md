# spo-wasm

WASM bindings for the SCPN Phase Orchestrator Kuramoto integrator.
Runs a self-contained Euler-step Kuramoto simulation in any browser or edge runtime.

## Prerequisites

```bash
# Install the wasm32 target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
cargo install wasm-pack
```

## Build

From the repository root:

```bash
cd spo-kernel
wasm-pack build crates/spo-wasm --target web --out-dir ../../../wasm-pkg
```

This produces `wasm-pkg/` containing:
- `spo_wasm_bg.wasm` — compiled WebAssembly module
- `spo_wasm.js` — ES-module JS bindings
- `spo_wasm.d.ts` — TypeScript declarations
- `package.json` — npm-compatible metadata

## Demo

Serve the example page with any static file server:

```bash
# From repository root
python -m http.server 8080
# Open http://localhost:8080/spo-kernel/crates/spo-wasm/example/index.html
```

The demo initialises 8 oscillators, runs 100 Kuramoto steps, and logs
the order parameter R at every 10th step. R should converge toward 1.0
as oscillators synchronise.

## API

| Function | Signature | Description |
|----------|-----------|-------------|
| `init` | `(n: usize)` | Create `n` oscillators at zero phase |
| `step` | `(omega_json: &str, coupling: f64, dt: f64) -> f64` | Advance one Euler step, return order parameter R |
| `get_phases` | `() -> String` | Current phases as JSON array |

## License

AGPL-3.0-or-later | Commercial license available.
