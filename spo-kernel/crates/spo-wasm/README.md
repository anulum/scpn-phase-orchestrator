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

The example page is an **interactive playground**: sliders for the oscillator
count, coupling, time step, and frequency spread drive a live simulation that
renders the phase ring, the mean-field vector, and the order-parameter time
series on a canvas. Raise the coupling past the critical point and watch R climb
toward 1.0 as the oscillators synchronise. Build `wasm-pkg/` first (see above) so
the page can load the compiled module.

The page logic is split into a pure, DOM-free helper module
(`example/simulation.mjs`) so it can be unit tested without a browser:

```bash
node --test spo-kernel/crates/spo-wasm/example/
```

The helper suite also drives the compiled `WasmEngine` directly (skipped when
`wasm-pkg/` is not built) and cross-checks the WASM order parameter against the
pure helper.

## API

The module exports a single `WasmEngine` class:

| Member | Signature | Description |
|--------|-----------|-------------|
| `new WasmEngine` | `(n: number)` | Create `n` oscillators at zero phase |
| `set_phases` | `(phases: Float64Array) -> void` | Replace the phase vector |
| `get_phases` | `() -> Float64Array` | View of the current phases |
| `step` | `(omegas: Float64Array, coupling: number, dt: number) -> number` | Advance one mean-field Euler step, return the order parameter R |
| `run` | `(omegas: Float64Array, coupling: number, dt: number, n_steps: number) -> number` | Advance `n_steps` steps, return the final R |

## License

AGPL-3.0-or-later | Commercial license available.
