# Interactive Tools

SPO ships with two interactive interfaces for exploring phase dynamics
without writing code.

## SPO Studio (Streamlit)

A visual GUI for browsing domainpacks and tuning control knobs in real time.

### Quick Start

```bash
pip install scpn-phase-orchestrator streamlit
streamlit run tools/spo_studio.py
```

Opens at `http://localhost:8501`.

### Features

- **Domainpack browser** — select any of the 33 domainpacks from a dropdown
- **Universal knobs** — slide K (coupling strength), ζ (drive strength),
  Ψ frequency to see immediate effects on synchronisation
- **Live R(t) chart** — order parameter plotted over simulation steps
- **Regime timeline** — visual trace of nominal / degraded / critical transitions
- **Per-layer breakdown** — R value for each oscillator layer
- **Metrics dashboard** — final R, regime, oscillator count, layer count

### Use Cases

- **Teaching**: demonstrate Kuramoto synchronisation interactively
- **Domain exploration**: browse domainpacks before writing binding specs
- **Parameter tuning**: find good K/ζ values for a new domain
- **Demos**: live presentations without terminal commands

## WASM Browser Demo

A 66KB WebAssembly build of the Kuramoto engine that runs entirely
in the browser. No installation required.

### Access

When deployed via GitHub Pages, the demo is at:

```
https://anulum.github.io/scpn-phase-orchestrator/demo/
```

Or open `docs/demo/index.html` locally after cloning the repo.

### Features

- **Real-time simulation** — oscillators step at 60fps in the browser
- **R(t) chart** — live order parameter with regime colour bands
  (green = nominal, yellow = degraded, red = critical)
- **Phase portrait** — oscillators on the unit circle with mean-field arrow
- **Controls** — sliders for N (4-64), coupling K, frequency spread, dt

### Building from Source

The WASM binary is pre-built in `docs/wasm-pkg/`. To rebuild:

```bash
cd spo-kernel
wasm-pack build crates/spo-wasm --target web --out-dir ../../../docs/wasm-pkg
```

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/) and a Rust
toolchain with the `wasm32-unknown-unknown` target.

### Technical Details

- **Binary size**: 66KB (gzipped ~30KB)
- **Engine**: Euler integrator with mean-field coupling
- **API**: `init(n)`, `step(omega_json, coupling, dt)`, `get_phases()`
- **No dependencies**: pure Rust → WASM, no JavaScript framework

## CLI Demo

For terminal-based demos, use the `spo demo` command:

```bash
spo demo --domain plasma_control --steps 100
spo demo --domain cardiac_rhythm --steps 50
```

Lists available domainpacks if the specified domain is not found.
See the [domainpack gallery](../galleries/domainpack_gallery.md) for
benchmark results across all 33 domains.
