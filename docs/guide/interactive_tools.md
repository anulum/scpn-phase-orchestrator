# Interactive Tools

SPO ships with three interactive interfaces for exploring phase dynamics
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

- **Domainpack browser** — select any bundled domainpack from a dropdown
- **Raw-source import** — generate reviewable binding proposals from CSV,
  event-log JSON, or graph JSON
- **Universal knobs** — slide K, alpha, zeta, and Psi for local replay tuning
- **Live R(t) chart** — order parameter plotted over replay steps
- **Regime timeline** — visual trace of nominal / degraded / critical transitions
- **Oscillator table** — edit oscillator rows and export an
  `oscillator_edit_review.json` artefact for binding review
- **Hierarchy monitor** — inspect reduced layer metrics and watermarks
- **Export review** — download deployment-readiness, binding, audit, Docker,
  WASM, and project-state artefacts; deploy-like manifests are disabled when
  validation fails

### Use Cases

- **Teaching**: demonstrate Kuramoto synchronisation interactively
- **Domain exploration**: browse domainpacks before writing binding specs
- **Parameter tuning**: find good K/ζ values for a new domain
- **Demos**: live presentations without terminal commands

For the operator workflow, see
[SPO Studio Operator Guide](spo_studio_operator.md).

## Binding Spec Studio (Streamlit)

A compact helper for binding-spec editing and validation before running a domainpack.

### Quick Start

```bash
streamlit run tools/binding_spec_studio.py
```

Opens at `http://localhost:8501`.

### Features

- **YAML load/save loop** — open an existing `binding_spec.yaml`, edit, and
  save locally
- **Live validation** — parses YAML and runs binding schema checks
- **P/I/S mapping table** — inspect family/channel-to-driver assignments and
  extra declared channels
- **Preview output** — generate synthetic extractor previews for each family
- **Resolved defaults** — inspect computed runtime binding defaults for quick
  sanity checks
- **Minimal scaffold** — write `binding_spec.yaml` and emit a reproducible
  domainpack folder with baseline `README.md`

## Policy Studio (Streamlit)

An interactive builder for supervisor policy rules with validation and dry-run
analysis.

### Quick Start

```bash
streamlit run tools/policy_studio.py
```

Opens at `http://localhost:8501`.

### Features

- **Structured rule builder** — compose rule name, regimes, conditions, and knob
  actions interactively with validated schema fields
- **Cooldown + cap preview** — visualise per-rule fire limits over a step horizon
- **Dry-run diagnostics** — upload `audit.jsonl` and inspect unreachable rules,
  overlaps, collisions, and fire counts
- **Binding-aware diagnostics** — validate rules against selected/loading or
  uploaded binding specs before running production policies

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
benchmark results across the measured domainpack snapshot.

## Safe usage guidelines

All interactive tools are intended for exploration and review, not for implicit
actuation. In production, run them in this order:

- open the interface with a bound and reviewed domainpack,
- validate `spo validate` before loading any policy or control controls,
- compare outputs against a replayed CLI run,
- only then use outputs to build deployment documentation.

That sequence prevents a visual tuning pass from being mistaken for validated
operational change.

## Data handling expectation

Interactive sessions that read external files should not be used as policy
authorities. They are for assessment and recommendation. The project control
boundary remains replayed and approved policy execution.
