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

A WebAssembly build of the Kuramoto engine that runs entirely in the browser.
No Python installation is required after the WASM package has been built.

### Access

When deployed via GitHub Pages, the hosted demo target is:

```
https://anulum.github.io/scpn-phase-orchestrator/demo/
```

For local development, build `wasm-pkg/` from the repository root and serve the
example page:

```bash
cd spo-kernel
wasm-pack build crates/spo-wasm --target web --out-dir ../../../wasm-pkg
cd ..
python -m http.server 8080
```

Then open:

```
http://localhost:8080/spo-kernel/crates/spo-wasm/example/index.html
```

### Features

- **Real-time simulation** — oscillators step at 60fps in the browser
- **R(t) chart** — live order parameter with regime colour bands
  (green = nominal, yellow = degraded, red = critical)
- **Phase portrait** — oscillators on the unit circle with mean-field arrow
- **Scenario presets** — deterministic starting points for weak coupling drift,
  the critical transition, strong synchronisation, and wide frequency dispersion
- **Controls** — sliders for oscillator count, coupling K, frequency spread, and
  time step after a scenario has seeded the validated parameters

### Building from Source

The local playground loads the generated package from repository-root
`wasm-pkg/`. To rebuild:

```bash
cd spo-kernel
wasm-pack build crates/spo-wasm --target web --out-dir ../../../wasm-pkg
```

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/) and a Rust
toolchain with the `wasm32-unknown-unknown` target.

### Technical Details

- **Engine**: Euler integrator with mean-field coupling
- **API**: `WasmEngine`, with `set_phases`, `get_phases`, `step`, and `run`
- **Browser logic**: `spo-kernel/crates/spo-wasm/example/simulation.mjs`
  contains DOM-free helper functions and the scenario catalogue, tested with
  `node --test`
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

## Recommended sequence for tool-based reviews

Use interactive tools only as a pre-action evidence layer:

1. validate and inspect a bound domainpack with CLI commands first,
2. open the interactive surface with fixed runtime defaults,
3. compare interface outputs against one replayed command sequence,
4. export a review packet before any operational proposal is discussed.

This keeps interface exploration from becoming an implicit controller change.

For teams with mixed expertise, this order also keeps domain authors, ML users,
and operations staff aligned on what was actually executed versus what was only
visualised.
