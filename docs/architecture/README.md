# SCPN Phase Orchestrator — Architecture Map

Canonical, evidence-based map of the SCPN Phase Orchestrator (SPO): what each
part consumes, produces, and depends on. It is the contract surface that sibling
SCPN repositories and the SCPN STUDIO consume; it is kept factual and current
against the code rather than aspirational.

Companion documents:

- [`backends.md`](backends.md) — acceleration and dispatch matrix (Python, Rust,
  JAX, Mojo, Julia, Go, WebGPU, WASM, FPGA).
- [`interfaces.md`](interfaces.md) — public interface contracts (CLI, REST,
  gRPC, Python library API, STUDIO surface).
- [`subsystems/`](subsystems/) — one document per subsystem with inputs,
  outputs, processing model, backend wiring, and honest scope boundaries.

Scope of this document set: **architecture only** (structure, data flow,
contracts). A state-of-the-art assessment of each component is maintained
separately and internally.

---

## 1. What SPO is

SPO is a domain-agnostic coherence-control toolkit for hierarchical oscillator
systems. It takes a declarative description of a domain (a *domainpack*),
extracts phase from raw signals, integrates coupled phase dynamics (the Universal
Phase Dynamics Equation, UPDE — a Kuramoto/Sakaguchi family with extensions),
observes the resulting dynamics through a large array of monitors, and proposes
bounded, audited control actions through a supervisory layer. The design
posture is **review-only by default**: the core produces control *proposals* and
a tamper-evident audit trail; it does not actuate hardware itself.

The package separates *topology* (the coupling structure and hierarchy declared
per domain) from *physics* (the integrator and observers), so the same engine
serves power-grid, rotating-machinery, neural, swarm, plasma, and other domains
by swapping a domainpack rather than code.

### Scale (verified 2026-06-23)

- ~655 Python modules, ~155k LOC under `src/scpn_phase_orchestrator/`.
- 25 subsystems; largest by LOC: `supervisor` (21.8k), `monitor` (18.5k),
  `experimental` accelerators (18.3k), `upde` (18.3k), `runtime` (17.4k).
- Rust kernel `spo-kernel/` — 6 crates. Differentiable JAX backend `nn/`.
  Polyglot accelerators in Mojo, Julia, Go (35 source files each), plus WebGPU,
  a WASM crate, and an FPGA Verilog core.
- 36 domainpacks under `domainpacks/`.

---

## 2. Canonical pipeline

```
 domainpack YAML
      │
      ▼
 binding/          load_binding_spec → BindingSpec → validate_binding_spec
      │            (topology, hierarchy, families, coupling, drivers,
      │             boundaries, actuators, channels)
      ▼
 oscillators/      raw signal + sample_rate → PhaseExtractor (P / I / S)
      │            → list[PhaseState] (theta, omega, amplitude, quality)
      ▼
 coupling/         CouplingBuilder → K_nm (N×N), alpha (Sakaguchi lags)
      │            (+ Hodge, plasticity, transfer-entropy adaptation, priors)
      ▼
 upde/             UPDEEngine.step(phases, omegas, K_nm, zeta, psi, alpha)
      │            → new phases   [Euler / RK4 / RK45; 14–15 engine variants]
      │            dispatch: Rust → WebGPU → Mojo → Julia → Go → Python
      ▼
 monitor/          30 observers + STL: order parameter R/ψ, chimera, Lyapunov,
      │            transfer entropy, twin-confidence + conformal gate, RQA, …
      ▼
 supervisor/       RegimeManager (hysteresis FSM) + PetriNet + Policy DSL + STL
      │            + predictive MPC (Ott–Antonsen) → ControlAction proposals
      ▼
 actuation/        ActionProjector: clamp to bounds + rate-limit;
      │            optional neural Control Barrier Function filter
      ▼
 runtime/audit     AuditLogger: SHA-256 hash-chained JSONL + protobuf stream,
                   optional HMAC signing; deterministic replay; assurance bundle
```

Optional and parallel tracks (not in the default step loop unless selected):

- **`ssgf/`** — a self-stabilising gauge-field closure that decodes a coupling
  matrix from a latent vector and drives it by a four-term cost.
- **`nn/`** — a JAX/Equinox differentiable re-implementation (learnable K and ω,
  inverse coupling inference, PPO supervisor). A research/optimisation track,
  wired only through the CLI verification command.
- **`autotune/`** — offline inference of a binding spec from time series
  (DMD frequency identification, least-squares coupling estimation, SINDy).

---

## 3. Subsystem index

| Subsystem | Role | Document |
|-----------|------|----------|
| `binding`, `oscillators`, `drivers`, `imprint` | Front end: domainpack → phase | [inputs.md](subsystems/inputs.md) |
| `coupling` | K_nm construction, adaptation, analysis | [coupling.md](subsystems/coupling.md) |
| `ssgf` | Self-stabilising gauge-field closure | [ssgf.md](subsystems/ssgf.md) |
| `autotune` | Offline binding-spec inference | [autotune.md](subsystems/autotune.md) |
| `upde` | Phase-ODE integrator family | [upde.md](subsystems/upde.md) |
| `monitor` | Dynamical observer array + STL | [monitor.md](subsystems/monitor.md) |
| `supervisor` | Regime FSM, policy, MPC, formal export | [supervisor.md](subsystems/supervisor.md) |
| `nn` | Differentiable JAX/Equinox backend | [nn.md](subsystems/nn.md) |
| `runtime`, `actuation`, `assurance`, `audit`, `meta`, `artifacts` | Trust + execution spine | [runtime-trust.md](subsystems/runtime-trust.md) |
| `adapters`, `apps`, `grpc_gen` | Ecosystem bridges + applications | [adapters.md](subsystems/adapters.md) |
| `studio`, `reporting`, `visualization` | Operator-facing surfaces | [studio-reporting.md](subsystems/studio-reporting.md) |
| `plugins`, `scaffold`, `domainpacks` | Extensibility + domain catalogue | [extensibility.md](subsystems/extensibility.md) |
| `experimental/accelerators` | Polyglot acceleration backends | [experimental-accelerators.md](subsystems/experimental-accelerators.md) |

---

## 4. Cross-repository integration

SPO is the topology/solver core of the SCPN ecosystem. Sibling repositories are
**optional** dependencies; SPO does not hard-import any of them. Integration is
through wire formats and domainpacks (see [adapters.md](subsystems/adapters.md)
for the full contract list):

| Sibling repo | Optional extra | Domainpack | Bridge | Coupling |
|--------------|----------------|------------|--------|----------|
| `scpn-control` (plasma) | `plasma` | `plasma_control` | `plasma_control_bridge` | wire format only (no import) |
| `scpn-quantum-control` | `quantum` | `quantum_simulation` | `quantum_control_bridge` | lazy import on demand only |
| `scpn-fusion-core` | `fusion` | `fusion_equilibrium` | `fusion_core_bridge` | wire format only (no import) |
| `sc-neurocore` | — | `neuroscience_eeg`, `brain_connectome` | `synapse_*`, `neurocore_bridge` | HTTP / validator only |
| `remanentia` | — | — | `remanentia_bridge` | HTTP client only |
| `scpn-studio` | `studio` | — | `studio/` surface | dataclass manifest (see below) |

Published wire formats (consumed by siblings / STUDIO): K_nm handshake,
phase-gossip JSON, `quantum_compiler_manifest` v1, `scpn_quantum_target_readiness_v1`,
plasma physics-invariant violations, coherence-memory snapshot, `PhaseState`.

---

## 5. How STUDIO consumes SPO

The `studio/` subsystem is a **review surface** with a live read-only ingestion
feed. It exposes builder functions and a panel registry whose panels are all
`execution_disabled=True` / `operator_review_required=True`, emits
Python-dataclass `ExportManifest` records, and exposes `/api/studio-feed` as a
`studio.control-feed.v1` JSON envelope with SPO runtime state under
`spo.studio-runtime-snapshot.v1`. The feed lets STUDIO ingest live SPO state; it
does not permit hardware writes, QPU execution, or policy promotion. Its
schema-A federation manifest is local-first, advertises the versioned live-feed
evidence schemas, ships no UI module, and passes the current STUDIO Platform
schema-A federation gate.
See [studio-reporting.md](subsystems/studio-reporting.md).

---

## 6. Honest scope boundaries

These are factual statements about what is wired versus library-only versus
declared, so consumers do not over-rely on a capability:

- **Rust kernel builds but is not active in every environment.** Dispatch falls
  back to pure Python (or another available backend) when `spo_kernel` is not
  importable; capability is per-process, computed at import. See `backends.md`.
- **The differentiable `nn/` track is parallel, not in the default loop.** It is
  reachable only through the CLI verification command. The former 1.0-blocking
  validation gaps are resolved; remaining `xfail`s are non-blocking precision,
  finite-size, heuristic-hardness, or test-design limitations.
- **`ssgf/` closure and `autotune/` are opt-in**, not part of the default step
  loop.
- **`experimental/` is a misnomer**: it holds the load-bearing polyglot
  accelerator backends (Mojo/Julia/Go/Rust/WebGPU), fully wired through
  `coupling/`, `monitor/`, and `upde/` — not aspirational research.
- **The declared `wavelet` and `zero_crossing` phase extractors are implemented**
  alongside Hilbert extraction; MQTT and OPC-UA runtime waveform tags dispatch
  through the declared extractor type.
- **The audit trail is real (SHA-256 chain + optional HMAC)**. Protobuf audit
  streams are verified once at run end when attached to `simulate()`; deterministic
  replay remains an opt-in CLI check. Signing is environment-gated, not structural.
- **Some supervisor capabilities are offline / review-only** (federated
  transport, causal counterfactual, evolutionary search, multiverse branches).
  The formal export covers PRISM, TLA+, and generated SMT-LIB feasibility
  models; optional Rust supervisor FFI readiness is probed separately from the
  default Python live-control path.
