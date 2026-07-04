# CLI Reference

The `spo` command is the primary entry point. Install with
`pip install scpn-phase-orchestrator` and the `spo` command becomes available.

## Operational position

The CLI is the orchestration boundary between raw domain inputs and reviewable
evidence. It is deliberately split into three planes:

- validation plane: binding parsing, schema checks, and resolved runtime defaults,
- execution plane: deterministic simulation, replayable audits, and proposal
  generation,
- reporting plane: deterministic summaries, regulator-oriented explanations, and
  transport checks.

A typical production workflow starts with `spo doctor` to confirm the
environment is ready, then `spo validate`, moves to `spo inspect` or `spo run`
on an accepted binding, and finally uses `spo replay` plus `spo report` before
any dashboard or platform review.

That separation keeps intent and actuation boundaries explicit. Even when the
same user runs each command on one terminal session, each command has an
independent evidence contract and explicit pass/fail output.

---

## `spo doctor`

Check that the current environment can run the orchestrator: the interpreter
version, the required runtime dependencies, the optional native compute backends
(Rust `spo_kernel`, Julia `juliacall`, the Go toolchain or prebuilt shared
libraries, and the Mojo toolchain), and the optional feature extras (`nn`,
`studio`, `queuewaves`, `plot`, `otel`, `notebook`). It also checks the
package-local review/export adapter surfaces for FMI co-simulation and hybrid
co-compilation manifests. Run it first after any install, upgrade, or move to a
new host.

```
spo doctor [--json-out]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json-out` | Emit the readiness record as a JSON audit object instead of text |

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Interpreter in range and all required dependencies present |
| 1 | Interpreter out of range or a required dependency missing |

Missing *optional* backends and extras are reported as warnings and never change
the exit code; only a missing *required* dependency or an unsupported
interpreter fails the check. The probe imports no heavy optional packages and
makes no network calls.

**Example:**

```bash
spo doctor
```

Example output (truncated):

```text
SCPN Phase Orchestrator environment diagnostics — PASS
  python   3.12.3  (Linux x86_64)

  [ ok ] python    Python 3.12.3 satisfies >=3.10,<3.14
  [ ok ] numpy     numpy 2.4.6
  [ ok ] rust      spo_kernel 0.5.10 importable (PyO3 FFI ready)
  [ ok ] go        prebuilt shared libraries: libhodge.so, libnpe.so, ...
  [ ok ] fmi-cosimulation  FMI 3.0 co-simulation export/review surface: ...
  [warn] streamlit not installed — install the 'studio' extra

PASS: all required dependencies are present (... optional components available).
```

The JSON form (`spo doctor --json-out`) returns a stable record with overall
`status`, per-component `checks`, and `missing_required` / `missing_optional`
lists for CI gating and provisioning automation.

---

## `spo validate`

Validate a binding specification YAML file against the schema.

```
spo validate <binding_spec>
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `binding_spec` | Path to a `binding_spec.yaml` file |

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Valid specification |
| 1 | Validation errors found |

**Output:** List of validation errors (one per line) or `Valid` followed by
the resolved binding configuration summary.

**Example:**

```bash
spo validate domainpacks/bio_stub/binding_spec.yaml
```

Example success output:

```text
Valid
Resolved configuration:
  domain: bio_stub v0.1.0 (research)
  timing: sample=0.01s control=0.1s interval=10 steps
  structure: layers=3 oscillators=6 channels=I, P, S
  engine: kuramoto features=none
```

---

## `spo run`

Run a simulation from a binding specification.

```
spo run <binding_spec> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `binding_spec` | Path to a `binding_spec.yaml` file |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--steps N` | 100 | Number of integration steps |
| `--audit PATH` | None | Write audit log to JSONL file |
| `--seed INT` | None | RNG seed for reproducibility |

**Output:** Resolved binding configuration, then final `R_good`, `R_bad`, and
regime. If `--audit` is set, the audit header contains the same resolved
configuration summary under `binding_summary` (with backward-compatible
`binding_config` key).

**Example:**

```bash
spo run domainpacks/bio_stub/binding_spec.yaml --steps 500 --audit run.jsonl --seed 42
```

---

## `spo replay`

Replay an audit log and optionally verify deterministic reproducibility.

```
spo replay <audit_log> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `audit_log` | Path to an `audit.jsonl` file produced by `spo run --audit` |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--verify` | off | Verify deterministic reproducibility (rebuild engine from header, compare states) |
| `--output PATH` | None | Write verification report to JSON file |

**Output:** Step-by-step comparison with pass/fail verdict.

**Example:**

```bash
spo replay run.jsonl --verify --output report.json
```

---

## `spo federated-transport-preflight`

Build review-only federated transport evidence from node update audit records.
The command reads newline-delimited node-update JSON records, wraps them in
signed/hash-linked transport envelopes, replays the ordered batch, validates the
transport declaration, and emits one deterministic preflight bundle. It does not
open sockets, export raw data, or permit live transport execution.

```
spo federated-transport-preflight <node_updates.jsonl> <transport.json> [--output PATH]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `node_updates.jsonl` | JSONL node update audit records accepted by `supervisor.federated_transport` |
| `transport.json` | Transport declaration with `transport`, endpoint, owner/auth/TLS approval fields, or JSONL replay evidence |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` | None | Write the emitted preflight bundle JSON while also printing it to stdout |

**Output:** `scpn_federated_transport_preflight_bundle_v1` JSON containing
`envelopes`, `replay_ledger`, `preflight_manifest`, and `bundle_hash`.

**Example:**

```bash
spo federated-transport-preflight updates.jsonl transport.json \
  --output federated_transport_preflight.json
```

---

## `spo federated-secure-aggregation-preflight`

Build review-only federated secure-aggregation evidence from node commitment
records. The command reads newline-delimited node-commitment JSON records, builds
the deterministic secure-aggregation manifest, validates custody/quorum evidence
and operator approval from the deployment declaration, and emits one deterministic
preflight bundle. It does not open sockets, export raw data, or permit live
secure-aggregation execution.

```
spo federated-secure-aggregation-preflight <commitments.jsonl> <deployment.json> [--output PATH]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `commitments.jsonl` | JSONL node commitment records accepted by `supervisor.federated_secure_aggregation` |
| `deployment.json` | Deployment declaration with optional `aggregation` policy plus `quorum_evidence`, `custody_rotation_policy`, `custody_records`, `accepted_node_threshold`, and operator approval fields |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` | None | Write the emitted preflight bundle JSON while also printing it to stdout |

**Output:** `scpn_federated_secure_aggregation_preflight_bundle_v1` JSON containing
`secure_aggregation_manifest`, `preflight_manifest`, and `bundle_hash`.

**Example:**

```bash
spo federated-secure-aggregation-preflight commitments.jsonl deployment.json \
  --output federated_secure_aggregation_preflight.json
```

---

## `spo federated-dp-noise-service-preflight`

Build review-only federated DP noise-service evidence from a DP-noise request and
a deployment declaration. The command builds the deterministic request and
response manifests, validates deployment prerequisites (mechanism, custody,
accountant, budget issuer, service endpoint, operator approval), and emits one
deterministic preflight bundle. Missing prerequisites are reported as a not-ready
readiness verdict rather than an error; only malformed inputs fail closed. It does
not open sockets, generate live noise, or permit live DP noise-service execution.

```
spo federated-dp-noise-service-preflight <request.json> <deployment.json> [--output PATH]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `request.json` | DP-noise request with `epsilon`, `delta`, `sensitivity`, `noise_multiplier`, `node_count`, `seed_hash`, `policy_keys`, and per-node `node_budgets` |
| `deployment.json` | Deployment declaration with `mechanism_label`, `privacy_accountant_owner`, `seed_custody_label`, `budget_issuer_label`, `service_endpoint_label`, and `operator_approved` |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` | None | Write the emitted preflight bundle JSON while also printing it to stdout |

**Output:** `scpn_federated_dp_noise_service_preflight_bundle_v1` JSON containing
`request_manifest`, `response_manifest`, `preflight_manifest`, `deployment_ready`,
and `bundle_hash`.

**Example:**

```bash
spo federated-dp-noise-service-preflight request.json deployment.json \
  --output federated_dp_noise_service_preflight.json
```

---

## `spo evolutionary-policy-dsl-search`

Run the deterministic offline policy-DSL mutation grammar over a compact policy-DSL
source and emit a review-only candidate bundle. It does not actuate, merge,
hot-patch, or execute any mutated candidate.

```
spo evolutionary-policy-dsl-search <policy.dsl> [--generations N] [--population N] [--mutation-step F] [--output PATH]
```

**Output:** `scpn_evolutionary_grammar_review_bundle_v1` JSON with `grammar`
(`policy-dsl`), the offline search `report`, and `bundle_hash`.

---

## `spo evolutionary-petri-mutation`

Run the deterministic offline Petri-net mutation grammar over a net-like JSON
payload (object or record array) and emit a review-only candidate bundle.

```
spo evolutionary-petri-mutation <net.json> [--generations N] [--candidates-per-generation N] [--mutation-step F] [--max-arc-weight N] [--max-token-bound N] [--output PATH]
```

**Output:** `scpn_evolutionary_grammar_review_bundle_v1` JSON with `grammar`
(`petri`), the offline mutation `report`, and `bundle_hash`.

---

## `spo evolutionary-topology-mutation`

Run the deterministic offline topology mutation grammar over a topology JSON with
`nodes` and `edges` arrays and emit a review-only candidate bundle.

```
spo evolutionary-topology-mutation <topology.json> [--generations N] [--population N] [--mutation-step F] [--min-edge-weight F] [--max-edge-weight F] [--edge-add-base-weight F] [--max-add-candidates N] [--output PATH]
```

**Output:** `scpn_evolutionary_grammar_review_bundle_v1` JSON with `grammar`
(`topology`), the offline mutation `report`, and `bundle_hash`.

---

## `spo assurance-case`

Assemble a review-only assurance-case bundle from audit and evidence records.

```
spo assurance-case --system NAME [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--audit-log PATH` | None | Add an audit-chain integrity evidence item from a JSONL audit log |
| `--evidence-file PATH` | None | Add JSON evidence records; repeatable |
| `--output PATH` | None | Write the bundle JSON instead of printing to stdout |

The command emits a deterministic `scpn_assurance_case_bundle_v1` record with a
bundle hash, per-standard clause coverage summary, evidence items, conformance
records, and the regulatory disclaimer. The output is a technical
evidence-mapping aid; it does not permit actuation or claim compliance.

**Example:**

```bash
spo assurance-case --system grid-review \
  --audit-log run.jsonl \
  --evidence-file twin_confidence.json \
  --output assurance_bundle.json
```

---

## `spo certification-evidence`

Assemble a standards-shaped review package around the assurance-case bundle.

```
spo certification-evidence --system NAME --output-dir DIR [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--audit-log PATH` | None | Add an audit-chain integrity evidence item from a JSONL audit log |
| `--evidence-file PATH` | None | Add JSON evidence records; repeatable |
| `--output-dir DIR` | Required | Write `manifest.json`, `assurance_bundle.json`, and `test_vectors.json` |

The command refuses to write into a non-empty output directory. The manifest
seals the package with SHA-256 file digests, the assurance bundle hash, the
standards covered, and a package hash. `test_vectors.json` lets reviewers
recompute evidence content hashes and clause-rationale hashes without trusting
incidental file ordering.

**Example:**

```bash
spo certification-evidence --system grid-review \
  --audit-log run.jsonl \
  --evidence-file twin_confidence.json \
  --output-dir review_package
```

---

## `spo koopman-mpc`

Run the review-only dVOC oscillation-damping demonstration and emit the sealed
before/after PRC oscillation evidence when `--output` is set. The command builds
a local underdamped oscillator, fits a Koopman predictor, applies offline
Koopman-MPC damping, and maps both ringdowns to PRC-028-1 / PRC-030-1 review
evidence. It does not actuate a live plant.

```
spo koopman-mpc [--frequency-hz HZ] [--damping-ratio ZETA] [--dt S] [--horizon N] [--output PATH]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--frequency-hz HZ` | `0.5` | Natural oscillator frequency |
| `--damping-ratio ZETA` | `0.02` | Open-loop damping ratio |
| `--dt S` | `0.02` | Sampling interval in seconds |
| `--horizon N` | `300` | Ringdown steps for each pass |
| `--output PATH` | None | Write `scpn_dvoc_oscillation_damping_audit_v1` JSON |

**Example:**

```bash
spo koopman-mpc --output dvoc_oscillation_damping.json
```

---

## `spo pmu-ieee-adapt`

Convert an IEEE-format multi-header PMU concentrator CSV into the two-column
`time_s,frequency_hz` series the ringdown screener consumes. Phasor measurement
concentrators and the oscillation-detection literature export a wide layout — a
label row, a quantity-type row (`T` for time, `F` for frequency, `VM`/`VA`/`IM`/
`IA` for the phasor channels), a unit row, and a secondary-label row, then one
time column and five channels per unit. The command parses that layout, counts
each frequency channel's dropouts, selects the channel that is free of dropouts
and within `--plausible-band-hz` of the nominal frequency (breaking ties toward
the largest peak-to-peak swing, which carries the most oscillation content), and
writes the selected channel with a SHA-256 provenance line linking the derived
CSV back to the source capture. The derived CSV is then screened with
`spo pmu-ringdown`.

```
spo pmu-ieee-adapt <csv_path> <output_path> [--nominal-frequency-hz HZ] [--plausible-band-hz HZ] [--time-column NAME] [--frequency-column NAME]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `<csv_path>` | Required | IEEE-format multi-header PMU concentrator CSV |
| `<output_path>` | Required | Destination for the derived two-column ingester CSV |
| `--nominal-frequency-hz HZ` | `60.0` | Nominal grid frequency used to reject out-of-band channels |
| `--plausible-band-hz HZ` | `2.0` | Half-width in hertz of the accepted band about the nominal frequency |
| `--time-column NAME` | `time_s` | Timestamp column name written to the derived CSV |
| `--frequency-column NAME` | `frequency_hz` | Frequency column name written to the derived CSV |

**Example:**

```bash
spo pmu-ieee-adapt concentrator_capture.csv ringdown.csv \
  --nominal-frequency-hz 60.0 \
  --plausible-band-hz 2.0

spo pmu-ringdown ringdown.csv \
  --event-id PMU-EVT-001 \
  --captured-at 2026-07-04T10:00:00Z \
  --signal-source PMU/BUS-42/frequency \
  --analysis-rate-hz 5.0 \
  --output pmu_prc.json
```

---

## `spo pmu-ringdown`

Screen a local PMU or historian frequency ringdown CSV for PRC oscillation
review evidence. The command validates uniformly sampled finite frequency data,
subtracts the nominal grid frequency, mean-detrends the deviation to remove the
operating-point offset, optionally decimates an over-sampled capture, estimates
oscillation modes, and writes a hash-sealed `scpn_pmu_ringdown_prc_audit_v1`
record when `--output` is set. Timestamp uniformity is measured against the
best-fit uniform grid, so a capture exported with decimal-rounded timestamps is
accepted. The record keeps both the raw capture rate and the post-decimation
analysis rate.

```
spo pmu-ringdown <csv_path> --event-id ID --captured-at TIMESTAMP --signal-source LABEL [--nominal-frequency-hz HZ] [--detrend MODE] [--analysis-rate-hz HZ] [--output PATH]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--event-id ID` | Required | Operator event identifier stamped into the evidence |
| `--captured-at TIMESTAMP` | Required | Capture timestamp stamped into the evidence |
| `--signal-source LABEL` | Required | PMU or historian signal label |
| `--time-column` | `time_s` | Timestamp column in seconds |
| `--frequency-column` | `frequency_hz` | Frequency column in hertz |
| `--nominal-frequency-hz HZ` | `60.0` | Nominal frequency subtracted before screening |
| `--detrend MODE` | `mean` | Deviation detrend: `mean` removes the operating-point offset, `none` disables it |
| `--analysis-rate-hz HZ` | None | Decimate an over-sampled capture to this rate before estimation (about ten times the highest mode of interest) |
| `--output PATH` | None | Write the sealed JSON evidence record |

**Example:**

```bash
spo pmu-ringdown pmu_ringdown.csv \
  --event-id PMU-EVT-001 \
  --captured-at 2026-07-04T10:00:00Z \
  --signal-source PMU/BUS-42/frequency \
  --output pmu_prc.json
```

---

## `spo ibr-ride-through`

Screen a local operator CSV for NERC PRC-029-1 ride-through review evidence.
The command reads timestamp, voltage, and frequency columns, applies the
review-only PRC-029 voltage/frequency screening tables, and writes a
hash-sealed `scpn_ibr_ride_through_prc029_audit_v1` record when `--output` is
set. It does not actuate or certify compliance.

```
spo ibr-ride-through <csv_path> --event-id ID --captured-at TIMESTAMP --signal-source LABEL [--ibr-category other_ibr|ac_wind] [--output PATH]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--event-id ID` | Required | Operator event identifier stamped into the evidence |
| `--captured-at TIMESTAMP` | Required | Capture timestamp stamped into the evidence |
| `--signal-source LABEL` | Required | Operator-facing IBR measurement source label |
| `--ibr-category` | `other_ibr` | PRC-029 voltage table selector (`other_ibr` or `ac_wind`) |
| `--time-column` | `time_s` | Timestamp column in seconds |
| `--voltage-column` | `voltage_pu` | Voltage column in per unit |
| `--frequency-column` | `frequency_hz` | Frequency column in hertz |
| `--output PATH` | None | Write the sealed JSON evidence record |

**Example:**

```bash
spo ibr-ride-through ride_through.csv \
  --event-id IBR-EVT-001 \
  --captured-at 2026-07-04T13:45:00Z \
  --signal-source IBR-17/high-side-transformer \
  --output prc029_ride_through.json
```

---

## `spo power-grid-prc-bundle`

Bind the review-only dVOC, PMU ringdown, and IBR ride-through evidence files
into one assessor handoff package. The command verifies each child schema,
claim boundary, content hash, and source-file SHA-256 before writing
`scpn_power_grid_prc_audit_bundle_v1`. It is a packaging surface only: it does
not re-screen data, actuate, or certify compliance.

```
spo power-grid-prc-bundle --bundle-id ID --created-at TIMESTAMP --operator-context TEXT --dvoc-evidence PATH --pmu-ringdown PATH --ibr-ride-through PATH [--output PATH]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--bundle-id ID` | Required | Operator bundle identifier |
| `--created-at TIMESTAMP` | Required | Bundle creation timestamp |
| `--operator-context TEXT` | Required | Human-readable assessor review context |
| `--dvoc-evidence PATH` | Required | `scpn_dvoc_oscillation_damping_audit_v1` JSON |
| `--pmu-ringdown PATH` | Required | `scpn_pmu_ringdown_prc_audit_v1` JSON |
| `--ibr-ride-through PATH` | Required | `scpn_ibr_ride_through_prc029_audit_v1` JSON |
| `--output PATH` | None | Write the sealed bundle JSON |

**Example:**

```bash
spo power-grid-prc-bundle \
  --bundle-id PG-REVIEW-001 \
  --created-at 2026-07-04T15:10:00Z \
  --operator-context "western interconnection post-event review" \
  --dvoc-evidence dvoc_oscillation_damping.json \
  --pmu-ringdown pmu_prc.json \
  --ibr-ride-through prc029_ride_through.json \
  --output power_grid_prc_bundle.json
```

---

## `spo scaffold`

Scaffold a new domainpack directory with starter files.

```
spo scaffold <name>
spo scaffold <name> --llm --description "..."
spo scaffold <name> --llm --description "..." --llm-response-json proposal.json
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `name` | Domain name (used as directory name under `domainpacks/`) |

**Creates:**

```
domainpacks/<name>/
    binding_spec.yaml
    README.md
    llm_scaffold_audit.json   # only with --llm
```

**Example:**

```bash
spo scaffold power_grid
spo scaffold traffic_grid --llm \
  --description "I am modelling traffic lights in a 4-intersection grid"
```

LLM mode requires a configured provider through `SPO_LLM_ENDPOINT` and
`SPO_LLM_MODEL`, or an offline captured proposal via `--llm-response-json`.
The generated JSON proposal is normalised, converted to binding YAML, and
validated before any domainpack files are written.

---

## `spo report`

Generate a coherence summary from an audit log.

```
spo report <audit_log> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `audit_log` | Path to an `audit.jsonl` file |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--json-out` | off | Output as JSON instead of human-readable text |

**Output:** R_global statistics (mean, min, max, std), regime distribution
(fraction of steps in each regime), and action counts.

**Example:**

```bash
spo report run.jsonl --json-out > summary.json
```

---

## `spo explain`

Generate a human-readable explanation report from an audit log.

```
spo explain <audit_log> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `audit_log` | Path to an `audit.jsonl` file |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--markdown-out PATH` | None | Write a Markdown explanation report |
| `--pdf-out PATH` | None | Write a dependency-free text PDF report |
| `--max-actions N` | 12 | Maximum control actions to explain |

**Output:** regime distribution, regime transitions, metric evidence,
control-action explanations, event summaries, and hash-chain status.

**Example:**

```bash
spo explain run.jsonl --markdown-out explain.md --pdf-out explain.pdf
```

---

## `spo queuewaves serve`

Start the QueueWaves cascade failure detection server.

```
spo queuewaves serve [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config PATH` | None | QueueWaves YAML configuration file |
| `--host HOST` | `0.0.0.0` | Bind address |
| `--port PORT` | `8080` | Bind port |

**Example:**

```bash
spo queuewaves serve --config qw_config.yaml --port 9090
```

---

## `spo queuewaves check`

Run a one-shot health check against configured QueueWaves endpoints.

```
spo queuewaves check [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config PATH` | None | QueueWaves YAML configuration file |

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | All endpoints healthy |
| 1 | One or more endpoints degraded or unreachable |

**Example:**

```bash
spo queuewaves check --config qw_config.yaml
```

## Choosing commands by objective

Use this sequence when you need a production-level audit trail:

1. validate that the domain spec is structurally correct,
2. inspect resolved settings before any simulation run,
3. execute with `--audit` so later replay can confirm every step,
4. report and explain before handing results to operators.

Use `spo run --audit` for deterministic replay support.
Use `spo replay --verify` whenever a new dependency, backend, or host profile is
introduced.
