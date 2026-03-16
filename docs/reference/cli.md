# CLI Reference

The `spo` command is the primary entry point. Install with
`pip install scpn-phase-orchestrator` and the `spo` command becomes available.

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

**Output:** List of validation errors (one per line) or `Valid`.

**Example:**

```bash
spo validate domainpacks/bio_stub/binding_spec.yaml
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

**Output:** Per-step regime, R_global, boundary violations, and policy actions
printed to stdout.

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

## `spo scaffold`

Scaffold a new domainpack directory with starter files.

```
spo scaffold <name>
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `name` | Domain name (used as directory name under `domainpacks/`) |

**Creates:**

```
domainpacks/<name>/
    binding_spec.yaml
    policy.yaml
    run.py
    README.md
```

**Example:**

```bash
spo scaffold power_grid
```

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
