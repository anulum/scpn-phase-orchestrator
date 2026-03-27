# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.5.x   | Yes                |
| 0.4.x   | Security fixes only |
| < 0.4   | No                 |

## Reporting a Vulnerability

**Preferred**: Use [GitHub Security Advisories](https://github.com/anulum/scpn-phase-orchestrator/security/advisories) to report vulnerabilities privately.

**Alternative**: Email **protoscience@anulum.li** with:

1. Description of the vulnerability
2. Steps to reproduce
3. Impact assessment (which components are affected)
4. Suggested severity (Critical / High / Medium / Low)

## Response SLA

| Phase | Timeline |
|-------|----------|
| Acknowledgement | 48 hours |
| Initial assessment + severity | 7 days |
| Fix timeline communicated | After assessment |
| Patch release (Critical/High) | 14 days target |
| Patch release (Medium/Low) | Next scheduled release |

## Severity Classification

We follow [CVSS v3.1](https://www.first.org/cvss/) scoring:

| Severity | CVSS | Examples |
|----------|------|----------|
| **Critical** (9.0-10.0) | Remote code execution, audit chain bypass | Arbitrary code via binding spec, SHA256 chain forgery |
| **High** (7.0-8.9) | Data exfiltration, control parameter injection | Coupling matrix injection via API, phase data leak |
| **Medium** (4.0-6.9) | Denial of service, information disclosure | OOM via large N, timing side-channel in audit |
| **Low** (0.1-3.9) | Minor info leak, hardening gap | Version string disclosure, verbose error messages |

## Threat Model

SPO operates in environments ranging from research simulations to
safety-critical control systems. The threat model covers:

### Input Validation
- **Binding specifications**: validated against JSON schema at load time.
  Malformed YAML cannot reach the engine.
- **Phase arrays**: checked for finite values, correct dimensions.
  NaN/Inf inputs are rejected by the Rust kernel.
- **Coupling matrices**: validated for shape, diagonal zeros, and
  non-negativity (where required by geometry constraints).

### Audit Integrity
- **SHA256 hash chain**: each audit record includes the hash of the
  previous record. Tampering with any record breaks the chain.
- **Replay verification**: deterministic replay detects divergences
  from the audit trail, catching both tampering and non-determinism.

### Dependency Security
- All CI dependencies are **SHA-pinned** (not floating tags).
- Bandit security scanner runs on every commit (pre-commit + CI).
- Ruff linter flags common security anti-patterns.
- No `eval()`, `exec()`, or `pickle.loads()` in the codebase.

### Control System Safety
- **Rate limits**: actuation layer enforces maximum parameter change
  per step, preventing discontinuous jumps that destabilise dynamics.
- **Regime FSM**: hard violations force CRITICAL regime regardless of
  supervisor policy, preventing unsafe control actions.
- **Boundary observer**: monitors safety limits independently of the
  supervisor, providing defense-in-depth.

## Disclosure Policy

- Do not disclose publicly before a fix is released.
- We will credit reporters in the CHANGELOG unless anonymity is requested.
- Security advisories are published via GitHub after the fix is available.
- CVE identifiers are requested for Critical and High severity issues.

## Contact

**Email**: protoscience@anulum.li
**GitHub Security Advisories**: [Report here](https://github.com/anulum/scpn-phase-orchestrator/security/advisories)
**PGP**: Available on request for encrypted communication.
