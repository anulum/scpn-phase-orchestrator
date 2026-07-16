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
- **HMAC record signing**: each record is HMAC-signed when `SPO_AUDIT_KEY`
  is configured (symmetric; verifier needs the shared key).
- **Post-quantum chain seal**: `runtime.audit_pqc` signs the chain tip with
  ML-DSA (FIPS 204) — an additive, publicly verifiable, post-quantum seal over
  the whole log. `verify_audit_log_seal` rejects it if the log changed after
  sealing. Needs the `pqc` extra and an OpenSSL 3.5+ backend (not bundled by
  every platform wheel).
- **Replay verification**: deterministic replay detects divergences
  from the audit trail, catching both tampering and non-determinism.

#### Signing key management

The HMAC signing key is supplied by environment, so it can be injected from any
secrets manager (Vault, AWS/GCP KMS, a Kubernetes secret) without the library
depending on one:

- **`SPO_AUDIT_KEY`** — the current operational signing key. New records are signed
  with it; a verifier needs it to check them.
- **`SPO_AUDIT_KEYRING`** — an optional JSON object mapping key id → historical key
  material, so records signed with a rotated-out key still verify. Key ids are
  derived deterministically (`key_id_for_secret`); a keyring entry whose id does not
  match its key material is rejected (fail-closed).

To rotate: mint a new key, move the outgoing key into `SPO_AUDIT_KEYRING` under its
derived id, and set the new key as `SPO_AUDIT_KEY`. Past records keep verifying
against the keyring while new records use the new key — no re-signing of history and
no verification gap. Deliver both variables from your secrets manager at process
start; never commit key material or bake it into an image. Because the HMAC key is a
shared secret, anyone who can verify can also sign — for third-party verifiability
layer the post-quantum chain seal above on top.

### Dependency Security
- All CI dependencies are **SHA-pinned** (not floating tags).
- Bandit security scanner runs on every commit (pre-commit + CI).
- Ruff linter flags common security anti-patterns.
- No `eval()`, `exec()`, or `pickle.loads()` in the codebase.
- **Vulnerability suppressions are temporary and tracked**: a `--ignore-vuln`
  entry in the CI `pip-audit` step is allowed only while no fixed release
  exists, and must carry an inline comment naming the affected package, the
  reason, and a review-by date. A suppression is removed as soon as the fixed
  release lands in the lockfiles — verified by auditing
  `requirements/dev-lock.txt` and `requirements/audit-tools.txt` with no
  ignore flags.

### Supply-chain provenance
Released artefacts carry **SLSA build provenance**, signed keylessly through
sigstore (Fulcio certificate + Rekor transparency log) using the workflow's
OIDC identity — no long-lived signing secrets exist.

- **PyPI** — wheels and the sdist are published with PEP 740 attestations
  (`pypa/gh-action-pypi-publish`).
- **GitHub release** — the sdist and the CycloneDX SBOM (`sbom.json`) are
  attested with `actions/attest-build-provenance`.
- **Container image** (`ghcr.io/anulum/scpn-phase-orchestrator`) — pushed with
  BuildKit `provenance=mode=max` + SBOM attestations, attested with
  `actions/attest-build-provenance`, and signed keylessly with cosign.

Verify before trusting an artefact:

```bash
# sdist / wheel / SBOM (GitHub attestation)
gh attestation verify <file> --repo anulum/SCPN-PHASE-ORCHESTRATOR

# container image (cosign keyless signature)
cosign verify ghcr.io/anulum/scpn-phase-orchestrator:<version> \
  --certificate-identity-regexp 'https://github.com/anulum/SCPN-PHASE-ORCHESTRATOR/.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

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
