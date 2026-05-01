<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Reproducible dependency lock policy -->

# Dependency Locks

SPO uses `pip-tools` (`pip-compile`) as the authoritative lock generator.
`uv` was evaluated, but `pip-tools` is kept as the project standard because it
already drives the existing hash-pinned lockfiles used by local workflows,
Docker builds, and CI.

## Canonical lockfiles

All lockfiles are hash-pinned and committed under `requirements/`:

- `runtime-lock.txt` — runtime install set used by production image paths.
- `queuewaves-lock.txt` — QueueWaves deployment/install set.
- `dev-lock.txt` — primary Linux/macOS development + CI profile (Python 3.12).
- `dev-lock-py311.txt` — CI matrix profile for Python 3.11.
- `dev-lock-py313.txt` — CI matrix profile for Python 3.13.
- `dev-lock-windows-ffi-py311.txt` — Windows FFI CI profile (Python 3.11).
- `dev-lock-windows-ffi-py312.txt` — Windows FFI CI profile (Python 3.12).
- `audit-tools.txt` / `docs-tools.txt` / `ci-tools.txt` / `publish-tools.txt`
  / `release-tools.txt` — focused tooling lock sets.

## Refresh workflow

Prerequisites:

```bash
python -m pip install --upgrade pip pip-tools
```

Regenerate the lockfiles from `pyproject.toml`:

```bash
make lock-refresh
```

Run the lock verification checks:

```bash
make lock-check
```

## Manual compile commands

The `make lock-refresh` target runs the same `pip-compile` invocations recorded
in lockfile headers. Use it by default. Manual invocation is only needed when
debugging resolver drift.

## CI contract

- CI installs from committed lockfiles with:
  `pip install --require-hashes --no-deps -r <lockfile>`.
- Any dependency change in `pyproject.toml` or lock-input extras must be
  accompanied by regenerated lockfiles in the same PR.
- CI matrix profiles must keep their matching lockfiles in sync.

## Why `pip-tools` over `uv` for this repository

1. Existing cross-platform lock matrix is already in `pip-compile` format.
2. Hash-pinned `--require-hashes` installs are already enforced in workflows.
3. No migration cost or dual-tool drift risk for current CI and Docker paths.

## Related

- [Install Profiles](install_profiles.md)
- [Production Deployment](production.md)
