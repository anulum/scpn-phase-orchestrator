<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Reproducible dependency lock policy -->

# Dependency Locks

# Why this page exists

Dependency locking is a production control surface in this repository, not just a
build convenience. The same dependency set can execute differently across
machines and time unless constraints are explicit and immutable.

In practical terms, this page is the operational boundary for deterministic
reproducibility:

- every workflow that claims deterministic installation must install from a
  hash-pinned lockfile,
- every lockfile change must be reviewed with the corresponding dependency
  surface,
- every release run must be able to reproduce both the build inputs and the
  runtime environment from the repository snapshot that defines it.

That is why this project treats lockfile maintenance as part of release
readiness, not a post-hoc cleanup chore.

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
- `julia-lock.txt` / `mojo-lock.txt` / `studio-sdk.txt` — optional backend
  and Studio SDK sets, constrained by `dev-lock.txt`.
- `pqc-lock.txt` — post-quantum signing backend.
- `audit-tools.txt` / `release-tools.txt` / `precommit-tools.txt` /
  `publish-tools.txt` — focused tooling lock sets compiled from the matching
  `requirements/*.in` inputs. Publish tooling is installed with dependencies
  from the hashed lock because `build` and `twine` need their runtime
  dependency graphs during isolated artifact checks.
- `ci-tools.txt` / `docs-tools.txt` — hand-maintained hash-pinned lists without
  a generator header; `make lock-refresh` does not rewrite them.

## Operational impact of lock hygiene

Treat lockfile drift as an installation-level behavior change. If two environments run the same code with different resolved dependencies, they are no longer in the same reproducibility boundary, even if their source tree is identical.

For production and regulated contexts:

- lockfile updates are release-scope changes;
- `pip install --require-hashes` is mandatory in deterministic build lanes;
- security or CVE remediation must include lock regeneration in the same commit set.

This keeps dependency posture and behavior traceability part of the same review cycle as code changes.

## Refresh workflow

Prerequisite: [`uv`](https://docs.astral.sh/uv/) with `uvx` on `PATH`. The
refresh does not use a `pip-compile` from the active environment.

Regenerate every generated lockfile:

```bash
make lock-refresh
```

The target runs `tools/refresh_dependency_locks.py`, which:

- runs each `pip-compile` lock with pip-tools 7.6.1 through
  `uvx --python <version> --from pip-tools==7.6.1 pip-compile`, so the 3.11,
  3.12 and 3.13 locks are resolved by the interpreter their headers name;
- resolves the two Windows FFI locks with
  `uv pip compile --python-platform windows`;
- refreshes `dev-lock.txt` before the locks that use it as a `--constraint`;
- restores the licence block that precedes a generated header, and restores
  the committed bytes if a generator fails.

pip-tools 7.5.3 fails against the pip pinned in the development locks
(`ImportError: cannot import name 'stdlib_pkgs'`), which is why the release is
pinned in the tool rather than taken from the environment.

Refresh one lock, list the table, or print the commands without running them:

```bash
python tools/refresh_dependency_locks.py --only dev-lock-py311.txt
python tools/refresh_dependency_locks.py --list
python tools/refresh_dependency_locks.py --dry-run
```

The tool never passes `--upgrade`, so existing pins act as resolver
preferences and every pin that still satisfies the inputs is kept. A version
move is made deliberately, in `pyproject.toml` or the `requirements/*.in`
input, and then refreshed. Run the lock verification checks:

```bash
make lock-check
```

## Manual compile commands

`tests/test_tools_refresh_dependency_locks.py` asserts that the command the tool
runs for each lock equals the command recorded in that lock's committed header
(pip-tools 7.6.1 also records `--no-index` next to `--no-emit-index-url`,
although the tool does not pass it), and that every lock with a generator header
is in the tool's table. Use the
target by default; call a generator by hand only when debugging resolver drift.

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

## Operating with lockfiles in a live service chain

For on-call and release operators, the lockfiles reduce ambiguity during outages:

- if a runtime image diverges from expected behaviour, matching lockfile snapshots
  in `requirements/` lets you isolate whether behaviour changed because of code or
  dependency drift,
- if a CI matrix suddenly fails to resolve, the lock history helps identify if
  solver changes or upstream package yanking caused the break,
- if a security team requests origin traceability, the lock hashes provide
  install-time evidence for every direct and transitive pinned package version.

Use this as part of change review. A dependency bump without lock regeneration is
an installation change that is not auditable from the repository alone.

## Related

- [Install Profiles](install_profiles.md)
- [Production Deployment](production.md)

## Failure handling when lockfiles drift

When a lockfile mismatch appears during CI or deployment:

1. Compare `requirements/` snapshots by lane (Python version + OS + extra
   profile).
2. Confirm dependency constraints in `pyproject.toml` are unchanged.
3. Regenerate the affected lockfile set with the project-standard command.
4. Re-run `make lock-check` and the relevant matrix jobs before release.

If the same dependency set cannot be reproduced from committed inputs, treat the
change as a release risk even when simulation tests are still green. Lockfiles are
part of operational determinism, not optional metadata.

## Reproducibility boundary

A release is reproducible only when both source inputs and pinned install inputs
are unchanged for the target lane. Keep lockfile commits grouped with code changes
that rely on them so audit review can attribute behavior differences to the correct
surface.
