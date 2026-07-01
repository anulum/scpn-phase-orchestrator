#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator release provenance spec assembler

"""Assemble the SLSA provenance spec for a release from artefacts and CI context.

The release workflow builds the sdist and the SBOM, then needs a provenance spec to
hand to ``spo provenance-attest``. This module computes that spec deterministically:
each released artefact becomes a digest-pinned subject, the SBOM becomes a
digest-pinned byproduct, and the hash-pinned lock files the build resolved its
dependencies from become digest-pinned resolved dependencies. The build identity,
invocation, and builder version are read from the GitHub Actions environment.

Every digest is computed from the file bytes on disk — nothing is fabricated — and
the source commit is recorded verbatim in the external parameters (a git commit is a
SHA-1/SHA-256 object name, not an artefact content digest, so it belongs in the build
parameters rather than a content-addressed descriptor). The output is a JSON object
in exactly the shape ``spo provenance-attest`` consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_CHUNK = 1 << 20

# The build recipe this spec describes. GitHub-hosted Actions is the hosted build
# service that satisfies the SLSA Build L2 provenance-generation obligation.
BUILD_TYPE = "https://github.com/anulum/scpn-phase-orchestrator/release@v1"


def _sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 of the file at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_env(env: Mapping[str, str], key: str) -> str:
    """Return ``env[key]`` if non-empty, else raise ``KeyError`` naming the variable."""
    value = env.get(key, "")
    if not value:
        raise KeyError(f"required CI environment variable {key} is missing or empty")
    return value


def _descriptor(path: Path, uri: str) -> dict[str, str]:
    """Return a ``{uri, sha256, name}`` descriptor for the file at ``path``."""
    return {"uri": uri, "sha256": _sha256_file(path), "name": path.name}


def build_release_provenance_spec(
    *,
    artifacts: Sequence[Path],
    sbom_files: Sequence[Path],
    lock_files: Sequence[Path],
    env: Mapping[str, str],
) -> dict[str, object]:
    """Return the provenance spec for a release build.

    Parameters
    ----------
    artifacts:
        The released artefacts (e.g. the sdist tarball); each becomes a
        digest-pinned subject. Must be non-empty.
    sbom_files:
        Build byproducts to record (e.g. ``sbom.json``); each becomes a
        digest-pinned byproduct.
    lock_files:
        The hash-pinned dependency lock files the build resolved from; each becomes
        a digest-pinned resolved dependency.
    env:
        The CI environment (the GitHub Actions ``GITHUB_*`` / ``RUNNER_*`` variables).

    Returns
    -------
    dict[str, object]
        The spec object in the shape ``spo provenance-attest`` consumes.

    Raises
    ------
    ValueError
        If no artefacts are supplied.
    KeyError
        If a required CI environment variable is missing.
    """
    if not artifacts:
        raise ValueError("at least one release artefact is required")

    server = _require_env(env, "GITHUB_SERVER_URL").rstrip("/")
    repository = _require_env(env, "GITHUB_REPOSITORY")
    ref = _require_env(env, "GITHUB_REF")
    commit = _require_env(env, "GITHUB_SHA")
    run_id = _require_env(env, "GITHUB_RUN_ID")
    run_attempt = env.get("GITHUB_RUN_ATTEMPT", "1")
    workflow = env.get("GITHUB_WORKFLOW", "Release")
    repo_url = f"{server}/{repository}"

    def _blob_uri(name: str) -> str:
        """Return the repository URL of a lock file at the built commit."""
        return f"{repo_url}/blob/{commit}/requirements/{name}"

    tag = ref.rsplit("/", 1)[-1]

    def _release_uri(name: str) -> str:
        """Return the release-asset download URL for a byproduct."""
        return f"{repo_url}/releases/download/{tag}/{name}"

    subjects = [{"name": path.name, "sha256": _sha256_file(path)} for path in artifacts]
    byproducts = [_descriptor(path, _release_uri(path.name)) for path in sbom_files]
    resolved_dependencies = [
        _descriptor(path, _blob_uri(path.name)) for path in lock_files
    ]

    return {
        "subjects": subjects,
        "build_type": BUILD_TYPE,
        "external_parameters": {
            "ref": ref,
            "repository": repository,
            "workflow": workflow,
            "source_commit": commit,
        },
        "resolved_dependencies": resolved_dependencies,
        "byproducts": byproducts,
        "builder_id": f"{repo_url}/.github/workflows/release.yml@{ref}",
        "invocation_id": (f"{repo_url}/actions/runs/{run_id}/attempts/{run_attempt}"),
        "builder_version": {
            "github_run_id": run_id,
            "github_run_attempt": run_attempt,
            "runner_os": env.get("RUNNER_OS", ""),
            "python": env.get("PROVENANCE_PYTHON_VERSION", ""),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Assemble the spec from CLI arguments and the environment, and write it out.

    Parameters
    ----------
    argv:
        The argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description="Assemble a release provenance spec.")
    parser.add_argument(
        "--artifact",
        dest="artifacts",
        action="append",
        default=[],
        required=True,
        help="A released artefact to record as a subject (repeatable).",
    )
    parser.add_argument(
        "--sbom",
        dest="sbom_files",
        action="append",
        default=[],
        help="A build byproduct (e.g. sbom.json) to record (repeatable).",
    )
    parser.add_argument(
        "--lock",
        dest="lock_files",
        action="append",
        default=[],
        help="A hash-pinned lock file resolved from, under requirements/ (repeatable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the spec JSON.",
    )
    args = parser.parse_args(argv)

    import os

    spec = build_release_provenance_spec(
        artifacts=[Path(item) for item in args.artifacts],
        sbom_files=[Path(item) for item in args.sbom_files],
        lock_files=[Path(item) for item in args.lock_files],
        env=os.environ,
    )
    args.output.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
