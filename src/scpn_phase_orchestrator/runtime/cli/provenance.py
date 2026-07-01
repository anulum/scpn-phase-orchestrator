# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI SLSA provenance attestation commands

"""Generate and verify a post-quantum-signed SLSA provenance attestation.

``provenance-attest`` reads a build-provenance spec (produced artefacts, build
definition, builder identity) and a signing seed, and emits a DSSE envelope wrapping
a SLSA v1 in-toto statement, signed with ML-DSA. ``provenance-verify`` checks such an
envelope against a trusted public key. Both commands are offline and deterministic:
no timestamps are read and no network call is made, so a CI job can produce the same
attestation for the same inputs and any consumer can verify it later. Publishing the
envelope to a transparency log or verifying it with ``cosign`` is an optional
operator step that needs network and OIDC, and is left to the operator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from scpn_phase_orchestrator.assurance import (
    ArtifactSubject,
    BuildDefinition,
    DsseEnvelope,
    ResourceDescriptor,
    RunDetails,
    build_slsa_provenance_statement,
    sign_provenance_statement,
    verify_dsse_envelope,
)
from scpn_phase_orchestrator.runtime.audit_pqc import (
    DEFAULT_VARIANT,
    MLDSA_VARIANTS,
    signing_key_from_seed,
)
from scpn_phase_orchestrator.runtime.cli._app import main


def _load_spec(path: Path) -> dict[str, Any]:
    """Return the parsed provenance spec object, else raise ``ClickException``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"provenance spec is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException("provenance spec must be a JSON object")
    return payload


def _require_list(spec: dict[str, Any], key: str, *, required: bool) -> list[Any]:
    """Return ``spec[key]`` as a list, else raise ``ClickException``."""
    value = spec.get(key)
    if value is None:
        if required:
            raise click.ClickException(f"provenance spec is missing '{key}'")
        return []
    if not isinstance(value, list):
        raise click.ClickException(f"provenance spec '{key}' must be a list")
    return value


def _require_object(
    spec: dict[str, Any], key: str, *, required: bool
) -> dict[str, Any]:
    """Return ``spec[key]`` as an object, else raise ``ClickException``."""
    value = spec.get(key)
    if value is None:
        if required:
            raise click.ClickException(f"provenance spec is missing '{key}'")
        return {}
    if not isinstance(value, dict):
        raise click.ClickException(f"provenance spec '{key}' must be an object")
    return value


def _require_str(spec: dict[str, Any], key: str) -> str:
    """Return ``spec[key]`` as a non-empty string, else raise ``ClickException``."""
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        raise click.ClickException(
            f"provenance spec '{key}' must be a non-empty string"
        )
    return value


def _subjects(spec: dict[str, Any]) -> tuple[ArtifactSubject, ...]:
    """Build the artefact subjects from the spec, else raise ``ClickException``."""
    entries = _require_list(spec, "subjects", required=True)
    subjects: list[ArtifactSubject] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise click.ClickException("each subject must be an object")
        try:
            subjects.append(
                ArtifactSubject(
                    name=str(entry.get("name")), sha256=str(entry.get("sha256"))
                )
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    if not subjects:
        raise click.ClickException("provenance spec 'subjects' must be non-empty")
    return tuple(subjects)


def _descriptors(spec: dict[str, Any], key: str) -> tuple[ResourceDescriptor, ...]:
    """Build a descriptor list under ``key`` from the spec, else raise.

    Shared by ``resolved_dependencies``, ``builder_dependencies``, and
    ``byproducts`` — each an optional list of ``{uri, sha256, name?}`` objects.
    """
    entries = _require_list(spec, key, required=False)
    descriptors: list[ResourceDescriptor] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise click.ClickException(f"each {key} entry must be an object")
        try:
            descriptors.append(
                ResourceDescriptor(
                    uri=str(entry.get("uri")),
                    sha256=str(entry.get("sha256")),
                    name=str(entry.get("name", "")),
                )
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    return tuple(descriptors)


def _builder_version(spec: dict[str, Any]) -> dict[str, str]:
    """Return the builder version map from the spec, else raise ``ClickException``.

    Every key and value must be a string so the block round-trips through JSON and
    serialises deterministically.
    """
    entries = _require_object(spec, "builder_version", required=False)
    version: dict[str, str] = {}
    for key, value in entries.items():
        if not isinstance(value, str):
            raise click.ClickException(
                f"builder_version value for '{key}' must be a string"
            )
        version[key] = value
    return version


def _load_seed(signing_seed: str | None, signing_seed_file: Path | None) -> str:
    """Return the signing seed from a flag or file, else raise ``ClickException``."""
    if signing_seed and signing_seed_file:
        raise click.ClickException(
            "pass only one of --signing-seed or --signing-seed-file"
        )
    if signing_seed:
        return signing_seed
    if signing_seed_file:
        return signing_seed_file.read_text(encoding="utf-8").strip()
    raise click.ClickException("a signing seed is required (--signing-seed[-file])")


def _load_public_key(public_key: str | None, public_key_file: Path | None) -> str:
    """Return the trusted public key hex from a flag or file, else raise."""
    if public_key and public_key_file:
        raise click.ClickException("pass only one of --public-key or --public-key-file")
    if public_key:
        return public_key
    if public_key_file:
        return public_key_file.read_text(encoding="utf-8").strip()
    raise click.ClickException("a trusted public key is required (--public-key[-file])")


@main.command(name="provenance-attest")
@click.argument(
    "spec_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--signing-seed", default=None, help="ML-DSA signing seed (32-byte hex).")
@click.option(
    "--signing-seed-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File holding the ML-DSA signing seed (hex).",
)
@click.option(
    "--algorithm",
    default=DEFAULT_VARIANT,
    type=click.Choice(MLDSA_VARIANTS),
    help="ML-DSA variant to sign with.",
)
def provenance_attest(
    spec_json: Path,
    signing_seed: str | None,
    signing_seed_file: Path | None,
    algorithm: str,
) -> None:
    """Emit a post-quantum-signed DSSE SLSA provenance attestation.

    Parameters
    ----------
    spec_json : Path
        Path to the provenance spec JSON (subjects, build definition, run details).
    signing_seed : str | None
        The ML-DSA signing seed as hex, if supplied inline.
    signing_seed_file : Path | None
        A file holding the ML-DSA signing seed, as an alternative to the flag.
    algorithm : str
        The ML-DSA variant to sign with.

    Raises
    ------
    ClickException
        If the spec, seed, or signing backend is invalid or unavailable.
    """
    spec = _load_spec(spec_json)
    build_definition = BuildDefinition(
        build_type=_require_str(spec, "build_type"),
        external_parameters=_require_object(spec, "external_parameters", required=True),
        internal_parameters=_require_object(
            spec, "internal_parameters", required=False
        ),
        resolved_dependencies=_descriptors(spec, "resolved_dependencies"),
    )
    run_details = RunDetails(
        builder_id=_require_str(spec, "builder_id"),
        invocation_id=_require_str(spec, "invocation_id"),
        started_on=str(spec.get("started_on", "")),
        finished_on=str(spec.get("finished_on", "")),
        builder_version=_builder_version(spec),
        builder_dependencies=_descriptors(spec, "builder_dependencies"),
        byproducts=_descriptors(spec, "byproducts"),
    )
    # The subjects, build definition, and run details are all pre-validated above,
    # so assembling the statement cannot fail here.
    statement = build_slsa_provenance_statement(
        _subjects(spec), build_definition, run_details
    )

    seed = _load_seed(signing_seed, signing_seed_file)
    try:
        private_key = signing_key_from_seed(seed, algorithm=algorithm)
        envelope = sign_provenance_statement(
            statement, private_key, algorithm=algorithm
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc

    public_bytes = private_key.public_key().public_bytes_raw()
    output = {
        "envelope": envelope.to_dict(),
        "public_key_hex": public_bytes.hex(),
        "statement_subject": sorted(subject.name for subject in statement.subjects),
    }
    click.echo(json.dumps(output, indent=2, sort_keys=True))


@main.command(name="provenance-verify")
@click.argument(
    "envelope_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--public-key", default=None, help="Trusted ML-DSA public key (hex).")
@click.option(
    "--public-key-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File holding the trusted ML-DSA public key (hex).",
)
def provenance_verify(
    envelope_json: Path,
    public_key: str | None,
    public_key_file: Path | None,
) -> None:
    """Verify a DSSE SLSA provenance attestation against a trusted public key.

    Parameters
    ----------
    envelope_json : Path
        Path to the DSSE envelope JSON (as emitted by ``provenance-attest``; a bare
        envelope or the wrapped ``{"envelope": ...}`` output are both accepted).
    public_key : str | None
        The trusted ML-DSA public key as hex, if supplied inline.
    public_key_file : Path | None
        A file holding the trusted ML-DSA public key, as an alternative to the flag.

    Raises
    ------
    ClickException
        If the envelope is malformed or the signature does not verify.
    """
    payload = _load_spec(envelope_json)
    envelope_payload = payload.get("envelope", payload)
    if not isinstance(envelope_payload, dict):
        raise click.ClickException("envelope must be a JSON object")
    try:
        envelope = DsseEnvelope.from_dict(envelope_payload)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    trusted = _load_public_key(public_key, public_key_file)
    try:
        verified = verify_dsse_envelope(envelope, trusted)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not verified:
        raise click.ClickException("provenance attestation signature is not valid")

    try:
        statement = envelope.statement()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps({"verified": True, "statement": statement}, indent=2, sort_keys=True)
    )


__all__ = ["provenance_attest", "provenance_verify"]
