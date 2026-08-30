#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reproducible Python release builder

"""Build byte-reproducible Python wheel and source artifacts.

Setuptools currently preserves wall-clock metadata in source distributions.
This tool builds through the configured PEP 517 backend, then rewrites only the
sdist container metadata to a canonical, path-safe tar+gzip representation.
Package file bytes and names are not modified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
_COPY_BUFFER_BYTES = 1024 * 1024


def _positive_epoch(value: str) -> int:
    try:
        epoch = int(value)
    except ValueError as exc:
        raise ValueError("SOURCE_DATE_EPOCH must be a positive integer") from exc
    if epoch <= 0:
        raise ValueError("SOURCE_DATE_EPOCH must be a positive integer")
    return epoch


def resolve_source_date_epoch(
    explicit: str | None,
    *,
    environment: Mapping[str, str] = os.environ,
    root: Path = ROOT,
) -> int:
    """Resolve a fixed build epoch from CLI, environment, or exact Git HEAD."""
    if explicit is not None:
        return _positive_epoch(explicit)
    configured = environment.get("SOURCE_DATE_EPOCH", "").strip()
    if configured:
        return _positive_epoch(configured)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required to derive SOURCE_DATE_EPOCH")
    result = subprocess.run(
        [git, "show", "-s", "--format=%ct", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return _positive_epoch(result.stdout.strip())


def _safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe sdist member path: {name!r}")
    canonical = path.as_posix()
    if canonical in {"", "."} or canonical != name.rstrip("/"):
        raise ValueError(f"non-canonical sdist member path: {name!r}")
    return canonical


def _normalized_mode(member: tarfile.TarInfo) -> int:
    if member.isdir():
        return 0o755
    return 0o755 if member.mode & 0o111 else 0o644


def canonicalize_sdist(source: Path, destination: Path, *, epoch: int) -> None:
    """Rewrite one sdist with sorted entries and deterministic safe metadata."""
    if destination.exists():
        raise FileExistsError(f"release artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="spo-canonical-sdist-",
        dir=destination.parent,
    ) as temporary:
        temporary_path = Path(temporary)
        canonical_tar = temporary_path / "canonical.tar"
        staged_gzip = temporary_path / destination.name
        with tarfile.open(source, mode="r:gz") as archive:
            members = sorted(archive.getmembers(), key=lambda item: item.name)
            if not members:
                raise ValueError("sdist archive is empty")
            with tarfile.open(
                canonical_tar,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as output:
                for member in members:
                    name = _safe_member_name(member.name)
                    if not (member.isdir() or member.isfile()):
                        raise ValueError(
                            f"unsupported sdist member type: {member.name!r}"
                        )
                    canonical = tarfile.TarInfo(name=name)
                    canonical.type = (
                        tarfile.DIRTYPE if member.isdir() else tarfile.REGTYPE
                    )
                    canonical.mode = _normalized_mode(member)
                    canonical.uid = 0
                    canonical.gid = 0
                    canonical.uname = ""
                    canonical.gname = ""
                    canonical.mtime = epoch
                    canonical.pax_headers = {}
                    if member.isfile():
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise ValueError(
                                f"sdist member cannot be read: {member.name!r}"
                            )
                        canonical.size = member.size
                        output.addfile(canonical, extracted)
                    else:
                        canonical.size = 0
                        output.addfile(canonical)
        with (
            canonical_tar.open("rb") as raw_tar,
            staged_gzip.open("wb") as raw_gzip,
            gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_gzip,
                mtime=epoch,
            ) as compressed,
        ):
            shutil.copyfileobj(raw_tar, compressed, length=_COPY_BUFFER_BYTES)
        staged_gzip.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_COPY_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def build_release_artifacts(
    output_directory: Path,
    *,
    epoch: int,
    build_sdist: bool,
    build_wheel: bool,
    root: Path = ROOT,
) -> tuple[Path, ...]:
    """Build selected artifacts and return their sorted destination paths."""
    if not build_sdist and not build_wheel:
        raise ValueError("at least one release artifact kind must be selected")
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spo-raw-release-") as temporary:
        raw_output = Path(temporary)
        command = [sys.executable, "-m", "build"]
        if build_sdist:
            command.append("--sdist")
        if build_wheel:
            command.append("--wheel")
        command.extend(("--outdir", str(raw_output)))
        environment = dict(os.environ)
        environment["SOURCE_DATE_EPOCH"] = str(epoch)
        environment.setdefault("PYTHONHASHSEED", "0")
        subprocess.run(command, cwd=root, env=environment, check=True)

        raw_artifacts = sorted(
            path
            for path in raw_output.iterdir()
            if path.suffix == ".whl" or path.name.endswith(".tar.gz")
        )
        expected_count = int(build_sdist) + int(build_wheel)
        if len(raw_artifacts) != expected_count:
            raise ValueError(
                "build backend returned an unexpected release artifact set"
            )
        destinations: list[Path] = []
        for artifact in raw_artifacts:
            destination = output_directory / artifact.name
            if artifact.name.endswith(".tar.gz"):
                canonicalize_sdist(artifact, destination, epoch=epoch)
            else:
                if destination.exists():
                    raise FileExistsError(
                        f"release artifact already exists: {destination}"
                    )
                shutil.copyfile(artifact, destination)
            destinations.append(destination)
    return tuple(sorted(destinations))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("dist"),
        help="Artifact destination (default: dist).",
    )
    parser.add_argument(
        "--source-date-epoch",
        default=None,
        help="Positive UNIX epoch; defaults to SOURCE_DATE_EPOCH or Git HEAD.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sdist-only", action="store_true")
    selection.add_argument("--wheel-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build requested release artifacts and print immutable SHA-256 lines."""
    args = _parser().parse_args(argv)
    epoch = resolve_source_date_epoch(args.source_date_epoch)
    artifacts = build_release_artifacts(
        args.outdir,
        epoch=epoch,
        build_sdist=not args.wheel_only,
        build_wheel=not args.sdist_only,
    )
    for artifact in artifacts:
        print(f"{_sha256(artifact)}  {artifact}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by release commands
    raise SystemExit(main())
