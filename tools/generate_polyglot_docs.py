#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — native polyglot API documentation generator

"""Generate native API-documentation artifacts for every maintained backend.

The repository deliberately keeps Go accelerators as independent ``package
main`` translation units, Julia accelerators as independent modules, and Mojo
accelerators as independent executables.  This coordinator preserves those
boundaries instead of combining sources into synthetic packages solely for
documentation.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("rust", "go", "julia", "mojo")


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one native documentation command and fail with its diagnostics."""
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"required documentation tool is unavailable: {command[0]}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
        raise RuntimeError(
            f"documentation command failed ({' '.join(command)}):\n{details}"
        ) from exc


def _generate_rust(repo_root: Path, output_root: Path) -> None:
    """Build rustdoc HTML for the workspace with warnings promoted to errors."""
    destination = output_root / "rust"
    with tempfile.TemporaryDirectory(prefix="spo-rustdoc-") as temporary:
        target = Path(temporary) / "target"
        env = os.environ.copy()
        env["RUSTDOCFLAGS"] = "-D warnings"
        _run(
            (
                "cargo",
                "doc",
                "--locked",
                "--no-deps",
                "--workspace",
                "--target-dir",
                str(target),
            ),
            cwd=repo_root / "spo-kernel",
            env=env,
        )
        shutil.copytree(target / "doc", destination, dirs_exist_ok=True)


def _generate_go(repo_root: Path, output_root: Path) -> None:
    """Run ``go doc`` independently for every C-shared translation unit."""
    source_root = repo_root / "go"
    destination = output_root / "go"
    destination.mkdir(parents=True, exist_ok=True)
    sources = sorted(source_root.glob("*.go"))
    if not sources:
        raise RuntimeError(f"no Go documentation sources found under {source_root}")

    for source in sources:
        with tempfile.TemporaryDirectory(
            prefix=f"spo-godoc-{source.stem}-"
        ) as temporary:
            unit = Path(temporary)
            shutil.copy2(source, unit / source.name)
            for module_file in ("go.mod", "go.sum"):
                candidate = source_root / module_file
                if candidate.exists():
                    shutil.copy2(candidate, unit / module_file)
            result = _run(("go", "doc", "-all", "-cmd", "."), cwd=unit)
        rendered = result.stdout.strip()
        if not rendered:
            raise RuntimeError(f"go doc produced no output for {source}")
        (destination / f"{source.stem}.txt").write_text(
            f"{rendered}\n", encoding="utf-8"
        )


def _generate_julia(repo_root: Path, output_root: Path) -> None:
    """Render the independent Julia modules through Julia reflection."""
    destination = output_root / "julia" / "api.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    script = repo_root / "tools" / "generate_julia_docs.jl"
    julia = shutil.which("julia")
    if julia:
        _run((julia, "--startup-file=no", str(script), str(destination)), cwd=repo_root)
        return

    # The Julia CI lane obtains its runtime through juliacall, which intentionally
    # does not expose a global ``julia`` executable.  Execute the same Julia source
    # through that already-resolved runtime rather than downloading another one.
    previous_output = os.environ.get("SPO_JULIA_DOC_OUTPUT")
    os.environ["SPO_JULIA_DOC_OUTPUT"] = str(destination)
    try:
        juliacall = importlib.import_module("juliacall")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Julia documentation requires julia or juliacall") from exc
    try:
        juliacall.Main.include(str(script))
    finally:
        if previous_output is None:
            os.environ.pop("SPO_JULIA_DOC_OUTPUT", None)
        else:
            os.environ["SPO_JULIA_DOC_OUTPUT"] = previous_output


def _generate_mojo(repo_root: Path, output_root: Path) -> None:
    """Compile every Mojo executable into the toolchain's JSON format."""
    source_root = repo_root / "mojo"
    destination = output_root / "mojo"
    destination.mkdir(parents=True, exist_ok=True)
    sources = sorted(source_root.glob("*.mojo"))
    if not sources:
        raise RuntimeError(f"no Mojo documentation sources found under {source_root}")
    adjacent_mojo = Path(sys.executable).with_name("mojo")
    mojo = str(adjacent_mojo) if adjacent_mojo.is_file() else "mojo"
    for source in sources:
        _run(
            (mojo, "doc", str(source), "-o", str(destination / f"{source.stem}.json")),
            cwd=repo_root,
        )


GENERATORS = {
    "rust": _generate_rust,
    "go": _generate_go,
    "julia": _generate_julia,
    "mojo": _generate_mojo,
}


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=(*LANGUAGES, "all"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/polyglot-docs"),
        help="artifact root (default: build/polyglot-docs)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the requested native documentation artifact set."""
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_root = args.output
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    selected = LANGUAGES if args.language == "all" else (args.language,)
    for language in selected:
        GENERATORS[language](repo_root, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
