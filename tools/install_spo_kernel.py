#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — active-environment spo-kernel installer

"""Install or verify ``spo_kernel`` in the selected Python environment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MANIFEST = Path("spo-kernel") / "crates" / "spo-ffi" / "Cargo.toml"


def absolute_without_dereference(path: Path) -> Path:
    """Return an absolute path without following interpreter symlinks.

    Parameters
    ----------
    path : Path
        User-provided path.

    Returns
    -------
    Path
        Absolute path that preserves virtual-environment interpreter symlinks.
    """
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


@dataclass(frozen=True, slots=True)
class KernelInstallPlan:
    """Resolved ``spo_kernel`` installation command.

    Attributes
    ----------
    python : Path
        Python interpreter whose environment receives the built extension.
    manifest : Path
        Cargo manifest for the PyO3/maturin FFI crate.
    release : bool
        Whether to build with ``maturin --release``.
    editable : bool
        Whether to run ``maturin develop`` instead of building a wheel only.
    """

    python: Path
    manifest: Path
    release: bool
    editable: bool

    def command(self) -> list[str]:
        """Return the maturin command for this plan.

        Returns
        -------
        list[str]
            Argument vector safe to pass to ``subprocess.run``.
        """
        subcommand = "develop" if self.editable else "build"
        command = [str(self.python), "-m", "maturin", subcommand]
        if self.release:
            command.append("--release")
        command.extend(["-m", str(self.manifest)])
        return command

    def to_record(self, *, dry_run: bool) -> dict[str, object]:
        """Return a JSON-safe plan record.

        Parameters
        ----------
        dry_run : bool
            Whether the command will be skipped.

        Returns
        -------
        dict[str, object]
            JSON-safe install-plan record.
        """
        return {
            "python": str(self.python),
            "manifest": str(self.manifest),
            "release": self.release,
            "editable": self.editable,
            "dry_run": dry_run,
            "command": self.command(),
        }


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the active-environment installer.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build/install spo_kernel into the selected Python environment using "
            "that interpreter's `python -m maturin`."
        )
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter that owns the target environment.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to spo-ffi Cargo.toml.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Build a release extension. This is the default.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build a debug extension instead of the default release extension.",
    )
    parser.add_argument(
        "--wheel-only",
        action="store_true",
        help="Run `maturin build` without installing into the interpreter.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify that the target interpreter can import the module.",
    )
    parser.add_argument(
        "--verify-module",
        default="spo_kernel",
        help="Module imported by --check-only after installation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan without running maturin.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON plan/status record.",
    )
    return parser


def resolve_plan(args: argparse.Namespace) -> KernelInstallPlan:
    """Resolve and validate an installation plan.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    KernelInstallPlan
        Validated plan.

    Raises
    ------
    FileNotFoundError
        If the selected interpreter or manifest path does not exist.
    """
    python = absolute_without_dereference(args.python)
    manifest = args.manifest.resolve()
    if not python.exists():
        raise FileNotFoundError(f"python interpreter not found: {python}")
    if not manifest.exists():
        raise FileNotFoundError(f"spo-ffi manifest not found: {manifest}")
    return KernelInstallPlan(
        python=python,
        manifest=manifest,
        release=not args.debug,
        editable=not args.wheel_only,
    )


def run_check(python: Path, module_name: str) -> subprocess.CompletedProcess[str]:
    """Verify that ``module_name`` imports in ``python``.

    Parameters
    ----------
    python : Path
        Python interpreter to execute.
    module_name : str
        Module name to import.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed import check process.
    """
    if not module_name.strip() or not all(
        part.isidentifier() for part in module_name.split(".")
    ):
        raise ValueError(f"invalid module name for import check: {module_name!r}")
    snippet = (
        "import importlib; "
        f"module = importlib.import_module({module_name!r}); "
        "print(module.__name__)"
    )
    return subprocess.run(
        [str(python), "-c", snippet],
        check=True,
        text=True,
        capture_output=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the installer CLI.

    Parameters
    ----------
    argv : list[str] | None
        Optional argument vector excluding the program name.

    Returns
    -------
    int
        Process exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.release and args.debug:
            raise ValueError("--release and --debug are mutually exclusive")
        plan = resolve_plan(args)
        if args.check_only:
            check = run_check(plan.python, args.verify_module)
            record = {
                **plan.to_record(dry_run=True),
                "check_only": True,
                "module": args.verify_module,
                "stdout": check.stdout.strip(),
            }
        elif args.dry_run:
            record = {**plan.to_record(dry_run=True), "check_only": False}
        else:
            completed = subprocess.run(plan.command(), check=True)
            check = run_check(plan.python, args.verify_module)
            record = {
                **plan.to_record(dry_run=False),
                "check_only": False,
                "returncode": completed.returncode,
                "module": args.verify_module,
                "stdout": check.stdout.strip(),
            }
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output = {"ok": True, **record}
    if args.json:
        print(json.dumps(output, sort_keys=True))
    else:
        print(" ".join(str(part) for part in record["command"]))
        if "stdout" in record:
            print(record["stdout"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
