#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — active-environment spo-kernel installer

"""Install or verify ``spo_kernel`` in the selected Python environment.

``maturin develop`` installs the built wheel through uv when the target
environment was created by uv, and uv applies the configuration of the project it
is run from. This repository's ``[tool.uv] exclude-dependencies`` lists
``spo-kernel``, so an install run from the checkout is skipped while maturin still
reports it as installed. ``maturin develop`` also installs into the environment
named by ``VIRTUAL_ENV`` or found as a ``.venv`` above the working directory,
not into the interpreter that runs it. The installer therefore runs maturin from
an empty working directory outside any project with ``VIRTUAL_ENV`` set to the
selected interpreter's prefix, and after installing it compares the extension
the environment would import with the library cargo just built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MANIFEST = Path("spo-kernel") / "crates" / "spo-ffi" / "Cargo.toml"

#: File name cargo gives a ``cdylib`` on platforms that differ from ``lib{}.so``.
_LIBRARY_FILENAME = {"win32": "{}.dll", "darwin": "lib{}.dylib"}


class KernelInstallError(RuntimeError):
    """The environment does not import the extension that was just built."""


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


def built_extension(plan: KernelInstallPlan) -> tuple[str, Path]:
    """Return the module name and path of the library cargo builds for ``plan``.

    Parameters
    ----------
    plan : KernelInstallPlan
        Resolved installation plan.

    Returns
    -------
    tuple[str, Path]
        The ``cdylib`` target name, which is also the Python module name, and the
        library file under the cargo target directory for the plan's profile.

    Raises
    ------
    FileNotFoundError
        If ``cargo`` is not on ``PATH``.
    KernelInstallError
        If cargo does not list the manifest's package, or the package declares no
        ``cdylib`` target.
    """
    cargo = shutil.which("cargo")
    if cargo is None:
        raise FileNotFoundError("cargo not found on PATH")
    completed = subprocess.run(
        [
            cargo,
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
            str(plan.manifest),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    metadata = json.loads(completed.stdout)
    package = next(
        (
            package
            for package in metadata["packages"]
            if Path(package["manifest_path"]) == plan.manifest
        ),
        None,
    )
    if package is None:
        raise KernelInstallError(f"cargo metadata does not list {plan.manifest}")
    names = [
        target["name"]
        for target in package["targets"]
        if "cdylib" in target["crate_types"]
    ]
    if not names:
        raise KernelInstallError(f"{plan.manifest} declares no cdylib target")
    name = names[0]
    filename = _LIBRARY_FILENAME.get(sys.platform, "lib{}.so").format(name)
    profile = "release" if plan.release else "debug"
    # An empty CARGO_BUILD_TARGET adds no path component.
    triple = os.environ.get("CARGO_BUILD_TARGET", "")
    return name, Path(metadata["target_directory"], triple, profile, filename)


def installed_extensions(python: Path, module_name: str) -> dict[str, str]:
    """Return the SHA-256 of each extension file ``python`` would load for a module.

    The module is located with ``importlib.util.find_spec`` and not executed, so a
    broken or stale extension is still reported rather than failing to load.

    Parameters
    ----------
    python : Path
        Python interpreter that owns the environment.
    module_name : str
        Top-level module to locate.

    Returns
    -------
    dict[str, str]
        SHA-256 hex digest by extension path; empty when the module is absent or
        ships no extension file.
    """
    snippet = (
        "import hashlib, importlib.machinery, importlib.util, json, pathlib, sys\n"
        "spec = importlib.util.find_spec(sys.argv[1])\n"
        "suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)\n"
        "files = []\n"
        "if spec is not None and spec.origin:\n"
        "    origin = pathlib.Path(spec.origin)\n"
        "    if origin.name.endswith(suffixes):\n"
        "        files = [origin]\n"
        "    elif spec.submodule_search_locations:\n"
        "        files = sorted(\n"
        "            path for path in origin.parent.iterdir()\n"
        "            if path.name.endswith(suffixes)\n"
        "        )\n"
        "print(json.dumps({\n"
        "    str(path): hashlib.sha256(path.read_bytes()).hexdigest()\n"
        "    for path in files\n"
        "}))\n"
    )
    completed = subprocess.run(
        [str(python), "-c", snippet, module_name],
        check=True,
        text=True,
        capture_output=True,
    )
    digests: dict[str, str] = json.loads(completed.stdout)
    return digests


def verify_installed_extension(
    python: Path, module_name: str, library: Path
) -> tuple[str, str]:
    """Confirm ``python`` would load ``library`` as ``module_name``.

    Parameters
    ----------
    python : Path
        Python interpreter that owns the environment.
    module_name : str
        Module the library is installed as.
    library : Path
        The library file cargo built.

    Returns
    -------
    tuple[str, str]
        The installed extension path and its SHA-256 digest.

    Raises
    ------
    KernelInstallError
        If the library is missing, or no extension of the module in the
        environment has the library's bytes.
    """
    if not library.is_file():
        raise KernelInstallError(f"built library not found: {library}")
    expected = hashlib.sha256(library.read_bytes()).hexdigest()
    installed = installed_extensions(python, module_name)
    for path, digest in installed.items():
        if digest == expected:
            return path, digest
    found = ", ".join(sorted(installed)) or "no extension"
    raise KernelInstallError(
        f"{python} does not load the {module_name} that was just built "
        f"(sha256 {expected}); it has {found}"
    )


def environment_prefix(python: Path) -> str:
    """Return ``sys.prefix`` of the environment that owns ``python``.

    Parameters
    ----------
    python : Path
        Python interpreter to query.

    Returns
    -------
    str
        The interpreter's ``sys.prefix``.
    """
    completed = subprocess.run(
        [str(python), "-c", "import sys; print(sys.prefix)"],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def run_install(plan: KernelInstallPlan) -> subprocess.CompletedProcess[bytes]:
    """Run the plan's maturin command against the plan's environment.

    ``maturin develop`` installs into the environment named by ``VIRTUAL_ENV``
    (or ``CONDA_PREFIX``, or a ``.venv`` above the working directory), not into
    the interpreter running it, so ``VIRTUAL_ENV`` is set to the selected
    interpreter's prefix. The command runs from an empty directory outside any
    project, so no project's uv configuration applies to the install.

    Parameters
    ----------
    plan : KernelInstallPlan
        Resolved installation plan.

    Returns
    -------
    subprocess.CompletedProcess[bytes]
        Completed maturin process.
    """
    env = {name: value for name, value in os.environ.items() if name != "CONDA_PREFIX"}
    env["VIRTUAL_ENV"] = environment_prefix(plan.python)
    with tempfile.TemporaryDirectory(prefix="spo-kernel-install-") as workdir:
        return subprocess.run(plan.command(), check=True, cwd=workdir, env=env)


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
            completed = run_install(plan)
            record = {
                **plan.to_record(dry_run=False),
                "check_only": False,
                "returncode": completed.returncode,
            }
            if plan.editable:
                name, library = built_extension(plan)
                path, digest = verify_installed_extension(plan.python, name, library)
                record["extension"] = path
                record["extension_sha256"] = digest
            check = run_check(plan.python, args.verify_module)
            record["module"] = args.verify_module
            record["stdout"] = check.stdout.strip()
    except (
        FileNotFoundError,
        ValueError,
        subprocess.CalledProcessError,
        KernelInstallError,
    ) as exc:
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
