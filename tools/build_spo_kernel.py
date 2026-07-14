#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — local spo-kernel (Rust) build helper

"""Build and install the in-repo Rust acceleration wheel into the active venv.

``pip install scpn-phase-orchestrator[rust]`` cannot work for outsiders because
``spo-kernel`` is deliberately not published to public PyPI (E0.1 / Option C).
This helper makes the local build path a single command:

    python tools/build_spo_kernel.py

It runs ``maturin develop --release`` from the correct crate directory
(``spo-kernel/crates/spo-ffi``) and then imports ``spo_kernel`` to confirm the
FFI surface is alive. If anything is missing it prints actionable diagnostics.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFI_CRATE = ROOT / "spo-kernel" / "crates" / "spo-ffi"


def _check(cmd: str) -> str | None:
    return shutil.which(cmd)


def run() -> int:
    maturin = _check("maturin")
    if maturin is None:
        print("ERROR: maturin is not on PATH.", file=sys.stderr)
        print("Install it, e.g.:  pip install maturin", file=sys.stderr)
        return 1

    if _check("cargo") is None:
        print("ERROR: cargo is not on PATH.", file=sys.stderr)
        print("Install Rust: https://rustup.rs", file=sys.stderr)
        return 1

    if not FFI_CRATE.is_dir():
        print(
            f"ERROR: expected crate directory does not exist: {FFI_CRATE}",
            file=sys.stderr,
        )
        return 1

    print(f"Building spo-kernel from {FFI_CRATE} ...")
    result = subprocess.run(
        [maturin, "develop", "--release"],
        cwd=FFI_CRATE,
        check=False,
    )
    if result.returncode != 0:
        print("ERROR: maturin develop --release failed.", file=sys.stderr)
        return result.returncode

    print("Verifying spo_kernel import ...")
    verify = subprocess.run(
        [sys.executable, "-c", "import spo_kernel; print('spo_kernel import OK')"],
        cwd=ROOT,
        check=False,
    )
    if verify.returncode != 0:
        print(
            "ERROR: built wheel installed but spo_kernel cannot be imported.",
            file=sys.stderr,
        )
        return verify.returncode

    print("spo-kernel local build succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
