# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Generate Python gRPC stubs from proto

"""Cross-platform replacement for tools/generate_grpc.sh.

Runs grpc_tools.protoc against proto/spo.proto and writes the generated
modules into src/scpn_phase_orchestrator/grpc_gen/. Works identically on
Linux, macOS and Windows — the shell script previously excluded Windows
developers who did not have WSL or Git Bash installed.

Note: the committed stubs in grpc_gen/ are post-processed by ruff (quote
style, line wrapping) and patched to use relative imports. After running
this script, run ``ruff format src/scpn_phase_orchestrator/grpc_gen/``
and convert any ``import spo_pb2`` to ``from . import spo_pb2`` before
committing.

Usage::

    python tools/generate_grpc.py

Exit codes:
    0  success
    1  grpc_tools not installed or protoc invocation failed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO / "proto"
OUT_DIR = REPO / "src" / "scpn_phase_orchestrator" / "grpc_gen"
PROTO_FILE = PROTO_DIR / "spo.proto"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="generate_grpc",
        description=(
            "Regenerate Python gRPC stubs from proto/spo.proto. "
            "Cross-platform replacement for generate_grpc.sh."
        ),
    )
    parser.parse_args()

    if not PROTO_FILE.exists():
        print(f"Proto file not found: {PROTO_FILE}", file=sys.stderr)
        return 1

    try:
        from grpc_tools import protoc  # type: ignore[import-untyped]
    except ImportError:
        print(
            "grpc_tools not installed. Install with: pip install grpcio-tools",
            file=sys.stderr,
        )
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    args = [
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        str(PROTO_FILE),
    ]
    rc = int(protoc.main(args))
    if rc != 0:
        print(f"protoc failed with exit code {rc}", file=sys.stderr)
        return rc

    print(f"Generated stubs in {OUT_DIR}")
    print(
        "Reminder: run `ruff format` over the generated files and convert "
        "`import spo_pb2` to `from . import spo_pb2` before committing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
