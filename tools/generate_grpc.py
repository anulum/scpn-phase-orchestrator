# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Generate Python gRPC stubs from proto

"""Cross-platform replacement for tools/generate_grpc.sh.

Runs grpc_tools.protoc against proto/spo.proto, normalises the generated sibling
import for package use, and publishes the two generated modules only after the
complete temporary output exists. Works identically on Linux, macOS, and Windows.
``--check`` regenerates into a temporary directory and fails if the committed
files differ, so generated sources never require hand editing.

Usage::

    python tools/generate_grpc.py
    python tools/generate_grpc.py --check

Exit codes:
    0  success
    1  grpc_tools not installed or protoc invocation failed
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO / "proto"
OUT_DIR = REPO / "src" / "scpn_phase_orchestrator" / "runtime" / "grpc_gen"
PROTO_FILE = PROTO_DIR / "spo.proto"
EXPECTED_STUBS = ("spo_pb2.py", "spo_pb2_grpc.py")
_ABSOLUTE_SIBLING_IMPORT = "import spo_pb2 as spo__pb2"
_RELATIVE_SIBLING_IMPORT = "from . import spo_pb2 as spo__pb2"


def _generate_into(output_dir: Path) -> int:
    """Run protoc into ``output_dir`` and return its process-style status."""
    from grpc_tools import protoc

    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        str(PROTO_FILE),
    ]
    return int(protoc.main(args))


def _normalise_package_import(output_dir: Path) -> None:
    """Convert the sibling import and emit canonical LF-only source text."""
    grpc_path = output_dir / "spo_pb2_grpc.py"
    text = grpc_path.read_text(encoding="utf-8")
    occurrences = text.count(_ABSOLUTE_SIBLING_IMPORT)
    if occurrences != 1:
        msg = (
            "expected one generated spo_pb2 sibling import, "
            f"found {occurrences} in {grpc_path}"
        )
        raise RuntimeError(msg)
    grpc_path.write_text(
        text.replace(_ABSOLUTE_SIBLING_IMPORT, _RELATIVE_SIBLING_IMPORT),
        encoding="utf-8",
        newline="\n",
    )


def _stale_outputs(generated_dir: Path, output_dir: Path) -> list[Path]:
    """Return committed outputs that differ from deterministic regeneration."""
    stale: list[Path] = []
    for filename in EXPECTED_STUBS:
        generated = generated_dir / filename
        committed = output_dir / filename
        if not committed.exists() or committed.read_bytes() != generated.read_bytes():
            stale.append(committed)
    return stale


def _publish(generated_dir: Path, output_dir: Path) -> None:
    """Copy the complete generated file set into the requested output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in EXPECTED_STUBS:
        shutil.copyfile(generated_dir / filename, output_dir / filename)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_grpc",
        description=(
            "Regenerate Python gRPC stubs from proto/spo.proto. "
            "Cross-platform replacement for generate_grpc.sh."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when committed stubs differ from deterministic regeneration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="generated-stub destination or comparison directory",
    )
    options = parser.parse_args(argv)

    if not PROTO_FILE.exists():
        print(f"Proto file not found: {PROTO_FILE}", file=sys.stderr)
        return 1

    try:
        import grpc_tools  # noqa: F401
    except ImportError:
        print(
            "grpc_tools not installed. Install with: pip install grpcio-tools",
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory(prefix="spo-grpc-regen-") as temp_dir:
        generated_dir = Path(temp_dir)
        rc = _generate_into(generated_dir)
        if rc != 0:
            print(f"protoc failed with exit code {rc}", file=sys.stderr)
            return rc

        missing = [
            generated_dir / filename
            for filename in EXPECTED_STUBS
            if not (generated_dir / filename).is_file()
        ]
        if missing:
            print(f"protoc omitted expected stubs: {missing}", file=sys.stderr)
            return 1

        _normalise_package_import(generated_dir)
        if options.check:
            stale = _stale_outputs(generated_dir, options.output_dir)
            if stale:
                for path in stale:
                    print(f"stale generated stub: {path}", file=sys.stderr)
                return 1
            print(f"Generated stubs are current in {options.output_dir}")
            return 0

        _publish(generated_dir, options.output_dir)

    print(f"Generated stubs in {options.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
