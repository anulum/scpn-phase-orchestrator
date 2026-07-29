# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — tests for tools/generate_grpc.py

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "generate_grpc.py"
PROTO = REPO / "proto" / "spo.proto"

_HAS_GRPC_TOOLS = importlib.util.find_spec("grpc_tools") is not None


class TestGenerateGrpcScript:
    def test_script_file_exists(self) -> None:
        assert SCRIPT.exists(), f"tool script missing: {SCRIPT}"

    def test_script_is_cross_platform(self) -> None:
        """No bash / shell-script artefacts in the replacement tool."""
        text = SCRIPT.read_text(encoding="utf-8")
        assert not text.startswith("#!/usr/bin/env bash"), (
            "Python replacement must not start with a bash shebang"
        )
        assert "set -euo pipefail" not in text
        # Uses pathlib, not hard-coded forward-slash paths.
        assert "from pathlib import Path" in text

    def test_script_has_spdx_header(self) -> None:
        lines = SCRIPT.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "# SPDX-License-Identifier: AGPL-3.0-or-later"
        assert lines[1] == "# Commercial license available"

    @pytest.mark.skipif(not _HAS_GRPC_TOOLS, reason="grpcio-tools not installed")
    def test_protoc_invocation_produces_stubs(self, tmp_path: Path) -> None:
        """The underlying protoc call must regenerate the two expected
        modules (spo_pb2.py, spo_pb2_grpc.py) in a fresh output directory.
        """
        from grpc_tools import protoc

        out = tmp_path / "grpc_gen"
        out.mkdir()
        rc = protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{PROTO.parent}",
                f"--python_out={out}",
                f"--grpc_python_out={out}",
                str(PROTO),
            ]
        )
        assert rc == 0
        assert (out / "spo_pb2.py").exists()
        assert (out / "spo_pb2_grpc.py").exists()

    @pytest.mark.skipif(not _HAS_GRPC_TOOLS, reason="grpcio-tools not installed")
    def test_script_help_flag_exits_zero(self) -> None:
        """`python generate_grpc.py --help` must render argparse usage
        without trying to run protoc.
        """
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "generate_grpc" in result.stdout.lower() or "usage" in (
            result.stdout.lower()
        )

    @pytest.mark.skipif(not _HAS_GRPC_TOOLS, reason="grpcio-tools not installed")
    def test_committed_stubs_match_deterministic_regeneration(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.skipif(not _HAS_GRPC_TOOLS, reason="grpcio-tools not installed")
    def test_script_generates_package_relative_repeatable_stubs(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "grpc_gen"
        generate = subprocess.run(
            [sys.executable, str(SCRIPT), "--output-dir", str(out)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert generate.returncode == 0, generate.stdout + generate.stderr
        assert (out / "spo_pb2.py").is_file()
        grpc_text = (out / "spo_pb2_grpc.py").read_text(encoding="utf-8")
        assert "from . import spo_pb2 as spo__pb2" in grpc_text
        assert "\nimport spo_pb2 as spo__pb2\n" not in grpc_text

        check = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--check",
                "--output-dir",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert check.returncode == 0, check.stdout + check.stderr

        (out / "spo_pb2.py").write_text("# stale\n", encoding="utf-8")
        stale = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--check",
                "--output-dir",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert stale.returncode == 1
        assert "stale generated stub" in stale.stderr

    def test_package_import_normalisation_emits_lf_newlines(
        self, tmp_path: Path
    ) -> None:
        """Rewriting the gRPC stub must not inherit Windows CRLF output."""
        grpc_path = tmp_path / "spo_pb2_grpc.py"
        grpc_path.write_bytes(b"# generated\r\nimport spo_pb2 as spo__pb2\r\n# end\r\n")
        spec = importlib.util.spec_from_file_location("generate_grpc", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        module._normalise_package_import(tmp_path)

        normalised = grpc_path.read_bytes()
        assert b"\r\n" not in normalised
        assert b"from . import spo_pb2 as spo__pb2\n" in normalised


# Pipeline wiring: generate_grpc.py replaces a bash-only script; the
# cross-platform tests above ensure Windows / CI runners can both execute
# the regenerator and verify its inputs without invoking bash.
