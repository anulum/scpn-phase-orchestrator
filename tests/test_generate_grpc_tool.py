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

    @pytest.mark.skipif(
        not _HAS_GRPC_TOOLS, reason="grpcio-tools not installed"
    )
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

    @pytest.mark.skipif(
        not _HAS_GRPC_TOOLS, reason="grpcio-tools not installed"
    )
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


# Pipeline wiring: generate_grpc.py replaces a bash-only script; the
# cross-platform tests above ensure Windows / CI runners can both execute
# the regenerator and verify its inputs without invoking bash.
