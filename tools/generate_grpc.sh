#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Generate Python gRPC stubs from proto

# Thin compatibility wrapper; the cross-platform implementation lives in
# generate_grpc.py so Windows developers can run it without WSL.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "No python interpreter found on PATH" >&2
    exit 1
fi

exec "${PYTHON}" "${REPO_ROOT}/tools/generate_grpc.py" "$@"
