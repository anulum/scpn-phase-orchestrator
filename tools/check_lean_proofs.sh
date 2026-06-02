#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lean proof gate

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEAN_DIR="$ROOT_DIR/formal/lean"

cd "$LEAN_DIR"

if grep -R -n -E '\b(sorry|admit|axiom|unsafe)\b' -- SPOFormal.lean lakefile.lean SPOFormal; then
  printf 'Lean proof gate rejected proof placeholders or unsafe declarations.\n' >&2
  exit 1
fi

lake env lean SPOFormal/Projector.lean
lake env lean SPOFormal/Regime.lean
lake build SPOFormal
lake env lean SPOFormal.lean
