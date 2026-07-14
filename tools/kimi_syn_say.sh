#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Kimi outbound Synapse helper
#
# Lightweight wrapper around syn-say so the kimi harness can send outbound
# peer messages on the SCPN-PHASE-ORCHESTRATOR channel.
#
# Usage:
#   tools/kimi_syn_say.sh <target> <message>
# Examples:
#   tools/kimi_syn_say.sh all "Status update..."
#   tools/kimi_syn_say.sh SCPN-STUDIO "Handover to Claude..."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="${1:-}"
shift || true
MESSAGE="$*"

if [[ -z "${TARGET}" || -z "${MESSAGE}" ]]; then
    echo "Usage: ${0} <target> <message>" >&2
    exit 1
fi

# syn-say uses the machine identity and --as-project to speak as the project.
exec syn-say --as-project "${TARGET}" "${MESSAGE}"
