# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI entry point

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

from scpn_phase_orchestrator.runtime.cli import assurance as assurance
from scpn_phase_orchestrator.runtime.cli import audit as audit
from scpn_phase_orchestrator.runtime.cli import binding as binding
from scpn_phase_orchestrator.runtime.cli import diagnostics as diagnostics
from scpn_phase_orchestrator.runtime.cli import digital_twin as digital_twin
from scpn_phase_orchestrator.runtime.cli import evaluation as evaluation
from scpn_phase_orchestrator.runtime.cli import (
    evolutionary_grammar as evolutionary_grammar,
)
from scpn_phase_orchestrator.runtime.cli import (
    federated_dp_noise_service as federated_dp_noise_service,
)
from scpn_phase_orchestrator.runtime.cli import (
    federated_secure_aggregation as federated_secure_aggregation,
)
from scpn_phase_orchestrator.runtime.cli import (
    federated_transport as federated_transport,
)
from scpn_phase_orchestrator.runtime.cli import koopman_mpc as koopman_mpc
from scpn_phase_orchestrator.runtime.cli import meta as meta
from scpn_phase_orchestrator.runtime.cli import monitoring as monitoring
from scpn_phase_orchestrator.runtime.cli import plugins as plugins
from scpn_phase_orchestrator.runtime.cli import power_grid as power_grid
from scpn_phase_orchestrator.runtime.cli import provenance as provenance
from scpn_phase_orchestrator.runtime.cli import queuewaves as queuewaves
from scpn_phase_orchestrator.runtime.cli import quickstart as quickstart
from scpn_phase_orchestrator.runtime.cli import run as run
from scpn_phase_orchestrator.runtime.cli import scaffold as scaffold
from scpn_phase_orchestrator.runtime.cli import (
    supervisor_candidate as supervisor_candidate,
)
from scpn_phase_orchestrator.runtime.cli import verification as verification
from scpn_phase_orchestrator.runtime.cli._app import main
from scpn_phase_orchestrator.runtime.cli.meta import (
    meta_transfer_manifest as meta_transfer_manifest,
)
from scpn_phase_orchestrator.runtime.simulation import simulate as simulate

__all__ = ["main", "meta_transfer_manifest", "simulate"]
