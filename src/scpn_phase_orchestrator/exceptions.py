# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

"""Exception hierarchy for scpn-phase-orchestrator."""

from __future__ import annotations

__all__ = [
    "SPOError",
    "BindingError",
    "ValidationError",
    "ExtractorError",
    "EngineError",
    "PolicyError",
    "AuditError",
]


class SPOError(Exception):
    """Base exception for all scpn-phase-orchestrator errors."""


class BindingError(SPOError, ValueError):
    """Failed to load or parse a binding spec."""


class ValidationError(SPOError, ValueError):
    """Binding spec failed validation."""


class ExtractorError(SPOError, RuntimeError):
    """Phase extractor encountered an unrecoverable signal condition."""


class EngineError(SPOError, RuntimeError):
    """UPDE or Stuart-Landau integrator diverged or hit a numerical fault."""


class PolicyError(SPOError, ValueError):
    """Malformed policy rule or unresolvable action."""


class AuditError(SPOError, RuntimeError):
    """Audit log integrity violation (hash chain mismatch, missing header)."""
