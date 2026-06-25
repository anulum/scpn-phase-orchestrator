# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Load-bearing accelerator namespace

"""Compatibility namespace for load-bearing polyglot accelerator implementations.

The package name is historical. Production subsystems import selected backend
bridges from this namespace through their own dispatchers; callers should use
the owning ``coupling``, ``monitor``, or ``upde`` API rather than importing
accelerator modules directly.
"""
