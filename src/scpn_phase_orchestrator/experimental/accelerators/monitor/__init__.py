# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Load-bearing monitor accelerator namespace

"""Private monitor backend bridges used by validated monitor dispatchers.

Monitor modules own the public API and input/output validation. These backend
bridges are load-bearing implementation details for optional Go, Julia, Mojo, or
Rust parity paths and are not standalone user-facing APIs.
"""
