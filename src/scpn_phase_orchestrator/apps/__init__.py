# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Application layer

"""Application-layer entry points built on top of the SPO core.

The apps package groups optional, operator-facing applications such as
QueueWaves. Importing this package performs no network binding, background
tasks, scraping, alerting, or control-plane activation; concrete applications
own their configuration validation, dependency checks, and runtime side effects.
"""
