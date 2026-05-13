<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Scaffold API -->

# Scaffold API

The scaffold API converts operator intent into reviewable domainpack artefacts.
The LLM-guided path is intentionally fail-closed: generated output must be a
strict JSON object, is normalised into deterministic binding YAML, and is then
round-tripped through the production binding loader and validator before files
are written.

Live provider mode requires `SPO_LLM_ENDPOINT` and `SPO_LLM_MODEL`. For
deterministic review, tests, and regulated environments, pass a captured JSON
proposal through `spo scaffold --llm --llm-response-json`.

::: scpn_phase_orchestrator.scaffold.llm
