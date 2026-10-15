<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Policy YAML fuzzing -->

# Policy YAML fuzzing

`policy_yaml_fuzzer.py` feeds bounded UTF-8 YAML inputs through the canonical
`scpn_phase_orchestrator.supervisor.policy_rules.load_policy_rules` function,
with its actual policy records, action types and imported dependencies.
Atheris instruments the production imports and reports unhandled faults.
Each input is capped at 8192 bytes before UTF-8 decoding; invalid byte
sequences are ignored. The resulting temporary policy file is removed after
either acceptance or refusal.

The pinned ClusterFuzzLite image supplies CPython 3.11 and Atheris. Its build
installs `requirements/runtime-lock-py311.txt` with hash enforcement and checks
the installed dependency graph. Regenerate that lock with the command in
[Contributing](../CONTRIBUTING.md#fuzz-runtime-dependencies).

The image build first checks the canonical loader's actual inputs and results
for valid, malformed, oversized, invalid-UTF-8 and empty data, including file
cleanup. It then executes a bounded smoke campaign against valid domainpack
policy and malformed YAML seeds, first through the Python harness and then through
the frozen executable. ClusterFuzzLite subsequently runs its configured
120-second campaign. Matching pull requests use code-change mode; an explicit
workflow dispatch uses batch mode against the selected revision.

Keep discovered inputs as regression cases in the policy loader's owning
tests. A bounded fuzz campaign supplies additional runtime evidence; coverage
gates use the separately measured package test profiles.
