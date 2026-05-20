# Codex session: formal checker readiness audit

Date: 2026-05-20T05:59:18+02:00
Repo: SCPN-PHASE-ORCHESTRATOR
Branch policy: main only

## Public-roadmap slice
Advanced formal supervisor verification by adding a deterministic, non-executing external-checker readiness audit for formal verification packages.

## Files changed
- src/scpn_phase_orchestrator/supervisor/formal_export.py
- src/scpn_phase_orchestrator/supervisor/__init__.py
- tests/test_formal_export.py
- ROADMAP.md
- docs/roadmap.md

## Implementation
- Added FormalCheckerAvailability audit records.
- Added audit_formal_checker_availability() with injectable executable-path mapping and default shutil.which lookup.
- Hardened FormalCheckerCommand validation so command manifests reject unsafe execution and malformed command parts.
- Exported the new API through the supervisor facade.
- Documented the readiness-audit milestone on public roadmap surfaces.

## Verification
- ruff check --no-cache src/scpn_phase_orchestrator/supervisor/formal_export.py src/scpn_phase_orchestrator/supervisor/__init__.py tests/test_formal_export.py: PASS
- PYTHONPATH=src .venv/bin/python -m pytest tests/test_formal_export.py tests/test_formal_export_residual.py -q: PASS, 61 passed
- git diff --check: PASS

## Push state
No push performed.
