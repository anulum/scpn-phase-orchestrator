# Codex session: formal checker readiness CLI

Date: 2026-05-20
Repo: SCPN-PHASE-ORCHESTRATOR
Branch policy: main only

## Public-roadmap slice
Exposed non-executing external-checker readiness evidence through the formal package CLI path.

## Files changed
- src/scpn_phase_orchestrator/runtime/cli.py
- tests/test_cli.py
- docs/reference/api/supervisor.md
- ROADMAP.md
- docs/roadmap.md

## Implementation
- Added spo formal-export --export package --include-checker-readiness.
- Added deterministic --checker-path executable=/path and --checker-path executable= overrides for CI readiness evidence.
- Kept default package JSON stable unless readiness is explicitly requested.
- Rejected readiness flags outside package export and rejected malformed resolver overrides.
- Documented the new operator surface on API and roadmap pages.

## Verification
- ruff check --no-cache src/scpn_phase_orchestrator/runtime/cli.py tests/test_cli.py: PASS
- PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli.py -q -k 'formal_export': PASS, 8 passed
- git diff --check: PASS

## Push state
No push performed.
