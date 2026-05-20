# Codex session: reference benchmark snapshot refresh

Date: 2026-05-20T06:04:48+02:00
Repo: SCPN-PHASE-ORCHESTRATOR
Branch policy: main only

## Public-roadmap slice
Refreshed the public reference benchmark snapshot after adding formal checker-readiness benchmark gates.

## Files changed
- benchmarks/results/reference_suite.json
- docs/galleries/reference_benchmark_snapshot.md

## Implementation
- Regenerated benchmarks/results/reference_suite.json with the current reference suite.
- Updated docs/galleries/reference_benchmark_snapshot.md historical timing rows from the regenerated JSON.
- Added formal checker-readiness metrics to the public Formal-Export Acceptance Gates section.

## Verification
- PYTHONPATH=src .venv/bin/python benchmarks/reference_suite.py: PASS
- PYTHONPATH=src .venv/bin/python -m pytest tests/test_reference_benchmark_snapshot_docs.py tests/test_reference_benchmark_suite.py -q: PASS, 23 passed
- ruff check --no-cache tests/test_reference_benchmark_snapshot_docs.py tests/test_reference_benchmark_suite.py benchmarks/reference_suite.py: PASS
- git diff --check: PASS

## Push state
No push performed.
