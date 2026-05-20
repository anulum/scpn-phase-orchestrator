# Codex session: semantic retrieval benchmark snapshot refresh

Date: 2026-05-20
Repo: SCPN-PHASE-ORCHESTRATOR
Branch policy: main only

## Public-roadmap slice
Refreshed the public reference benchmark snapshot after adding semantic retrieval ranking gates.

## Files changed
- benchmarks/results/reference_suite.json
- docs/galleries/reference_benchmark_snapshot.md

## Implementation
- Regenerated benchmarks/results/reference_suite.json with the current reference suite.
- Added semantic_retrieval_ranking_quality to the public historical results table.
- Added a dedicated Semantic Retrieval Ranking Acceptance Gates section.
- Synchronized timing rows and summary values with the regenerated JSON artefact.

## Verification
- PYTHONPATH=src .venv/bin/python benchmarks/reference_suite.py: PASS
- PYTHONPATH=src .venv/bin/python -m pytest tests/test_reference_benchmark_snapshot_docs.py tests/test_reference_benchmark_suite.py -q: PASS, 27 passed
- ruff check --no-cache tests/test_reference_benchmark_snapshot_docs.py tests/test_reference_benchmark_suite.py benchmarks/reference_suite.py: PASS
- git diff --check: PASS

## Push state
No push performed.
