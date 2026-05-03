# Codex session: publish workflow security scanning

Date: 2026-05-03
Project: SCPN-PHASE-ORCHESTRATOR

## Scope

- Continue roadmap completion work by adding container image security scanning coverage.
- Complete the Docker multi-stage build + security scans checklist item.

## Changes

- Added a Grype scan step in publish CI:
  - `.github/workflows/publish.yml`
  - `Scan image with Grype` step now runs after Trivy and fails the job on HIGH/CRITICAL vulnerabilities.
- Updated production guide to mention both Trivy and Grype gates in publish pipeline:
  - `docs/guide/production.md`
- Marked roadmap item complete:
  - `Docker multi-stage build with security scanning (Trivy/Grype)`

## Validation

- `rg` confirms workflow now contains both Trivy and Grype image-scan stages.
- `sed` confirms production deployment guide states the scan gate uses both tools.

## Commit

- `docs: add Grype scan in publish workflow` (this session’s commit)
