# Handover — Publish Security Scanning

## Completed
- Added Grype scan step to `.github/workflows/publish.yml` for container image security gating.
- Updated `docs/guide/production.md` to document Trivy + Grype publish gating.
- Updated `ROADMAP.md` to mark Docker security scanning item as done.

## Validation
- `rg -n "trivy|grype" .github/workflows/publish.yml` confirms both scanners now run in build container job.
- `rg -n "Scan" docs/guide/production.md` confirms publish pipeline docs describe both scanners.

## Handoff Notes
- Grype container image used: `anchore/grype:v0.82.0` with `--fail-on high` gate.
- Trivy remains `exit-code: 1` with `severity: CRITICAL,HIGH`.
