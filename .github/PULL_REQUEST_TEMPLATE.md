## Summary

<!-- Describe what this PR does and why. -->

## Test plan

- [ ] `pytest tests/ -v` passes
- [ ] `cargo test` passes (if Rust changed)
- [ ] `ruff check src/ tests/` clean
- [ ] `ruff format --check src/ tests/` clean
- [ ] `mypy src/scpn_phase_orchestrator/ --ignore-missing-imports` clean
- [ ] `cargo clippy --workspace -- -D warnings` clean (if Rust changed)
- [ ] CHANGELOG updated (if user-facing)
- [ ] No anti-slop violations (see CLAUDE.md policy)
