## Summary

<!-- Describe what this PR does and why. -->

## Test plan

- [ ] `pytest tests/ -v` passes
- [ ] `cargo test` passes (if Rust changed)
- [ ] `ruff check src/ tests/` clean
- [ ] `ruff format --check src/ tests/` clean
- [ ] `mypy src/scpn_phase_orchestrator/` clean
- [ ] `cargo clippy --workspace -- -D warnings` clean (if Rust changed)
- [ ] CHANGELOG updated (if user-facing)
- [ ] `python tools/coverage_guard.py` passes (if tests changed)
- [ ] `python tools/check_test_module_linkage.py` passes (if modules added)
- [ ] `bandit -r src/ -c pyproject.toml` clean
- [ ] Benchmarks checked (if performance-sensitive)
- [ ] No anti-slop violations (see CLAUDE.md policy)
