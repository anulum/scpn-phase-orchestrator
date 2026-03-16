# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Build targets

.DEFAULT_GOAL := help

.PHONY: help install install-dev test test-rust test-all lint fmt bandit sast \
        preflight preflight-fast docs docs-build bench bench-rust bridge \
        build docker-build docker-run clean install-hooks

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install with dev dependencies
	pip install -e ".[dev,queuewaves,plot,notebook]"

test:  ## Run Python tests with coverage
	pytest tests/ -v --tb=short --cov=scpn_phase_orchestrator --cov-report=term-missing

test-rust:  ## Run Rust engine tests
	cd spo-kernel && cargo test --workspace --exclude spo-ffi

test-all: test test-rust  ## Run Python + Rust tests

lint:  ## Check code style
	ruff check src/ tests/
	ruff format --check src/ tests/

fmt:  ## Auto-format Python + Rust
	ruff format src/ tests/
	ruff check --fix src/ tests/
	cd spo-kernel && cargo fmt

bandit:  ## Security static analysis
	bandit -r src/ -c pyproject.toml

sast: bandit  ## Alias for bandit

preflight:  ## Full CI-equivalent gate (10 checks)
	python tools/preflight.py

preflight-fast:  ## Lint-only (~5s)
	python tools/preflight.py --no-tests

docs:  ## Live docs preview
	mkdocs serve

docs-build:  ## Build docs (strict)
	mkdocs build --strict

bench:  ## Python benchmarks
	python bench/run_benchmarks.py

bench-rust:  ## Rust Criterion benchmarks
	cd spo-kernel && cargo bench -p spo-engine

bridge:  ## Build Rust FFI via maturin
	maturin develop --release -m spo-kernel/crates/spo-ffi/Cargo.toml

build:  ## Build sdist + wheel
	python -m build

docker-build:  ## Build Docker image
	docker build -t scpn-phase-orchestrator .

docker-run:  ## Run Docker image
	docker run --rm -it scpn-phase-orchestrator spo info

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install-hooks:  ## Install git hooks
	git config core.hooksPath .githooks
	@echo "Hooks installed from .githooks/"
