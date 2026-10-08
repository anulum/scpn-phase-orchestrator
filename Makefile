# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Build targets

.DEFAULT_GOAL := help

.PHONY: help install install-dev quickstart test test-rust test-all lint fmt bandit sast \
        preflight preflight-fast docs docs-build bench bench-rust bridge bridge-check \
        lock-refresh lock-check \
        build docker-build docker-run clean install-hooks phase-sindy-quality attention-residuals-quality

PYTHON ?= python

PHASE_SINDY_QUALITY_FILES := src/scpn_phase_orchestrator/autotune/sindy.py \
    tests/test_autotune_sindy.py tests/test_sindy.py \
    benchmarks/phase_sindy_benchmark.py native-tests/test_sindy_runtime_profiles.py \
    native-tests/helpers/sindy_cli_probe.py tools/phase_sindy_coverage.py \
    tests/test_phase_sindy_coverage.py

ATTENTION_RESIDUALS_QUALITY_FILES := src/scpn_phase_orchestrator/coupling/attention_residuals.py \
    src/scpn_phase_orchestrator/coupling/_attnres_validation.py \
    src/scpn_phase_orchestrator/experimental/accelerators/coupling/_attnres_go.py \
    src/scpn_phase_orchestrator/experimental/accelerators/coupling/_attnres_julia.py \
    src/scpn_phase_orchestrator/experimental/accelerators/coupling/_attnres_mojo.py \
    tests/test_attention_residuals.py tests/test_attention_residuals_backends.py \
    tests/test_attention_residuals_real_runtime.py \
    tests/test_attention_residuals_stability.py \
    tests/test_attnres_modulation_benchmark.py \
    benchmarks/attnres_reference.py benchmarks/attnres_modulation_benchmark.py \
    native-tests/test_attention_residuals_profiles.py

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install with dev dependencies
	pip install -e ".[dev,queuewaves,plot,notebook]"

quickstart:  ## Create dev venv, install extras, run minimal audited demo
	python -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev,queuewaves,plot,notebook]"
	.venv/bin/spo validate domainpacks/minimal_domain/binding_spec.yaml
	.venv/bin/spo run domainpacks/minimal_domain/binding_spec.yaml --steps 50 --audit /tmp/spo-quickstart-audit.jsonl
	.venv/bin/spo report /tmp/spo-quickstart-audit.jsonl

test:  ## Run Python tests with coverage
	pytest tests/ -v --tb=short --cov=scpn_phase_orchestrator --cov-report=term-missing

test-rust:  ## Run Rust engine tests
	cd spo-kernel && cargo test --workspace --exclude spo-ffi

test-all: test test-rust  ## Run Python + Rust tests

lint:  ## Check code style
	ruff check src/ tests/
	ruff format --check src/ tests/

phase-sindy-quality:  ## Check Phase-SINDy source, tests and runtime diagnostics
	$(PYTHON) -m ruff check --isolated --select E,F,W,I,UP,B,SIM,N,C4,RET,PTH,D --ignore N803,N806,D105 --config 'lint.pydocstyle.convention="numpy"' $(PHASE_SINDY_QUALITY_FILES)
	$(PYTHON) -m ruff format --check $(PHASE_SINDY_QUALITY_FILES)
	$(PYTHON) -m mypy --config-file tools/phase_sindy_mypy.ini $(PHASE_SINDY_QUALITY_FILES)
	$(PYTHON) tools/phase_sindy_coverage.py --python-report tests/fixtures/phase_sindy_coverage/python.json --native-report tests/fixtures/phase_sindy_coverage/native.json

attention-residuals-quality:  ## Check phase attention source, tests and diagnostics
	$(PYTHON) -m ruff check --isolated --select E,F,W,I,UP,B,SIM,N,C4,RET,PTH,D --ignore N803,N806,D105 --config 'lint.pydocstyle.convention="numpy"' $(ATTENTION_RESIDUALS_QUALITY_FILES)
	$(PYTHON) -m ruff format --check $(ATTENTION_RESIDUALS_QUALITY_FILES)
	$(PYTHON) -m mypy --config-file tools/attention_residuals_mypy.ini $(ATTENTION_RESIDUALS_QUALITY_FILES)

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

# lock-refresh regenerates every generated hash-pinned lockfile under
# requirements/. tools/refresh_dependency_locks.py runs each pip-compile lock
# through `uvx` with the pinned pip-tools release and the lock's own Python
# version (3.11/3.12/3.13), resolves the Windows FFI locks with
# `uv pip compile --python-platform windows` so Unix-only deps (e.g. uvloop) are
# excluded and Windows-only deps (pywin32) are included, and restores the licence
# block that precedes each generated header. Requires uv/uvx on PATH.
# ci-tools.txt and docs-tools.txt are hand-maintained pin lists.
lock-refresh:  ## Regenerate every generated hash-pinned lockfile (needs uv/uvx)
	$(PYTHON) tools/refresh_dependency_locks.py

lock-check:  ## Verify lockfiles are hash-pinned and installable
	pip install --require-hashes --no-deps -r requirements/dev-lock.txt
	pip install --require-hashes --no-deps -r requirements/runtime-lock.txt
	pip install --require-hashes --no-deps -r requirements/queuewaves-lock.txt

bench:  ## Python benchmarks
	python bench/run_benchmarks.py

bench-rust:  ## Rust Criterion benchmarks
	cd spo-kernel && cargo bench -p spo-engine

bridge:  ## Build Rust FFI into the selected Python env
	$(PYTHON) tools/install_spo_kernel.py --release

bridge-check:  ## Verify real native phase and amplitude calculations
	$(PYTHON) tools/install_spo_kernel.py --check-only

build:  ## Build sdist + wheel
	python -m build

docker-build:  ## Build Docker image
	docker build -t scpn-phase-orchestrator .

docker-run:  ## Serve locally until interrupted; remove the container on exit
	docker run --rm -it --restart=no -p 127.0.0.1:8000:8000 scpn-phase-orchestrator

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install-hooks:  ## Install git hooks
	git config core.hooksPath .githooks
	@echo "Hooks installed from .githooks/"

BASIN_BIFURCATION_QUALITY_FILES := src/scpn_phase_orchestrator/upde/basin_stability.py \
    src/scpn_phase_orchestrator/upde/bifurcation.py \
    src/scpn_phase_orchestrator/upde/_basin_stability_validation.py \
    src/scpn_phase_orchestrator/experimental/accelerators/upde/_basin_stability_go.py \
    src/scpn_phase_orchestrator/experimental/accelerators/upde/_basin_stability_julia.py \
    src/scpn_phase_orchestrator/experimental/accelerators/upde/_basin_stability_mojo.py \
    tests/test_basin_stability.py tests/test_basin_stability_algorithm.py \
    tests/test_basin_stability_backends.py tests/test_basin_stability_stability.py \
    tests/test_prop_basin_stability.py tests/test_bifurcation.py \
    tests/test_bifurcation_dispatch.py tests/test_basin_bifurcation_real_runtime.py \
    tests/test_basin_stability_benchmark.py \
    native-tests/test_basin_bifurcation_profiles.py \
    benchmarks/basin_stability_benchmark.py benchmarks/kuramoto_trial_reference.py

.PHONY: basin-bifurcation-quality
basin-bifurcation-quality:  ## Check whole basin and coupling-sweep source/tests/diagnostics
	$(PYTHON) -m ruff check --isolated --select E,F,W,I,UP,B,SIM,N,C4,RET,PTH,D --ignore N802,N803,N806,D105 --config 'lint.pydocstyle.convention="numpy"' $(BASIN_BIFURCATION_QUALITY_FILES)
	$(PYTHON) -m ruff format --check $(BASIN_BIFURCATION_QUALITY_FILES)
	$(PYTHON) -m mypy --config-file tools/basin_bifurcation_mypy.ini $(BASIN_BIFURCATION_QUALITY_FILES)

CHIMERA_QUALITY_FILES := src/scpn_phase_orchestrator/monitor/chimera.py \
    src/scpn_phase_orchestrator/monitor/_chimera_validation.py \
    src/scpn_phase_orchestrator/experimental/accelerators/monitor/_chimera_go.py \
    src/scpn_phase_orchestrator/experimental/accelerators/monitor/_chimera_julia.py \
    src/scpn_phase_orchestrator/experimental/accelerators/monitor/_chimera_mojo.py \
    tests/test_chimera.py \
    tests/test_chimera_algorithm.py \
    tests/test_chimera_backends.py \
    tests/test_chimera_dispatch_contracts.py \
    tests/test_chimera_measurement_inputs.py \
    tests/test_chimera_stability.py \
    tests/test_prop_chimera_winding.py \
    native-tests/test_chimera_boundary.py \
    benchmarks/chimera_benchmark.py \
    tests/test_chimera_real_runtime.py \
    tests/test_chimera_benchmark.py \
    native-tests/test_chimera_runtime_profiles.py \
    benchmarks/chimera_local_order_reference.py \
    src/scpn_phase_orchestrator/nn/chimera.py \
    tests/test_nn_chimera.py \
    examples/neuroscience_eeg.py \
    examples/eeg_file_ingestion.py \
    benchmarks/chimera_comparison.py

.PHONY: chimera-quality
chimera-quality:  ## Check complete chimera owners, tests, benchmarks and consumers
	$(PYTHON) -m ruff check --no-cache --isolated --select E,F,W,I,UP,B,SIM,N,C4,RET,PTH,D --ignore N802,N803,N806,D105 --config 'lint.pydocstyle.convention="numpy"' $(CHIMERA_QUALITY_FILES)
	$(PYTHON) -m ruff format --check $(CHIMERA_QUALITY_FILES)
	$(PYTHON) -m mypy --cache-dir=/dev/null --config-file tools/chimera_mypy.ini $(CHIMERA_QUALITY_FILES)

ETHICAL_COST_QUALITY_FILES := src/scpn_phase_orchestrator/ssgf/ethical.py \
    tests/test_closure_ethical.py tests/test_ssgf_ethical_cost_inputs.py \
    tests/test_ssgf_modules.py tests/test_ethical_cost_real_runtime.py \
    tests/test_ethical_cost_benchmark.py native-tests/test_ethical_cost_runtime_profiles.py \
    benchmarks/ethical_cost_benchmark.py benchmarks/ethical_cost_reference.py

.PHONY: ethical-cost-quality
ethical-cost-quality:  ## Check ethical-cost owners, original consumers and measured comparisons
	$(PYTHON) -m ruff check --no-cache --isolated --select E,F,W,I,UP,B,SIM,N,C4,RET,PTH,D --ignore N802,N803,N806,D105 --config 'lint.pydocstyle.convention="numpy"' $(ETHICAL_COST_QUALITY_FILES)
	$(PYTHON) -m ruff format --no-cache --check $(ETHICAL_COST_QUALITY_FILES)
	$(PYTHON) -m mypy --cache-dir=/dev/null --config-file tools/ethical_cost_mypy.ini $(ETHICAL_COST_QUALITY_FILES)

CONNECTOME_QUALITY_FILES := src/scpn_phase_orchestrator/coupling/connectome.py \
    src/scpn_phase_orchestrator/coupling/_connectome_validation.py \
    tests/test_connectome.py tests/test_connectome_python_fallback.py \
    tests/test_connectome_validation_guards.py tests/test_connectome_measurement_inputs.py \
    tests/test_connectome_real_runtime.py tests/test_connectome_benchmark.py \
    native-tests/test_connectome_runtime_profiles.py tests/test_ci_workflow_modularity.py \
    benchmarks/connectome_benchmark.py benchmarks/connectome_reference.py

.PHONY: connectome-quality
connectome-quality:  ## Check connectome owners, original consumers and installed comparisons
	$(PYTHON) -m ruff check --no-cache $(CONNECTOME_QUALITY_FILES)
	$(PYTHON) -m ruff check --no-cache --isolated --select D --config 'lint.pydocstyle.convention="numpy"' $(CONNECTOME_QUALITY_FILES)
	$(PYTHON) -m ruff format --no-cache --check $(CONNECTOME_QUALITY_FILES)
	$(PYTHON) -m mypy --strict --cache-dir=/dev/null --config-file tools/connectome_mypy.ini $(CONNECTOME_QUALITY_FILES)
