# Codex handover — dimension exact backend contracts

Timestamp: 2026-05-31T135310
Project: SCPN-PHASE-ORCHESTRATOR

## Completed slice

The U1 monitor-dimension backend boundary now checks exact NumPy reference preservation for direct Go/Julia/Mojo correlation-integral and Kaplan-Yorke outputs, and the public dispatcher falls back when accelerated backends diverge from the reference.

## Files touched

- `src/scpn_phase_orchestrator/monitor/dimension.py`
- `src/scpn_phase_orchestrator/experimental/accelerators/monitor/_dimension_validation.py`
- `src/scpn_phase_orchestrator/experimental/accelerators/monitor/_dimension_go.py`
- `src/scpn_phase_orchestrator/experimental/accelerators/monitor/_dimension_julia.py`
- `src/scpn_phase_orchestrator/experimental/accelerators/monitor/_dimension_mojo.py`
- `tests/test_dimension_backends.py`
- `docs/reference/api/monitor_dimension.md`
- `benchmarks/dimension_benchmark.py`
- `docs/roadmap.md`

## Notes for next cycle

Continue the residual U1 public-boundary validation sweep with another coherent physics/math surface, likely `chimera`, `embedding`, `psychedelic`, or PID/integrated-information if still open in the roadmap evidence row. Do not create bucket tests.
