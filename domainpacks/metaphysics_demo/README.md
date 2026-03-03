# metaphysics_demo domainpack

Demonstrates all three oscillator channels (P/I/S), imprint modulation,
geometry projection, and policy-driven Ψ control in a single run.

## Features Exercised

- **P/I/S oscillators**: 3 physical, 2 informational, 2 symbolic.
- **Imprint model**: decay_rate=0.01, saturation=2.0, modulates K and alpha.
- **Geometry prior**: symmetric + non-negative projection on K_nm.
- **Policy rules**: boost coupling, shift target phase, apply damping.
- **Boundary observer**: R floor + per-layer R_2 ceiling.

## Ablation

`run.py` executes the simulation twice (same seed, same geometry):

1. **Imprint ON** — ImprintModel active, modulating coupling each step.
2. **Imprint OFF** — no imprint modulation.

The delta between final R_good values quantifies imprint's contribution.

## Usage

```bash
python domainpacks/metaphysics_demo/run.py
```

With `matplotlib` installed (`pip install -e ".[plot]"`), a comparison plot
is saved to `domainpacks/metaphysics_demo/ablation.png`.
