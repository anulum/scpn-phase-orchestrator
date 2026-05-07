<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Financial Markets Domainpack -->

# Financial Markets Domainpack

Represents cross-asset synchronisation as a phase-dynamics problem.
Equities, bonds, commodities, and foreign-exchange layers expose phase
coherence that can be used for regime analysis and contagion-risk studies.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|-------------|---------|---------|
| equities | 2 | P | Equity index phase |
| bonds | 2 | P | Sovereign bond phase |
| commodities | 2 | P | Gold and oil phase |
| forex | 2 | P | Currency-pair phase |

## Boundaries

- `correlation_ceiling`: high cross-asset synchronisation guard.
- `volatility_floor`: low implied-volatility sanity boundary.

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time
contagion-control checks. It bounds cross-asset coupling and equity rebalance
lag steps, then falls back to a zero-lag safe hold when a candidate action
exceeds those priors.

This template is for simulation, replay, and policy review. It is not a
trading or investment system.

## Run

```bash
spo validate domainpacks/financial_markets/binding_spec.yaml
spo run domainpacks/financial_markets/binding_spec.yaml --steps 100
python domainpacks/financial_markets/run.py
```

## Read Next

- [UPDE Market API](../../docs/reference/api/upde_market.md)
- [Market Regime Example](../../examples/market_regime_detection.py)
- [Advanced Dynamics](../../docs/guide/advanced_dynamics.md)
