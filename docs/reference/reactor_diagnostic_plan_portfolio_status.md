# Reactor diagnostic-plan portfolio status

This snapshot records whether exact diagnostic-plan objects from the 20 Reactor
Systems device repositories pass SPO's public review-only intake. Each observed
local head was independently matched to public `origin/main`, and each exact
manifest/fixture pair was passed to
`device_diagnostic_plan_review_from_producer_bytes()` without importing or
executing producer source.

## Result

- **20 producers** were examined at `2026-09-01T23:02:00+02:00`.
- **20 fixtures** are structurally accepted; the partition is **20 accepted / 0 refused**.
- **11 fixtures** have byte-identical SPO custody.
- **9 fixtures** remain digest-pinned public producer objects rather than local custody.
- **0 fixtures** constitute a qualified physical observation or physical phase.
- **0 fixtures** create CONTROL intent, action, execution, actuation, or machine-protection authority.

Structural acceptance means only that a synthetic design declaration obeys the
exact envelope `1.1.0` exchange contract. It does not validate hardware,
calibration, geometry, clocks, observability, measurements, regimes, or
operation. A public-object row pins exact revision, manifest and fixture
digests but deliberately does not claim SPO holds identical fixture bytes.

## Byte-identical SPO fixture custody

SPO holds exact fixtures for `SCPN-BEAM-TARGET-CORE`,
`SCPN-DENSE-PLASMA-FOCUS-CORE`, `SCPN-ICF-BEAM-CORE`,
`SCPN-ICF-IMPACT-CORE`, `SCPN-ICF-LASER-CORE`, `SCPN-MIF-LINER-CORE`,
`SCPN-MIF-MAGLIF-CORE`, `SCPN-MIF-PLASMA-JET-CORE`,
`SCPN-THETA-PINCH-CORE`, `SCPN-TOKAMAK-CORE`, and `SCPN-Z-PINCH-CORE`.
Their `custody_fixture_path` values are hash-checked by the repository test
surface.

## Digest-pinned public producer objects

The following exact public heads passed intake but are not described as local
SPO custody:

| Producer | Exact public head | Fixture SHA-256 |
|---|---|---|
| `SCPN-FRC-CORE` | `d05913d3219cf9c7e2f5ae3b148d3d183530f33f` | `ad4ba517069cf41e64fdbd7e4156a981cb2d6726878ead7a5abe56cb3676da05` |
| `SCPN-FUSION-FISSION-HYBRID-CORE` | `66f3975995334c0e92ece68d221b90d9a081d00c` | `d9b57391c37114f98d73b160a2f18060dd928b1542001e460bf407ab6f6008de` |
| `SCPN-IEC-CORE` | `02f1cdfe947336de5fb5cfebe982b4a426431cf7` | `486b410afbed0aa2e3fcad4d43533b54cd3784d5552a20f9500d6d4eff54000b` |
| `SCPN-LEVITATED-DIPOLE-CORE` | `2b3c687a4062497891a5a8ad800bdc18bd5941cc` | `48778cb069b2835291adce635041d30511447939ef84c44855a137627399f9e2` |
| `SCPN-MAGNETIC-CUSP-CORE` | `725044ae7622afbd081ae30bbb87c4c72bcf3d91` | `b9a1f9a46a4d034b25d2cebd89b97e3729696f7e67910e6549c37057cf479b07` |
| `SCPN-MIRROR-CORE` | `4fe976e4fd3438ce008ce964c44e9c47bd61cc70` | `257cedfee5f7b061ea6b8368e46d1b229bac8dc24f1391d88315462bc4457ba6` |
| `SCPN-RFP-CORE` | `360b9d7d2ac951a9c4e9dd83c102a773d9ff6a6f` | `bea1f321ea800dc1f31d8cfba4c327ad5fd46f8986b1e33684055527c26723a2` |
| `SCPN-SPHEROMAK-CORE` | `46dcd71680a2ac7a63ea26370d0054818d5b6b4d` | `325a7a196c279a6ea3c0954678611d07a583acee4d41cc5e47a81ddcc2d2a8f5` |
| `SCPN-STELLARATOR-CORE` | `8d5c57eb23435a445772837ad0a69f57896a8692` | `789b27619f17678e61b31db7cfa45147661bbd33ef97e6e94af1dc734a5629e9` |

The previously missing `timing_uncertainty_s` member is now explicit in every
producer channel. A separately repaired SPO compatibility-table omission for
`direct_cyclic` facility-clock channels was required before the final 20/20
replay; producer bytes were not changed for that consumer defect.

## Machine-readable status

The [digest-sealed status data](data/reactor_diagnostic_plan_portfolio_status.v1.json)
records all 20 revisions, manifest and fixture hashes, configurations, plans,
custody states, and fail-closed authority fields. Its
[Draft 2020-12 schema](../specs/reactor_diagnostic_plan_portfolio_status.schema.json)
enforces schema version `1.1.0`, the 20/0 structural split, the 11/9 custody
split, and zero physical or control authority.

This register complements the
[reactor configuration evidence coverage](reactor_configuration_evidence_coverage.md):
an accepted design plan can still correctly remain producerless for physical
evidence and `not_declared` for semantic ingress.
