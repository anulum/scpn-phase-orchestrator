# Reactor diagnostic-plan portfolio status

This snapshot records whether exact diagnostic-plan objects from the 20 Reactor
Systems device repositories pass SPO's public review-only intake. Each observed
local head was independently matched to public `origin/main`, and each exact
manifest/fixture pair was passed to
`device_diagnostic_plan_review_from_producer_bytes()` without importing or
executing producer source.

## Result

- **20 producers** were examined at `2026-09-02T01:11:19+02:00`.
- **20 fixtures** are structurally accepted; the partition is **20 accepted / 0 refused**.
- **0 current fixtures** have byte-identical SPO custody; retained `1.1.0` fixtures remain a compatibility corpus.
- **20 fixtures** are digest-pinned public producer objects rather than current local custody.
- **140/140 hosted workflows** completed successfully at the exact 20 public heads.
- **0 fixtures** constitute a qualified physical observation or physical phase.
- **0 fixtures** create CONTROL intent, action, execution, actuation, or machine-protection authority.

Structural acceptance means only that a synthetic design declaration obeys the
exact envelope `1.2.0` exchange contract. It does not validate hardware,
calibration, geometry, clocks, observability, measurements, regimes, or
operation. A public-object row pins exact revision, manifest and fixture
digests but deliberately does not claim SPO holds identical fixture bytes.

## Current 1.2 public-object evidence

Every current envelope `1.2.0` object is verified at its public `main`
HEAD. SPO retains older `1.1.0` fixtures as compatibility inputs, but those
bytes are not classified as custody of the current object. Each row below
pins the current fixture and the seven successful hosted run IDs
(`CI`, `CodeQL`, `Docs`, `Pre-commit`, `SBOM`, `Scorecard`, and
`Security audit`). Hosted success corroborates repository gates only; it
does not upgrade scientific or operational evidence.

| Producer | Exact public head | Fixture SHA-256 | Hosted run IDs |
|---|---|---|---|
| `SCPN-BEAM-TARGET-CORE` | `30c62db23565c5e57bc5f7df9f28f5d700399384` | `6318fee21c7395945f82f5920b09545982fb7a50c1284a053656d5a2f43ac6e9` | `33568210643, 33568210646, 33568210650, 33568210658, 33568210700, 33568210713, 33568210923` |
| `SCPN-DENSE-PLASMA-FOCUS-CORE` | `45db7b7d294be857719143b150a0ed3a00643a5f` | `2f2b708854f2f507fb5fc0f9a68aeaf928f393e1ef58d256338957e8f8762929` | `33568168490, 33568168494, 33568168501, 33568168512, 33568168535, 33568168629, 33568168894` |
| `SCPN-FRC-CORE` | `be5ecd333712e18ed4021f3b5e27b55a32668e68` | `7b38f13e56d8da075ab1f24b91d1d5fb4a8c478f5fad4a9af8544f0363c13234` | `33568141949, 33568141955, 33568141966, 33568141971, 33568141985, 33568142008, 33568142291` |
| `SCPN-FUSION-FISSION-HYBRID-CORE` | `e2deab2501fdb4f1ee3ee98928c63541dc81e2ee` | `e79737b73b783ba467fcfe42a3c2f3851fa4519d55b91769bb5e0cb55426f359` | `33568215410, 33568215449, 33568215462, 33568215546, 33568215566, 33568215574, 33568215711` |
| `SCPN-ICF-BEAM-CORE` | `69d55ebf703773a24e64e5a8be5676d3c214ee8a` | `01048901561c9dab00efedc01cf5eb56a2b4199736aa0a1c251fe15293aff9f1` | `33568178765, 33568178769, 33568178770, 33568178779, 33568178790, 33568178814, 33568178826` |
| `SCPN-ICF-IMPACT-CORE` | `ceb0eafa26b30e24d516a0a56103926ee5cd9dc8` | `bfc2fbb31b0c172b4acb5554d5ead8324153454cceae214f062d599c2ec62e39` | `33568183812, 33568183820, 33568183908, 33568183990, 33568184065, 33568184085, 33568184254` |
| `SCPN-ICF-LASER-CORE` | `3ff396d6d69d891f53fb0210053d259bd42ccc6e` | `bc2edd2c12f7f5b9da4fc09c1d0747850fbb29ab76170db0d43ab1ad92be0f7c` | `33568178530, 33568178647, 33568178648, 33568178729, 33568178760, 33568178811, 33568179462` |
| `SCPN-IEC-CORE` | `3b9dbf6cc3bbf5defd35bf81d146de3a30f66e09` | `87492cb79c32aa496bcca12ebd92f8d4eec767a53e7da64fee0f498ed1197f9b` | `33568206585, 33568206606, 33568206609, 33568206624, 33568206631, 33568206644, 33568207050` |
| `SCPN-LEVITATED-DIPOLE-CORE` | `8b286c031b1e9dbfd1b037f2406aaba9781a3548` | `abd1345cbd38e343a092bbdd166f0abbc235ed8218c3893f1b1ece09ab7050f5` | `33568154917, 33568154926, 33568154934, 33568154940, 33568154976, 33568154988, 33568155128` |
| `SCPN-MAGNETIC-CUSP-CORE` | `afa7c32e62c652e6d655a4699a846c55ecc4a6db` | `eb4a3ea6d8213872686aa1ad5a148f5855dd43c43408376970953b3c836ad61a` | `33568150866, 33568150883, 33568150886, 33568150915, 33568150922, 33568150953, 33568151071` |
| `SCPN-MIF-LINER-CORE` | `7e4b6dbc2c1d174b584cec568f78c339498c6a52` | `754514c545f52932c430684cdc4c6b31270fc40b005c65c7461146dfab59f8d3` | `33568211003, 33568211004, 33568211028, 33568211055, 33568211092, 33568211093, 33568211280` |
| `SCPN-MIF-MAGLIF-CORE` | `90c8503db3eeacd3ef86b54d852a70b887773df2` | `46dfcd0f1c417b3a3ba876fb735a578cd2118bfe739d0316d7510a82c43bac2a` | `33568194083, 33568194113, 33568194160, 33568194532, 33568194574, 33568194588, 33568194741` |
| `SCPN-MIF-PLASMA-JET-CORE` | `43cea01a5f663fcd1f4d0f3413809229666439ff` | `d9ac83a603c3b5b63c54a60b04379638c904836c3ac38df70996007823e0c7f8` | `33568194790, 33568194810, 33568194813, 33568194868, 33568194893, 33568194973, 33568195086` |
| `SCPN-MIRROR-CORE` | `75a5b624a833c84d3100aa79f488f2e841bbf20a` | `efb5634818a256a9ab5d68c2cdb346503f2c581326f23d7a06c8a0d2da724d48` | `33568145289, 33568145293, 33568145297, 33568145298, 33568145369, 33568145447, 33568145747` |
| `SCPN-RFP-CORE` | `f5a35424b05531399e2759cc3713bc25b647c59d` | `5f88e694faf5b73e0e0ce3000fb643a6b0b5c84993cd373a917f9d633aa35d42` | `33568127967, 33568127988, 33568127995, 33568127997, 33568128022, 33568128025, 33568128270` |
| `SCPN-SPHEROMAK-CORE` | `5712e5d9535477033c83a3d6e1586bcef288eb25` | `21232f9966169dee29798171f71faef825732ac1df30fca97dd10bf5752fefd3` | `33568137201, 33568137202, 33568137219, 33568137275, 33568137289, 33568137326, 33568137722` |
| `SCPN-STELLARATOR-CORE` | `afe989f3683380de144468ceae4531b758e9b11e` | `4cc94911b526d1220a3bd123d91a48603e9e134b9f7ef7b9e311fc65a38c6ba7` | `33568122776, 33568122777, 33568122797, 33568122802, 33568122826, 33568122829, 33568122883` |
| `SCPN-THETA-PINCH-CORE` | `2221f9043efb20193cb69842daa45ea2b5658127` | `5f8fe33fa4ce7c1fe591a6aefee852e91c74cdff0424c19b3592c0b0604f0611` | `33568167933, 33568167964, 33568167969, 33568167975, 33568168025, 33568168028, 33568168056` |
| `SCPN-TOKAMAK-CORE` | `7402191c43e8fe57cffda1dd5b3cf4319d6d398d` | `8e0c0d51f6c7aece428a6e761adf20f820f44aa6946b05921912cc4c87790253` | `33568117723, 33568117746, 33568117753, 33568117801, 33568117806, 33568117808, 33568117848` |
| `SCPN-Z-PINCH-CORE` | `fb050319f5397f85f2e19844b4f2e40ad1aa29a9` | `cfe64254564245f064bada4ca91dbc1debf11e6d75e29b2cbe073492207c89a1` | `33568159756, 33568159764, 33568159766, 33568159772, 33568159798, 33568159804, 33568159881` |

The previously missing `timing_uncertainty_s` member is now explicit in every
producer channel. A separately repaired SPO compatibility-table omission for
`direct_cyclic` facility-clock channels was required before the final 20/20
replay; producer bytes were not changed for that consumer defect.

## Machine-readable status

The [digest-sealed status data](data/reactor_diagnostic_plan_portfolio_status.v1.json)
records all 20 revisions, manifest and fixture hashes, configurations, plans,
custody states, and fail-closed authority fields. Its
[Draft 2020-12 schema](../specs/reactor_diagnostic_plan_portfolio_status.schema.json)
enforces schema version `1.2.0`, the 20/0 structural split, the 0/20 current-custody/public-object
split, exact 140-run evidence, and zero physical or control authority.

This register complements the
[reactor configuration evidence coverage](reactor_configuration_evidence_coverage.md):
an accepted design plan can still correctly remain producerless for physical
evidence and `not_declared` for semantic ingress.
