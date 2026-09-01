# Reactor signal occurrence ledger

- **Snapshot:** 2026-09-01
- **Authority:** `review_only`; `actionable=false`; direct actuation not authorized
- **Machine record:** [reactor_signal_occurrence_ledger.v1.json](data/reactor_signal_occurrence_ledger.v1.json)
- **Schema:** [reactor_signal_occurrence_ledger.schema.json](../specs/reactor_signal_occurrence_ledger.schema.json)

This ledger is the exact-source companion to the cross-reactor observability
atlas. It answers a narrower question: where do externally meaningful phase,
mode, regime, event, and clock concepts actually occur today in
SCPN-PHASE-ORCHESTRATOR (SPO), SCPN-FUSION-CORE, SCPN-MIF-CORE, and
SCPN-CONTROL, and what can each occurrence honestly mean?

The snapshot contains 39 stable public or cross-project occurrence groups:
12 SPO, 11 FUSION, 7 MIF, and 9 CONTROL. Every group is bound to an exact Git
revision, source path, public or boundary symbol, and SHA-256 of the exact Git
blob. The payload itself is canonical-JSON sealed. No occurrence in this
snapshot admits a physical phase observation or authorizes direct actuation.

## Reading rule

An occurrence is evidence that a concept or value exists in source. It is not
by itself evidence that the concept was measured in a reactor. The ledger
therefore separates:

- `source_markers`: direct names or boundary statements present in the source;
- `assessment_basis=explicit_source_boundary`: the source states its own
  limitation;
- `assessment_basis=inferred_by_spo_audit`: SPO classifies the carrier from the
  implemented fields and behavior without upgrading its maturity;
- `physical_source_present`: immutable physical-source custody exists;
- `physical_observation_admitted`: the complete diagnostic evidence gate passed;
- `physical_phase_eligible`: an admitted observation also has cyclic/modal
  reference, operator, clock, uncertainty, quality, validity, and observability
  support.

Plain substring hits are excluded. In particular, CLI mode, delivery mode,
failure mode, test names, documentation repeats, and fixture copies do not
become separate reactor-semantic rows. Python/Rust mirrors and repeated helper
calls are grouped under the stable outward symbol.

## Exact revisions

| Project | Exact scanned revision | Occurrence groups |
|---|---|---:|
| SCPN-PHASE-ORCHESTRATOR | `386c6537b22a3e36fd10402dbe68cffc8721a360` | 12 |
| SCPN-FUSION-CORE | `c30fb3932b47a812dc26d5846761030cdd0bc94c` | 11 |
| SCPN-MIF-CORE | `f3132574b0d4f45b29e2c27cfc2c830ee868c13e` | 7 |
| SCPN-CONTROL | `a3b39652f8d97cbdd057afb4e9e5e2859369ab79` | 9 |

The audit read exact tracked Git objects. Uncommitted sibling work was neither
read as canonical evidence nor modified.

## SPO authority surfaces

| ID | Stable surface | Meaning at this revision | Evidence boundary |
|---|---|---|---|
| SPO-001 | U0 vocabulary and contracts | canonical carrier, clock, observable, phase, relation, and regime grammar | contract only |
| SPO-002 | reactor registry | 32 configuration identities | identity is not implementation or readiness |
| SPO-003 | observability profiles | cross-family evidence requirements and failure dispositions | declaration, not observation |
| SPO-004 | regime/mode ontology | separated physical and numerical mode domains plus closed regime axes | definition, not extraction |
| SPO-005 | regime assessment | sealed eight-axis vector | current builder abstains; no classification |
| SPO-006 | FUSION transport handoff | 12 noncyclic bounded quantities | zero phase observability; UNKNOWN regime |
| SPO-007 | MIF merge-compression handoff | numerical oscillator phase plus bounded/categorical merge evidence | simulation only; UNKNOWN regime |
| SPO-008 | device diagnostic-plan review | synthetic signal/frame/clock design | creates no observation or classifier result |
| SPO-009 | FAIR-MAST source review | exact physical archive and qualification custody | not admitted and not phase-eligible |
| SPO-010 | reference portfolio | nine cross-family semantic examples | scaffold only |
| SPO-011 | PHA-C handoff/timeline | numerical phase dispersion and lock-event replay | local model evidence |
| SPO-012 | `FusionCoreBridge.observables_to_phases()` | normalized legacy angles | not valid U0 semantic ingress |

SPO's ownership is consequently precise: it owns semantic identity,
comparability rules, abstention, evidence requirements, and review envelopes.
It does not own producer truth, CONTROL admission, actuator execution, or
machine-protection veto.

## FUSION producer and model surfaces

| ID | Stable surface | Carrier | Evidence state |
|---|---|---|---|
| FUS-001 | `GlobalPsiDriver` | numerical phase | model tick |
| FUS-002 | `build_knm_plasma` presets | numerical phase/model scenario | hand-selected simulation model |
| FUS-003 | `interferometer_phase_shift` | optical field phase | explicitly synthetic diagnostic |
| FUS-004 | `RMFPhaseLockController` | numerical phase with physical intent | deterministic software horizon |
| FUS-005 | FRC tilt report | complex-mode identity and categorical model regime | reduced n=1 model; no parity claim |
| FUS-006 | locked-mode chain | harmonic amplitude, rotation lock, island and event model | no observed modal phase |
| FUS-007 | L/H and I-phase model | categorical model state | predator-prey simulation |
| FUS-008 | TORAX review envelope | bounded transport features | exact portable simulation evidence |
| FUS-009 | MAST archive envelope | physical magnetic source | unqualified for semantic ingress |
| FUS-010 | MAST qualification | measurement inventory and derived grids | phase inference explicitly absent |
| FUS-011 | stellarator replay contracts | bounded clocked replay features | synthetic replay, no facility-clock transform |

The FUS-003 source says it generates “synthetic raw diagnostic channels.” The
word `phase` in `interferometer_phase_rad` therefore identifies an optical
forward-model output; it does not identify an observed tearing-mode angle.
Likewise, FUS-004 accepts `plasma_phi`, but the public surface is a software
simulation and not a calibration record for a plant diagnostic.

FAIR-MAST is the only physical-source occurrence in this FUSION snapshot. Its
72 arrays, 11 measurement families, 132 qualified channel-correspondence
records, and four reproduced grids materially improve custody. They do not
supply calibration lineage, a validated physical geometry join, an observation
operator or harmonic basis, provider quality flags, uncertainty, an instrument
clock relation, a resolved event, or an observability threshold. Accordingly,
FUS-009, FUS-010, and SPO-009 all keep
`physical_observation_admitted=false` and `physical_phase_eligible=false`.

## MIF producer surfaces

| ID | Stable surface | Carrier | Evidence state |
|---|---|---|---|
| MIF-001 | merge-compression producer envelope | mixed numerical/bounded/categorical | exact portable simulation source |
| MIF-002 | Doppler-Kuramoto and moving-frame UPDE | numerical phase | simulation model |
| MIF-003 | merge-window monitor | categorical lock verdict over model values | simulation model |
| MIF-004 | streaming trigger | hold/fire/abort decision state | software decision model |
| MIF-005 | pulsed-shot FSM | protocol phase | eight named stages, not an angle |
| MIF-006 | named DAQ profiles | bounded fixture features | synthetic facility-shaped mock |
| MIF-007 | AER spike buffer | event timestamps | unbound event clock, not event-cycle phase |

The MIF producer states that it “owns MIF facts only” and does not assign
portable reactor phase meaning. SPO preserves that boundary: only the
Doppler-Kuramoto oscillator coordinates become `numerical_phase`; translation,
compression, lock, trigger, and gate results remain bounded or categorical.
`fire` is an implemented software decision, not evidence that a physical bank
fired and not permission for SPO to command one.

## CONTROL consumer and local-model surfaces

| ID | Stable surface | Meaning | Separation from action |
|---|---|---|---|
| CTRL-001 | generic semantic admission | admitted-for-review or rejected | always non-actionable |
| CTRL-002 | MIF semantic admission | exact numerical-phase and clock review | does not relabel phase physical |
| CTRL-003 | regime-assessment admission | exact custody and abstention review | no action approval |
| CTRL-004 | pulsed scenario scheduler | protocol state plus phase-lock guard | local model, not fed by SPO admission |
| CTRL-005 | Kuramoto/UPDE package | numerical phase | explicitly not reactor observation/actuation |
| CTRL-006 | locked-mode chain | reduced complex-mode/event model | no admitted physical mode phase |
| CTRL-007 | L/H and I-phase controller model | categorical model regime | no SPO admission link |
| CTRL-008 | RWM feedback model | reduced complex-mode regime | requires independent facility admission |
| CTRL-009 | geometry-neutral replay | bounded step/time diagnostics | lacks U0 evidence envelope |

CONTROL correctly owns admission and local action-model code, but those are
different surfaces. `admitted_for_review` is not an actuation decision. The
ledger found no path that turns an SPO semantic or regime-assessment record
into a hardware command, interlock, or machine-protection override.

## Changes from the 2026-08-30 atlas

Four former statements need explicit revision:

1. The “no U0 MIF producer handoff” architecture gap is closed by MIF-001 and
   SPO-007. The physical-observation gap remains open.
2. Open-text mode identity and unstandardized regime axes are partially closed
   by SPO-003 through SPO-005. Evidence-bearing physical bindings and validated
   classifiers are still missing.
3. FAIR-MAST now provides exact physical-source custody through FUS-009,
   FUS-010, and SPO-009. It does not yet provide phase observability.
4. CTRL-003 adds regime-assessment admission while intentionally requiring an
   abstaining assessment and retaining `review_only=true`, `actionable=false`.

These are source-state deltas, not claims about remote publication, reactor
operation, or technology readiness.

## Open gaps and next design consequences

| Gap | Immediate consequence |
|---|---|
| `OBS-01` | physical signals cannot enter phase semantics until operator, calibration, uncertainty, quality, validity, geometry, and provenance are complete |
| `CLK-01` | shot, simulation, replay, review, facility, and hardware timestamps cannot be compared without explicit transforms |
| `MODE-01` | a named or harmonic mode cannot yield physical phase without a validated spatial/modal operator and reference |
| `EVT-01` | a shot ID or timestamp does not define an event cycle |
| `DATA-01` | simulations and fixtures remain simulations and fixtures |
| `ACT-01` | semantic review remains intentionally disconnected from actuation and machine protection |
| `PROD-01` | mirror, ICF, IEC, beam-target, Z-pinch, spheromak, and other families remain architecture-only in this four-project ledger until a producer exists |
| `LEG-01` | callers must not treat normalized legacy angles as U0 reactor phase |

The highest-value next physical slice remains a diagnostic-specific tokamak
complex-mode observation operator over the MAST magnetic source, but only after
the listed MAST prerequisites are supplied. In parallel, producer projects for
other reactor families can adopt SPO-003/SPO-004 without pretending that their
bounded, event-relative, RF, modal, or protocol signals share one universal
phase.

## Verification contract

`tests/test_reactor_signal_occurrence_ledger.py` validates the JSON Schema,
canonical payload seal, exact revisions, row counts and order, source-digest
sentinels, gap references, and non-escalation implications. The checked-in
source hashes are SHA-256 values of `git show REV:path` bytes. Verification of
sibling blobs is an audit operation at refresh time; normal package CI does not
depend on sibling checkouts.
