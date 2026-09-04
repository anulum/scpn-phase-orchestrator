# Reactor diagnostic-plan portfolio status

This snapshot applies SPO's public, review-only diagnostic-plan validator to
immutable `origin/main` bytes from all 22 Reactor Systems device projects. It
reads each exact `reactor-domain.json` and
`tests/data/plan_envelope_fixture.json` object without importing or executing
producer code. Local worktree state is not used as remote evidence.

## Result

- **22 device projects** were examined at `2026-09-04T07:27:40Z`.
- **7 producer objects** are structurally accepted, **13 are refused**, and
  **2 architecture-only projects** have no declared diagnostic plan.
- **0 current fixtures** have byte-identical SPO custody.
- **7 fixtures** are digest-pinned public producer objects.
- **148/154 hosted workflows** completed successfully at the exact 22 public
  heads; six `CI` workflows failed and no workflow was cancelled.
- **0 fixtures** constitute a qualified physical observation or physical phase.
- No row creates CONTROL intent, action, execution, actuation, or
  machine-protection authority.

Structural acceptance proves only that a design declaration satisfies an
exact versioned manifest/envelope/plan contract. It does not validate hardware,
calibration, geometry, clocks, observability, measurements, regimes, or
operation. A failed hosted workflow is engineering evidence and stays visible;
it is never converted into scientific evidence.

## Exact project partition

| Project | Status | Exact public head | Current boundary |
|---|---|---|---|
| `SCPN-BEAM-TARGET-CORE` | refused | `de78bbdbb8e2e25465348cc11914c2b76f61a0ff` | canonical kernel-owner exclusion missing |
| `SCPN-DENSE-PLASMA-FOCUS-CORE` | refused | `35448d2627774e824013f4f61a7c5cffc1783fab` | source digest mismatch; CI failed |
| `SCPN-FRC-CORE` | refused | `2f59d99ab61b1111fe5bdaa676696be2b2d13fe3` | canonical kernel-owner exclusion missing |
| `SCPN-FUSION-FISSION-HYBRID-CORE` | refused | `9ed313551a52c0aceda206abeb4161522c1d01ce` | canonical kernel-owner exclusion missing |
| `SCPN-ICF-BEAM-CORE` | accepted | `8cea19a51f0aed7b47c7b475e631fac236ef79b8` | supply physical sample envelope |
| `SCPN-ICF-IMPACT-CORE` | accepted | `0c20361ced162f0e89e254aafbe9fbd0c182ef65` | supply physical sample envelope |
| `SCPN-ICF-LASER-CORE` | accepted | `6ef0faf241980cfc101af89fd35ac3cc7979f11f` | supply physical sample envelope |
| `SCPN-IEC-CORE` | accepted | `c32d4a06a3c9c0c0b8914916ee7d825df6961779` | supply physical sample envelope |
| `SCPN-LATTICE-FUSION-CORE` | not declared | `269e0b7c8f750f4d681b4bb1c6e9cbaf4e201722` | publish a versioned diagnostic plan |
| `SCPN-LEVITATED-DIPOLE-CORE` | accepted | `429b71be5c4aab72ce2974186b08863fb8a8e2a8` | supply physical sample envelope |
| `SCPN-MAGNETIC-CUSP-CORE` | accepted | `e0eb35b0df9d8c035b70c04d4ee00ff9b142ac20` | supply physical sample envelope |
| `SCPN-MIF-LINER-CORE` | refused | `d587bfadb81530a4da1626349b0af0c0788f0019` | canonical kernel-owner exclusion missing |
| `SCPN-MIF-MAGLIF-CORE` | refused | `638854a8f0c59bec5c950ef44a27c8d541d64d23` | canonical kernel-owner exclusion missing |
| `SCPN-MIF-PLASMA-JET-CORE` | refused | `f6ab528ca639b74081cb4d6ba1dac32c8420353c` | canonical kernel-owner exclusion missing |
| `SCPN-MIRROR-CORE` | refused | `b481450136ea65f7ddbd3127d4ee762757051cb5` | source digest mismatch; CI failed |
| `SCPN-MUON-FUSION-CORE` | not declared | `1dbbb283996f0b2771d62949410dec6290915202` | publish a versioned diagnostic plan |
| `SCPN-RFP-CORE` | refused | `e34ea503145c6e39cae5aed5c7a03325b4f0c825` | source digest mismatch; CI failed |
| `SCPN-SPHEROMAK-CORE` | refused | `b9f3fb14a731625ab941e2420f781c157a656f63` | source digest mismatch; CI failed |
| `SCPN-STELLARATOR-CORE` | accepted | `53455eee60fae820622b0568cd1af9c5d86cb093` | supply physical sample envelope |
| `SCPN-THETA-PINCH-CORE` | refused | `a5083ff1f3507636280df2b166eb06d0b7c6d82b` | source digest mismatch; CI failed |
| `SCPN-TOKAMAK-CORE` | refused | `8b40cdd943127da00017f61b1b02cef713299a50` | canonical kernel-owner exclusion missing |
| `SCPN-Z-PINCH-CORE` | refused | `80adc66392d6083ac8bc16a4d8bfc5d9bb40b652` | source digest mismatch; CI failed |

The seven ownership refusals have a `kernel_library` declaration but do not
exclude `shared_physics_geometry_and_numerics_kernels` in favour of the
canonical `SCPN-REACTOR-KERNELS` owner. The six digest refusals contain an
envelope that does not pin the supplied current manifest and plan bytes. Both
classes are producer-owned fix-forward work and remain refused until exact new
objects pass review.

The lattice and muon projects are intentionally architecture-only in this
snapshot. Registry membership and green repository workflows do not invent a
diagnostic plan, signal, observation, phase, or control path.

## Hosted verification boundary

Every row pins seven run IDs for `CI`, `CodeQL`, `Docs`, `Pre-commit`, `SBOM`,
`Scorecard`, and `Security audit`. The exact-head totals are 154 expected, 148
successful, 6 failed, and 0 cancelled. The six failed `CI` runs correspond to
the six stale digest objects above; all sibling workflows at those heads are
green. These failures were reported to the Reactor Systems owner and must not
be purged while unresolved.

## Machine-readable custody

The canonical
[`reactor_diagnostic_plan_portfolio_status.v1.json`](data/reactor_diagnostic_plan_portfolio_status.v1.json)
is validated by
[`reactor_diagnostic_plan_portfolio_status.schema.json`](../specs/reactor_diagnostic_plan_portfolio_status.schema.json).
Schema version `1.3.0` binds the 22-project partition, review contract `1.1.0`,
exact source revisions, artifact digests, hosted-run receipts, and fail-closed
authority fields. Full hashes and run IDs live in the sealed JSON rather than
being duplicated here.

This register complements the
[reactor configuration evidence coverage](reactor_configuration_evidence_coverage.md):
an accepted design plan can still correctly remain producerless for physical
evidence and `not_declared` for semantic ingress.
