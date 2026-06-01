<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Studio API reference -->

# Studio API

Studio helper functions are pure payload builders used by the Streamlit
operator surface and by tests that validate browser-facing contracts without
opening transports or writing hardware.

The public `scpn_phase_orchestrator.studio` facade exports the passive physics
review panels below alongside the lower-level helper module. Operator code can
therefore depend on the Studio package surface without importing private helper
paths or bypassing the review-only safety gates.

`build_studio_product_manifest()` returns the metadata-only standalone Studio
product manifest used by the packaged `tools/spo_studio.py` shell. The manifest
lists every passive physics review panel, the public builder name, the
claim/proof boundary, required evidence, disabled execution gates, and a
deterministic `manifest_sha256` over the canonical manifest payload while keeping
`network_opened`, `hardware_write_permitted`, `qpu_execution_permitted`, and
`actuation_permitted` false. It does not execute panel builders, import optional
runtimes, open transports, or touch hardware.

`build_integrated_information_panel(records)` renders passive
`integrated_information` monitor audit records into an operator payload with
latest Phi proxy values, normalised-Phi and total-integration ranges,
minimum-partition review cards, and explicit `actuation_permitted: false` plus
`consciousness_claim_permitted: false` gates. The helper requires the
`engineering_proxy_not_theoretical_iit` claim boundary, finite real-valued
non-negative information metrics, Phi/log-bin normalisation consistency,
integer-only minimum partitions, and symmetric bounded pairwise-MI matrices
when matrix evidence is supplied.

`build_strange_loop_studio_panel(records)` renders offline
`StrangeLoopSupervisor` drift-scenario result records into a Studio payload with
scenario pass counts, triggered-mode summaries, maximum drift/oscillation/
over-control scores, minimum control coherence, and failed-scenario IDs. It
requires the `strange_loop_drift_review_not_live_actuation` claim boundary,
`non_actuating: true`, `execution_disabled: true`, supported expected triggers,
finite non-negative metrics, unit-interval coherence, and SHA-256 scenario/
result hashes before Studio may display the evidence. The payload remains
review-only and sets `actuation_permitted: false`.

`build_morphogenetic_field_studio_panel(svg_artifact)` renders a passive
`render_morphogenetic_field_svg()` artefact into a Studio payload with SVG
metadata, fixed-width ASCII heatmap rows, field-energy statistics, strongest
off-diagonal topology edges, and explicit `actuation_permitted: false`. The
helper requires a complete SVG document, square snapshot shape, bounded
unit-interval field statistics, shape-compatible heatmap rows, sorted
off-diagonal top-edge records, and finite non-negative L2 energy before Studio
may expose the field rendering to an operator.

`build_multiverse_counterfactual_studio_panel(manifest, risk_report)` joins a
passive multiverse rollout manifest with the matching branch-risk report and
renders branch comparison rows, approval counts, rejected-branch IDs,
coherence ranges, safest-branch metadata, and immutable manifest/report hashes.
The helper requires non-actuating execution-disabled rollout and risk claim
boundaries, supported NumPy or JAX vectorised backends, unit-interval `R`
metrics with ordered intervals, matching branch hashes across rollout and risk
records, finite topology pressure metrics, and explicit
`actuation_permitted: false` output.

`build_hybrid_order_studio_panel(records, scenarios=...)` renders passive
hybrid classical-quantum order-parameter audit records and deterministic
scenario fixtures into a Studio payload. It exposes classical `R`/`Psi`,
bipartition Von Neumann entropy, normalised entropy, participation-ratio
ranges, strongest-entanglement review cards, simulator backend summaries, and
scenario candidate rows. The helper requires the
`quantum_cosimulation_monitor_not_qpu_execution` claim boundary, true
`non_actuating` and `execution_disabled` flags, finite unit-interval
coherence and normalised-entropy metrics, positive qubit counts, valid
two-group bipartitions, supported local NumPy simulator backends, and matching
record SHA-256 hashes before Studio may display the evidence. The payload sets
both `actuation_permitted: false` and `qpu_execution_permitted: false`.

`build_information_geometry_studio_panel(records, scenarios=...)` renders
passive information-geometry control proposal audit records and deterministic
domain scenarios into a Studio payload. It exposes Fisher-Rao and Wasserstein
distance ranges, metric-tensor and positive metric-diagonal ranges,
natural-gradient norm ranges, curvature/geodesic metrics, backend summaries,
latest review proposal cards, and scenario candidate rows. The helper requires the
`information_geometry_control_not_live_actuation` claim boundary, true
`non_actuating` and `execution_disabled` flags, supported NumPy/JAX information
geometry backends, SHA-256 proposal/scenario hashes, unit-mass simplex
coordinates, matching target/tangent dimensions, positive finite metric
diagonals, symmetric metric tensors, matching geodesic/Fisher-Rao metrics, and
strictly positive review TTL/max-step values before Studio may display the
evidence. The payload sets `actuation_permitted: false` and never exposes
executable control actions.

`build_topos_semantic_binding_studio_panel(symbolic_reports, policy_reports,
examples=...)` renders passive Topos-theoretic semantic-binding evidence into
an operator payload. It joins symbolic-binding functor validation reports,
policy-composition category reports, and deterministic domain obligation
examples while preserving the prototype proof boundary. The helper requires
the `categorical_validation_prototype_not_formal_topos_proof` proof boundary,
true `non_actuating` flags, supported validation schemas, SHA-256 report and
example hashes, matching object/morphism counts, deterministic morphisms,
well-formed obligation records, and passed domain examples before Studio may
display the evidence. The payload sets `actuation_permitted: false` and
`formal_proof_claim_permitted: false`; it is a review surface, not a
machine-checked proof or executable policy channel.

`build_evolutionary_supervisor_policy_search_studio_panel(reports,
examples=..., dsl_reports=...)` renders passive evolutionary supervisor policy
search evidence into an operator payload. It joins offline replay-search
reports, enriched deterministic domain examples, and optional policy-DSL
mutation reports while preserving review-only gates. The helper requires the
`offline_evolutionary_supervisor_review_not_live_actuation` claim boundary for
core search reports, `evolutionary_supervisor_search_not_live_actuation` for
domain examples, true `non_actuating` and `execution_disabled` flags where the
source schema carries them, disabled live merge, disabled hot patching,
operator review requirements, SHA-256 report/candidate/scenario hashes,
matching candidate and accepted/rejected counts, finite replay/STL metrics,
and candidate-level actuation denial before Studio may display the evidence.
The payload sets `actuation_permitted: false`,
`hot_patch_permitted: false`, and `live_merge_permitted: false`.

`build_autopoietic_lineage_studio_panel(manifests)` renders passive
autopoietic lineage sandbox evidence into an operator payload. It preserves
offline lineage manifests, domain-labelled replay corpus rows, child-policy
SHA-256 hashes, accepted child rows, rejected child rows, and blocked-reason
evidence under the `autopoietic_lineage_sandbox_review_not_live_merge` claim
boundary. The helper requires the lineage sandbox schema, true review and
execution-disabled flags, disabled live merge, disabled hot patching, disabled
actuation, canonical SHA-256 hashes, matching child/replay/domain counts,
finite replay metrics, and candidate-level policy diffs before Studio may
display the evidence. The payload sets `actuation_permitted: false`,
`hot_patch_permitted: false`, `live_merge_permitted: false`, and
`operator_review_required: true`; it is a review surface for comparing
resource-bounded child-policy proposals, not a self-modifying runtime.

::: scpn_phase_orchestrator.studio.workflow

::: scpn_phase_orchestrator.studio.product

::: scpn_phase_orchestrator.studio.ui_helpers
