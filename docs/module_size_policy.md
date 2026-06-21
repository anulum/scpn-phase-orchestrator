# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Module size policy

# Module size policy

Line count is a **proxy, never the gate.** Single-responsibility code is fine at
any length; multi-responsibility code must be split at any length. The number is
a *trigger* for a review, not a budget to satisfy.

## The real criterion

> One module = one responsibility. If you need the word "and" to describe what a
> module does, split it.

The decisive test is the **AST call-graph**, not the line count:

- **Split it** when the module decomposes into a few cohesive clusters with only
  a handful of cross-edges — i.e. an acyclic dependency graph of sub-modules.
  This is a god-module: the responsibilities are merely co-located.
- **Keep it whole** when the module is one densely interconnected cluster — a
  solver, parser, state machine, or single class whose methods share state.
  Splitting it would invent artificial coupling and import ceremony; if such a
  class is itself doing too much, the fix is to *extract collaborators*, not to
  scatter its code across files.

## Trigger thresholds (`tools/check_module_size.py`)

| Lines | Action |
|---|---|
| < 600 | Nothing — this matches the size of a healthy single-responsibility module. |
| 600–900 | Fine if single-responsibility; a glance is enough. |
| 900–1199 | **Review trigger.** Run the AST call-graph: does it split into clean clusters? |
| ≥ 1200 | Strong multi-responsibility signal. **Default-split** unless the call-graph shows one cohesive cluster; if kept whole, record it in the allowlist with a justification. |

These numbers are calibrated to this repository. The god-file refactor campaign
split ten modules in the 1185–1882 range, and **every one was genuinely
multi-responsibility** — it decomposed into 4–9 single-responsibility sub-modules
with a clean DAG and byte-identical (AST-level) relocation. The resulting
sub-modules cluster around 100–620 lines, with the largest cohesive units
(`boundary` 586, `runtime` 620, `compiler` 451) comfortable as single
responsibilities. So in practice, ≥ 1200 lines reliably meant "split me", while
the 900–1199 band is the judgement zone.

## Exemptions

- Test modules and generated trees (`grpc_gen`) are exempt — their size is not a
  design choice.
- A package `__init__.py` that is a thin re-export façade is exempt by intent.

## The guard

`tools/check_module_size.py` reports modules by tier:

- **Default** (`python tools/check_module_size.py`) prints the report and exits
  0 — a warning, not an error.
- **Ratchet** (`python tools/check_module_size.py --check`) exits non-zero only
  when a module is ≥ 1200 lines *and* absent from
  `tools/large_module_allowlist.json`. New god-modules are blocked; modules
  reviewed and kept whole are grandfathered with a recorded reason.

To keep a module above the split threshold, add an entry to the allowlist:

```json
{
  "reviewed_large_modules": [
    {"module": "scpn_phase_orchestrator.pkg.solver", "reason": "one cohesive solver; methods share state"}
  ]
}
```

The reason is mandatory — an allowlist entry is a documented design decision, not
a silence switch.

## Standing record (for reviewers)

File size in this repository is an actively audited design decision, not an
oversight. Two facts establish that:

1. **The god-file campaign is complete.** Every module that exceeded the split
   threshold was decomposed into a responsibility-scoped package — `runtime/cli`,
   `studio/ui_helpers`, `nn/supervisor`, `plugins/registry`,
   `binding/digital_twin`, `supervisor/formal_export`, `supervisor/hierarchy`,
   `runtime/cli/_payloads`, `binding/semantic`, and `monitor/stl`. Each split was
   derived from an AST call-graph, produced a clean acyclic module graph, and
   relocated every node byte-identically at the AST level (proven per node). The
   commits are in the git history. No module above the split threshold remains.

2. **The largest remaining modules were reviewed and kept by responsibility.**
   The figures below were current at the last review; run
   `python tools/check_module_size.py` for live numbers. Each sits in the
   900–1199 review band and was assessed with the AST call-graph and found to be
   a single responsibility — splitting it would invent coupling, not remove it.

   | Module | ~Lines | Why it is one responsibility |
   |---|---|---|
   | `upde/moving_frame` | ~1120 | One moving-frame UPDE solver engine plus its run entry points and state; the engine's methods share state. |
   | `upde/pha_c_formal_obligation` | ~1110 | One PHA-C kinematic proof obligation — its construction, verification, and serialisation with the kinematic sub-checks. |
   | `supervisor/federated_transport` | ~1100 | One signed-transport lifecycle (build → validate → replay) over a shared hashing/normalisation core, with a deployment-preflight feature; the envelope, replay ledger, and ~20 helpers are densely shared. |
   | `monitor/twin_confidence` | ~1080 | One twin-confidence pipeline: divergence → calibration → scoring → summary → Prometheus export, tightly coupled. |
   | `runtime/cli/plugins/scheduler_control` | ~1050 | One CLI command group — the scheduler-control sub-commands over shared payload loaders. |
   | `nn/functional` | ~1015 | The flat `nn.functional` namespace of integrator and metric primitives (Kuramoto / Winfree / simplicial / Stuart–Landau steps, order parameter, PLV, …), kept flat by the conventional functional-namespace contract. |

   Modules in the 900–1000 band (`supervisor/evolutionary_petri_grammar`,
   `supervisor/causal`, `upde/pha_c_acceptance`, `autotune/reward`) were assessed
   on the same basis and are likewise single-responsibility.

A line-count objection to any of these is answered here: the criterion is the
call-graph, the threshold guard (`--check`) prevents regressions, and the
remaining large modules are cohesive by deliberate review. If a future change
makes one genuinely multi-responsibility, split it; if it crosses the split
threshold and is still one responsibility, record it in the allowlist with a
reason.
