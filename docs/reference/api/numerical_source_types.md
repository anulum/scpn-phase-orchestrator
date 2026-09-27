<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — Numerical source type contracts
-->

# Numerical source types

Measurement arrays retain their original type until validation. Numeric text,
booleans, datetime values and duration values are refused before conversion to
`float64`, including booleans and durations inside object arrays. Integer and
floating arrays remain supported. Where an API already accepts numeric object
arrays, every original element must be a plain real number. Shape, finiteness,
index, probability and domain bounds remain owned by the individual API.
Coupling prior, spectral coupling, transfer entropy and delayed dynamics refuse
real numeric object values outside the `float64` range with the owning input
error before native dispatch.

This contract applies to coupling, phase extraction, monitor inputs, replay
policy arrays, imprint updates, SSGF, supervisor distributions and gradients,
phase gossip, delayed dynamics, plasma coupling imports, PHA-C acceptance
evidence and numerical visualisations. For hybrid quantum monitoring, classical phases obey this rule;
the quantum state retains its complex statevector or density-matrix contract.

Recurrence outputs are binary masks and accept boolean arrays as well as valid
numeric 0/1 arrays. Other original measurement types remain refused. Generated
activation masks, random fixtures, histograms and validated numerical results
are internal calculations with their own types.

## Language boundaries

Go, Julia and Mojo Python bridges share the same original-source checks before
packing numerical buffers. Their kernels receive typed numerical buffers. Rust
measurement and metadata extractors reject source aliases before conversion.
The named-call wrappers retain positional, keyword and mixed calls and their
existing introspection signatures. Negative unsigned counts retain the existing
`OverflowError`. Plain measurement/metadata extractors raise `ValueError` for
source aliases; incompatible ndarray ABI types are rejected by PyO3 type checks.

The flat hypergraph encoding contains one start offset and strength per edge,
without a terminal offset. Offsets start at zero and strictly increase; each
edge contains at least two distinct in-range nodes. Invalid offsets, node
indices, buffer dimensions and non-finite values raise Python errors before
integration. Empty native pairwise coupling and alpha buffers preserve their
existing meanings: absent pairwise coupling and zero phase shifts. A zero-step
run validates its inputs and returns the initial state.

## Measurements

[Delayed dynamics and transfer entropy comparisons](../data/measurement_source_contracts_polyglot_2026-09-27.json)
record the subsequent overflow and diagnostic corrections against all five
available backends, with the corrected source hashes.

[Final language comparisons](../data/coercion_sweep_final_polyglot_2026-09-27.json)
record real calls to the available Python, Rust, Go, Julia and Mojo backends.
Parity gates compare numerical results; the separate trajectory timing rows
measure elapsed time and do not establish numerical parity. Records include
fixture arguments and the installed extension hash. Earlier dated measurements
remain historical observations of their recorded source hashes.

The host is shared. These small fixtures establish input and numerical contracts;
they do not establish production scaling or controlled speedup ratios.
