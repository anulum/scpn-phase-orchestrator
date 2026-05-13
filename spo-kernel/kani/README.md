<!--
SPDX-License-Identifier: AGPL-3.0-or-later
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Kani Formal Verification

The executable proof harnesses live in the owning Rust crate:

- `crates/spo-supervisor/src/formal_safety.rs`

They are compiled only under `cfg(kani)` and call the same crate functions used
at runtime:

- `projector::project_value`
- `projector::compute_adaptive_rate_limit_fixed`
- `projector::project_fixed_point_value`
- `regime::classify_regime_from_summary`

Run all supervisor proofs:

```bash
cargo kani -p spo-supervisor -Z function-contracts
```

Run one proof:

```bash
cargo kani -p spo-supervisor -Z function-contracts \
  --harness nominal_safe_summary_never_classifies_critical
```

The normal Rust regression layer is:

```bash
cargo test -p spo-supervisor
```
