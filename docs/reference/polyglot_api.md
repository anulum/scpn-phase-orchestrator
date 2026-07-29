<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Polyglot API documentation -->

# Polyglot API Documentation

SCPN Phase Orchestrator has four maintained native accelerator surfaces in
addition to its public Python API. Each surface keeps its native source model and
documentation format; the documentation build does not merge independent
accelerators into a synthetic package.

## Generated artifacts

| Backend | Native generator | Artifact | Contract |
| --- | --- | --- | --- |
| Rust | `cargo doc` / rustdoc | HTML under `rust/` | All Cargo workspace crates, including the internal PyO3 wrapper crate |
| Go | `go doc -all -cmd` | One text file per `go/*.go` unit | Each C-shared translation unit is documented independently |
| Julia | Julia `Base.Docs` and reflection | `julia/api.md` | Docstrings and signatures for module-owned, non-private callables |
| Mojo | `mojo doc` | One JSON file per `mojo/*.mojo` unit | Toolchain-native docstring output for each executable module |

CI uploads these as `polyglot-docs-rust`, `polyglot-docs-go`,
`polyglot-docs-julia`, and `polyglot-docs-mojo`. Rustdoc warnings fail the Rust
documentation step. Compilation remains a separate gate for every backend, so a
documentation artifact is not evidence of numerical parity or runtime support.

## Build locally

With all four toolchains available, generate the complete set:

```bash
python tools/generate_polyglot_docs.py all --output build/polyglot-docs
```

Generate one backend when only that toolchain is installed:

```bash
python tools/generate_polyglot_docs.py rust --output build/polyglot-docs
python tools/generate_polyglot_docs.py go --output build/polyglot-docs
python tools/generate_polyglot_docs.py julia --output build/polyglot-docs
.venv/bin/python tools/generate_polyglot_docs.py mojo --output build/polyglot-docs
```

The Julia command accepts either a `julia` executable or the `juliacall` runtime
used by the Julia CI lane. The Mojo command should run from the Python environment
that contains the pinned Mojo toolchain in `requirements/mojo-lock.txt`.

## Source boundaries

- Rust is a Cargo workspace. The artifact includes all six crates; `spo-ffi`
  remains an internal Python binding surface even though its Rust items are
  inspectable.
- Go sources are separate `package main` C-shared libraries. Combining them would
  create duplicate declarations and document a package that cannot be built.
- Julia sources are separate modules loaded by the Python backend bridge. The
  generated Markdown records module descriptions, attached `Base.Docs`
  docstrings, and module-owned callable signatures.
- Mojo sources are separate subprocess executables, so each has its own JSON
  artifact. The format is owned by the pinned Mojo compiler and may evolve with
  that toolchain.

For the stable user-facing contract, use the [Python API reference](api/index.md).
For backend availability, fallback, and promotion policy, use the
[Backend Strategy](../guide/backend_strategy.md) and
[Backend Fallback Chain](../guide/backend_fallbacks.md).
