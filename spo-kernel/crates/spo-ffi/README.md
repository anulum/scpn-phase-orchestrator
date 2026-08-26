<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# spo-kernel

Rust-accelerated UPDE kernel for
[scpn-phase-orchestrator](https://pypi.org/project/scpn-phase-orchestrator/)
— the synchronisation-analysis and honest-evaluation toolkit built on
Kuramoto/UPDE phase dynamics.

This package ships the compiled extension module (`spo_kernel`, PyO3,
stable ABI `abi3-py310`: one wheel per platform, CPython 3.10+). It is an
optional accelerator: `scpn-phase-orchestrator` runs without it and uses
it automatically when installed.

```bash
pip install spo-kernel
```

The Python surface (steppers, coupling builders, monitors, projectors) is
documented in the main package's API reference:
<https://anulum.github.io/scpn-phase-orchestrator/>. Source, sealed
evidence artefacts, and the issue tracker live in the repository:
<https://github.com/anulum/scpn-phase-orchestrator> (directory
`spo-kernel/`).

License: AGPL-3.0-or-later (commercial license available —
protoscience@anulum.li).
