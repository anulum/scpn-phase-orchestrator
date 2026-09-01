<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Z-pinch diagnostic-plan fixture custody -->

# Z-pinch diagnostic-plan fixture custody

These files are exact committed producer objects from
`SCPN-Z-PINCH-CORE` revision
`acf5dfe49de175e856c3d575bcfd11dead60d0fd`:

- `reactor-domain.json` SHA-256
  `2fd87b7e83de9f3a43ad64c7cfe0be96d91277c8745f51f028d7d67cf9f3cf35`;
- `plan_envelope_fixture.json` SHA-256
  `69a068ded2d3db9b9c64080547cedd204893637996b9fab91638ba3491b940a0`.

The exact producer revision was archived from a clean checkout and built as
`scpn_z_pinch_core-0.1.0.dev0-py3-none-any.whl`; the installed-wheel identity
supplied to SPO is SHA-256
`a413f77a4c60888980690a35d0dd3fdee2e9d2d1eb4868aee430bb1c6d8718fd`.
Two independent PEP 517 builds with `SOURCE_DATE_EPOCH=1788281779`, the source
commit timestamp, produced byte-identical wheels.

The fixture is a synthetic diagnostic design declaration. It is not a
measurement, physical observation, physical phase, regime classification,
CONTROL intent, action, execution permission, actuator command, or machine-
protection artefact. SPO consumes the producer documents as bytes and never
executes the sibling checkout.
