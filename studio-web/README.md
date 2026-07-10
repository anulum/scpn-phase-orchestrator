<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# SPO studio remote (`studio-web/`)

The SCPN Phase Orchestrator studio as a **Module Federation remote**: the UI
panel the [SCPN Studio](../../SCPN-STUDIO/docs/studio-hosting.md) Hub loads at
runtime from this repository's published `remoteEntry.js`. It is served under
`https://www.anulum.org/studios/scpn-phase-orchestrator/` and inherits the
portal's session — it builds no login, account, or billing of its own.

## What the panel shows

`SpoStudioPanel` is a **pure renderer**: it computes nothing and upgrades no
claim. Its one surface is the **evidence-coverage map** — how each of SPO's six
assurance evidence categories (audit logging, replay determinism, formal
verification, twin confidence, conformal gate, control envelope) contributes to
EU AI Act, ISO/IEC 42001 and ANSI/UL 4600 clauses, at the honest `addressed` /
`partially_addressed` boundary. A snapshot that fails its guard renders as a loud
`unverifiable` block, never a blank or a silently downgraded card.

## Single source of truth

The panel inlines `src/panel/evidence_coverage.json`, produced by the Python
module `scpn_phase_orchestrator.studio.panel_data` from the same clause map the
assurance-case bundle uses. Regenerate it after any assurance clause-map change:

```bash
python ../tools/build_studio_panel_data.py           # rewrite the snapshot
python ../tools/build_studio_panel_data.py --check    # verify it is in sync (CI + lint gate)
```

The federation contract (name, exposed module, remote-entry URL) is mirrored in
`scpn_phase_orchestrator.studio.federation_manifest` (`ui_module`) and guarded by
`tests/test_studio_federation_manifest.py`.

## Develop, test, build

```bash
pnpm install --frozen-lockfile
pnpm typecheck        # tsc --noEmit (strict)
pnpm test:coverage    # vitest, 100% thresholds
pnpm build            # tsc + vite build -> dist/remoteEntry.js
pnpm dev              # local portal shell at src/main.tsx
```

`node_modules/`, `dist/`, and `coverage/` are gitignored; the committed tree is
the source, the config, the lockfile, and the evidence snapshot. The `studio-web`
CI job runs the typecheck / coverage / build chain and asserts the built
`remoteEntry.js` carries the `scpn_phase_orchestrator` federation name.
