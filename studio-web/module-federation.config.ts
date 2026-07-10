// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-phase-orchestrator — studio-web Module Federation contract

/**
 * The federation contract of the SPO studio remote.
 *
 * Locked by the platform v1 contract and mirrored in the Python capability
 * manifest (`studio/federation_manifest.py`): the federation name is
 * `scpn_phase_orchestrator` (the underscored studio id), the remote exposes
 * exactly `./SpoStudioPanel`, and react/react-dom are shared as version-pinned
 * singletons so the Hub never mounts a second React. Keep this object
 * additive-only; renaming any field is a breaking change to the Hub contract.
 */

export const FEDERATION_NAME = "scpn_phase_orchestrator";
export const PANEL_EXPOSE_KEY = "./SpoStudioPanel";
export const REACT_VERSION = "19.2.7";

export const moduleFederationConfig = {
  name: FEDERATION_NAME,
  filename: "remoteEntry.js",
  exposes: {
    [PANEL_EXPOSE_KEY]: "./src/SpoStudioPanel.tsx",
  },
  shared: {
    react: { singleton: true, requiredVersion: REACT_VERSION },
    "react-dom": { singleton: true, requiredVersion: REACT_VERSION },
  },
} as const;
