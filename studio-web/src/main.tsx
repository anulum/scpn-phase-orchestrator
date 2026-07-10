// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-phase-orchestrator — studio portal entry

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import SpoStudioPanel from "./SpoStudioPanel";

const container = document.getElementById("root");
if (container === null) {
  throw new Error("portal shell is missing its #root container");
}

createRoot(container).render(
  <StrictMode>
    <SpoStudioPanel />
  </StrictMode>,
);
