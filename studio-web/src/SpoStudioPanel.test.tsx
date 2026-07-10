// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-phase-orchestrator — SpoStudioPanel render test (committed evidence)

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SpoStudioPanel } from "./SpoStudioPanel";

describe("SpoStudioPanel", () => {
  it("renders the committed evidence-coverage map at its honest boundary", () => {
    render(<SpoStudioPanel />);
    expect(
      screen.getByRole("heading", { name: "SCPN Phase Orchestrator" }),
    ).toBeDefined();
    // The regulatory disclaimer is shown, never elided.
    expect(screen.getByText(/does not constitute legal advice/i)).toBeDefined();
    // The six assurance evidence categories are each rendered as a card.
    expect(screen.getAllByRole("table").length).toBe(6);
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
