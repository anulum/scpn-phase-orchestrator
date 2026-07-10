// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-phase-orchestrator — SpoStudioPanel fail-closed branch test

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// Force the committed snapshot to have failed its guard so the panel must render
// the loud fail-closed block instead of any partial or blank surface.
vi.mock("./panel/data", () => ({
  evidenceCoverage: { ok: false, reason: "snapshot schema is not recognised" },
}));

import { SpoStudioPanel } from "./SpoStudioPanel";

describe("SpoStudioPanel (fail-closed)", () => {
  it("renders an unverifiable block when the snapshot fails its guard", () => {
    render(<SpoStudioPanel />);
    const alert = screen.getByRole("alert");
    expect(alert.textContent).toContain("unverifiable");
    expect(alert.textContent).toContain("evidence_coverage.json");
    expect(alert.textContent).toContain("snapshot schema is not recognised");
    expect(screen.queryByRole("table")).toBeNull();
  });
});
