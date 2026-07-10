// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-phase-orchestrator — evidence-coverage renderer tests

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { EvidenceCoverageTable } from "./EvidenceCoverageTable";
import type { EvidenceCoveragePanelView } from "./data";

const PANEL: EvidenceCoveragePanelView = {
  schema: "spo.studio.evidence-coverage.v1",
  studio: "scpn-phase-orchestrator",
  disclaimer: "review-only evidence mapping",
  categories: [
    {
      category: "audit_logging",
      clauseCount: 2,
      addressedCount: 1,
      partiallyAddressedCount: 1,
      clauses: [
        {
          standard: "EU AI Act 2024/1689",
          clauseId: "Article 12",
          title: "Record-Keeping",
          status: "addressed",
          rationale: "hash-chained log",
        },
        {
          standard: "ISO/IEC 42001:2023",
          clauseId: "Clause 8",
          title: "Operation",
          status: "partially_addressed",
          rationale: "operational records",
        },
      ],
    },
  ],
  summary: {
    categoryCount: 1,
    clauseMappingCount: 2,
    addressedCount: 1,
    partiallyAddressedCount: 1,
    standardsCovered: ["EU AI Act 2024/1689", "ISO/IEC 42001:2023"],
  },
};

describe("EvidenceCoverageTable", () => {
  it("renders the summary counts and the standards spanned", () => {
    render(<EvidenceCoverageTable panel={PANEL} />);
    expect(screen.getByText("1 evidence categories")).toBeDefined();
    expect(screen.getByText("2 clause mappings")).toBeDefined();
    expect(
      screen.getByText("EU AI Act 2024/1689, ISO/IEC 42001:2023"),
    ).toBeDefined();
  });

  it("renders each clause with its resolved title and honest status label", () => {
    render(<EvidenceCoverageTable panel={PANEL} />);
    const table = screen.getByRole("table");
    expect(within(table).getByText("Article 12 — Record-Keeping")).toBeDefined();
    expect(within(table).getByText("Clause 8 — Operation")).toBeDefined();

    const addressed = within(table).getByText("addressed");
    const partial = within(table).getByText("partially addressed");
    expect(addressed.getAttribute("data-status")).toBe("addressed");
    expect(partial.getAttribute("data-status")).toBe("partially_addressed");
  });
});
