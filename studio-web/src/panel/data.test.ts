// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-phase-orchestrator — studio-web loader guard tests

import { describe, expect, it } from "vitest";

import { EXPECTED_SCHEMA, evidenceCoverage, loadEvidenceCoverage } from "./data";

function validSnapshot(): Record<string, unknown> {
  return {
    schema: EXPECTED_SCHEMA,
    studio: "scpn-phase-orchestrator",
    disclaimer: "review-only evidence mapping",
    categories: [
      {
        category: "audit_logging",
        clause_count: 2,
        addressed_count: 1,
        partially_addressed_count: 1,
        clauses: [
          {
            standard: "EU AI Act 2024/1689",
            clause_id: "Article 12",
            title: "Record-Keeping",
            status: "addressed",
            rationale: "hash-chained log",
          },
          {
            standard: "ISO/IEC 42001:2023",
            clause_id: "Clause 8",
            title: "Operation",
            status: "partially_addressed",
            rationale: "operational records",
          },
        ],
      },
    ],
    summary: {
      category_count: 1,
      clause_mapping_count: 2,
      addressed_count: 1,
      partially_addressed_count: 1,
      standards_covered: ["EU AI Act 2024/1689", "ISO/IEC 42001:2023"],
    },
  };
}

function mutate(fn: (draft: Record<string, unknown>) => void): unknown {
  const draft = structuredClone(validSnapshot());
  fn(draft);
  return draft;
}

function firstCategory(draft: Record<string, unknown>): Record<string, unknown> {
  return (draft["categories"] as Record<string, unknown>[])[0]!;
}

function firstClause(draft: Record<string, unknown>): Record<string, unknown> {
  return (firstCategory(draft)["clauses"] as Record<string, unknown>[])[0]!;
}

function summary(draft: Record<string, unknown>): Record<string, unknown> {
  return draft["summary"] as Record<string, unknown>;
}

describe("loadEvidenceCoverage", () => {
  it("accepts the committed snapshot", () => {
    expect(evidenceCoverage.ok).toBe(true);
    if (evidenceCoverage.ok) {
      expect(evidenceCoverage.value.schema).toBe(EXPECTED_SCHEMA);
      expect(evidenceCoverage.value.categories.length).toBeGreaterThan(0);
    }
  });

  it("accepts a well-formed hand-built snapshot with both statuses", () => {
    const result = loadEvidenceCoverage(validSnapshot());
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.studio).toBe("scpn-phase-orchestrator");
      expect(result.value.categories[0]?.clauses.map((c) => c.status)).toEqual([
        "addressed",
        "partially_addressed",
      ]);
      expect(result.value.summary.standardsCovered).toHaveLength(2);
    }
  });

  const rejections: ReadonlyArray<readonly [string, unknown, string]> = [
    ["non-object payload", 42, "not an object"],
    ["array payload", [], "not an object"],
    ["wrong schema", mutate((d) => (d["schema"] = "other")), "schema is not recognised"],
    ["studio not a string", mutate((d) => (d["studio"] = 1)), "studio or disclaimer"],
    [
      "disclaimer not a string",
      mutate((d) => (d["disclaimer"] = null)),
      "studio or disclaimer",
    ],
    ["categories not an array", mutate((d) => (d["categories"] = {})), "no categories"],
    ["categories empty", mutate((d) => (d["categories"] = [])), "no categories"],
    [
      "category not a record",
      mutate((d) => ((d["categories"] as unknown[])[0] = 7)),
      "malformed category",
    ],
    [
      "category name missing",
      mutate((d) => delete firstCategory(d)["category"]),
      "malformed category",
    ],
    [
      "clause_count as a string",
      mutate((d) => (firstCategory(d)["clause_count"] = "2")),
      "malformed category",
    ],
    [
      "clause_count non-integer",
      mutate((d) => (firstCategory(d)["clause_count"] = 1.5)),
      "malformed category",
    ],
    [
      "addressed_count negative",
      mutate((d) => (firstCategory(d)["addressed_count"] = -1)),
      "malformed category",
    ],
    [
      "partially_addressed_count missing",
      mutate((d) => delete firstCategory(d)["partially_addressed_count"]),
      "malformed category",
    ],
    [
      "clauses not an array",
      mutate((d) => (firstCategory(d)["clauses"] = "nope")),
      "malformed category",
    ],
    [
      "clause not a record",
      mutate((d) => ((firstCategory(d)["clauses"] as unknown[])[0] = 3)),
      "malformed category",
    ],
    [
      "clause standard missing",
      mutate((d) => delete firstClause(d)["standard"]),
      "malformed category",
    ],
    [
      "clause clause_id missing",
      mutate((d) => delete firstClause(d)["clause_id"]),
      "malformed category",
    ],
    [
      "clause title missing",
      mutate((d) => delete firstClause(d)["title"]),
      "malformed category",
    ],
    [
      "clause rationale missing",
      mutate((d) => delete firstClause(d)["rationale"]),
      "malformed category",
    ],
    [
      "clause status not recognised",
      mutate((d) => (firstClause(d)["status"] = "maybe")),
      "malformed category",
    ],
    [
      "clause count mismatch",
      mutate((d) => (firstCategory(d)["clause_count"] = 9)),
      "malformed category",
    ],
    ["summary not a record", mutate((d) => (d["summary"] = 5)), "summary is malformed"],
    [
      "summary category_count missing",
      mutate((d) => delete summary(d)["category_count"]),
      "summary is malformed",
    ],
    [
      "summary clause_mapping_count missing",
      mutate((d) => delete summary(d)["clause_mapping_count"]),
      "summary is malformed",
    ],
    [
      "summary addressed_count missing",
      mutate((d) => delete summary(d)["addressed_count"]),
      "summary is malformed",
    ],
    [
      "summary partially_addressed_count missing",
      mutate((d) => delete summary(d)["partially_addressed_count"]),
      "summary is malformed",
    ],
    [
      "summary standards_covered not a list",
      mutate((d) => (summary(d)["standards_covered"] = "EU")),
      "summary is malformed",
    ],
    [
      "summary standards_covered has a non-string",
      mutate((d) => (summary(d)["standards_covered"] = ["EU", 9])),
      "summary is malformed",
    ],
  ];

  it.each(rejections)("rejects %s", (_name, payload, reason) => {
    const result = loadEvidenceCoverage(payload);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toContain(reason);
    }
  });
});
