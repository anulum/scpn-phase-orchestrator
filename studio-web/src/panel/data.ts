// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-phase-orchestrator — studio-web committed-evidence loader (fail-closed)

/**
 * Typed, fail-closed view over the committed evidence-coverage snapshot the
 * panel renders. The snapshot is produced by the Python single source of truth
 * (`scpn_phase_orchestrator.studio.panel_data`) and kept in lock-step by a
 * drift-guard test, so this loader only has to defend the wire shape: a
 * malformed snapshot renders as a loud `unverifiable` block, never as a silent
 * blank or a downgraded card.
 */

import evidenceCoverageJson from "./evidence_coverage.json";

export const EXPECTED_SCHEMA = "spo.studio.evidence-coverage.v1";

export type ClauseStatus = "addressed" | "partially_addressed";

export interface ClauseView {
  readonly standard: string;
  readonly clauseId: string;
  readonly title: string;
  readonly status: ClauseStatus;
  readonly rationale: string;
}

export interface CategoryView {
  readonly category: string;
  readonly clauseCount: number;
  readonly addressedCount: number;
  readonly partiallyAddressedCount: number;
  readonly clauses: readonly ClauseView[];
}

export interface SummaryView {
  readonly categoryCount: number;
  readonly clauseMappingCount: number;
  readonly addressedCount: number;
  readonly partiallyAddressedCount: number;
  readonly standardsCovered: readonly string[];
}

export interface EvidenceCoveragePanelView {
  readonly schema: string;
  readonly studio: string;
  readonly disclaimer: string;
  readonly categories: readonly CategoryView[];
  readonly summary: SummaryView;
}

export type Loaded<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly reason: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function str(obj: Record<string, unknown>, key: string): string | null {
  const value = obj[key];
  return typeof value === "string" ? value : null;
}

function count(obj: Record<string, unknown>, key: string): number | null {
  const value = obj[key];
  return typeof value === "number" && Number.isInteger(value) && value >= 0
    ? value
    : null;
}

function stringList(value: unknown): readonly string[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  return value.every((item) => typeof item === "string")
    ? (value as string[])
    : null;
}

function parseClause(raw: unknown): ClauseView | null {
  if (!isRecord(raw)) {
    return null;
  }
  const standard = str(raw, "standard");
  const clauseId = str(raw, "clause_id");
  const title = str(raw, "title");
  const status = str(raw, "status");
  const rationale = str(raw, "rationale");
  if (
    standard === null ||
    clauseId === null ||
    title === null ||
    rationale === null
  ) {
    return null;
  }
  if (status !== "addressed" && status !== "partially_addressed") {
    return null;
  }
  return { standard, clauseId, title, status, rationale };
}

function parseCategory(raw: unknown): CategoryView | null {
  if (!isRecord(raw)) {
    return null;
  }
  const category = str(raw, "category");
  const clauseCount = count(raw, "clause_count");
  const addressedCount = count(raw, "addressed_count");
  const partiallyAddressedCount = count(raw, "partially_addressed_count");
  const clausesRaw = raw["clauses"];
  if (
    category === null ||
    clauseCount === null ||
    addressedCount === null ||
    partiallyAddressedCount === null ||
    !Array.isArray(clausesRaw)
  ) {
    return null;
  }
  const clauses: ClauseView[] = [];
  for (const entry of clausesRaw) {
    const clause = parseClause(entry);
    if (clause === null) {
      return null;
    }
    clauses.push(clause);
  }
  if (clauses.length !== clauseCount) {
    return null;
  }
  return {
    category,
    clauseCount,
    addressedCount,
    partiallyAddressedCount,
    clauses,
  };
}

function parseSummary(raw: unknown): SummaryView | null {
  if (!isRecord(raw)) {
    return null;
  }
  const categoryCount = count(raw, "category_count");
  const clauseMappingCount = count(raw, "clause_mapping_count");
  const addressedCount = count(raw, "addressed_count");
  const partiallyAddressedCount = count(raw, "partially_addressed_count");
  const standardsCovered = stringList(raw["standards_covered"]);
  if (
    categoryCount === null ||
    clauseMappingCount === null ||
    addressedCount === null ||
    partiallyAddressedCount === null ||
    standardsCovered === null
  ) {
    return null;
  }
  return {
    categoryCount,
    clauseMappingCount,
    addressedCount,
    partiallyAddressedCount,
    standardsCovered,
  };
}

/**
 * Parse an evidence-coverage snapshot into a typed view or a fail-closed reason.
 */
export function loadEvidenceCoverage(
  raw: unknown,
): Loaded<EvidenceCoveragePanelView> {
  if (!isRecord(raw)) {
    return { ok: false, reason: "snapshot is not an object" };
  }
  if (raw["schema"] !== EXPECTED_SCHEMA) {
    return { ok: false, reason: "snapshot schema is not recognised" };
  }
  const studio = str(raw, "studio");
  const disclaimer = str(raw, "disclaimer");
  if (studio === null || disclaimer === null) {
    return { ok: false, reason: "snapshot studio or disclaimer is malformed" };
  }
  const categoriesRaw = raw["categories"];
  if (!Array.isArray(categoriesRaw) || categoriesRaw.length === 0) {
    return { ok: false, reason: "snapshot carries no categories" };
  }
  const categories: CategoryView[] = [];
  for (const entry of categoriesRaw) {
    const category = parseCategory(entry);
    if (category === null) {
      return { ok: false, reason: "snapshot carries a malformed category" };
    }
    categories.push(category);
  }
  const summary = parseSummary(raw["summary"]);
  if (summary === null) {
    return { ok: false, reason: "snapshot summary is malformed" };
  }
  return {
    ok: true,
    value: { schema: EXPECTED_SCHEMA, studio, disclaimer, categories, summary },
  };
}

/** The committed evidence-coverage snapshot, parsed once at module load. */
export const evidenceCoverage: Loaded<EvidenceCoveragePanelView> =
  loadEvidenceCoverage(evidenceCoverageJson);
