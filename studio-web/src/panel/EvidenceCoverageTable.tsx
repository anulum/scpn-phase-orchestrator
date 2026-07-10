// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-phase-orchestrator — studio-web evidence-coverage renderer

import type {
  CategoryView,
  ClauseStatus,
  EvidenceCoveragePanelView,
} from "./data";

const STATUS_LABELS: Record<ClauseStatus, string> = {
  addressed: "addressed",
  partially_addressed: "partially addressed",
};

function CategoryCard({ category }: { category: CategoryView }) {
  return (
    <section className="spo-category">
      <h3>
        {category.category} — {category.addressedCount} addressed,{" "}
        {category.partiallyAddressedCount} partial ({category.clauseCount}{" "}
        clauses)
      </h3>
      <table className="spo-clauses">
        <thead>
          <tr>
            <th scope="col">Standard</th>
            <th scope="col">Clause</th>
            <th scope="col">Status</th>
            <th scope="col">Rationale</th>
          </tr>
        </thead>
        <tbody>
          {category.clauses.map((clause) => (
            <tr key={`${clause.standard}::${clause.clauseId}`}>
              <td>{clause.standard}</td>
              <td>
                {clause.clauseId} — {clause.title}
              </td>
              <td className="spo-status" data-status={clause.status}>
                {STATUS_LABELS[clause.status]}
              </td>
              <td>{clause.rationale}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

/**
 * Render the evidence-coverage map: one card per assurance evidence category,
 * each listing the regulatory clauses it contributes to and whether the
 * contribution is addressed or partially addressed.
 */
export function EvidenceCoverageTable({
  panel,
}: {
  panel: EvidenceCoveragePanelView;
}) {
  return (
    <div className="spo-coverage">
      <ul className="spo-summary">
        <li>{panel.summary.categoryCount} evidence categories</li>
        <li>{panel.summary.clauseMappingCount} clause mappings</li>
        <li>{panel.summary.addressedCount} addressed</li>
        <li>{panel.summary.partiallyAddressedCount} partially addressed</li>
        <li>{panel.summary.standardsCovered.join(", ")}</li>
      </ul>
      {panel.categories.map((category) => (
        <CategoryCard key={category.category} category={category} />
      ))}
    </div>
  );
}
