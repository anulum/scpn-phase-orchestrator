// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-phase-orchestrator — SpoStudioPanel (Module Federation expose)

import "./tokens.css";

import { EvidenceCoverageTable } from "./panel/EvidenceCoverageTable";
import { Unverifiable } from "./panel/Unverifiable";
import { evidenceCoverage } from "./panel/data";

/**
 * The SPO studio panel the Hub mounts through Module Federation.
 *
 * It renders the committed evidence-coverage map verbatim: how each of SPO's
 * six assurance evidence categories contributes to EU AI Act, ISO/IEC 42001 and
 * ANSI/UL 4600 clauses, at the honest `addressed` / `partially_addressed`
 * boundary. It computes nothing and upgrades nothing; a snapshot that fails its
 * guard renders as a loud `unverifiable` block.
 */
export function SpoStudioPanel() {
  return (
    <article className="spo-panel">
      <header className="spo-header">
        <h2>SCPN Phase Orchestrator</h2>
        <p className="spo-banner">
          Assurance evidence-to-clause coverage, rendered at its honest boundary.
          Partial rows are explicit, not failures — no row is upgraded to a
          conformity claim.
        </p>
      </header>
      {evidenceCoverage.ok ? (
        <>
          <p className="spo-disclaimer">{evidenceCoverage.value.disclaimer}</p>
          <EvidenceCoverageTable panel={evidenceCoverage.value} />
        </>
      ) : (
        <Unverifiable
          surface="evidence_coverage.json"
          reason={evidenceCoverage.reason}
        />
      )}
    </article>
  );
}

export default SpoStudioPanel;
