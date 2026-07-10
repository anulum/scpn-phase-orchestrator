# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — choosing-an-engine guide drift guard

"""Keep the engine shortlist guide in sync with the actual UPDE engine surface.

The ``docs/guide/choosing_an_engine.md`` guide promises to be *the* shortlist of
every phase-dynamics engine. This drift guard fails if a new engine is added to
the ``upde`` public facade without appearing in the guide (or if the guide names
an engine the facade no longer exports), so the shortlist can never silently rot.
"""

from __future__ import annotations

from pathlib import Path

import scpn_phase_orchestrator.upde as upde

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide" / "choosing_an_engine.md"


def _engine_names() -> set[str]:
    return {name for name in upde.__all__ if name.endswith("Engine")}


def test_guide_lists_every_public_upde_engine() -> None:
    doc = GUIDE.read_text(encoding="utf-8")
    missing = {name for name in _engine_names() if name not in doc}
    assert not missing, f"engines missing from the choosing-an-engine guide: {missing}"


def test_guide_names_only_real_engines() -> None:
    # Every ``*Engine`` token in the guide's tables must be a real public engine,
    # so the shortlist never advertises a renamed or removed engine.
    import re

    doc = GUIDE.read_text(encoding="utf-8")
    cited = set(re.findall(r"`([A-Za-z0-9]+Engine)`", doc))
    unknown = cited - _engine_names()
    assert not unknown, f"guide cites non-existent engines: {unknown}"


def test_guide_states_the_golden_path() -> None:
    doc = GUIDE.read_text(encoding="utf-8")
    assert "UPDEEngine" in doc
    assert "simulate()" in doc
    # The honesty note must keep the validated-niche caveat.
    assert "modal damping" in doc
