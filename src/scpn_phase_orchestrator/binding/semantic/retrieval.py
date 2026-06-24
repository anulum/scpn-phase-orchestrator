# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Semantic compiler retrieval evidence

"""Local, domainpack, and docs retrieval evidence ranking for the compiler."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalEvidence:
    """Local domainpack evidence used during symbolic binding generation."""

    domainpack: str
    path: str
    score: float
    matched_terms: list[str]
    summary: str
    source: str = "domainpack"
    rank: int = 0
    ranking_features: dict[str, float] = field(default_factory=dict)

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-safe retrieval evidence record.

        Returns
        -------
        dict[str, Any]
            Deterministic, JSON-safe audit mapping of the RetrievalEvidence fields.
        """
        return {
            "domainpack": self.domainpack,
            "path": self.path,
            "rank": self.rank,
            "score": self.score,
            "matched_terms": self.matched_terms,
            "summary": self.summary,
            "source": self.source,
            "ranking_features": dict(sorted(self.ranking_features.items())),
        }


def _retrieve_local_evidence(
    prompt: str,
    *,
    domainpack_root: str | Path | None,
    docs_root: str | Path | None,
    limit_per_source: int = 3,
) -> list[RetrievalEvidence]:
    """Retrieve evidence from the local store."""
    domainpack_evidence = _retrieve_domainpack_evidence(
        prompt,
        domainpack_root,
        limit=limit_per_source,
    )
    docs_evidence = _retrieve_docs_evidence(
        prompt,
        docs_root,
        limit=limit_per_source,
    )
    return _rank_retrieval_evidence([*domainpack_evidence, *docs_evidence])


def _rank_retrieval_evidence(
    evidence: list[RetrievalEvidence],
) -> list[RetrievalEvidence]:
    """Rank the retrieved evidence by relevance."""
    ranked = sorted(
        evidence,
        key=lambda item: (
            -item.score,
            -item.ranking_features.get("source_priority", 0.0),
            -item.ranking_features.get("matched_term_count", 0.0),
            -item.ranking_features.get("name_match_count", 0.0),
            item.source,
            item.domainpack,
            item.path,
        ),
    )
    return [replace(item, rank=index) for index, item in enumerate(ranked, start=1)]


def _retrieve_domainpack_evidence(
    prompt: str,
    root: str | Path | None,
    *,
    limit: int = 3,
) -> list[RetrievalEvidence]:
    """Retrieve evidence from the domainpack."""
    if root is None:
        return []
    base = Path(root)
    if not base.exists() or not base.is_dir():
        return []

    prompt_terms = _terms(prompt)
    if not prompt_terms:
        return []

    scored: list[RetrievalEvidence] = []
    for spec_path in sorted(base.glob("*/binding_spec.yaml")):
        domain_dir = spec_path.parent
        text_parts = [domain_dir.name.replace("_", " ")]
        text_parts.append(_safe_read(spec_path, max_chars=12000))
        readme_path = domain_dir / "README.md"
        if readme_path.exists():
            text_parts.append(_safe_read(readme_path, max_chars=4000))
        corpus = " ".join(text_parts).lower()
        corpus_terms = set(_terms(corpus))
        matched = sorted(prompt_terms & corpus_terms)
        if not matched:
            continue
        domain_phrase = domain_dir.name.lower().replace("_", " ")
        name_bonus = sum(1 for term in prompt_terms if term in domain_phrase)
        phrase_bonus = 2 if domain_phrase in prompt.lower() else 0
        score = (len(matched) + name_bonus + phrase_bonus) / max(
            len(prompt_terms),
            1,
        )
        scored.append(
            RetrievalEvidence(
                domainpack=domain_dir.name,
                path=str(spec_path),
                score=round(min(score, 1.0), 3),
                matched_terms=matched[:12],
                summary=_evidence_summary(domain_dir.name, matched),
                source="domainpack",
                ranking_features={
                    "matched_term_count": float(len(matched)),
                    "name_match_count": float(name_bonus),
                    "phrase_match": float(phrase_bonus > 0),
                    "prompt_term_count": float(len(prompt_terms)),
                    "source_priority": 1.0,
                    "term_density": round(
                        len(matched) / max(len(corpus_terms), 1),
                        6,
                    ),
                },
            )
        )

    return sorted(scored, key=lambda item: (-item.score, item.domainpack))[:limit]


def _retrieve_docs_evidence(
    prompt: str,
    root: str | Path | None,
    *,
    limit: int = 3,
) -> list[RetrievalEvidence]:
    """Retrieve evidence from the documentation."""
    if root is None:
        return []
    base = Path(root)
    if not base.exists() or not base.is_dir():
        return []

    prompt_terms = _terms(prompt)
    if not prompt_terms:
        return []

    scored: list[RetrievalEvidence] = []
    for doc_path in sorted(base.rglob("*.md")):
        if "internal" in doc_path.parts:
            continue
        text = _safe_read(doc_path, max_chars=20000)
        corpus_terms = set(_terms(text))
        matched = sorted(prompt_terms & corpus_terms)
        if not matched:
            continue
        title_bonus = sum(1 for term in prompt_terms if term in doc_path.stem.lower())
        score = (len(matched) + title_bonus) / max(len(prompt_terms), 1)
        scored.append(
            RetrievalEvidence(
                domainpack=doc_path.stem,
                path=str(doc_path),
                score=round(min(score, 1.0), 3),
                matched_terms=matched[:12],
                summary=_evidence_summary(doc_path.stem, matched),
                source="docs",
                ranking_features={
                    "matched_term_count": float(len(matched)),
                    "name_match_count": float(title_bonus),
                    "phrase_match": 0.0,
                    "prompt_term_count": float(len(prompt_terms)),
                    "source_priority": 0.75,
                    "term_density": round(
                        len(matched) / max(len(corpus_terms), 1),
                        6,
                    ),
                },
            )
        )
    return sorted(scored, key=lambda item: (-item.score, item.path))[:limit]


def _safe_read(path: Path, *, max_chars: int) -> str:
    """Read a file safely, returning empty on failure."""
    try:
        return path.read_text(encoding="utf-8")[:max_chars]
    except UnicodeDecodeError:
        return ""


def _terms(text: str) -> set[str]:
    """Return the search terms extracted from a query."""
    stopwords = {
        "and",
        "for",
        "from",
        "into",
        "model",
        "orchestrate",
        "phase",
        "system",
        "the",
        "under",
        "with",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9]{3,}", text.lower())
        if term not in stopwords
    }


def _evidence_summary(domainpack: str, matched_terms: list[str]) -> str:
    """Return a summary of the retrieved evidence."""
    terms = ", ".join(matched_terms[:5])
    return f"{domainpack} matched local terms: {terms}"
