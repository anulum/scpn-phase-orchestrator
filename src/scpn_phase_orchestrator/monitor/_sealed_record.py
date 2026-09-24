# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — canonical record hashing and sealed-JSON loading

"""Canonical-JSON record hashing and verified loading of sealed JSON artefacts.

This lives in the core boundary so core monitors can verify a sealed operating
point without importing the runtime ``assurance`` package, which re-exports
these functions for its own records.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

__all__ = ["canonical_record_hash", "load_sealed_json"]


def canonical_record_hash(record: Mapping[str, object]) -> str:
    """Return the SHA-256 of a record under canonical JSON serialisation.

    The canonical form sorts object keys, removes incidental whitespace, and
    rejects non-finite numbers so every accepted record is strict JSON rather
    than Python's extended ``NaN`` / ``Infinity`` dialect.

    Parameters
    ----------
    record:
        A JSON-safe mapping.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.

    Raises
    ------
    ValueError
        If the record contains ``NaN`` or infinite numbers.
    """
    try:
        serialised = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ValueError as exc:
        if "Out of range float values" in str(exc):
            raise ValueError("record must contain only finite JSON numbers") from exc
        raise
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def load_sealed_json(path: str | Path) -> dict[str, object]:
    """Load a sealed JSON artefact and verify its ``content_hash``, fail-closed.

    Parameters
    ----------
    path : str | Path
        Path to a JSON object carrying a ``content_hash`` field computed by
        :func:`canonical_record_hash` over the rest of the object.

    Returns
    -------
    dict[str, object]
        The verified payload, ``content_hash`` included.

    Raises
    ------
    ValueError
        If the payload is not a JSON object, carries no ``content_hash``, or
        the hash does not recompute from the record.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence must be a JSON object; refusing to trust it")
    record = copy.deepcopy(payload)
    sealed = record.pop("content_hash", None)
    if not isinstance(sealed, str):
        raise ValueError("evidence carries no content_hash; refusing to trust it")
    if canonical_record_hash(record) != sealed:
        raise ValueError(
            "evidence content_hash does not recompute from the record; "
            "refusing to configure a monitor from a tampered artefact"
        )
    return payload
