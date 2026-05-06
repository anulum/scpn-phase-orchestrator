# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - GitHub Action reference guard

"""Validate that pinned GitHub Action references in workflow files resolve.

The publish workflow pins actions by commit SHA for supply-chain stability.
GitHub Actions fails during job setup when a pin is not an actual commit or
tag ref in the referenced repository. This guard catches those failures in the
preflight job before expensive release jobs fan out.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USES_RE = re.compile(r"^\s*-\s*uses:\s*([^\s#]+)", re.MULTILINE)
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ActionRef:
    """A remote GitHub Action reference extracted from a workflow."""

    path: Path
    line: int
    repo: str
    ref: str


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _is_remote_github_action(spec: str) -> bool:
    """Return true for owner/repo actions, not local or docker actions."""
    if spec.startswith("./") or spec.startswith("docker://"):
        return False
    if "@" not in spec:
        return False
    repo, _ = spec.rsplit("@", 1)
    return repo.count("/") == 1


def extract_action_refs(paths: Iterable[Path]) -> list[ActionRef]:
    """Extract remote ``owner/repo@ref`` action refs from workflow files."""
    refs: list[ActionRef] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in USES_RE.finditer(text):
            spec = match.group(1)
            if not _is_remote_github_action(spec):
                continue
            repo, ref = spec.rsplit("@", 1)
            refs.append(
                ActionRef(
                    path=path,
                    line=_line_number(text, match.start()),
                    repo=repo,
                    ref=ref,
                )
            )
    return refs


def _gh_api(path: str) -> subprocess.CompletedProcess[str]:
    gh_bin = shutil.which("gh")
    if gh_bin is None:
        return subprocess.CompletedProcess(["gh", "api", path], 127, "", "gh missing")
    return subprocess.run(
        [gh_bin, "api", path],
        capture_output=True,
        text=True,
        check=False,
    )


def ref_resolves(repo: str, ref: str) -> bool:
    """Return true when ``ref`` resolves as a commit or tag in ``repo``."""
    commit = _gh_api(f"repos/{repo}/commits/{ref}")
    if commit.returncode == 0:
        return True

    tag = _gh_api(f"repos/{repo}/git/ref/tags/{ref}")
    return tag.returncode == 0


def validate_refs(refs: Sequence[ActionRef]) -> tuple[list[ActionRef], list[ActionRef]]:
    """Split refs into missing refs and non-SHA refs."""
    missing: list[ActionRef] = []
    non_sha: list[ActionRef] = []

    for action_ref in refs:
        if not FULL_SHA_RE.fullmatch(action_ref.ref):
            non_sha.append(action_ref)
        if not ref_resolves(action_ref.repo, action_ref.ref):
            missing.append(action_ref)

    return missing, non_sha


def _format_ref(action_ref: ActionRef) -> str:
    rel = (
        action_ref.path.relative_to(ROOT)
        if action_ref.path.is_relative_to(ROOT)
        else action_ref.path
    )
    return f"{rel}:{action_ref.line}: {action_ref.repo}@{action_ref.ref}"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    paths = (
        [ROOT / ".github/workflows/publish.yml"]
        if not args
        else [Path(a) for a in args]
    )

    if shutil.which("gh") is None:
        print("ERROR: GitHub CLI `gh` is required to validate action refs")
        return 1

    refs = extract_action_refs(paths)
    missing, non_sha = validate_refs(refs)

    if non_sha:
        print("ERROR: action refs must be pinned to full 40-character commit SHAs")
        for action_ref in non_sha:
            print(f"  {_format_ref(action_ref)}")
        return 1

    if missing:
        print("ERROR: action refs do not resolve in GitHub")
        for action_ref in missing:
            print(f"  {_format_ref(action_ref)}")
        return 1

    print(f"OK: {len(refs)} GitHub Action refs resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
