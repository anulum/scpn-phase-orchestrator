# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reproducible release builder tests

"""Fail-closed tests for deterministic wheel and sdist assembly."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "build_reproducible_release.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_build_reproducible_release_test_mod", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()


def _raw_sdist(
    path: Path,
    *,
    mtime: int = 100,
    reverse: bool = False,
    member_type: bytes | None = None,
) -> None:
    entries = [
        ("package-1.0.0/", b""),
        ("package-1.0.0/module.py", b"VALUE = 1\n"),
    ]
    if reverse:
        entries.reverse()
    with tarfile.open(path, "w:gz") as archive:
        for index, (name, payload) in enumerate(entries):
            info = tarfile.TarInfo(name)
            info.mtime = mtime + index
            info.uid = 1000 + index
            info.gid = 2000 + index
            info.uname = "builder"
            info.gname = "group"
            if name.endswith("/"):
                info.type = tarfile.DIRTYPE
                info.mode = 0o700
                archive.addfile(info)
            else:
                info.type = member_type or tarfile.REGTYPE
                info.mode = 0o664
                info.size = len(payload)
                if info.isreg():
                    archive.addfile(info, io.BytesIO(payload))
                else:
                    archive.addfile(info)


def test_canonical_sdist_is_byte_reproducible_and_metadata_normalized(
    tmp_path: Path,
) -> None:
    first_raw = tmp_path / "first.tar.gz"
    second_raw = tmp_path / "second.tar.gz"
    first = tmp_path / "first-canonical.tar.gz"
    second = tmp_path / "second-canonical.tar.gz"
    _raw_sdist(first_raw, mtime=100, reverse=False)
    _raw_sdist(second_raw, mtime=999, reverse=True)

    mod.canonicalize_sdist(first_raw, first, epoch=123456789)
    mod.canonicalize_sdist(second_raw, second, epoch=123456789)

    assert first.read_bytes() == second.read_bytes()
    assert mod._sha256(first) == mod._sha256(second)
    assert int.from_bytes(first.read_bytes()[4:8], "little") == 123456789
    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == [
        "package-1.0.0",
        "package-1.0.0/module.py",
    ]
    assert all(member.mtime == 123456789 for member in members)
    assert all(member.uid == member.gid == 0 for member in members)
    assert all(member.uname == member.gname == "" for member in members)
    assert members[0].mode == 0o755
    assert members[1].mode == 0o644


@pytest.mark.parametrize(
    "name",
    ("", ".", "/absolute", "../escape", "root/../escape", "root//file"),
)
def test_member_path_validation_refuses_unsafe_or_noncanonical_names(name: str) -> None:
    with pytest.raises(ValueError, match="sdist member path"):
        mod._safe_member_name(name)


def test_canonical_sdist_refuses_empty_unsupported_and_existing_output(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.tar.gz"
    with tarfile.open(empty, "w:gz"):
        pass
    with pytest.raises(ValueError, match="archive is empty"):
        mod.canonicalize_sdist(empty, tmp_path / "empty-out.tar.gz", epoch=1)

    link = tmp_path / "link.tar.gz"
    _raw_sdist(link, member_type=tarfile.SYMTYPE)
    with pytest.raises(ValueError, match="unsupported sdist member"):
        mod.canonicalize_sdist(link, tmp_path / "link-out.tar.gz", epoch=1)

    raw = tmp_path / "raw.tar.gz"
    destination = tmp_path / "existing.tar.gz"
    _raw_sdist(raw)
    destination.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="already exists"):
        mod.canonicalize_sdist(raw, destination, epoch=1)
    assert destination.read_bytes() == b"preserve"


@pytest.mark.parametrize("value", ("", "0", "-1", "not-a-number"))
def test_source_date_epoch_refuses_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        mod.resolve_source_date_epoch(value)


def test_source_date_epoch_precedence_and_git_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert mod.resolve_source_date_epoch("10", environment={}) == 10
    assert (
        mod.resolve_source_date_epoch(None, environment={"SOURCE_DATE_EPOCH": "20"})
        == 20
    )

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess(args[0], 0, stdout="30\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod.resolve_source_date_epoch(None, environment={}, root=tmp_path) == 30

    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="git executable"):
        mod.resolve_source_date_epoch(None, environment={}, root=tmp_path)


def test_canonical_sdist_refuses_unreadable_regular_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"placeholder")
    original_open = mod.tarfile.open

    class UnreadableArchive:
        def __enter__(self) -> UnreadableArchive:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getmembers(self) -> list[tarfile.TarInfo]:
            member = tarfile.TarInfo("package-1.0.0/module.py")
            member.type = tarfile.REGTYPE
            return [member]

        def extractfile(self, member: tarfile.TarInfo) -> None:
            return None

    def fake_tar_open(name: object, mode: str, **kwargs: object) -> object:
        if mode == "r:gz":
            return UnreadableArchive()
        return original_open(name, mode, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(mod.tarfile, "open", fake_tar_open)
    with pytest.raises(ValueError, match="cannot be read"):
        mod.canonicalize_sdist(source, tmp_path / "output.tar.gz", epoch=1)


def _fake_build_run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    check: bool,
) -> subprocess.CompletedProcess[str]:
    assert cwd.is_dir()
    assert env["SOURCE_DATE_EPOCH"] == "123"
    assert check is True
    output = Path(command[command.index("--outdir") + 1])
    if "--sdist" in command:
        _raw_sdist(output / "package-1.0.0.tar.gz")
    if "--wheel" in command:
        (output / "package-1.0.0-py3-none-any.whl").write_bytes(b"wheel")
    return subprocess.CompletedProcess(command, 0)


def test_build_release_artifacts_selects_and_canonicalizes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod.subprocess, "run", _fake_build_run)

    artifacts = mod.build_release_artifacts(
        tmp_path / "dist",
        epoch=123,
        build_sdist=True,
        build_wheel=True,
        root=tmp_path,
    )

    assert [path.name for path in artifacts] == [
        "package-1.0.0-py3-none-any.whl",
        "package-1.0.0.tar.gz",
    ]
    assert artifacts[0].read_bytes() == b"wheel"
    with tarfile.open(artifacts[1], "r:gz") as archive:
        assert all(member.mtime == 123 for member in archive.getmembers())


def test_build_release_artifacts_refuses_empty_selection_unexpected_set_and_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        mod.build_release_artifacts(
            tmp_path,
            epoch=123,
            build_sdist=False,
            build_wheel=False,
        )

    def empty_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(mod.subprocess, "run", empty_run)
    with pytest.raises(ValueError, match="unexpected release artifact set"):
        mod.build_release_artifacts(
            tmp_path / "empty",
            epoch=123,
            build_sdist=True,
            build_wheel=False,
            root=tmp_path,
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_build_run)
    destination = tmp_path / "collision"
    destination.mkdir()
    wheel = destination / "package-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="already exists"):
        mod.build_release_artifacts(
            destination,
            epoch=123,
            build_sdist=False,
            build_wheel=True,
            root=tmp_path,
        )
    assert wheel.read_bytes() == b"preserve"


def test_main_prints_artifact_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(mod.subprocess, "run", _fake_build_run)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    output = tmp_path / "dist"

    assert mod.main(["--outdir", str(output), "--source-date-epoch", "123"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert all(str(output) in line for line in lines)


def test_tag_workflows_use_the_canonical_builder_for_every_python_artifact() -> None:
    root = Path(__file__).resolve().parents[1]
    publish = (root / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    release = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert publish.count("tools/build_reproducible_release.py") == 4
    assert "--sdist-only --outdir dist" in publish
    assert "--wheel-only --outdir dist" in publish
    assert "cmp dist/*.tar.gz dist-recheck/*.tar.gz" in publish
    assert "cmp dist/*.whl dist-recheck/*.whl" in publish
    assert "python -m build --sdist" not in publish
    assert "python -m build --wheel" not in publish
    assert release.count("tools/build_reproducible_release.py") == 2
    assert "--sdist-only --outdir dist" in release
    assert "cmp dist/*.tar.gz dist-recheck/*.tar.gz" in release
    assert "python -m build --sdist" not in release
