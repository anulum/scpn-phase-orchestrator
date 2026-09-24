# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spo-kernel installer verification tests

"""Prove the kernel installer installs what it built, or says it did not.

``maturin develop`` installs through uv in a uv-created environment, and uv run
from this checkout applies ``[tool.uv] exclude-dependencies``, which lists
``spo-kernel``. The install was then skipped while maturin reported success, and
an environment kept importing an older kernel. These tests use real throwaway
environments, a real cargo, and real files; nothing is substituted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "spo-kernel" / "crates" / "spo-ffi" / "Cargo.toml"

UV = shutil.which("uv")

requires_cargo = pytest.mark.skipif(
    shutil.which("cargo") is None, reason="cargo is not installed"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_install_spo_kernel_test_mod", REPO_ROOT / "tools" / "install_spo_kernel.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load()


def _plan(manifest: Path, *, release: bool = True) -> object:
    return tool.KernelInstallPlan(
        python=Path(sys.executable),
        manifest=manifest.resolve(),
        release=release,
        editable=True,
    )


def _environment(root: Path) -> tuple[Path, Path]:
    """Create a real environment and return its interpreter and site-packages."""
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root)], check=True
    )
    python = root / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    purelib = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return python, Path(purelib)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@requires_cargo
@pytest.mark.parametrize(("release", "profile"), [(True, "release"), (False, "debug")])
def test_built_extension_names_the_profile_library(release: bool, profile: str) -> None:
    name, library = tool.built_extension(_plan(MANIFEST, release=release))

    assert name == "spo_kernel"
    assert library.parent.name == profile
    assert library.parent.parent == REPO_ROOT / "spo-kernel" / "target"
    assert library.name == tool._LIBRARY_FILENAME.get(sys.platform, "lib{}.so").format(
        "spo_kernel"
    )


def test_built_extension_needs_cargo_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """An environment whose ``PATH`` holds no cargo; nothing else is altered."""
    monkeypatch.setenv("PATH", "")

    with pytest.raises(FileNotFoundError, match="cargo not found on PATH"):
        tool.built_extension(_plan(MANIFEST))


@requires_cargo
def test_built_extension_rejects_a_workspace_manifest() -> None:
    with pytest.raises(tool.KernelInstallError, match="cargo metadata does not list"):
        tool.built_extension(_plan(REPO_ROOT / "spo-kernel" / "Cargo.toml"))


@requires_cargo
def test_built_extension_rejects_a_crate_without_cdylib(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("", encoding="utf-8")
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "plain"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[lib]\ncrate-type = ["rlib"]\n',
        encoding="utf-8",
    )

    with pytest.raises(tool.KernelInstallError, match="declares no cdylib target"):
        tool.built_extension(_plan(tmp_path / "Cargo.toml"))


def test_verify_accepts_the_package_extension_that_was_built(tmp_path: Path) -> None:
    python, site = _environment(tmp_path / "env")
    package = site / "kernelpkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    installed = package / "kernelpkg.abi3.so"
    installed.write_bytes(b"new build")
    library = tmp_path / "libkernelpkg.so"
    library.write_bytes(b"new build")

    path, digest = tool.verify_installed_extension(python, "kernelpkg", library)

    assert Path(path) == installed
    assert digest == _sha256(library)


def test_verify_accepts_a_single_file_extension_module(tmp_path: Path) -> None:
    python, site = _environment(tmp_path / "env")
    installed = site / "kernelmod.abi3.so"
    installed.write_bytes(b"new build")
    library = tmp_path / "libkernelmod.so"
    library.write_bytes(b"new build")

    path, _ = tool.verify_installed_extension(python, "kernelmod", library)

    assert Path(path) == installed


def test_verify_rejects_a_stale_extension(tmp_path: Path) -> None:
    """The case maturin reported as installed: the old kernel is still there."""
    python, site = _environment(tmp_path / "env")
    package = site / "kernelpkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernelpkg.abi3.so").write_bytes(b"old build")
    library = tmp_path / "libkernelpkg.so"
    library.write_bytes(b"new build")

    with pytest.raises(tool.KernelInstallError, match=r"kernelpkg\.abi3\.so"):
        tool.verify_installed_extension(python, "kernelpkg", library)


@pytest.mark.parametrize("module", ["absentmod", "plainmod"])
def test_verify_rejects_a_module_without_the_extension(
    tmp_path: Path, module: str
) -> None:
    python, site = _environment(tmp_path / "env")
    (site / "plainmod.py").write_text("", encoding="utf-8")
    library = tmp_path / "libkernel.so"
    library.write_bytes(b"new build")

    with pytest.raises(tool.KernelInstallError, match="it has no extension"):
        tool.verify_installed_extension(python, module, library)


def test_verify_rejects_a_missing_built_library(tmp_path: Path) -> None:
    python, _ = _environment(tmp_path / "env")

    with pytest.raises(tool.KernelInstallError, match="built library not found"):
        tool.verify_installed_extension(python, "kernelpkg", tmp_path / "absent.so")


def _pinned_maturin() -> str:
    text = (REPO_ROOT / "requirements" / "ci-tools.txt").read_text(encoding="utf-8")
    match = re.search(r"^maturin==([^\s\\]+)", text, flags=re.MULTILINE)
    assert match is not None
    return f"maturin=={match.group(1)}"


@pytest.mark.integration
@requires_cargo
@pytest.mark.skipif(UV is None, reason="uv is not installed")
def test_installer_run_from_the_checkout_installs_into_a_uv_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run the installer from the repository root into a fresh uv environment.

    The environment is created by uv, as the repository's own is, so maturin
    installs through uv; the working directory is the checkout, whose uv
    configuration excludes ``spo-kernel``.
    """
    assert UV is not None
    env = tmp_path / "uvenv"
    subprocess.run(
        [UV, "venv", "--quiet", "--python", sys.executable, str(env)],
        check=True,
        cwd=tmp_path,
    )
    python = env / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    provisioned = subprocess.run(
        [UV, "pip", "install", "--offline", "--python", str(python), _pinned_maturin()],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if provisioned.returncode != 0:
        pytest.skip(
            f"{_pinned_maturin()} is not available offline: {provisioned.stderr}"
        )
    monkeypatch.chdir(REPO_ROOT)

    argv = ["--python", str(python), "--manifest", str(MANIFEST), "--debug", "--json"]
    assert tool.main(argv) == 0
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    _, library = tool.built_extension(_plan(MANIFEST, release=False))
    assert record["ok"] is True
    assert record["stdout"] == "spo_kernel"
    assert record["extension_sha256"] == _sha256(library)
    assert Path(record["extension"]).is_relative_to(env)

    assert tool.main([*argv, "--wheel-only"]) == 0
    wheel_record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert wheel_record["ok"] is True
    assert wheel_record["editable"] is False
    assert "extension" not in wheel_record


def _run_main(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    status = tool.main(argv)
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_cli_dry_run_prints_the_release_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status, out, _ = _run_main(["--dry-run", "--manifest", str(MANIFEST)], capsys)

    assert status == 0
    assert out.split() == [
        sys.executable,
        "-m",
        "maturin",
        "develop",
        "--release",
        "-m",
        str(MANIFEST),
    ]


def test_cli_check_only_prints_the_imported_module(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status, out, _ = _run_main(
        ["--check-only", "--verify-module", "json", "--manifest", str(MANIFEST)], capsys
    )

    assert status == 0
    assert out.splitlines()[-1] == "json"


def test_cli_resolves_a_relative_interpreter_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    python, _ = _environment(tmp_path / "env")
    monkeypatch.chdir(tmp_path)
    relative = python.relative_to(tmp_path)

    status, out, _ = _run_main(
        ["--dry-run", "--json", "--python", str(relative), "--manifest", str(MANIFEST)],
        capsys,
    )

    assert status == 0
    assert json.loads(out)["python"] == str(tmp_path / relative)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--release", "--debug"], "mutually exclusive"),
        (["--python", "/nonexistent/python"], "python interpreter not found"),
        (["--manifest", "/nonexistent/Cargo.toml"], "spo-ffi manifest not found"),
        (["--check-only", "--verify-module", "os;x"], "invalid module name"),
    ],
)
def test_cli_reports_errors_as_text_and_json(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--manifest", str(MANIFEST), *argv]

    status, _, err = _run_main(base, capsys)
    assert status == 1
    assert err.startswith("ERROR: ") and message in err

    status, out, _ = _run_main([*base, "--json"], capsys)
    payload = json.loads(out)
    assert status == 1
    assert set(payload) == {"ok", "error"}
    assert payload["ok"] is False
    assert message in payload["error"]


def test_script_entry_point_exits_with_the_cli_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run the file as a script; ``sys.argv`` is the real CLI input channel."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["install_spo_kernel.py", "--dry-run", "--manifest", str(MANIFEST)],
    )

    with pytest.raises(SystemExit) as exited:
        runpy.run_path(
            str(REPO_ROOT / "tools" / "install_spo_kernel.py"), run_name="__main__"
        )

    assert exited.value.code == 0
    assert "maturin develop --release" in capsys.readouterr().out
