# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — fetch helper for citation-only real corpora

"""Download and verify the citation-only corpora used by the early-warning capstones.

This tool fetches only data that is publicly available and redistributable under
its own terms.  The raw files are placed under ``data/``; the downstream capstones
read them but never copy them into the repository.

Supported corpora
-----------------
* ``dakos`` — Dakos et al. 2008 palaeoclimate transition records from the
  Early-Warning-Signals Toolbox dataset repository
  (``earlywarningtoolbox/datasets``).
* ``psml`` — Zheng et al. 2021 PSML 23-bus power-system co-simulation dataset
  from Zenodo record 5130612.
* ``physionet-chbmit`` — CHB-MIT scalp EEG (subject chb01) from PhysioNet.
* ``physionet-afdb`` — MIT-BIH Atrial Fibrillation Database from PhysioNet.

The CHB-MIT and AFDB PhysioNet records used here are public-access, so no
account is required for those two corpora.  Credentials are accepted for
other protected PhysioNet databases via ``--physionet-user``/
``--physionet-password`` or the environment variables
``PHYSIONET_USER``/``PHYSIONET_PASSWORD``.

Examples
--------
Download the small Dakos corpus (no credentials)::

    python tools/fetch_real_corpora.py dakos

Download the PSML archive (5.2 GB)::

    python tools/fetch_real_corpora.py psml

Download CHB-MIT and AFDB records (public access, credentials optional)::

    python tools/fetch_real_corpora.py physionet-chbmit physionet-afdb

Check what would be downloaded without writing anything::

    python tools/fetch_real_corpora.py --dry-run all
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# Corpus metadata                                                             #
# --------------------------------------------------------------------------- #

DAKOS_RECORDS = (
    "Eo_Gl",
    "Vostok1deut",
    "Vostok2deut",
    "Vostok3deut",
    "Vostok4deut",
    "GBA_temp",
    "YD2PB_grayscale",
    "terrigenous",
)

CHBMIT_RECORDS = (
    "chb01_01",
    "chb01_02",
    "chb01_03",
    "chb01_04",
    "chb01_05",
    "chb01_06",
    "chb01_07",
    "chb01_15",
    "chb01_16",
    "chb01_18",
    "chb01_21",
    "chb01_26",
)

AFDB_RECORDS = (
    "04015",
    "04126",
    "04043",
    "04048",
    "04746",
    "04908",
    "05091",
    "07879",
)

PSML_ZENODO_URL = "https://zenodo.org/records/5130612/files/PSML.zip"
PSML_ZIP_SIZE = 5_179_159_297  # bytes, as reported by Zenodo on 2026-07-09

CHBMIT_BASE = "https://physionet.org/files/chbmit/1.0.0"
AFDB_BASE = "https://physionet.org/files/afdb/1.0.0"
DAKOS_RAW_BASE = "https://raw.githubusercontent.com/earlywarningtoolbox/datasets/master"


# --------------------------------------------------------------------------- #
# Low-level download helpers                                                  #
# --------------------------------------------------------------------------- #


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"{current} B"
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {pct * 100:5.1f}%  {current:_} / {total:_} B"


def _human_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def _basic_auth_header(user: str, password: str) -> dict[str, str]:
    creds = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    return {"Authorization": f"Basic {creds}"}


def download_file(
    url: str,
    dest: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    headers: dict[str, str] | None = None,
    dry_run: bool = False,
    chunk_size: int = 262_144,
    timeout: int = 60,
    quiet: bool = False,
) -> Path:
    """Download ``url`` to ``dest`` with resume, progress, and verification.

    Parameters
    ----------
    url
        Source URL.
    dest
        Destination path.
    expected_size
        Expected file size in bytes; warned if mismatched.
    expected_sha256
        Optional hex SHA-256; verified after download.
    headers
        Extra headers, e.g. for PhysioNet Basic Auth.
    dry_run
        If True, only print what would be done and return ``dest``.
    chunk_size
        Download chunk size.
    timeout
        Socket timeout for the initial connection.
    quiet
        If True, suppress per-chunk progress bars (useful when many files
        download concurrently).

    Returns
    -------
    Path
        The destination path.
    """
    headers = dict(headers or {})
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing_size = dest.stat().st_size if dest.exists() else 0
    if existing_size and expected_size and existing_size == expected_size:
        print(f"  ✓ already complete: {dest}")
        if expected_sha256 and not _sha256_matches(dest, expected_sha256):
            print(f"  ! checksum mismatch, re-downloading: {dest}")
            existing_size = 0
            dest.unlink()
        else:
            return dest

    if dry_run:
        print(f"  would download: {url} -> {dest} ({_human_size(expected_size or 0)})")
        return dest

    req_headers = dict(headers)
    start = 0
    mode = "wb"
    if existing_size:
        # Try to resume; the server may ignore the Range header.
        req_headers["Range"] = f"bytes={existing_size}-"
        start = existing_size
        mode = "ab"
        print(f"  resuming {dest} from {_human_size(existing_size)}")
    else:
        print(f"  downloading {url}")

    request = urllib.request.Request(url, headers=req_headers)  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            total = expected_size
            if total is None and "Content-Length" in response.headers:
                reported = int(response.headers["Content-Length"])
                if start and response.status == 206:
                    total = start + reported
                else:
                    total = reported
                    if start and response.status != 206:
                        start = 0
                        mode = "wb"

            current = start
            last_print = start
            print_threshold = max(1, (total or 0) // 100) if total else 262_144
            chunks_since_print = 0
            with dest.open(mode) as fh:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    current += len(chunk)
                    chunks_since_print += 1
                    if (
                        not quiet
                        and total
                        and (
                            current - last_print >= print_threshold
                            or chunks_since_print >= 50
                        )
                    ):
                        print(
                            f"\r    {_progress_bar(current, total)}",
                            end="",
                            flush=True,
                        )
                        last_print = current
                        chunks_since_print = 0
            if not quiet:
                if total:
                    print(f"\r    {_progress_bar(current, total)}")
                else:
                    print()
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing_size:
            print(
                f"  ✓ already complete (server reports range not satisfiable): {dest}"
            )
        else:
            raise

    actual_size = dest.stat().st_size
    if expected_size and actual_size != expected_size:
        print(
            f"  ! size mismatch for {dest}: expected {expected_size}, got {actual_size}"
        )

    if expected_sha256 and not _sha256_matches(dest, expected_sha256):
        raise RuntimeError(f"checksum mismatch after download: {dest}")

    return dest


def _sha256_matches(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower() == expected.lower()


# --------------------------------------------------------------------------- #
# Corpus fetchers                                                             #
# --------------------------------------------------------------------------- #


def fetch_dakos(data_root: Path, *, dry_run: bool = False) -> None:
    """Download the Dakos et al. 2008 proxy text files."""
    dest_dir = data_root / "dakos_climate_transitions"
    print(f"Dakos palaeoclimate records -> {dest_dir}")
    for stem in DAKOS_RECORDS:
        for suffix in ("_Y1int.txt", "_Yt.txt"):
            url = f"{DAKOS_RAW_BASE}/{stem}{suffix}"
            dest = dest_dir / f"{stem}{suffix}"
            download_file(url, dest, dry_run=dry_run, timeout=30)


def fetch_psml(data_root: Path, *, dry_run: bool = False) -> None:
    """Download and extract the PSML Zenodo archive."""
    dest_dir = data_root / "psml_grid_oscillation"
    zip_path = dest_dir / "PSML.zip"
    print(f"PSML Zenodo archive -> {zip_path}")
    download_file(
        PSML_ZENODO_URL,
        zip_path,
        expected_size=PSML_ZIP_SIZE,
        dry_run=dry_run,
        timeout=120,
        chunk_size=256 * 1024,
    )
    if dry_run:
        return

    if not zip_path.exists() or zip_path.stat().st_size != PSML_ZIP_SIZE:
        print("  ! PSML zip incomplete; skipping extraction")
        return

    extract_marker = dest_dir / ".extracted"
    if extract_marker.exists():
        print("  ✓ already extracted")
        return

    print("  extracting PSML.zip (this may take a few minutes)")
    unzip_path = shutil.which("unzip")
    if unzip_path:
        subprocess.run(
            [unzip_path, "-q", str(zip_path), "-d", str(dest_dir)],
            check=True,
        )
    else:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    extract_marker.write_text("extracted\n", encoding="utf-8")
    print(f"  ✓ extracted to {dest_dir}")


def _physionet_auth(args: argparse.Namespace) -> dict[str, str]:
    """Return Basic-Auth headers if credentials are supplied, otherwise {}.

    CHB-MIT and AFDB are public-access PhysioNet records, so anonymous
    downloads are allowed; credentials are only needed for protected corpora.
    """
    user = args.physionet_user or os.environ.get("PHYSIONET_USER")
    password = args.physionet_password or os.environ.get("PHYSIONET_PASSWORD")
    if user and password:
        return _basic_auth_header(user, password)
    return {}


def _download_chbmit_record(
    record: str, *, base: str, dest_dir: Path, headers: dict[str, str], dry_run: bool
) -> None:
    """Download a single CHB-MIT record (worker for the parallel fetch)."""
    url = f"{base}/chb01/{record}.edf"
    dest = dest_dir / f"{record}.edf"
    download_file(
        url,
        dest,
        headers=headers,
        dry_run=dry_run,
        timeout=120,
        chunk_size=1_048_576,
        quiet=True,
    )


def fetch_physionet_chbmit(
    data_root: Path, *, headers: dict[str, str] | None = None, dry_run: bool = False
) -> None:
    """Download CHB-MIT subject chb01 records used by the EEG capstone.

    Large EDFs are fetched in parallel with a small worker pool; the server
    connection is the bottleneck, so concurrency modestly increases throughput
    while still respecting the remote host.
    """
    dest_dir = data_root / "chb01_seizures"
    print(f"CHB-MIT scalp EEG -> {dest_dir}")
    headers = headers or {}
    workers = min(4, len(CHBMIT_RECORDS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _download_chbmit_record,
                record,
                base=CHBMIT_BASE,
                dest_dir=dest_dir,
                headers=headers,
                dry_run=dry_run,
            )
            for record in CHBMIT_RECORDS
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def fetch_physionet_afdb(
    data_root: Path, *, headers: dict[str, str] | None = None, dry_run: bool = False
) -> None:
    """Download MIT-BIH AFDB records used by the cardiac capstone."""
    dest_dir = data_root / "afdb_atrial_fibrillation"
    print(f"MIT-BIH AFDB -> {dest_dir}")
    headers = headers or {}
    for record in AFDB_RECORDS:
        for ext in (".dat", ".hea", ".atr"):
            url = f"{AFDB_BASE}/{record}{ext}"
            dest = dest_dir / f"{record}{ext}"
            download_file(url, dest, headers=headers, dry_run=dry_run, timeout=120)


# --------------------------------------------------------------------------- #
# Verification                                                                #
# --------------------------------------------------------------------------- #


def verify_corpus(data_root: Path) -> dict[str, dict[str, object]]:
    """Return a status report for each expected corpus subdirectory."""
    report: dict[str, dict[str, object]] = {}
    dakos_dir = data_root / "dakos_climate_transitions"
    dakos_files = [
        dakos_dir / f"{stem}{suffix}"
        for stem in DAKOS_RECORDS
        for suffix in ("_Y1int.txt", "_Yt.txt")
    ]
    report["dakos"] = {
        "expected": len(dakos_files),
        "present": sum(1 for f in dakos_files if f.exists()),
        "size_bytes": sum(
            f.stat().st_size for f in dakos_dir.glob("*.txt") if f.exists()
        ),
    }

    psml_dir = data_root / "psml_grid_oscillation"
    psml_zip = psml_dir / "PSML.zip"
    report["psml"] = {
        "zip_present": psml_zip.exists(),
        "zip_size_bytes": psml_zip.stat().st_size if psml_zip.exists() else 0,
        "extracted": (psml_dir / ".extracted").exists(),
        "scenario_count": len(list((psml_dir).rglob("trans.csv")))
        if psml_dir.exists()
        else 0,
    }

    chbmit_dir = data_root / "chb01_seizures"
    chbmit_files = [chbmit_dir / f"{record}.edf" for record in CHBMIT_RECORDS]
    report["chbmit"] = {
        "expected": len(chbmit_files),
        "present": sum(1 for f in chbmit_files if f.exists()),
        "size_bytes": sum(f.stat().st_size for f in chbmit_files if f.exists()),
    }

    afdb_dir = data_root / "afdb_atrial_fibrillation"
    afdb_files = [
        afdb_dir / f"{record}{ext}"
        for record in AFDB_RECORDS
        for ext in (".dat", ".hea", ".atr")
    ]
    report["afdb"] = {
        "expected": len(afdb_files),
        "present": sum(1 for f in afdb_files if f.exists()),
        "size_bytes": sum(f.stat().st_size for f in afdb_files if f.exists()),
    }

    return report


def print_report(report: dict[str, dict[str, object]]) -> None:
    print("\nCorpus verification report:")
    for name, info in report.items():
        line = f"  {name:12}"
        if "present" in info and "expected" in info:
            line += f"  {info['present']}/{info['expected']} files"
        if "zip_present" in info:
            line += f"  zip={'yes' if info['zip_present'] else 'no'}"
        if "extracted" in info:
            line += f"  extracted={'yes' if info['extracted'] else 'no'}"
        if "scenario_count" in info:
            line += f"  scenarios={info['scenario_count']}"
        if "size_bytes" in info:
            line += f"  ({_human_size(int(info['size_bytes']))})"
        print(line)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "commands",
        nargs="+",
        choices=(
            "dakos",
            "psml",
            "physionet-chbmit",
            "physionet-afdb",
            "all",
            "verify",
        ),
        help="corpus action(s) to perform",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="root directory for raw corpora (default: ./data)",
    )
    parser.add_argument(
        "--physionet-user",
        default=os.environ.get("PHYSIONET_USER"),
        help="PhysioNet username (or PHYSIONET_USER env var)",
    )
    parser.add_argument(
        "--physionet-password",
        default=os.environ.get("PHYSIONET_PASSWORD"),
        help="PhysioNet password (or PHYSIONET_PASSWORD env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be downloaded without writing files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    print(f"Data root: {data_root}")

    commands = set(args.commands)
    if "all" in commands:
        commands = {"dakos", "psml", "physionet-chbmit", "physionet-afdb"}
    run_verify = "verify" in commands
    commands.discard("verify")

    auth = _physionet_auth(args)

    for command in sorted(commands):
        print()
        if command == "dakos":
            fetch_dakos(data_root, dry_run=args.dry_run)
        elif command == "psml":
            fetch_psml(data_root, dry_run=args.dry_run)
        elif command == "physionet-chbmit":
            fetch_physionet_chbmit(data_root, headers=auth, dry_run=args.dry_run)
        elif command == "physionet-afdb":
            fetch_physionet_afdb(data_root, headers=auth, dry_run=args.dry_run)

    if run_verify or not args.dry_run:
        print()
        report = verify_corpus(data_root)
        print_report(report)

    if args.dry_run:
        print("\nDry run complete; no files were written.")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
