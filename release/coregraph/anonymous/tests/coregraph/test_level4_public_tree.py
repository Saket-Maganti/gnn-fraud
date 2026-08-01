from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from scripts.github_publish.validate_public_tree import scan


def _source_snapshot(root: Path, filename: str) -> Path:
    directory = root / "release" / "level4_source_snapshot"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", "payload-free source fixture\n")
    return path


def test_exact_checksum_verified_source_snapshots_are_the_only_zip_exception(
    tmp_path: Path,
) -> None:
    (tmp_path / "LICENSE").write_text("fixture\n", encoding="utf-8")
    snapshots = [
        _source_snapshot(tmp_path, "coregraph_source_snapshot.zip"),
        _source_snapshot(tmp_path, "curated_source_snapshot.zip"),
    ]
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in snapshots
    )
    (snapshots[0].parent / "CHECKSUMS.sha256").write_text(
        checksums, encoding="utf-8"
    )
    assert scan(tmp_path) == []

    with zipfile.ZipFile(tmp_path / "provider_archive.zip", "w") as archive:
        archive.writestr("prediction.csv", "forbidden\n")
    findings = scan(tmp_path)
    assert any(
        item.category == "duplicate_archive" and item.path == "provider_archive.zip"
        for item in findings
    )
