"""Checksum-first, extraction-free access to canonical prediction archives."""

from __future__ import annotations

import hashlib
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Iterator, Mapping


CANONICAL_ARCHIVE_HASHES: dict[str, str] = {
    "dgraphfin_10seed_inductive_isolated.zip": "6ce0d2e37893a7a162d6d575347f6606eb63f90b7940cc3a259dc309cf88b8c8",
    "dgraphfin_10seed_strict_inductive.zip": "e0055d3482107d16c7d52574b0a32adc1e7ae9236b67dbdf12f57327c0e6bce5",
    "dgraphfin_10seed_transductive.zip": "6d0167aae53b681bb7ffc037b84c723869ba718f30feb955c333890bfe8783d5",
    "elliptic_10seed_inductive_isolated.zip": "20f25a1f93604ea5eb8537c8808f9b69dae2fc82eccbccfd36c50c443aee94e8",
    "elliptic_10seed_strict_inductive.zip": "24752f5ffdc082dc79ca5084701fccd04d2ac9588b4b15712598bdfe8daa1e4a",
    "elliptic_10seed_transductive.zip": "99d2f7ad1ad95fd7c30c193da9003c091b0c3fdce028dccd2bd0019f35869c08",
}


class ArchiveIntegrityError(RuntimeError):
    """An archive or member failed a fail-closed integrity check."""


def sha256_stream(handle: IO[bytes], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


@dataclass(frozen=True)
class ArchiveRecord:
    name: str
    sha256: str

    def __post_init__(self) -> None:
        if PurePosixPath(self.name).name != self.name or not self.name.endswith(".zip"):
            raise ValueError("archive record name must be a ZIP basename")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("archive record requires a lowercase SHA-256")


class ArchiveStore:
    """Read local canonical ZIPs without exposing an extraction operation."""

    def __init__(
        self,
        cache_root: Path,
        records: Mapping[str, str] | None = None,
    ) -> None:
        self.cache_root = cache_root.resolve()
        self.archives_root = self.cache_root / "archives"
        source = records or CANONICAL_ARCHIVE_HASHES
        self.records = {name: ArchiveRecord(name, digest) for name, digest in source.items()}
        self._verified: dict[str, tuple[int, int]] = {}
        self._verified_members: set[tuple[str, str, str]] = set()

    def archive_path(self, archive_name: str) -> Path:
        if archive_name not in self.records:
            raise ArchiveIntegrityError(f"archive is not canonical: {archive_name}")
        return self.archives_root / archive_name

    def verify_archive(self, archive_name: str, *, force: bool = False) -> Path:
        path = self.archive_path(archive_name)
        if not path.is_file():
            raise ArchiveIntegrityError(f"canonical archive is absent: {path}")
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if not force and self._verified.get(archive_name) == signature:
            return path
        observed = sha256_path(path)
        expected = self.records[archive_name].sha256
        if observed != expected:
            raise ArchiveIntegrityError(
                f"archive checksum mismatch for {archive_name}: expected {expected}, observed {observed}"
            )
        try:
            with zipfile.ZipFile(path) as archive:
                corrupt = archive.testzip()
        except zipfile.BadZipFile as exc:
            raise ArchiveIntegrityError(f"invalid ZIP archive {archive_name}: {exc}") from exc
        if corrupt is not None:
            raise ArchiveIntegrityError(f"ZIP CRC failure in {archive_name}: {corrupt}")
        self._verified[archive_name] = signature
        return path

    def list_members(self, archive_name: str) -> tuple[str, ...]:
        path = self.verify_archive(archive_name)
        with zipfile.ZipFile(path) as archive:
            return tuple(info.filename for info in archive.infolist() if not info.is_dir())

    @contextmanager
    def open_member(
        self,
        archive_name: str,
        member_name: str,
        *,
        expected_sha256: str,
    ) -> Iterator[IO[bytes]]:
        """Verify once by streaming, then yield an extraction-free stream."""

        if not expected_sha256:
            raise ArchiveIntegrityError("member access requires an expected SHA-256")
        path = self.verify_member(
            archive_name,
            member_name,
            expected_sha256=expected_sha256,
        )
        with zipfile.ZipFile(path) as archive:
            with archive.open(member_name, "r") as source:
                yield source

    def verify_member(
        self,
        archive_name: str,
        member_name: str,
        *,
        expected_sha256: str,
        force: bool = False,
    ) -> Path:
        """Verify an indexed member without extracting it and cache immutable success."""

        if not expected_sha256:
            raise ArchiveIntegrityError("member access requires an expected SHA-256")
        path = self.verify_archive(archive_name, force=force)
        if PurePosixPath(member_name).is_absolute() or ".." in PurePosixPath(member_name).parts:
            raise ArchiveIntegrityError(f"unsafe member path: {member_name}")
        identity = (archive_name, member_name, expected_sha256)
        if not force and identity in self._verified_members:
            return path
        try:
            with zipfile.ZipFile(path) as archive:
                with archive.open(member_name, "r") as source:
                    observed = sha256_stream(source)
        except (KeyError, zipfile.BadZipFile) as exc:
            raise ArchiveIntegrityError(
                f"cannot open {member_name} from {archive_name}: {exc}"
            ) from exc
        if observed != expected_sha256:
            raise ArchiveIntegrityError(
                f"member checksum mismatch for {archive_name}:{member_name}: "
                f"expected {expected_sha256}, observed {observed}"
            )
        self._verified_members.add(identity)
        return path
