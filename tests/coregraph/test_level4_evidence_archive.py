from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from coregraph.evidence.archive_store import ArchiveIntegrityError, ArchiveStore
from coregraph.evidence.cache_validation import validate_cache
from coregraph.evidence.member_index import MemberIndex, MemberRecord
from coregraph.evidence.prediction_reader import PredictionReader


def _archive(tmp_path: Path) -> tuple[Path, str, str, str]:
    cache = tmp_path / "cache"
    archives = cache / "archives"
    archives.mkdir(parents=True)
    member = "predictions/demo.csv"
    payload = (
        "node_id,score,y_true,split,label_known\n"
        "a,0.1,0,train,true\n"
        "b,0.8,1,test,true\n"
        "c,0.2,-1,test,false\n"
        "d,0.7,1,test,true\n"
    ).encode()
    path = archives / "demo.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, payload)
    return cache, hashlib.sha256(path.read_bytes()).hexdigest(), member, hashlib.sha256(payload).hexdigest()


def test_archive_store_and_prediction_reader_stream_without_extraction(tmp_path: Path) -> None:
    cache, archive_hash, member, member_hash = _archive(tmp_path)
    store = ArchiveStore(cache, {"demo.zip": archive_hash})
    assert store.list_members("demo.zip") == (member,)
    index = MemberIndex(
        [MemberRecord("demo", "strict", "mlp", 1, "demo.zip", member, member_hash)]
    )
    reader = PredictionReader(store, index)
    chunks = list(
        reader.iter_chunks(
            dataset="demo",
            protocol="strict",
            expert="mlp",
            seed=1,
            chunk_size=1,
            id_column="node_id",
            require_monotonic_ids=True,
        )
    )
    assert [[row["node_id"] for row in chunk.rows] for chunk in chunks] == [["b"], ["d"]]
    assert not (cache / "predictions").exists()
    assert validate_cache(cache, expected={"demo.zip": archive_hash})[0].verified


def test_archive_and_member_integrity_fail_closed(tmp_path: Path) -> None:
    cache, archive_hash, member, member_hash = _archive(tmp_path)
    store = ArchiveStore(cache, {"demo.zip": archive_hash})
    with pytest.raises(ArchiveIntegrityError, match="member checksum mismatch"):
        with store.open_member("demo.zip", member, expected_sha256="0" * 64):
            pass
    with pytest.raises(ArchiveIntegrityError, match="unsafe member path"):
        with store.open_member("demo.zip", "../demo.csv", expected_sha256=member_hash):
            pass
    path = cache / "archives" / "demo.zip"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ArchiveIntegrityError, match="archive checksum mismatch"):
        store.verify_archive("demo.zip", force=True)


def test_member_index_csv_and_canonical_grid(tmp_path: Path) -> None:
    records = [
        MemberRecord(dataset, protocol, expert, seed, "a.zip", f"{dataset}/{protocol}/{expert}/{seed}.csv", "a" * 64)
        for dataset in ("dgraphfin", "elliptic")
        for protocol in ("isolated_inductive", "strict_inductive", "transductive_structure")
        for expert in ("feature_mlp", "gcn", "graphsage")
        for seed in range(1, 11)
    ]
    index = MemberIndex(records)
    index.validate_canonical_grid()
    assert index.locate("elliptic", "strict_inductive", "gcn", 5).seed == 5
    path = tmp_path / "members.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "protocol", "expert", "seed", "archive_name", "member_name", "member_sha256"))
        writer.writeheader()
        writer.writerow({"dataset": "d", "protocol": "p", "expert": "e", "seed": 1, "archive_name": "a.zip", "member_name": "m.csv", "member_sha256": "b" * 64})
    assert MemberIndex.from_csv(path).locate("d", "p", "e", 1).member_name == "m.csv"
    with pytest.raises(ValueError, match="duplicate member coordinate"):
        MemberIndex([records[0], records[0]])


def test_prediction_schema_and_label_known_validation(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "archives").mkdir(parents=True)
    payload = b"score,y_true,split,label_known\n0.2,0,test,maybe\n"
    path = cache / "archives" / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bad.csv", payload)
    store = ArchiveStore(cache, {"bad.zip": hashlib.sha256(path.read_bytes()).hexdigest()})
    index = MemberIndex([MemberRecord("d", "p", "e", 1, "bad.zip", "bad.csv", hashlib.sha256(payload).hexdigest())])
    with pytest.raises(ArchiveIntegrityError, match="invalid label_known"):
        list(PredictionReader(store, index).iter_chunks(dataset="d", protocol="p", expert="e", seed=1))
    assert validate_cache(tmp_path / "missing", expected={"bad.zip": "0" * 64})[0].status == "BLOCKED_ARCHIVE_ABSENT"
