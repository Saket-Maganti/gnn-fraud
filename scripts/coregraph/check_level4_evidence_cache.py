#!/usr/bin/env python3
"""Run an SSD-free checksum and representative-member evidence smoke."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.evidence import ArchiveStore, MemberIndex, PredictionReader, validate_cache
from coregraph.io.path_resolution import resolve_paths


def main() -> int:
    paths = resolve_paths(start=ROOT)
    index_path = paths.evidence_cache / "indexes" / "RB09V3_MEMBER_INDEX.csv"
    failures: list[str] = []
    validations = validate_cache(paths.evidence_cache)
    if len(validations) != 6 or any(
        not item.verified or item.member_count != 31 for item in validations
    ):
        failures.append("canonical_archive_validation")
    try:
        index = MemberIndex.from_csv(index_path)
        index.validate_canonical_grid()
    except (FileNotFoundError, ValueError) as exc:
        failures.append(f"member_index:{exc}")
        index = MemberIndex(())
    store = ArchiveStore(paths.evidence_cache)
    reader = PredictionReader(store, index)
    by_archive = defaultdict(list)
    for record in index.records:
        by_archive[record.archive_name].append(record)
    representative_rows: list[dict[str, object]] = []
    for archive_name in sorted(by_archive):
        record = sorted(by_archive[archive_name], key=lambda item: item.coordinate)[0]
        try:
            chunks = reader.iter_chunks(
                dataset=record.dataset,
                protocol=record.protocol,
                expert=record.expert,
                seed=record.seed,
                splits=("test",),
                require_label_known=True,
                chunk_size=2,
            )
            first = next(chunks)
            chunks.close()
        except (RuntimeError, StopIteration, ValueError) as exc:
            failures.append(f"representative_member:{archive_name}:{exc}")
            continue
        representative_rows.append(
            {
                "archive": archive_name,
                "member": record.member_name,
                "coordinate": list(record.coordinate),
                "sample_rows": len(first.rows),
                "member_sha256": record.member_sha256,
                "metric_computation_performed": False,
            }
        )
    if len(representative_rows) != 6:
        failures.append(f"representative_count:{len(representative_rows)}!=6")
    report = {
        "schema": "coregraph_level4_evidence_offline_smoke_v1",
        "status": "PASS" if not failures else "FAIL",
        "archives": [item.to_dict() for item in validations],
        "member_index_count": len(index.records),
        "representative_members": representative_rows,
        "ssd_source_used": False,
        "permanent_extractions": 0,
        "target_metrics_computed": 0,
        "target_oracles_computed": 0,
        "failures": failures,
    }
    output = ROOT / "results/coregraph_build/LEVEL4_EVIDENCE_OFFLINE_SMOKE.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
