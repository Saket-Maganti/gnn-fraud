from __future__ import annotations

import csv
from pathlib import Path

from coregraph.experts.base import OfficialStatus
from coregraph.experts.official_adapters.external import (
    load_official_adapters,
    validate_prediction_parity,
)


def test_all_named_official_adapters_are_pinned() -> None:
    adapters = load_official_adapters("external_baselines/BASELINE_REGISTRY.yaml")
    assert {"mowst", "graphmetro", "ciga", "eerm", "tgn", "good"} <= set(adapters)
    assert all(len(adapter.commit) == 40 for adapter in adapters.values())
    assert adapters["tgn"].status is OfficialStatus.PENDING_INTEGRATION
    assert adapters["graphmetro"].status is OfficialStatus.UNAVAILABLE_LICENSE


def test_parity_schema_and_identifier_alignment(tmp_path: Path) -> None:
    adapter = load_official_adapters(
        "external_baselines/BASELINE_REGISTRY.yaml"
    )["mowst"]
    output = tmp_path / "predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("node_id", "score", "expert_id", "official_status"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "node_id": "node:2",
                "score": 0.8,
                "expert_id": "mowst",
                "official_status": "OFFICIAL_CODE",
            }
        )
        writer.writerow(
            {
                "node_id": "node:1",
                "score": 0.2,
                "expert_id": "mowst",
                "official_status": "OFFICIAL_CODE",
            }
        )
    result = validate_prediction_parity(
        adapter,
        output,
        expected_ids=("node:1", "node:2"),
        id_column="node_id",
    )
    assert result.status == "PARITY_SCHEMA_PASS"
    assert result.output_sha256
