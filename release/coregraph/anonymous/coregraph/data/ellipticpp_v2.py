"""Elliptic++ V2 schema and join validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class EllipticPlusPlusSchemaReport:
    feature_rows: int
    class_rows: int
    edge_rows: int
    occurrence_unique: bool
    join_rate: float
    correlated_domain: bool = True


def validate_ellipticpp_frames(
    features: pd.DataFrame,
    classes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    address_column: str,
    time_column: str,
    class_address_column: str,
    edge_source_column: str,
    edge_target_column: str,
    minimum_join_rate: float = 0.95,
) -> EllipticPlusPlusSchemaReport:
    required = {
        "features": ({address_column, time_column}, set(features.columns)),
        "classes": ({class_address_column}, set(classes.columns)),
        "edges": ({edge_source_column, edge_target_column}, set(edges.columns)),
    }
    missing = {
        name: sorted(expected - actual)
        for name, (expected, actual) in required.items()
        if expected - actual
    }
    if missing:
        raise ValueError(f"Elliptic++ schema mismatch: {missing}")
    occurrence_unique = not features.duplicated([address_column, time_column]).any()
    if not occurrence_unique:
        raise ValueError("Elliptic++ (address,time) occurrences must be unique")
    addresses = set(features[address_column].astype(str))
    edge_endpoints = pd.concat(
        [
            edges[edge_source_column].astype(str),
            edges[edge_target_column].astype(str),
        ],
        ignore_index=True,
    )
    join_rate = float(edge_endpoints.isin(addresses).mean()) if len(edge_endpoints) else 1.0
    if join_rate < minimum_join_rate:
        raise ValueError(
            f"Elliptic++ occurrence/edge join rate {join_rate:.3f} "
            f"is below required {minimum_join_rate:.3f}"
        )
    return EllipticPlusPlusSchemaReport(
        feature_rows=len(features),
        class_rows=len(classes),
        edge_rows=len(edges),
        occurrence_unique=True,
        join_rate=join_rate,
    )


def read_csv_in_chunks(path: str, *, chunksize: int = 100_000) -> Iterable[pd.DataFrame]:
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    return pd.read_csv(path, chunksize=chunksize)
