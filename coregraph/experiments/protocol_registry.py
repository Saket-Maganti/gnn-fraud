"""Frozen semantic protocol aliases for V4 prediction manifests."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from coregraph.contracts.axes import VisibilityAxis, VisibilitySpec
from coregraph.contracts.serialization import to_primitive


def load_protocol_registry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("schema_version") != "coregraph_contract_protocol_registry_v4":
        raise ValueError("protocol registry must use the frozen V4 schema")
    if registry.get("frozen_before_manifest_conversion") is not True:
        raise ValueError("protocol registry must be frozen before conversion")
    protocols = registry.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("protocol registry must contain protocol records")
    identifiers = [str(record.get("protocol_id", "")) for record in protocols]
    if any(not identifier for identifier in identifiers):
        raise ValueError("protocol registry identifiers cannot be empty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("protocol registry contains duplicate aliases")
    expected_identifiers = {
        VisibilityAxis.STRICT_INDUCTIVE.value,
        VisibilityAxis.ISOLATED_INDUCTIVE.value,
        VisibilityAxis.TRANSDUCTIVE_STRUCTURE.value,
    }
    if set(identifiers) != expected_identifiers:
        raise ValueError("protocol registry must contain the exact frozen aliases")
    for record in protocols:
        roles = record.get("allowed_contract_roles")
        if not isinstance(roles, list) or not roles:
            raise ValueError("protocol registry roles are required")
        if not set(roles).issubset({"source", "target"}):
            raise ValueError("protocol registry contains an invalid contract role")
        profile = str(record.get("visibility_profile", ""))
        if profile not in {
            VisibilityAxis.STRICT_INDUCTIVE.value,
            VisibilityAxis.ISOLATED_INDUCTIVE.value,
            VisibilityAxis.TRANSDUCTIVE_STRUCTURE.value,
        }:
            raise ValueError("protocol registry visibility profile is invalid")
    return registry


def _records_by_id(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    protocols = registry.get("protocols", ())
    return {
        str(record["protocol_id"]): record
        for record in protocols
        if isinstance(record, Mapping)
    }


def _expected_visibility(profile: str) -> dict[str, Any]:
    return to_primitive(VisibilitySpec.from_v2(VisibilityAxis(profile)))


def validate_protocol_bindings(
    artifacts: Sequence[Any],
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate aliases independently of coordinate and artifact identities."""

    records = _records_by_id(registry)
    if not records:
        raise ValueError("protocol registry contains no usable records")
    target_bindings: dict[
        tuple[str, str, int, str],
        dict[str, set[str]],
    ] = defaultdict(lambda: {"coordinates": set(), "contract_ids": set()})
    reverse: dict[str, set[str]] = defaultdict(set)
    for artifact in artifacts:
        protocol_id = str(artifact.protocol_id)
        if protocol_id not in records:
            raise ValueError(f"protocol alias {protocol_id!r} is not frozen")
        record = records[protocol_id]
        if artifact.contract_role not in set(record["allowed_contract_roles"]):
            raise ValueError(
                f"protocol alias {protocol_id!r} is incompatible with "
                f"{artifact.contract_role!r} role"
            )
        expected = _expected_visibility(str(record["visibility_profile"]))
        actual = to_primitive(artifact.deployment_contract.visibility)
        if actual != expected:
            raise ValueError(
                f"protocol alias {protocol_id!r} visibility does not match "
                "the deployment contract"
            )
        if artifact.contract_role != "target":
            continue
        key = (
            str(artifact.dataset),
            protocol_id,
            int(artifact.expert_prediction_seed),
            str(artifact.fold),
        )
        target_bindings[key]["coordinates"].add(
            str(artifact.contract_coordinate_hash)
        )
        target_bindings[key]["contract_ids"].add(str(artifact.contract_id))
        reverse[str(artifact.contract_coordinate_hash)].add(protocol_id)
    output: list[dict[str, Any]] = []
    for key, values in sorted(target_bindings.items()):
        if len(values["coordinates"]) != 1:
            raise ValueError(
                f"protocol alias collision maps {key} to multiple coordinate hashes"
            )
        if len(values["contract_ids"]) != 1:
            raise ValueError(
                f"protocol alias collision maps {key} to multiple target contract IDs"
            )
        dataset, protocol_id, seed, fold = key
        output.append(
            {
                "dataset": dataset,
                "target_protocol_id": protocol_id,
                "expert_prediction_seed": seed,
                "fold": fold,
                "target_contract_coordinate_hash": next(
                    iter(values["coordinates"])
                ),
                "target_contract_id": next(iter(values["contract_ids"])),
            }
        )
    collisions = {
        key: aliases for key, aliases in reverse.items() if len(aliases) > 1
    }
    if collisions:
        raise ValueError(
            "one coordinate hash maps to multiple incompatible protocol aliases: "
            f"{collisions}"
        )
    return output
