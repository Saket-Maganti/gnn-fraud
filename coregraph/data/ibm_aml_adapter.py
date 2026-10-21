"""IBM AML transaction-edge task adapter."""

from __future__ import annotations

from enum import Enum

import numpy as np

from coregraph.contracts.contract import DeploymentContract
from coregraph.data.contract_dataset import ContractDataset, DatasetManifest
from coregraph.data.elliptic_v2 import _scale_train_only
from coregraph.tasks.transaction_task import TransactionTaskAdapter


class IBMAMLSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


def build_ibm_aml_transaction_dataset(
    *,
    transaction_ids: np.ndarray,
    account_ids: np.ndarray,
    transaction_features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    contract: DeploymentContract,
    size: IBMAMLSize,
    train_cutoff: float,
    validation_cutoff: float,
    label_visibility_fraction: float = 0.5,
    covariate_visibility_fraction: float = 0.6,
    labels_are_canonical: bool = False,
    raw_checksum: str = "fixture",
) -> ContractDataset:
    if not 0 < label_visibility_fraction <= covariate_visibility_fraction <= 1:
        raise ValueError("require 0 < label visibility <= covariate visibility <= 1")
    if size is IBMAMLSize.LARGE:
        raise RuntimeError(
            "IBM AML Large is RESOURCE_BLOCKED in the pre-run build; use Small/Medium "
            "or execute the declared later large-resource plan"
        )
    ids = np.asarray(transaction_ids)
    accounts = np.asarray(account_ids, dtype=int)
    if accounts.ndim != 2 or accounts.shape[1] != 2 or len(accounts) != len(ids):
        raise ValueError("account_ids must have [source,destination] per transaction")
    raw_y = np.asarray(labels, dtype=int)
    if labels_are_canonical:
        if not set(np.unique(raw_y)).issubset({0, 1, 2}):
            raise ValueError("canonical IBM AML labels must use {0,1,2}")
        y = raw_y
    else:
        if not set(np.unique(raw_y)).issubset({0, 1}):
            raise ValueError("raw IBM AML labels must be binary 0=normal 1=laundering")
        y = np.where(raw_y == 1, 1, 2)
    times = np.asarray(timestamps)
    known = y != 0
    train = known & (times <= train_cutoff)
    validation = known & (times > train_cutoff) & (times <= validation_cutoff)
    test = known & (times > validation_cutoff)
    scaled, scaler_provenance = _scale_train_only(transaction_features, train)
    task = TransactionTaskAdapter()
    batch = task.build_batch(
        transaction_ids=ids,
        transaction_features=scaled,
        labels=y,
        train_mask=train,
        validation_mask=validation,
        test_mask=test,
        timestamps=times,
        graph_view=None,
        contract_id=contract.contract_id,
    )
    unique_accounts, inverse = np.unique(accounts.reshape(-1), return_inverse=True)
    edge_index = inverse.reshape(-1, 2).T
    return ContractDataset(
        manifest=DatasetManifest(
            dataset_id="ibm_aml",
            variant=f"{size.value}_transaction_edge_v2",
            source="IBM AML synthetic transaction dataset",
            licence_status="dataset_terms_review_required",
            raw_checksum=raw_checksum,
            timestamp_quality="provider_transaction_timestamp",
            notes=(
                f"labels_visible={label_visibility_fraction:.2f}; "
                f"covariates_visible={covariate_visibility_fraction:.2f}"
            ),
        ),
        task_adapter=task,
        batch=batch,
        node_ids=unique_accounts,
        node_features=np.empty((len(unique_accounts), 0), dtype=float),
        edge_index=edge_index,
        edge_timestamps=times,
        edge_types=None,
        edge_attributes=scaled,
        provenance=scaler_provenance
        + (
            ("prediction_unit", "transaction"),
            ("label_mapping", "raw_1_fraud_raw_0_normal_to_canonical_1_2"),
            ("label_visibility_fraction", str(label_visibility_fraction)),
            ("covariate_visibility_fraction", str(covariate_visibility_fraction)),
        ),
    )
