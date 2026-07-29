"""Strict score-domain metadata and conversions."""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch


class ScoreType(str, Enum):
    PROBABILITY = "PROBABILITY"
    LOGIT = "LOGIT"
    RANK_SCORE = "RANK_SCORE"


def validate_scores(values: torch.Tensor, score_type: ScoreType) -> torch.Tensor:
    if not torch.isfinite(values).all():
        raise ValueError("scores must be finite")
    if score_type is ScoreType.PROBABILITY and (
        bool((values < 0).any()) or bool((values > 1).any())
    ):
        raise ValueError("declared probabilities must lie in [0,1]")
    return values


def validate_numpy_scores(values: np.ndarray, score_type: ScoreType) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("scores must be finite")
    if score_type is ScoreType.PROBABILITY and (
        np.any(array < 0) or np.any(array > 1)
    ):
        raise ValueError("declared probabilities must lie in [0,1]")
    return array


def probabilities(values: torch.Tensor, score_type: ScoreType) -> torch.Tensor:
    validate_scores(values, score_type)
    if score_type is ScoreType.PROBABILITY:
        return values
    if score_type is ScoreType.LOGIT:
        return torch.sigmoid(values)
    raise ValueError("rank scores cannot be converted to calibrated probabilities")


def numpy_probabilities(values: np.ndarray, score_type: ScoreType) -> np.ndarray:
    array = validate_numpy_scores(values, score_type)
    if score_type is ScoreType.PROBABILITY:
        return array
    if score_type is ScoreType.LOGIT:
        return 1.0 / (1.0 + np.exp(-np.clip(array, -50, 50)))
    raise ValueError("rank scores cannot be converted to calibrated probabilities")
