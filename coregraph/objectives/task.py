"""Task-loss facade retained for the Level-4 package layout."""

from coregraph.objectives.classification import (
    binary_cross_entropy,
    binary_cross_entropy_values,
    class_balanced_loss,
    focal_loss,
)

__all__ = [
    "binary_cross_entropy",
    "binary_cross_entropy_values",
    "class_balanced_loss",
    "focal_loss",
]
