"""
src/losses package
Physical-layer multi-task loss functions and regularization terms.
"""
from src.train.losses import (
    compute_task_loss,
    compute_rank_complexity_penalty
)

__all__ = [
    "compute_task_loss",
    "compute_rank_complexity_penalty"
]
