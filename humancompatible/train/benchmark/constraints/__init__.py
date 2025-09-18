from .constraint import FairnessConstraint
from .constraint_fns import (
    fairret_stat_equality,
    ppv_equality,
    acc_equality,
    tpr_equality,
    abs_loss_equality,
    loss_equality,
    abs_diff_tpr,
    abs_diff_fpr,
    abs_max_dev_from_overall_tpr,
    abs_diff_pr
)

__all__ = ["FairnessConstraint, loss_equality"]
