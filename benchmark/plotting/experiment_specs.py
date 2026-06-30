import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join("../", REPO_ROOT, "results")

from aggregate_results import ExperimentSpec

specs = {
    "E1": ExperimentSpec(
        name="E1",
        data="folktables",
        task="weight_norm",
        bound=2.0,
        seeds=(0, 1, 2, 3, 4),
        results_root=RESULTS,
    ),
    "E2": ExperimentSpec(
        name="E2",
        data="folktables",
        task="folktables_positive_rate_vec",
        bound=0.2,
        seeds=(0, 1, 2, 3, 4),
        results_root=RESULTS,
    ),
    "E3": ExperimentSpec(
        name="E3",
        data="folktables",
        task="folktables_positive_rate_pair",
        bound=0.1,
        seeds=(0, 1, 2, 3, 4),
        results_root=RESULTS,
    ),
    "E4": ExperimentSpec(
        name="E4",
        data="dutch",
        task="folktables_positive_rate_pair",
        bound=0.1,
        seeds=(0, 1, 2, 3, 4),
        results_root=RESULTS,
    ),
    "E5": ExperimentSpec(
        name="E5",
        data="folktables",
        task="cifar10",
        bound=0.1,
        seeds=(0, 1, 2, 3, 4),
        results_root=RESULTS,
    )
}