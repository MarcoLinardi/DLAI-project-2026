"""Model merging primitives: interpolation, soup, TIES, permutation alignment."""

from src.merging.lmc import (
    error_barrier,
    evaluate_on_alpha_grid,
    interpolate_state_dicts,
    reset_bn_stats,
)
from src.merging.permute import (
    PARAM_PERM_MAP,
    PERM_DIMS,
    PERM_NAMES,
    Permutations,
    align_pair,
    apply_permutations,
    find_perms_activation_matching,
)
from src.merging.soup import (
    average_state_dicts,
    greedy_soup,
    uniform_soup,
)
from src.merging.ties import (
    compute_task_vectors,
    disjoint_merge,
    elect_signs,
    ties_merge,
    trim_by_magnitude,
)

__all__ = [
    "PARAM_PERM_MAP",
    "PERM_DIMS",
    "PERM_NAMES",
    "Permutations",
    "align_pair",
    "apply_permutations",
    "average_state_dicts",
    "compute_task_vectors",
    "disjoint_merge",
    "elect_signs",
    "error_barrier",
    "evaluate_on_alpha_grid",
    "find_perms_activation_matching",
    "greedy_soup",
    "interpolate_state_dicts",
    "reset_bn_stats",
    "ties_merge",
    "trim_by_magnitude",
    "uniform_soup",
]
