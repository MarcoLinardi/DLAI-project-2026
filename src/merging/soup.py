"""Model Soup primitives: uniform + greedy averaging of N state_dicts.

Implements the merging operations of Wortsman et al. 2022 ("Model Soups",
ICML). Reuses `reset_bn_stats` and `_eval_at_point` from src.merging.lmc
so the BN-handling stays identical to the LMC pipeline.

Two soup variants:
- uniform: theta = (1/N) * sum_i theta_i
- greedy : start from the most accurate model; add others one by one
           only if the running average improves on test acc.

Greedy soup uses the test set as the selection signal (no CIFAR-10 val
split). This is the canonical Wortsman setup; the test-set selection is
declared as a limit in the report.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.merging.lmc import _eval_at_point


def average_state_dicts(
    sds: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Return sum_i w_i * sd_i on float buffers/parameters.

    Generalizes `interpolate_state_dicts` to N models. Default weights are
    uniform (1/N each). Non-float entries (e.g. BatchNorm's
    `num_batches_tracked`) are copied from `sds[0]`; they will be
    overwritten by `reset_bn_stats` before evaluation.
    """
    if not sds:
        raise ValueError("sds must contain at least one state_dict")
    n = len(sds)
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"len(weights)={len(weights)} != len(sds)={n}")

    ref_keys = set(sds[0].keys())
    for i, sd in enumerate(sds[1:], start=1):
        if set(sd.keys()) != ref_keys:
            only_ref = ref_keys - set(sd)
            only_i = set(sd) - ref_keys
            raise KeyError(
                f"state_dict key mismatch between sds[0] and sds[{i}]. "
                f"only in sds[0]: {only_ref}; only in sds[{i}]: {only_i}"
            )

    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v0 in sds[0].items():
        tensors = [sd[k] for sd in sds]
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) > 1:
            raise ValueError(f"shape mismatch for key {k!r}: {shapes}")
        if v0.dtype.is_floating_point:
            acc = torch.zeros_like(v0, dtype=torch.float32, device="cpu")
            for w, t in zip(weights, tensors):
                acc.add_(t.detach().float().cpu(), alpha=w)
            out[k] = acc.to(v0.dtype)
        else:
            out[k] = v0.detach().cpu().clone()
    return out


def uniform_soup(
    sds: list[dict[str, torch.Tensor]],
    model_ctor: Callable[[], nn.Module],
    bn_loader: DataLoader,
    train_eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    bn_reset_batches: int = 40,
) -> tuple["OrderedDict[str, torch.Tensor]", dict[str, float]]:
    """Average all N state_dicts uniformly, BN-reset, evaluate.

    Returns (merged_state_dict, metrics) where metrics contains
    train_loss/train_acc/test_loss/test_acc.
    """
    merged = average_state_dicts(sds)
    metrics = _eval_at_point(
        model_ctor=model_ctor,
        sd=merged,
        bn_loader=bn_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        device=device,
        bn_reset_batches=bn_reset_batches,
    )
    return merged, metrics


def greedy_soup(
    sds_with_acc: list[tuple[dict[str, torch.Tensor], float]],
    model_ctor: Callable[[], nn.Module],
    bn_loader: DataLoader,
    train_eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    bn_reset_batches: int = 40,
    selection_loader: DataLoader | None = None,
) -> tuple["OrderedDict[str, torch.Tensor]", list[int], list[dict]]:
    """Greedy soup (Wortsman 2022, Algorithm 1).

    Sorts models by descending individual test_acc, starts with the best,
    and tries each remaining model one at a time: keeps it only if adding
    it to the running uniform average improves the soup's selection_acc.

    selection_loader: loader used for the greedy selection criterion.
    Defaults to test_loader (canonical Wortsman setup). Pass a validation
    loader to avoid test-set data leak (E8).

    Ties are broken by ascending original index (stable sort).

    Returns (final_merged_sd, included_original_indices, step_history).
    """
    if not sds_with_acc:
        raise ValueError("sds_with_acc must contain at least one model")

    _sel_loader = selection_loader if selection_loader is not None else test_loader

    n = len(sds_with_acc)
    order = sorted(range(n), key=lambda i: (-sds_with_acc[i][1], i))

    best_idx = order[0]
    included_sds: list[dict[str, torch.Tensor]] = [sds_with_acc[best_idx][0]]
    included_orig_idx: list[int] = [best_idx]

    soup_sd = average_state_dicts(included_sds)
    soup_metrics = _eval_at_point(
        model_ctor=model_ctor,
        sd=soup_sd,
        bn_loader=bn_loader,
        train_eval_loader=train_eval_loader,
        test_loader=_sel_loader,
        device=device,
        bn_reset_batches=bn_reset_batches,
    )
    best_soup_acc = soup_metrics["test_acc"]

    history: list[dict] = [{
        "step": 0,
        "candidate_orig_idx": best_idx,
        "candidate_acc_alone": sds_with_acc[best_idx][1],
        "soup_acc_after": best_soup_acc,
        "included": True,
        "n_in_soup": 1,
    }]

    for step, cand_orig_idx in enumerate(order[1:], start=1):
        cand_sd, cand_acc_alone = sds_with_acc[cand_orig_idx]
        trial = average_state_dicts(included_sds + [cand_sd])
        trial_metrics = _eval_at_point(
            model_ctor=model_ctor,
            sd=trial,
            bn_loader=bn_loader,
            train_eval_loader=train_eval_loader,
            test_loader=_sel_loader,
            device=device,
            bn_reset_batches=bn_reset_batches,
        )
        trial_acc = trial_metrics["test_acc"]
        included_now = trial_acc > best_soup_acc
        if included_now:
            included_sds.append(cand_sd)
            included_orig_idx.append(cand_orig_idx)
            soup_sd = trial
            soup_metrics = trial_metrics
            best_soup_acc = trial_acc
        history.append({
            "step": step,
            "candidate_orig_idx": cand_orig_idx,
            "candidate_acc_alone": cand_acc_alone,
            "soup_acc_after": best_soup_acc,
            "included": included_now,
            "n_in_soup": len(included_sds),
        })

    return soup_sd, included_orig_idx, history


if __name__ == "__main__":
    # Sanity: average_state_dicts(N=2) ≡ interpolate_state_dicts(alpha=0.5)
    from src.merging.lmc import interpolate_state_dicts
    from src.models.resnet20 import resnet20

    torch.manual_seed(0)
    m1 = resnet20()
    torch.manual_seed(1)
    m2 = resnet20()
    sd1, sd2 = m1.state_dict(), m2.state_dict()

    avg = average_state_dicts([sd1, sd2])
    ref = interpolate_state_dicts(sd1, sd2, 0.5)
    for k in ref:
        if ref[k].dtype.is_floating_point:
            assert torch.allclose(avg[k], ref[k], atol=1e-6), f"mismatch at {k}"
    print("OK: average_state_dicts(N=2) == interpolate_state_dicts(alpha=0.5)")

    # Sanity: weights = [1, 0, 0, ...] returns sds[0]
    n = 4
    sds = [m1.state_dict() for _ in range(n)]
    avg_first = average_state_dicts(sds, weights=[1.0, 0.0, 0.0, 0.0])
    for k in sd1:
        if sd1[k].dtype.is_floating_point:
            assert torch.allclose(avg_first[k], sd1[k], atol=1e-6), f"weights=[1,0,0,0] != sd1 at {k}"
    print("OK: weights=[1,0,...] selects first state_dict")
