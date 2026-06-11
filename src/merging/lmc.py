"""Linear Mode Connectivity primitives.

Implements the operations needed by experiment E1 (Frankle, Dziugaite, Roy,
Carbin 2020): linear interpolation of two model state_dicts, BatchNorm
running-stats re-estimation on the merged model, evaluation along an alpha
grid, and the error_barrier metric.

The same primitives are reused by Phase 3 (model soup) and Phase 4
(permutation alignment) through the `_eval_at_point` helper.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.utils import accuracy, AverageMeter


def interpolate_state_dicts(
    sd1: dict[str, torch.Tensor],
    sd2: dict[str, torch.Tensor],
    alpha: float,
) -> "OrderedDict[str, torch.Tensor]":
    """Return (1 - alpha) * sd1 + alpha * sd2 on float buffers/parameters.

    Non-float entries (e.g. BatchNorm's `num_batches_tracked` int64 counter)
    are copied from `sd1`; they will be overwritten by `reset_bn_stats`.
    """
    if set(sd1.keys()) != set(sd2.keys()):
        only1 = set(sd1) - set(sd2)
        only2 = set(sd2) - set(sd1)
        raise KeyError(f"state_dict key mismatch. only in sd1: {only1}; only in sd2: {only2}")

    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v1 in sd1.items():
        v2 = sd2[k]
        if v1.shape != v2.shape:
            raise ValueError(f"shape mismatch for key {k!r}: {v1.shape} vs {v2.shape}")
        if v1.dtype.is_floating_point:
            mixed = (1.0 - alpha) * v1.detach().float().cpu() + alpha * v2.detach().float().cpu()
            out[k] = mixed.to(v1.dtype)
        else:
            out[k] = v1.detach().cpu().clone()
    return out


def reset_bn_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 40,
) -> None:
    """Re-estimate BatchNorm running mean/var by streaming `num_batches` batches.

    Without this step, evaluating an interpolated model gives garbage: the
    running stats inherited from the parents do not match the activations
    produced by the merged weights.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bn_modules:
        return
    for m in bn_modules:
        m.reset_running_stats()

    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            x = x.to(device, non_blocking=True)
            model(x)


def _eval_at_point(
    model_ctor: Callable[[], nn.Module],
    sd: dict[str, torch.Tensor],
    bn_loader: DataLoader,
    train_eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    bn_reset_batches: int = 40,
) -> dict[str, float]:
    """Build a fresh model, load `sd`, reset BN stats, evaluate on train+test.

    Generic over how `sd` was built — reused for alpha interpolation (E1),
    weighted averaging (E3 soup), and permuted merging (E4).
    """
    model = model_ctor()
    model.load_state_dict(sd, strict=True)
    model.to(device)
    reset_bn_stats(model, bn_loader, device, num_batches=bn_reset_batches)

    criterion = nn.CrossEntropyLoss()
    tr_loss, tr_acc = _evaluate(model, train_eval_loader, criterion, device)
    te_loss, te_acc = _evaluate(model, test_loader, criterion, device)
    return {
        "train_loss": tr_loss,
        "train_acc": tr_acc,
        "test_loss": te_loss,
        "test_acc": te_acc,
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Mirror of src.training.train.evaluate, duplicated to avoid a circular
    dependency between merging and training modules."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(accuracy(logits, y), n=x.size(0))
    return loss_meter.avg, acc_meter.avg


def evaluate_on_alpha_grid(
    model_ctor: Callable[[], nn.Module],
    sd1: dict[str, torch.Tensor],
    sd2: dict[str, torch.Tensor],
    alphas: list[float],
    bn_loader: DataLoader,
    train_eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    bn_reset_batches: int = 40,
) -> list[dict[str, float]]:
    """Evaluate the linear-interpolation curve theta(alpha) on a grid."""
    rows: list[dict[str, float]] = []
    for a in alphas:
        sd = interpolate_state_dicts(sd1, sd2, a)
        metrics = _eval_at_point(
            model_ctor, sd, bn_loader, train_eval_loader, test_loader, device,
            bn_reset_batches=bn_reset_batches,
        )
        metrics["alpha"] = float(a)
        rows.append(metrics)
        print(
            f"  alpha={a:.2f} | train_loss={metrics['train_loss']:.4f} "
            f"acc={metrics['train_acc']*100:5.2f}% | "
            f"test_loss={metrics['test_loss']:.4f} "
            f"acc={metrics['test_acc']*100:5.2f}%"
        )
    return rows


def error_barrier(rows: list[dict[str, float]], metric: str = "test_loss") -> float:
    """max_alpha metric(theta(alpha)) - 0.5*(metric(alpha=0) + metric(alpha=1)).

    Endpoints are taken from the evaluated curve (post-BN-reset), not from
    the nominal checkpoint values — this is the definition used in
    Frankle 2020.
    """
    by_alpha = sorted(rows, key=lambda r: r["alpha"])
    if by_alpha[0]["alpha"] != 0.0 or by_alpha[-1]["alpha"] != 1.0:
        raise ValueError("alpha grid must include both 0.0 and 1.0")
    end_avg = 0.5 * (by_alpha[0][metric] + by_alpha[-1][metric])
    if metric.endswith("_acc"):
        # For accuracy the "barrier" is a drop, so we flip the sign.
        worst = min(r[metric] for r in by_alpha)
        return end_avg - worst
    worst = max(r[metric] for r in by_alpha)
    return worst - end_avg
