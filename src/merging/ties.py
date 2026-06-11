"""TIES-Merging primitives (Yadav et al. 2023, NeurIPS).

Implements a simplified 3-stage pipeline on task vectors tau_i = theta_i - theta_base:

  1. TRIM:    per-model, keep only the top-k% parameters by absolute magnitude
              (threshold computed globally across the concatenated tensor of
              the task vector, NOT per-layer; this is the Yadav 2023 §3.1
              definition).
  2. ELECT:   per-parameter element, pick the sign majority weighted by
              magnitude across the N trimmed task vectors:
                sign_sum[k] = sum_i sign(tau_i[k]) * |tau_i[k]|
                signs[k]    = sign(sign_sum[k])  in {-1, 0, +1}
              The 0 case is rare post-trim (only when contributions cancel
              exactly element-wise).
  3. MERGE:   per-parameter element, average over those models whose sign
              agrees with the elected sign:
                mask_i[k]     = sign(tau_i[k]) == signs[k]
                tau_merged[k] = sum_i tau_i[k] * mask_i[k] / max(sum_i mask_i[k], 1)

Final merged state_dict: theta_TIES = theta_base + lam * tau_merged.

Non-float buffers (e.g. BatchNorm's num_batches_tracked) get tau_i = 0;
they are copied from theta_base in the final result and will be
overwritten by `reset_bn_stats` before evaluation.
"""

from __future__ import annotations

from collections import OrderedDict

import torch


def compute_task_vectors(
    finetuned_sds: list[dict[str, torch.Tensor]],
    base_sd: dict[str, torch.Tensor],
) -> list["OrderedDict[str, torch.Tensor]"]:
    """tau_i[k] = sd_i[k] - base_sd[k] for float keys; 0 elsewhere.

    Output tensors live on CPU in float32 to keep TIES bookkeeping simple.
    """
    task_vectors: list[OrderedDict[str, torch.Tensor]] = []
    for i, sd in enumerate(finetuned_sds):
        if set(sd.keys()) != set(base_sd.keys()):
            raise KeyError(f"key mismatch between finetuned_sds[{i}] and base_sd")
        tv: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v_base in base_sd.items():
            v_ft = sd[k]
            if v_ft.shape != v_base.shape:
                raise ValueError(f"shape mismatch at {k!r}: ft {v_ft.shape} vs base {v_base.shape}")
            if v_base.dtype.is_floating_point:
                tv[k] = (v_ft.detach().float().cpu() - v_base.detach().float().cpu())
            else:
                tv[k] = torch.zeros_like(v_base, dtype=torch.float32, device="cpu")
        task_vectors.append(tv)
    return task_vectors


def _float_keys(tv: dict[str, torch.Tensor]) -> list[str]:
    """Names of entries that participate in TIES math (float, non-zero shape)."""
    return [k for k, v in tv.items() if v.dtype.is_floating_point and v.numel() > 0]


def trim_by_magnitude(
    task_vectors: list["OrderedDict[str, torch.Tensor]"],
    keep_ratio: float = 0.20,
) -> list["OrderedDict[str, torch.Tensor]"]:
    """For each task vector, zero out all but the top-`keep_ratio` fraction
    of elements by absolute magnitude.

    Threshold is computed globally on the concatenated tensor of all float
    entries (Yadav 2023 §3.1: "magnitude across the whole task vector").
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if keep_ratio == 1.0:
        return [OrderedDict((k, v.clone()) for k, v in tv.items()) for tv in task_vectors]

    out: list[OrderedDict[str, torch.Tensor]] = []
    for tv in task_vectors:
        float_keys = _float_keys(tv)
        if not float_keys:
            out.append(OrderedDict((k, v.clone()) for k, v in tv.items()))
            continue

        flat = torch.cat([tv[k].abs().flatten() for k in float_keys])
        k_keep = max(1, int(round(keep_ratio * flat.numel())))
        threshold = torch.topk(flat, k_keep, largest=True, sorted=False).values.min()

        new_tv: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in tv.items():
            if v.dtype.is_floating_point and v.numel() > 0:
                mask = v.abs() >= threshold
                new_tv[k] = v * mask.to(v.dtype)
            else:
                new_tv[k] = v.clone()
        out.append(new_tv)
    return out


def elect_signs(
    task_vectors: list["OrderedDict[str, torch.Tensor]"],
) -> "OrderedDict[str, torch.Tensor]":
    """For each float entry, return the element-wise majority sign in {-1, 0, +1}.

    Majority is magnitude-weighted: sum_i sign(tau_i) * |tau_i|. The signum
    of that sum is the elected sign.
    """
    if not task_vectors:
        raise ValueError("task_vectors must be non-empty")
    signs: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v0 in task_vectors[0].items():
        if not v0.dtype.is_floating_point:
            continue
        acc = torch.zeros_like(v0)
        for tv in task_vectors:
            acc.add_(tv[k])  # sum of signed magnitudes == sum_i sign(tau)*|tau|
        signs[k] = torch.sign(acc)
    return signs


def disjoint_merge(
    task_vectors: list["OrderedDict[str, torch.Tensor]"],
    signs: dict[str, torch.Tensor],
) -> "OrderedDict[str, torch.Tensor]":
    """For each float entry, average only the contributions whose sign agrees
    with the elected sign. Falls back to 0 where no model agrees.
    """
    if not task_vectors:
        raise ValueError("task_vectors must be non-empty")
    merged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v0 in task_vectors[0].items():
        if not v0.dtype.is_floating_point:
            merged[k] = v0.clone()
            continue
        sign_k = signs[k]
        sum_vals = torch.zeros_like(v0)
        count = torch.zeros_like(v0)
        for tv in task_vectors:
            v = tv[k]
            agree = (torch.sign(v) == sign_k) & (sign_k != 0)
            sum_vals.add_(v * agree.to(v.dtype))
            count.add_(agree.to(v.dtype))
        merged[k] = sum_vals / count.clamp_min(1.0)
    return merged


def ties_merge(
    finetuned_sds: list[dict[str, torch.Tensor]],
    base_sd: dict[str, torch.Tensor],
    keep_ratio: float = 0.20,
    lam: float = 1.0,
) -> "OrderedDict[str, torch.Tensor]":
    """End-to-end TIES pipeline. Returns theta_base + lam * tau_merged.

    The output state_dict can be loaded directly into a fresh model via
    `model.load_state_dict(...)`. BN running stats are inherited from
    base_sd (tau is 0 on non-float keys); call `reset_bn_stats` before
    evaluation, as for any merged model.
    """
    task_vectors = compute_task_vectors(finetuned_sds, base_sd)
    trimmed = trim_by_magnitude(task_vectors, keep_ratio=keep_ratio)
    signs = elect_signs(trimmed)
    tau_merged = disjoint_merge(trimmed, signs)

    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v_base in base_sd.items():
        if v_base.dtype.is_floating_point:
            update = lam * tau_merged[k]
            out[k] = (v_base.detach().float().cpu() + update).to(v_base.dtype)
        else:
            out[k] = v_base.detach().cpu().clone()
    return out


if __name__ == "__main__":
    # Sanity 1: opposite task vectors with keep_ratio=1.0 cancel → merged ≈ base
    from src.models.resnet20 import resnet20

    torch.manual_seed(0)
    m = resnet20()
    base = m.state_dict()
    ft_plus = OrderedDict()
    ft_minus = OrderedDict()
    for k, v in base.items():
        if v.dtype.is_floating_point:
            ft_plus[k] = v + 0.1
            ft_minus[k] = v - 0.1
        else:
            ft_plus[k] = v.clone()
            ft_minus[k] = v.clone()
    merged = ties_merge([ft_plus, ft_minus], base, keep_ratio=1.0, lam=1.0)
    # With opposite signs everywhere, no model agrees with majority (sign=0 case)
    # → tau_merged ≈ 0 → merged ≈ base. Allow small numerical slack.
    for k, v in merged.items():
        if v.dtype.is_floating_point:
            diff = (v - base[k]).abs().max().item()
            assert diff < 1e-5, f"opposite-cancel failed at {k}: max diff {diff}"
    print("OK: opposite task vectors cancel (merged ≈ base)")

    # Sanity 2: identical task vectors with keep_ratio=1.0 → merged = base + lam * tau
    ft = OrderedDict()
    for k, v in base.items():
        if v.dtype.is_floating_point:
            ft[k] = v + 0.05
        else:
            ft[k] = v.clone()
    merged_identical = ties_merge([ft, ft, ft], base, keep_ratio=1.0, lam=1.0)
    for k, v in merged_identical.items():
        if v.dtype.is_floating_point:
            expected = base[k] + 0.05
            diff = (v - expected).abs().max().item()
            assert diff < 1e-5, f"identical-agree failed at {k}: max diff {diff}"
    print("OK: identical task vectors propagate fully")

    # Sanity 3: no NaN/Inf with a realistic (small) keep_ratio
    torch.manual_seed(1)
    fts = []
    for _ in range(3):
        ft_rand = OrderedDict()
        for k, v in base.items():
            if v.dtype.is_floating_point:
                ft_rand[k] = v + 0.01 * torch.randn_like(v)
            else:
                ft_rand[k] = v.clone()
        fts.append(ft_rand)
    out = ties_merge(fts, base, keep_ratio=0.20, lam=1.0)
    for k, v in out.items():
        assert not torch.isnan(v).any(), f"NaN at {k}"
        assert not torch.isinf(v).any(), f"Inf at {k}"
    print("OK: keep_ratio=0.20 on 3 random task vectors yields finite output")
