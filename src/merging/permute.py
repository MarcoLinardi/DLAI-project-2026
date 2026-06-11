"""Git Re-Basin permutation alignment for ResNet-20 (Ainsworth et al. 2023).

Implements **Activation Matching** (Ainsworth 2023, §3.3): given two trained
models θ_A and θ_B with identical architecture, find a per-layer
permutation of θ_B's channels that maximises the channel-wise correlation
of the two models' activations on a small calibration batch. Applying
that permutation produces θ_B' whose function is identical to θ_B's but
whose channels are aligned with θ_A's, so the average (θ_A + θ_B') / 2
can stay inside a low-loss basin even when (θ_A + θ_B) / 2 does not.

The skip connections of ResNet-20 force shared output permutations across
every block of a stage (residual + shortcut sum). We expose 12 distinct
permutations: 3 stage-output (16/32/64 channels) and 9 block-internal
(one per BasicBlock, between conv1 and conv2). See the per-tensor
`PARAM_PERM_MAP` for the exact mapping of state_dict keys to the
(out_dim_perm, in_dim_perm) pair to apply.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Permutation registry (12 permutations, dims fixed by ResNet-20 architecture)
# ---------------------------------------------------------------------------

PERM_DIMS: dict[str, int] = {
    "P_stage1": 16, "P_stage2": 32, "P_stage3": 64,
    "P_l1b0i": 16, "P_l1b1i": 16, "P_l1b2i": 16,
    "P_l2b0i": 32, "P_l2b1i": 32, "P_l2b2i": 32,
    "P_l3b0i": 64, "P_l3b1i": 64, "P_l3b2i": 64,
}

PERM_NAMES: tuple[str, ...] = tuple(PERM_DIMS.keys())

# Input channels of stage k come from stage k-1 (and stage 1 comes from conv1,
# which itself has P_stage1 on its output). So "input to stage k" == "stage k-1 output".
_STAGE_IN_PERM = {1: "P_stage1", 2: "P_stage1", 3: "P_stage2"}
_STAGE_OUT_PERM = {1: "P_stage1", 2: "P_stage2", 3: "P_stage3"}


def _block_internal_perm(stage: int, block: int) -> str:
    return f"P_l{stage}b{block}i"


def _build_param_perm_map() -> dict[str, tuple[str | None, str | None]]:
    """For each state_dict key of ResNet-20 (option='B'), return the pair
    (out_dim_perm, in_dim_perm). `None` means "leave that dim alone"."""
    m: dict[str, tuple[str | None, str | None]] = {}

    # Stem conv + BN
    m["conv1.weight"] = ("P_stage1", None)
    for sub in ("weight", "bias", "running_mean", "running_var"):
        m[f"bn1.{sub}"] = ("P_stage1", None)
    m["bn1.num_batches_tracked"] = (None, None)

    for stage in (1, 2, 3):
        stage_in = _STAGE_IN_PERM[stage]
        stage_out = _STAGE_OUT_PERM[stage]
        for block in (0, 1, 2):
            internal = _block_internal_perm(stage, block)
            # The input to block b is the stage-input perm when b==0 (it comes
            # from the previous stage / stem), and the stage-output perm
            # otherwise (it comes from the previous block in this stage).
            block_input = stage_in if block == 0 else stage_out

            # conv1: out -> internal, in -> previous activation
            m[f"layer{stage}.{block}.conv1.weight"] = (internal, block_input)
            for sub in ("weight", "bias", "running_mean", "running_var"):
                m[f"layer{stage}.{block}.bn1.{sub}"] = (internal, None)
            m[f"layer{stage}.{block}.bn1.num_batches_tracked"] = (None, None)

            # conv2: out -> stage_out (must agree with shortcut for the add), in -> internal
            m[f"layer{stage}.{block}.conv2.weight"] = (stage_out, internal)
            for sub in ("weight", "bias", "running_mean", "running_var"):
                m[f"layer{stage}.{block}.bn2.{sub}"] = (stage_out, None)
            m[f"layer{stage}.{block}.bn2.num_batches_tracked"] = (None, None)

            # Shortcut conv+BN exists only on block 0 of stages 2 and 3
            # (downsample / channel-bump). Block 0 of stage 1 has nn.Identity
            # because in_planes==planes and stride==1.
            if block == 0 and stage in (2, 3):
                m[f"layer{stage}.0.shortcut.0.weight"] = (stage_out, stage_in)
                for sub in ("weight", "bias", "running_mean", "running_var"):
                    m[f"layer{stage}.0.shortcut.1.{sub}"] = (stage_out, None)
                m[f"layer{stage}.0.shortcut.1.num_batches_tracked"] = (None, None)

    # Classifier head
    m["fc.weight"] = (None, "P_stage3")  # out = classes (no perm), in = pool features
    m["fc.bias"] = (None, None)
    return m


PARAM_PERM_MAP: dict[str, tuple[str | None, str | None]] = _build_param_perm_map()


# ---------------------------------------------------------------------------
# Permutations container
# ---------------------------------------------------------------------------


@dataclass
class Permutations:
    """Holds the 12 per-layer permutations for ResNet-20."""

    perms: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def identity(cls) -> "Permutations":
        return cls(perms={name: torch.arange(dim, dtype=torch.long)
                          for name, dim in PERM_DIMS.items()})

    def get(self, name: str) -> torch.Tensor:
        return self.perms[name]

    def is_identity(self) -> bool:
        for name, dim in PERM_DIMS.items():
            if not torch.equal(self.perms[name], torch.arange(dim, dtype=torch.long)):
                return False
        return True


# ---------------------------------------------------------------------------
# Apply permutations to a state_dict
# ---------------------------------------------------------------------------


def apply_permutations(
    sd: dict[str, torch.Tensor],
    perms: Permutations,
) -> "OrderedDict[str, torch.Tensor]":
    """Apply `perms` to every tensor of `sd` according to PARAM_PERM_MAP.

    Keys not in the map (unexpected — would only happen if the architecture
    changes) are passed through unchanged with a printed warning.
    """
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v in sd.items():
        if k not in PARAM_PERM_MAP:
            print(f"warn: apply_permutations: unknown key {k!r}, copied unchanged")
            out[k] = v.detach().clone()
            continue
        out_p, in_p = PARAM_PERM_MAP[k]
        new_v = v.detach().clone()
        if out_p is not None:
            P = perms.get(out_p).to(new_v.device)
            if P.numel() != new_v.shape[0]:
                raise ValueError(
                    f"perm {out_p!r} has size {P.numel()} but tensor {k!r} has dim0={new_v.shape[0]}"
                )
            new_v = torch.index_select(new_v, dim=0, index=P)
        if in_p is not None:
            if new_v.ndim < 2:
                raise ValueError(f"in_perm for {k!r}: tensor has shape {tuple(new_v.shape)}, no dim 1")
            P = perms.get(in_p).to(new_v.device)
            if P.numel() != new_v.shape[1]:
                raise ValueError(
                    f"perm {in_p!r} has size {P.numel()} but tensor {k!r} has dim1={new_v.shape[1]}"
                )
            new_v = torch.index_select(new_v, dim=1, index=P)
        out[k] = new_v
    return out


# ---------------------------------------------------------------------------
# Activation matching
# ---------------------------------------------------------------------------


def _gather_calibration_batch(loader: DataLoader, n_samples: int) -> torch.Tensor:
    """Concatenate enough batches to reach `n_samples` examples."""
    chunks: list[torch.Tensor] = []
    seen = 0
    for x, _ in loader:
        chunks.append(x)
        seen += x.size(0)
        if seen >= n_samples:
            break
    return torch.cat(chunks, dim=0)[:n_samples]


def _flatten_conv_act(x: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) -> (N*H*W, C). Used to build correlation matrices."""
    return x.permute(0, 2, 3, 1).reshape(-1, x.size(1))


def _register_activation_hooks(model: nn.Module, sink: dict) -> list:
    """Register forward hooks on `model` so each forward pass populates `sink`
    with the 12 activation tensors keyed by permutation name. Returns the list
    of handles so the caller can release them."""

    def stage_hook(name: str):
        def hook(_module, _inputs, output):
            sink[name] = _flatten_conv_act(output.detach()).cpu()
        return hook

    def block_internal_hook(name: str):
        def hook(_module, inputs):  # forward_pre_hook signature: (module, inputs)
            x = inputs[0].detach()
            sink[name] = _flatten_conv_act(x).cpu()
        return hook

    handles = []
    for stage in (1, 2, 3):
        layer = getattr(model, f"layer{stage}")
        handles.append(layer.register_forward_hook(stage_hook(f"P_stage{stage}")))
        for b in (0, 1, 2):
            handles.append(
                layer[b].conv2.register_forward_pre_hook(block_internal_hook(f"P_l{stage}b{b}i"))
            )
    return handles


def _capture_activations(
    model: nn.Module,
    x_batch: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Forward `x_batch` through `model` (eval, no-grad) and return the 12
    activation tensors keyed by permutation name (CPU floats)."""
    sink: dict[str, torch.Tensor] = {}
    handles = _register_activation_hooks(model, sink)
    try:
        model.to(device).eval()
        with torch.no_grad():
            _ = model(x_batch.to(device))
    finally:
        for h in handles:
            h.remove()
    return sink


def find_perms_activation_matching(
    model_A: nn.Module,
    model_B: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 512,
    max_iters: int = 5,
    verbose: bool = True,
) -> Permutations:
    """Find the 12 permutations of model_B that maximise channel-wise
    correlation with model_A on a calibration batch.

    Iterative AM (Ainsworth 2023 §3.3): at each iteration we apply the
    permutations accumulated so far to model_B, recompute its activations,
    and solve the 12 linear-assignment problems again. Permutations from
    successive iterations are composed via `perm_total[k] = perm_total_old[local_P[k]]`.

    Stops when the latest iteration yields identity local permutations
    (converged) or when `max_iters` is reached. Empirically 3–5 iters suffice
    for ResNet-20 on CIFAR-10.

    The forward pass on model_A is done only once (it never changes).
    Per-permutation cost is dominated by `linear_sum_assignment` on a
    16x16 / 32x32 / 64x64 matrix → microseconds.
    """
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")

    model_A.to(device).eval()
    model_B.to(device).eval()
    sd_B_orig = {k: v.detach().clone() for k, v in model_B.state_dict().items()}

    x_batch = _gather_calibration_batch(loader, n_samples).to(device)

    # model_A activations: captured once (model_A is fixed across iterations).
    acts_A = _capture_activations(model_A, x_batch, device)

    perm_total: dict[str, torch.Tensor] = {
        name: torch.arange(dim, dtype=torch.long) for name, dim in PERM_DIMS.items()
    }

    for it in range(max_iters):
        # Apply accumulated perms to B and reload it.
        sd_B_current = apply_permutations(sd_B_orig, Permutations(perms=perm_total))
        model_B.load_state_dict(sd_B_current, strict=True)
        acts_B = _capture_activations(model_B, x_batch, device)

        # For each permutation, solve assignment between A and B_current channels.
        local_perms: dict[str, torch.Tensor] = {}
        n_local_moved = 0
        for name in PERM_NAMES:
            A = acts_A[name].float()
            B = acts_B[name].float()
            C = (A.T @ B).numpy()
            _row_ind, col_ind = linear_sum_assignment(-C)
            local_P = torch.from_numpy(col_ind).long()
            local_perms[name] = local_P
            if not torch.equal(local_P, torch.arange(local_P.numel(), dtype=torch.long)):
                n_local_moved += 1

        # Compose: perm_total_new[k] = perm_total_old[local_P[k]]
        new_perm_total = {
            name: perm_total[name][local_perms[name]] for name in PERM_NAMES
        }
        perm_total = new_perm_total

        if verbose:
            n_total_moved = sum(
                int(not torch.equal(perm_total[n], torch.arange(perm_total[n].numel(),
                                                                dtype=torch.long)))
                for n in perm_total
            )
            print(f"  AM iter {it + 1}/{max_iters}: {n_local_moved}/12 local perms moved, "
                  f"{n_total_moved}/12 cumulative differ from identity")

        if n_local_moved == 0:
            if verbose:
                print(f"  AM converged at iter {it + 1}")
            # Restore B state_dict before returning (caller doesn't expect mutation).
            model_B.load_state_dict(sd_B_orig, strict=True)
            return Permutations(perms=perm_total)

    # Restore original B
    model_B.load_state_dict(sd_B_orig, strict=True)
    return Permutations(perms=perm_total)


def align_pair(
    sd_A: dict[str, torch.Tensor],
    sd_B: dict[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    model_ctor=None,
    n_samples: int = 512,
    max_iters: int = 5,
) -> "OrderedDict[str, torch.Tensor]":
    """Find perms with iterative activation matching, apply them to sd_B,
    return sd_B_aligned.

    Requires `model_ctor` (callable returning a fresh ResNet-20) so we can
    instantiate two models, load the state_dicts, and run the forward passes.
    """
    if model_ctor is None:
        from src.models.resnet20 import resnet20
        model_ctor = lambda: resnet20(num_classes=10)

    model_A = model_ctor()
    model_B = model_ctor()
    model_A.load_state_dict(sd_A, strict=True)
    model_B.load_state_dict(sd_B, strict=True)

    perms = find_perms_activation_matching(
        model_A, model_B, loader, device, n_samples=n_samples, max_iters=max_iters,
    )
    sd_B_aligned = apply_permutations(sd_B, perms)
    return sd_B_aligned


# ---------------------------------------------------------------------------
# Sanity checks (run with `python -m src.merging.permute`)
# ---------------------------------------------------------------------------


def _all_state_keys_covered(sd: dict) -> tuple[set[str], set[str]]:
    """Return (missing, extra) keys between sd and PARAM_PERM_MAP."""
    return (set(sd) - set(PARAM_PERM_MAP), set(PARAM_PERM_MAP) - set(sd))


if __name__ == "__main__":
    from src.models.resnet20 import resnet20

    # 1. PARAM_PERM_MAP covers every key in a fresh ResNet-20 state_dict.
    m = resnet20()
    sd = m.state_dict()
    missing, extra = _all_state_keys_covered(sd)
    assert not missing, f"PARAM_PERM_MAP missing keys: {sorted(missing)}"
    if extra:
        print(f"note: PARAM_PERM_MAP has {len(extra)} unused keys: {sorted(extra)[:5]}...")
    print(f"OK: PARAM_PERM_MAP covers all {len(sd)} state_dict keys")

    # 2. Identity permutation does not change the state_dict.
    sd_id = apply_permutations(sd, Permutations.identity())
    for k in sd:
        if sd[k].dtype.is_floating_point:
            assert torch.allclose(sd_id[k], sd[k]), f"identity perm changed {k!r}"
    print("OK: identity permutation leaves state_dict invariant")

    # 3. Permuted model computes the same function (output invariance).
    torch.manual_seed(0)
    sd2 = resnet20().state_dict()
    # Manual random permutations
    rand_perms = Permutations(perms={
        name: torch.randperm(dim) for name, dim in PERM_DIMS.items()
    })
    sd2_permuted = apply_permutations(sd2, rand_perms)
    m_orig = resnet20()
    m_perm = resnet20()
    m_orig.load_state_dict(sd2, strict=True)
    m_perm.load_state_dict(sd2_permuted, strict=True)
    m_orig.eval()
    m_perm.eval()
    x_test = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        y_orig = m_orig(x_test)
        y_perm = m_perm(x_test)
    diff = (y_orig - y_perm).abs().max().item()
    assert diff < 1e-4, f"permutation broke functional invariance: max diff {diff}"
    print(f"OK: permuted model is functionally equivalent (max output diff {diff:.2e})")
