# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/trainers/pivot_advantage.py
"""Pivot-advantage rollout + normalization for online RL (NFT first).

Standard group-relative advantage centers each prompt's rewards on the group
**mean**, so ~half the samples get positive advantage and half negative.
Pivot-advantage replaces the mean with the reward of a purpose-built **pivot**
rollout:

1. **Sampling** — for each prompt group of size ``N``, sample the ``N`` initial
   latents, then build one extra *pivot* latent as the average of those ``N``
   latents rescaled by ``sqrt(N)`` so it is still a valid ``N(0, I)`` draw
   (``mean`` of ``N`` i.i.d. standard normals has variance ``1/N``; ``* sqrt(N)``
   restores unit variance). Roll out all ``N + 1``.
2. **Reward** — compute rewards for all ``N + 1`` samples as usual.
3. **Advantage** — center on the pivot reward instead of the group mean::

       A_i = (reward_i - reward_pivot) / std

   The ``/ std`` scaling is kept (reusing ``global_std``) because NFT/AWM map the
   advantage into ``[0, 1]`` via ``adv_clip_range``; an unscaled
   ``reward_i - reward_pivot`` (~0.05 for PickScore) would collapse the signal.
4. **Optimize** — train on the ``N`` samples only (the pivot is a reference and
   is dropped), so the returned per-epoch sample count is ``M * N`` — identical
   geometry to uniform NFT (gradient accumulation unchanged).

Design notes
------------
* **Isolated from the baseline.** Dispatched only when
  ``training_args.pivot_advantage`` is ``True``; otherwise the standard
  advantage path runs unchanged.
* **Latent injection.** Generation reuses ``trainer.sample_batch`` with an
  injected ``latents=`` (added to ``SD3_5Adapter.inference``); the initial
  latents are sampled up front via ``pipeline.prepare_latents`` so the pivot can
  be formed exactly.
* **Local groups.** ``group_contiguous`` is forced so every group's ``N``
  latents, its pivot, and their rewards live on one rank — the pivot latent and
  pivot-centered advantage are computed with no cross-rank communication (only
  ``global_std`` needs a single all-reduce).

v1 scope (fail-fast): NFT trainer, single training source, pointwise rewards.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch

from ..samples import BaseSample
from ..utils.base import create_generator
from ..utils.logger_utils import setup_logger

if TYPE_CHECKING:
    from .abc import BaseTrainer

logger = setup_logger(__name__)


def _validate_pivot_advantage(trainer: "BaseTrainer") -> None:
    """Fail fast on configurations pivot-advantage does not support (v1)."""
    if len(getattr(trainer, "train_dataloaders_by_source", {})) > 1:
        raise NotImplementedError(
            "pivot_advantage supports a single training source only; found "
            f"{len(trainer.train_dataloaders_by_source)} sources."
        )
    if not hasattr(trainer.dataloader, "dataset") or not hasattr(trainer.dataloader, "collate_fn"):
        raise NotImplementedError(
            "pivot_advantage requires a plain indexable training DataLoader (single source)."
        )
    groupwise = list(trainer.reward_processor._groupwise_models)
    if groupwise:
        raise NotImplementedError(
            "pivot_advantage supports pointwise reward models only; "
            f"found groupwise reward(s): {groupwise}."
        )
    pipeline = trainer.adapter.pipeline
    if not hasattr(pipeline, "prepare_latents") or not hasattr(pipeline, "transformer"):
        raise NotImplementedError(
            "pivot_advantage v1 requires a pipeline exposing `prepare_latents` and "
            "`transformer` (e.g. sd3-5). The active adapter does not."
        )


def _select_epoch_prompt_indices(trainer: "BaseTrainer", num_prompts: int) -> List[int]:
    """Pick ``num_prompts`` dataset indices for this epoch, identical on every rank."""
    dataset = trainer.dataloader.dataset
    dataset_size = len(dataset)
    if num_prompts > dataset_size:
        raise ValueError(
            f"unique_sample_num_per_epoch({num_prompts}) exceeds the training dataset "
            f"size ({dataset_size})."
        )
    generator = torch.Generator()
    generator.manual_seed(trainer.training_args.seed + trainer.epoch)
    return torch.randperm(dataset_size, generator=generator)[:num_prompts].tolist()


def _aggregate_rewards(trainer: "BaseTrainer", rewards: Dict[str, torch.Tensor]) -> np.ndarray:
    """Weighted-sum the per-model reward tensors into one scalar per sample."""
    reward_weights = trainer.advantage_processor.reward_weights
    aggregated: Optional[np.ndarray] = None
    for name, value in rewards.items():
        arr = torch.as_tensor(value).detach().cpu().numpy().astype(np.float64).reshape(-1)
        per_ds = reward_weights.get(name, {})
        weight = float(next(iter(per_ds.values()))) if per_ds else 1.0
        contrib = arr * weight
        aggregated = contrib if aggregated is None else aggregated + contrib
    if aggregated is None:
        raise RuntimeError("No rewards were produced for the pivot-advantage rollout.")
    return aggregated


def generate_samples_pivotadvantage(
    trainer: "BaseTrainer",
    *,
    compute_log_prob: bool = False,
    trajectory_indices: Optional[List[int]] = None,
    **extra_inference_kwargs: Any,
) -> List[BaseSample]:
    """Pivot-advantage replacement for ``BaseTrainer.generate_samples`` (NFT).

    Generates ``N + 1`` rollouts per prompt (``N`` normal + 1 pivot), computes
    pivot-centered advantages for the ``N`` normal samples, and returns only
    those ``N`` (with ``extra_kwargs['advantage']`` set). ``prepare_feedback``
    then skips its own advantage computation.

    Args:
        trainer: The active NFT trainer.
        compute_log_prob: Whether inference stores log-probabilities (NFT: False).
        trajectory_indices: Trajectory positions to store (NFT: ``[-1]``).
        **extra_inference_kwargs: Forwarded to ``adapter.inference``.

    Returns:
        The ``M * N / W`` (this rank's) normal samples, each carrying its
        pivot-advantage in ``extra_kwargs['advantage']``.
    """
    _validate_pivot_advantage(trainer)

    accelerator = trainer.accelerator
    ta = trainer.training_args
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    num_prompts = ta.unique_sample_num_per_epoch  # M
    group_size = ta.group_size  # N
    if num_prompts % world_size != 0:
        raise RuntimeError(
            f"pivot_advantage needs unique_sample_num_per_epoch ({num_prompts}) divisible "
            f"by world_size ({world_size}); group_contiguous alignment should ensure this."
        )
    groups_per_rank = num_prompts // world_size

    pipeline = trainer.adapter.pipeline
    device = accelerator.device
    dtype = pipeline.transformer.dtype
    num_channels_latents = pipeline.transformer.config.in_channels
    height, width = ta.height, ta.width

    infer_kwargs = {
        "compute_log_prob": compute_log_prob,
        "trajectory_indices": trajectory_indices,
        **extra_inference_kwargs,
    }

    prompt_indices = _select_epoch_prompt_indices(trainer, num_prompts)
    local_prompts = prompt_indices[rank * groups_per_rank : (rank + 1) * groups_per_rank]

    trainer.adapter.rollout()
    trainer.reward_buffer.clear()

    # ----- Sample per-group initial latents + pivot; flatten into jobs. -----
    job_prompt: List[int] = []
    job_latent: List[torch.Tensor] = []
    job_is_pivot: List[bool] = []
    for p in local_prompts:
        gen = create_generator(ta.seed, trainer.epoch, int(p), device=device)
        init = pipeline.prepare_latents(
            group_size, num_channels_latents, height, width, dtype, device, gen
        )  # (N, C, h, w)
        # Pivot latent = sqrt(N) * mean(init), computed in fp32 for stability.
        pivot = (init.float().mean(dim=0, keepdim=True) * math.sqrt(group_size)).to(init.dtype)
        for i in range(group_size):
            job_prompt.append(int(p))
            job_latent.append(init[i])
            job_is_pivot.append(False)
        job_prompt.append(int(p))
        job_latent.append(pivot[0])
        job_is_pivot.append(True)

    # ----- Generate all N+1 rollouts per group, injecting the latents. -----
    dataset = trainer.dataloader.dataset
    collate_fn = trainer.dataloader.collate_fn
    batch_size = ta.per_device_batch_size

    samples: List[BaseSample] = []
    sample_prompt: List[int] = []
    sample_is_pivot: List[bool] = []
    with torch.no_grad(), trainer.autocast():
        for start in range(0, len(job_prompt), batch_size):
            chunk_prompts = job_prompt[start : start + batch_size]
            chunk_latents = torch.stack(job_latent[start : start + batch_size], dim=0).to(device)
            batch = collate_fn([dataset[p] for p in chunk_prompts])
            sample_batch = trainer.sample_batch(
                batch,
                reward_buffer=trainer.reward_buffer,
                latents=chunk_latents,
                **infer_kwargs,
            )
            if len(sample_batch) != len(chunk_prompts):
                raise RuntimeError(
                    "pivot_advantage expects one sample per injected latent "
                    f"(got {len(sample_batch)} for {len(chunk_prompts)})."
                )
            samples.extend(sample_batch)
            sample_prompt.extend(chunk_prompts)
            sample_is_pivot.extend(job_is_pivot[start : start + batch_size])

    rewards = trainer.reward_buffer.finalize(store_to_samples=True, split="all")
    aggregated = _aggregate_rewards(trainer, rewards)  # (M/W * (N+1),)

    # ----- Group locally; center advantages on each group's pivot reward. -----
    groups: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"normal": [], "pivot": None})
    for idx, (p, is_pivot) in enumerate(zip(sample_prompt, sample_is_pivot)):
        if is_pivot:
            groups[p]["pivot"] = idx
        else:
            groups[p]["normal"].append(idx)

    normal_indices = [i for g in groups.values() for i in g["normal"]]
    std = _resolve_std(accelerator, aggregated[normal_indices], ta.global_std)

    returned: List[BaseSample] = []
    pivot_rewards: List[float] = []
    advantages: List[float] = []
    for p, g in groups.items():
        if g["pivot"] is None:
            raise RuntimeError(f"pivot_advantage: prompt group {p} has no pivot sample.")
        reward_pivot = float(aggregated[g["pivot"]])
        pivot_rewards.append(reward_pivot)
        group_std = std if ta.global_std else max(float(np.std(aggregated[g["normal"]])), 1e-6)
        for i in g["normal"]:
            adv = (float(aggregated[i]) - reward_pivot) / group_std
            samples[i].extra_kwargs["advantage"] = torch.tensor(adv, device=device)
            advantages.append(adv)
            returned.append(samples[i])

    expected = groups_per_rank * group_size
    if len(returned) != expected:
        raise RuntimeError(
            f"pivot_advantage produced {len(returned)} training samples on rank {rank} "
            f"but expected M/W * N = {expected}. Equal per-rank counts are required by "
            "the optimize loop's gradient accumulation."
        )

    _log_pivot_advantage(
        trainer, aggregated[normal_indices], np.asarray(pivot_rewards), np.asarray(advantages)
    )
    return returned


def _resolve_std(accelerator, normal_rewards: np.ndarray, global_std: bool) -> float:
    """Global std over all normal-sample rewards (all-reduced) when ``global_std``.

    Returns the scalar global std when ``global_std`` is True; otherwise returns
    ``0.0`` (unused — the caller computes a per-group std instead).
    """
    if not global_std:
        return 0.0
    r = np.asarray(normal_rewards, dtype=np.float64)
    t = torch.tensor(
        [float(len(r)), float(r.sum()), float((r**2).sum())],
        device=accelerator.device,
        dtype=torch.float64,
    )
    t = accelerator.reduce(t, reduction="sum")
    n, s, ss = t[0].item(), t[1].item(), t[2].item()
    return max((ss / n - (s / n) ** 2) ** 0.5, 1e-6)


def _log_pivot_advantage(
    trainer: "BaseTrainer",
    normal_rewards: np.ndarray,
    pivot_rewards: np.ndarray,
    advantages: np.ndarray,
) -> None:
    """Log global reward / pivot / advantage stats (main process).

    ``pivot_adv_frac_positive`` is the key diagnostic: unlike mean-centered
    advantage (~0.5), it reports how often a sample beats its group's pivot.
    """
    accelerator = trainer.accelerator

    def _gather(arr: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(arr, dtype=torch.float32, device=accelerator.device)
        return accelerator.gather(t).cpu().numpy()

    g_reward = _gather(normal_rewards)
    g_pivot = _gather(pivot_rewards)
    g_adv = _gather(advantages)

    if not accelerator.is_main_process:
        return
    trainer.log_data(
        {
            "train/reward_mean": float(np.mean(g_reward)),
            "train/reward_std": float(np.std(g_reward)),
            "train/pivot_reward_mean": float(np.mean(g_pivot)),
            "train/pivot_reward_std": float(np.std(g_pivot)),
            "train/pivot_adv_mean": float(np.mean(g_adv)),
            "train/pivot_adv_abs_mean": float(np.mean(np.abs(g_adv))),
            "train/pivot_adv_min": float(np.min(g_adv)),
            "train/pivot_adv_max": float(np.max(g_adv)),
            "train/pivot_adv_frac_positive": float(np.mean(g_adv > 0)),
        },
        step=trainer.step,
    )
