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

# src/flow_factory/trainers/dynamic_allocation.py
"""Dynamic (adaptive) rollout allocation for online RL.

Baseline online-RL trainers (FlowGRPO, DiffusionNFT, AWM) roll out the *same*
number of images ``K`` (``group_size``) for every prompt, implicitly assuming
every prompt contributes equally to learning. This module implements a
two-phase alternative that re-distributes a fixed rollout budget across prompts
based on a per-prompt metric measured from a first-phase rollout:

1. **Phase 1** — roll out a user-set fraction ``x%`` of the per-prompt budget
   for every prompt (``K1 = round(x% * K)`` rollouts each), then score each
   prompt with a configurable metric (e.g. mean or std of its phase-1 reward).
2. **Phase 2** — rank prompts by the metric and allocate the remaining budget
   unevenly, so prompts judged more useful receive more rollouts.

The epoch's **total** rollout count stays fixed at ``M * K`` (``M`` =
``unique_sample_num_per_epoch``); only the per-prompt counts change and become
unequal. This keeps the optimize-loop geometry (per-rank sample count, gradient
accumulation) identical to the uniform baseline, so the result is a
compute-matched comparison.

Design notes
------------
* **Isolated from the baseline.** Trainers dispatch here only when
  ``training_args.dynamic_allocation`` is ``True``; otherwise the legacy
  ``BaseTrainer.generate_samples`` path runs byte-identically.
* **Index-addressed generation.** Preprocessing is cached and the training
  dataset is index-addressable, so any rank can roll out any prompt by dataset
  index via ``dataloader.dataset[i]`` + ``dataloader.collate_fn``. This decouples
  prompt ownership from ranks and lets a globally-ranked allocation be sliced
  evenly across ranks.
* **Equal per-rank counts.** The phase split is balanced so every rank ends a
  sampling epoch with exactly ``M * K / W`` samples, preserving the
  ``distributed_k_repeat`` gather + reshape advantage path and the gradient
  accumulation step count.
* **Grouping is automatic.** All rollouts of a prompt share a content-derived
  ``unique_id`` (see ``samples.py``), so the downstream reward/advantage
  grouping handles variable group sizes with no change.

Extending the metric set
------------------------
Register a new per-prompt metric with :func:`register_dynamic_allocation_metric`::

    @register_dynamic_allocation_metric("entropy")
    def _entropy(phase1_rewards: np.ndarray) -> float:
        ...

and select it via ``training_args.dynamic_allocation_metric: "entropy"``.
"""

import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from accelerate.utils import gather_object

from ..logger import LogTable
from ..samples import BaseSample
from ..utils.logger_utils import setup_logger

if TYPE_CHECKING:
    from .abc import BaseTrainer

logger = setup_logger(__name__)


# ============================ Metric registry ============================
# A metric maps a 1-D array of a prompt's phase-1 (aggregated) rewards to a
# single scalar score. Higher/lower-gets-more-budget is decided separately by
# ``dynamic_allocation_metric_direction``.
DynamicAllocationMetric = Callable[[np.ndarray], float]

_METRIC_REGISTRY: Dict[str, DynamicAllocationMetric] = {}


def register_dynamic_allocation_metric(
    name: str,
) -> Callable[[DynamicAllocationMetric], DynamicAllocationMetric]:
    """Register a per-prompt allocation metric under *name* (case-insensitive).

    Args:
        name: Key used in ``training_args.dynamic_allocation_metric``.

    Returns:
        Decorator that registers and returns the metric function unchanged.
    """
    key = name.lower()

    def _decorator(func: DynamicAllocationMetric) -> DynamicAllocationMetric:
        if key in _METRIC_REGISTRY:
            raise ValueError(f"Dynamic-allocation metric '{key}' is already registered.")
        _METRIC_REGISTRY[key] = func
        return func

    return _decorator


def get_dynamic_allocation_metric(name: str) -> DynamicAllocationMetric:
    """Resolve a registered metric by name, raising with the valid options."""
    key = name.lower()
    if key not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown dynamic_allocation_metric '{name}'. "
            f"Registered metrics: {sorted(_METRIC_REGISTRY)}."
        )
    return _METRIC_REGISTRY[key]


@register_dynamic_allocation_metric("mean")
def _metric_mean(phase1_rewards: np.ndarray) -> float:
    """Mean phase-1 reward of a prompt."""
    return float(np.mean(phase1_rewards))


@register_dynamic_allocation_metric("std")
def _metric_std(phase1_rewards: np.ndarray) -> float:
    """Standard deviation of a prompt's phase-1 rewards.

    High std => the prompt produces a strong within-group advantage signal
    (a zero-std group contributes no gradient in GRPO/NFT/AWM), so allocating
    more budget to high-std prompts buys more useful learning signal.
    """
    return float(np.std(phase1_rewards))


@register_dynamic_allocation_metric("advantage")
def _metric_advantage(phase1_rewards: np.ndarray) -> float:
    """Mean absolute group-relative advantage of a prompt's phase-1 rewards.

    Computes the GRPO/NFT/AWM advantage *numerator* per rollout —
    ``advantage_i = reward_i - mean(rewards)`` ("use the mean and the reward",
    no std denominator) — then reduces to one per-prompt scalar as the mean
    absolute advantage ``mean(|advantage_i|)``. (The plain mean of the
    advantages is identically zero, so a magnitude reduction is required.)

    Like ``std`` it grows with within-group reward spread — so prompts whose
    phase-1 rollouts disagree most (the strongest, least-degenerate advantage
    signal) rank highest and earn more phase-2 budget — but uses the L1
    (mean-absolute) reduction instead of L2 (root-mean-square).
    """
    rewards = np.asarray(phase1_rewards, dtype=np.float64)
    return float(np.mean(np.abs(rewards - rewards.mean())))


# ======================== Allocation strategy registry ========================
# A strategy maps a per-prompt metric vector to an integer extra-budget vector
# that sums to ``total_extra`` (the phase-2 budget). ``direction`` is "higher"
# (high metric -> more budget) or "lower"; ``floor`` is the minimum extra budget
# guaranteed to every prompt.
AllocationStrategy = Callable[[np.ndarray, int, str, int], np.ndarray]

_STRATEGY_REGISTRY: Dict[str, AllocationStrategy] = {}


def register_dynamic_allocation_strategy(
    name: str,
) -> Callable[[AllocationStrategy], AllocationStrategy]:
    """Register a phase-2 allocation strategy under *name* (case-insensitive)."""
    key = name.lower()

    def _decorator(func: AllocationStrategy) -> AllocationStrategy:
        if key in _STRATEGY_REGISTRY:
            raise ValueError(f"Dynamic-allocation strategy '{key}' is already registered.")
        _STRATEGY_REGISTRY[key] = func
        return func

    return _decorator


def get_dynamic_allocation_strategy(name: str) -> AllocationStrategy:
    """Resolve a registered strategy by name, raising with the valid options."""
    key = name.lower()
    if key not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown dynamic_allocation_strategy '{name}'. "
            f"Registered strategies: {sorted(_STRATEGY_REGISTRY)}."
        )
    return _STRATEGY_REGISTRY[key]


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Integer apportionment of *total* across *weights* (largest-remainder).

    Returns a non-negative integer array summing exactly to ``total``. A
    degenerate (all-zero / non-finite) weight vector falls back to a uniform
    split.

    Args:
        weights: Non-negative weights, shape ``(M,)``.
        total: Integer total to apportion (``>= 0``).

    Returns:
        Integer array of counts, shape ``(M,)``, summing to ``total``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    if total <= 0 or len(weights) == 0:
        return np.zeros(len(weights), dtype=np.int64)
    if weights.sum() <= 0.0:
        weights = np.ones_like(weights)

    quota = weights / weights.sum() * total
    counts = np.floor(quota).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder > 0:
        frac = quota - counts
        winners = np.argsort(-frac)[:remainder]
        counts[winners] += 1
    return counts


def _apply_floor_and_apportion(weights: np.ndarray, total_extra: int, floor: int) -> np.ndarray:
    """Guarantee *floor* extra per prompt, apportion the rest by *weights*."""
    num_prompts = len(weights)
    reserved = floor * num_prompts
    if reserved > total_extra:
        raise ValueError(
            f"dynamic_allocation_min_per_prompt({floor}) * num_prompts({num_prompts}) "
            f"= {reserved} exceeds the phase-2 budget ({total_extra}). "
            "Lower `dynamic_allocation_min_per_prompt` or `dynamic_allocation_phase1_ratio`."
        )
    rest = _largest_remainder(weights, total_extra - reserved)
    return np.full(num_prompts, floor, dtype=np.int64) + rest


@register_dynamic_allocation_strategy("proportional")
def _strategy_proportional(
    metric: np.ndarray, total_extra: int, direction: str, floor: int
) -> np.ndarray:
    """Allocate phase-2 budget proportionally to the (shifted) metric."""
    if direction == "higher":
        weights = metric - metric.min()
    else:
        weights = metric.max() - metric
    return _apply_floor_and_apportion(weights, total_extra, floor)


@register_dynamic_allocation_strategy("rank_linear")
def _strategy_rank_linear(
    metric: np.ndarray, total_extra: int, direction: str, floor: int
) -> np.ndarray:
    """Allocate phase-2 budget by rank, linearly from top (most) to bottom.

    Magnitude-insensitive: only the ordering of the metric matters, so a single
    outlier prompt cannot capture the whole budget.
    """
    num_prompts = len(metric)
    order = np.argsort(metric)  # ascending
    ranks = np.empty(num_prompts, dtype=np.float64)
    ranks[order] = np.arange(1, num_prompts + 1)  # high metric -> high rank
    if direction == "lower":
        ranks = (num_prompts + 1) - ranks
    return _apply_floor_and_apportion(ranks, total_extra, floor)


# ============================ Generation helpers ============================


def _even_split(total: int, num_parts: int) -> List[int]:
    """Split *total* into *num_parts* near-equal integer parts summing to total."""
    base, rem = divmod(total, num_parts)
    return [base + 1 if r < rem else base for r in range(num_parts)]


def _select_epoch_prompt_indices(trainer: "BaseTrainer", num_prompts: int) -> List[int]:
    """Pick ``num_prompts`` dataset indices for this epoch, identical on every rank.

    Mirrors the samplers' deterministic ``randperm(seed + epoch)[:M]`` selection
    so the prompt set matches the uniform baseline for the same seed/epoch.
    """
    dataset = trainer.dataloader.dataset
    dataset_size = len(dataset)
    if num_prompts > dataset_size:
        raise ValueError(
            f"unique_sample_num_per_epoch({num_prompts}) exceeds the training dataset "
            f"size ({dataset_size}); cannot select that many unique prompts."
        )
    generator = torch.Generator()
    generator.manual_seed(trainer.training_args.seed + trainer.epoch)
    return torch.randperm(dataset_size, generator=generator)[:num_prompts].tolist()


def _generate_for_indices(
    trainer: "BaseTrainer",
    dataset_indices: List[int],
    reward_buffer: Any,
    infer_kwargs: Dict[str, Any],
) -> List[BaseSample]:
    """Roll out one image per dataset index, in ``per_device_batch_size`` chunks.

    Reuses ``trainer.sample_batch`` so dataset-metadata injection, CPU offload,
    and reward-buffer feeding match the baseline path exactly.

    Args:
        trainer: The active trainer (provides adapter, dataloader, args).
        dataset_indices: This rank's dataset indices to roll out (one image each).
        reward_buffer: Reward buffer fed via ``sample_batch`` (rewards computed
            at ``finalize``).
        infer_kwargs: Forwarded to ``adapter.inference`` (e.g. ``compute_log_prob``,
            ``trajectory_indices``).

    Returns:
        Generated samples, in the same order as ``dataset_indices``.
    """
    dataset = trainer.dataloader.dataset
    collate_fn = trainer.dataloader.collate_fn
    batch_size = trainer.training_args.per_device_batch_size

    samples: List[BaseSample] = []
    for start in range(0, len(dataset_indices), batch_size):
        chunk = dataset_indices[start : start + batch_size]
        batch = collate_fn([dataset[i] for i in chunk])
        sample_batch = trainer.sample_batch(batch, reward_buffer=reward_buffer, **infer_kwargs)
        if len(sample_batch) != len(chunk):
            raise RuntimeError(
                "Dynamic allocation expects one sample per dataset index "
                f"(got {len(sample_batch)} samples for {len(chunk)} indices). "
                "Adapters that emit multiple images per prompt row are not "
                "supported in dynamic-allocation mode."
            )
        samples.extend(sample_batch)
    return samples


def _aggregate_rewards(trainer: "BaseTrainer", rewards: Dict[str, torch.Tensor]) -> np.ndarray:
    """Weighted-sum the per-model reward tensors into one scalar per sample.

    Uses the same per-reward weights the ``AdvantageProcessor`` uses. For the
    single-source pointwise setting required by dynamic allocation, each reward's
    weight dict has one entry, so this is the (weighted) sum across reward models.
    """
    reward_weights = trainer.advantage_processor.reward_weights
    aggregated: Optional[np.ndarray] = None
    for name, value in rewards.items():
        arr = torch.as_tensor(value).detach().cpu().numpy().astype(np.float64).reshape(-1)
        per_ds = reward_weights.get(name, {})
        weight = float(next(iter(per_ds.values()))) if per_ds else 1.0
        contrib = arr * weight
        aggregated = contrib if aggregated is None else aggregated + contrib
    if aggregated is None:
        raise RuntimeError("No rewards were produced for the phase-1 rollout.")
    return aggregated


def _validate_dynamic_allocation(trainer: "BaseTrainer") -> None:
    """Fail fast on configurations dynamic allocation does not support (v1)."""
    ta = trainer.training_args

    # Single training source only: index-addressed generation uses the one
    # underlying DataLoader's dataset + collate_fn.
    if len(getattr(trainer, "train_dataloaders_by_source", {})) > 1:
        raise NotImplementedError(
            "dynamic_allocation supports a single training source only; "
            f"found {len(trainer.train_dataloaders_by_source)} sources. "
            "Use one `data.datasets` training entry."
        )
    if not hasattr(trainer.dataloader, "dataset") or not hasattr(trainer.dataloader, "collate_fn"):
        raise NotImplementedError(
            "dynamic_allocation requires a plain indexable training DataLoader "
            "(single source). The active dataloader is not index-addressable."
        )

    # Pointwise rewards only: the per-prompt metric does the grouping, and the
    # gather path scatters a prompt's rollouts across ranks (groupwise reward
    # models need group-complete ranks).
    groupwise = list(trainer.reward_processor._groupwise_models)
    if groupwise:
        raise NotImplementedError(
            "dynamic_allocation supports pointwise reward models only; "
            f"found groupwise reward(s): {groupwise}. "
            "Remove them or use the uniform baseline."
        )

    if ta.group_size < 3:
        raise ValueError(
            f"dynamic_allocation needs group_size >= 3 (got {ta.group_size}): "
            "phase 1 needs >= 2 rollouts per prompt for a meaningful metric and "
            "phase 2 needs a non-empty remaining budget."
        )


def _resolve_phase1_count(trainer: "BaseTrainer") -> int:
    """Resolve K1 (phase-1 rollouts per prompt) with valid-range clamping."""
    ta = trainer.training_args
    k = ta.group_size
    k1 = int(round(ta.dynamic_allocation_phase1_ratio * k))
    k1 = max(2, min(k1, k - 1))  # >= 2 for a metric; <= K-1 so phase 2 is non-empty
    return k1


# ============================ Public entry point ============================


def generate_samples_dynamicallocation(
    trainer: "BaseTrainer",
    *,
    compute_log_prob: bool = False,
    trajectory_indices: Optional[List[int]] = None,
    **extra_inference_kwargs: Any,
) -> List[BaseSample]:
    """Two-phase adaptive-allocation replacement for ``BaseTrainer.generate_samples``.

    Drop-in for the trainers' ``sample()`` body: takes the same
    ``compute_log_prob`` / ``trajectory_indices`` / ``extra_inference_kwargs``
    that ``generate_samples`` forwards to ``adapter.inference``, returns the full
    list of this rank's samples for the epoch, and stashes the precomputed
    per-sample rewards on ``trainer._dynamicallocation_rewards`` (aligned to the
    returned samples) so the trainer's ``prepare_feedback`` can skip a redundant
    reward pass.

    Args:
        trainer: The active GRPO/NFT/AWM trainer.
        compute_log_prob: Whether inference stores log-probabilities (GRPO: True).
        trajectory_indices: Which trajectory positions inference stores
            (``[-1]`` for NFT/AWM, full list for GRPO).
        **extra_inference_kwargs: Forwarded verbatim to ``adapter.inference``.

    Returns:
        This rank's samples for the epoch (phase-1 then phase-2 order), each
        carrying ``extra_kwargs['rewards']``.
    """
    _validate_dynamic_allocation(trainer)

    accelerator = trainer.accelerator
    ta = trainer.training_args
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    num_prompts = ta.unique_sample_num_per_epoch  # M
    group_size = ta.group_size  # K
    k1 = _resolve_phase1_count(trainer)  # K1
    total_budget = num_prompts * group_size  # N = M * K
    phase1_budget = num_prompts * k1  # N1 = M * K1
    phase2_budget = total_budget - phase1_budget  # N2 = M * (K - K1)

    if total_budget % world_size != 0:
        raise RuntimeError(
            f"Total rollout budget M*K = {total_budget} is not divisible by world_size "
            f"({world_size}); cannot keep per-rank sample counts equal. This should have "
            "been enforced by the distributed_k_repeat geometry alignment."
        )
    per_rank_total = total_budget // world_size

    infer_kwargs = {
        "compute_log_prob": compute_log_prob,
        "trajectory_indices": trajectory_indices,
        **extra_inference_kwargs,
    }

    # Identical prompt set + per-rank phase split on every rank. Phase 1 is split
    # evenly; phase 2 takes the per-rank remainder so every rank ends with exactly
    # `per_rank_total` samples (required by the distributed advantage reshape and
    # the gradient-accumulation step count). Both phases must be non-empty on
    # every rank: each rank calls `reward_buffer.finalize` (a barrier) once per
    # phase, so an empty phase on one rank would either crash that finalize or
    # desync the barrier.
    prompt_indices = _select_epoch_prompt_indices(trainer, num_prompts)
    phase1_counts = _even_split(phase1_budget, world_size)
    phase2_counts = [per_rank_total - n1 for n1 in phase1_counts]
    if min(phase1_counts) < 1 or min(phase2_counts) < 1:
        raise ValueError(
            "dynamic_allocation cannot give every rank >= 1 rollout in both phases "
            f"(world_size={world_size}, K1={k1}, phase1_per_rank={phase1_counts}, "
            f"phase2_per_rank={phase2_counts}). Increase unique_sample_num_per_epoch "
            f"(currently {num_prompts}) or group_size (currently {group_size}), or use "
            "fewer processes. As a rule of thumb, unique_sample_num_per_epoch >= "
            "world_size keeps both phases populated."
        )

    trainer.adapter.rollout()

    # ----- Phase 1: uniform K1 rollouts per prompt, this rank's slice. -----
    phase1_jobs = [idx for idx in prompt_indices for _ in range(k1)]  # length N1
    p1_start = sum(phase1_counts[:rank])
    local_phase1_jobs = phase1_jobs[p1_start : p1_start + phase1_counts[rank]]

    trainer.reward_buffer.clear()
    with torch.no_grad(), trainer.autocast():
        phase1_samples = _generate_for_indices(
            trainer, local_phase1_jobs, trainer.reward_buffer, infer_kwargs
        )
    phase1_rewards = trainer.reward_buffer.finalize(store_to_samples=True, split="all")
    phase1_agg = _aggregate_rewards(trainer, phase1_rewards)  # (n1_r,)

    # ----- Score prompts globally, then allocate the remaining budget. -----
    metric_fn = get_dynamic_allocation_metric(ta.dynamic_allocation_metric)
    metric_vec, phase1_rewards_by_idx, prompt_by_idx = _gather_phase1_records(
        metric_fn, local_phase1_jobs, phase1_agg, phase1_samples, prompt_indices
    )
    strategy = get_dynamic_allocation_strategy(ta.dynamic_allocation_strategy)
    extra_per_prompt = strategy(
        metric_vec,
        phase2_budget,
        ta.dynamic_allocation_metric_direction,
        ta.dynamic_allocation_min_per_prompt,
    )
    if int(extra_per_prompt.sum()) != phase2_budget:
        raise RuntimeError(
            f"Allocation strategy produced {int(extra_per_prompt.sum())} phase-2 rollouts "
            f"but the budget is {phase2_budget}. This is an allocation-strategy bug."
        )

    # ----- Phase 2: per-prompt extra rollouts, this rank's slice. -----
    phase2_jobs = [
        idx for idx, count in zip(prompt_indices, extra_per_prompt.tolist()) for _ in range(count)
    ]  # length N2
    p2_start = sum(phase2_counts[:rank])
    local_phase2_jobs = phase2_jobs[p2_start : p2_start + phase2_counts[rank]]

    trainer.reward_buffer.clear()
    with torch.no_grad(), trainer.autocast():
        phase2_samples = _generate_for_indices(
            trainer, local_phase2_jobs, trainer.reward_buffer, infer_kwargs
        )
    phase2_rewards = trainer.reward_buffer.finalize(store_to_samples=True, split="all")
    phase2_agg = _aggregate_rewards(trainer, phase2_rewards)  # (n2_r,)
    phase2_rewards_by_idx = _gather_phase2_records(local_phase2_jobs, phase2_agg, prompt_indices)

    # ----- Combine phase-1 + phase-2 and stash the merged rewards. -----
    samples = phase1_samples + phase2_samples
    merged_rewards = {
        name: torch.cat(
            [
                torch.as_tensor(phase1_rewards[name]).reshape(-1).cpu(),
                torch.as_tensor(phase2_rewards[name]).reshape(-1).cpu(),
            ]
        )
        for name in phase1_rewards
    }
    trainer._dynamicallocation_rewards = merged_rewards

    if len(samples) != per_rank_total:
        raise RuntimeError(
            f"Dynamic allocation produced {len(samples)} samples on rank {rank} but "
            f"expected exactly per_rank_total={per_rank_total}. Equal per-rank counts "
            "are required by the distributed advantage / gradient-accumulation paths."
        )

    _log_prompt_allocation(
        trainer,
        prompt_indices=prompt_indices,
        prompt_by_idx=prompt_by_idx,
        k1=k1,
        total_per_prompt=(k1 + extra_per_prompt),
        metric_vec=metric_vec,
        phase1_rewards_by_idx=phase1_rewards_by_idx,
        phase2_rewards_by_idx=phase2_rewards_by_idx,
    )
    return samples


def _gather_phase1_records(
    metric_fn: DynamicAllocationMetric,
    local_phase1_jobs: List[int],
    local_phase1_agg: np.ndarray,
    local_phase1_samples: List[BaseSample],
    prompt_indices: List[int],
) -> Tuple[np.ndarray, Dict[int, List[float]], Dict[int, str]]:
    """Gather phase-1 ``(index, prompt, reward)`` across ranks; reduce to a metric.

    Phase-1 rollouts of a prompt are scattered across ranks (the phase-1 job list
    is sliced evenly), so the per-prompt metric — and the prompt text + rewards
    used for logging — are assembled only after an all-gather. ``gather_object``
    is used (per-rank phase-1 counts may differ by one), keyed by dataset index.

    Args:
        metric_fn: Per-prompt metric over a prompt's phase-1 rewards.
        local_phase1_jobs: This rank's phase-1 dataset indices (one per sample).
        local_phase1_agg: This rank's aggregated phase-1 rewards, aligned to
            ``local_phase1_jobs`` and ``local_phase1_samples``.
        local_phase1_samples: This rank's phase-1 samples (for prompt text).
        prompt_indices: The epoch's full prompt index list (defines output order).

    Returns:
        ``(metric_vec, phase1_rewards_by_idx, prompt_by_idx)`` where ``metric_vec``
        is aligned to ``prompt_indices``.
    """
    local_records: List[Tuple[int, str, float]] = [
        (
            int(idx),
            (
                sample.prompt
                if getattr(sample, "prompt", None) is not None
                else f"<dataset_index={idx}>"
            ),
            float(reward),
        )
        for idx, sample, reward in zip(local_phase1_jobs, local_phase1_samples, local_phase1_agg)
    ]
    gathered: List[Tuple[int, str, float]] = gather_object(local_records)

    rewards_by_idx: Dict[int, List[float]] = {idx: [] for idx in prompt_indices}
    prompt_by_idx: Dict[int, str] = {}
    for idx, prompt, reward in gathered:
        # ``prompt_indices`` are unique within an epoch; every gathered record
        # maps to exactly one entry.
        rewards_by_idx[idx].append(reward)
        prompt_by_idx.setdefault(idx, prompt)

    metric_vec = np.empty(len(prompt_indices), dtype=np.float64)
    for i, idx in enumerate(prompt_indices):
        prompt_rewards = rewards_by_idx[idx]
        if not prompt_rewards:
            raise RuntimeError(
                f"Prompt (dataset index {idx}) received no phase-1 rollouts; "
                "cannot compute its allocation metric."
            )
        metric_vec[i] = metric_fn(np.asarray(prompt_rewards, dtype=np.float64))
    return metric_vec, rewards_by_idx, prompt_by_idx


def _gather_phase2_records(
    local_phase2_jobs: List[int],
    local_phase2_agg: np.ndarray,
    prompt_indices: List[int],
) -> Dict[int, List[float]]:
    """Gather phase-2 ``(index, reward)`` across ranks, grouped by dataset index.

    Used (with the phase-1 rewards) to report each prompt's average reward over
    **all** its rollouts. Prompts with no phase-2 rollouts map to an empty list.
    """
    local_records: List[Tuple[int, float]] = [
        (int(idx), float(reward)) for idx, reward in zip(local_phase2_jobs, local_phase2_agg)
    ]
    gathered: List[Tuple[int, float]] = gather_object(local_records)
    rewards_by_idx: Dict[int, List[float]] = {idx: [] for idx in prompt_indices}
    for idx, reward in gathered:
        rewards_by_idx[idx].append(reward)
    return rewards_by_idx


def _log_prompt_allocation(
    trainer: "BaseTrainer",
    *,
    prompt_indices: List[int],
    prompt_by_idx: Dict[int, str],
    k1: int,
    total_per_prompt: np.ndarray,
    metric_vec: np.ndarray,
    phase1_rewards_by_idx: Dict[int, List[float]],
    phase2_rewards_by_idx: Dict[int, List[float]],
) -> None:
    """Log per-prompt allocation (main process only).

    Emits, for every training prompt this epoch: the prompt text, the number of
    rollouts allocated to it, and its average reward over **all** its rollouts
    (phase 1 + phase 2). Three sinks:

    1. Scalar spread metrics (``train/dynalloc_*``) for curves.
    2. A per-prompt table (``train/dynalloc_prompts``) to the experiment tracker
       (rendered as a table in wandb / columns in SwanLab).
    3. A durable per-epoch JSONL under
       ``{save_dir}/{run_name}/dynamic_allocation/epoch_{epoch:04d}.jsonl`` for
       offline analysis (written only when ``save_dir`` is set).
    """
    if not trainer.accelerator.is_main_process:
        return

    metric_name = trainer.training_args.dynamic_allocation_metric
    rows: List[List[Any]] = []
    records: List[Dict[str, Any]] = []
    for i, idx in enumerate(prompt_indices):
        p1 = phase1_rewards_by_idx.get(idx, [])
        p2 = phase2_rewards_by_idx.get(idx, [])
        all_rewards = p1 + p2
        avg_reward = float(np.mean(all_rewards)) if all_rewards else float("nan")
        phase1_avg = float(np.mean(p1)) if p1 else float("nan")
        num_rollouts = int(total_per_prompt[i])
        prompt = prompt_by_idx.get(idx, f"<dataset_index={idx}>")
        rows.append(
            [
                prompt,
                num_rollouts,
                round(avg_reward, 6),
                round(phase1_avg, 6),
                round(float(metric_vec[i]), 6),
            ]
        )
        records.append(
            {
                "epoch": int(trainer.epoch),
                "dataset_index": int(idx),
                "prompt": prompt,
                "num_rollouts": num_rollouts,
                "avg_reward": avg_reward,
                "phase1_avg_reward": phase1_avg,
                f"phase1_{metric_name}": float(metric_vec[i]),
            }
        )

    columns = ["prompt", "num_rollouts", "avg_reward", "phase1_avg_reward", f"phase1_{metric_name}"]
    trainer.log_data(
        {
            "train/dynalloc_phase1_k1": k1,
            "train/dynalloc_metric_mean": float(np.mean(metric_vec)),
            "train/dynalloc_metric_std": float(np.std(metric_vec)),
            "train/dynalloc_budget_min": int(np.min(total_per_prompt)),
            "train/dynalloc_budget_max": int(np.max(total_per_prompt)),
            "train/dynalloc_budget_std": float(np.std(total_per_prompt)),
            "train/dynalloc_prompts": LogTable(columns=columns, rows=rows),
        },
        step=trainer.step,
    )

    save_dir = getattr(trainer.log_args, "save_dir", None)
    if save_dir:
        out_dir = os.path.join(str(save_dir), str(trainer.log_args.run_name), "dynamic_allocation")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"epoch_{int(trainer.epoch):04d}.jsonl")
        with open(out_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"[dynamic_allocation] wrote per-prompt allocation log -> {out_path}")
