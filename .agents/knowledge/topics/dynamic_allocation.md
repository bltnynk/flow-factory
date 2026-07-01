# Dynamic (Adaptive) Rollout Allocation

**Read when**: Touching `trainers/dynamic_allocation.py`, the `dynamic_allocation*`
fields in `hparams/training_args/_base.py`, or the `sample()` / `prepare_feedback()`
dispatch in `trainers/{grpo,nft,awm}.py`.

---

## What it does

Baseline online RL (FlowGRPO, DiffusionNFT, AWM) rolls out a fixed `K`
(`group_size`) images for **every** prompt. Dynamic allocation re-distributes a
**fixed** per-epoch rollout budget `M * K` (`M = unique_sample_num_per_epoch`)
across prompts based on a per-prompt metric, in two phases:

1. **Phase 1** — roll out `K1 = clamp(round(x * K), 2, K-1)` images per prompt
   (`x = dynamic_allocation_phase1_ratio`) and score each prompt with a metric
   over its phase-1 rewards.
2. **Phase 2** — allocate the remaining `M * (K - K1)` budget across prompts by
   the metric ranking; total stays `M * K`, only per-prompt counts change.

Opt-in via `training_args.dynamic_allocation` (default `False`). When off, the
trainers run the uniform `BaseTrainer.generate_samples` path byte-identically.

## Config fields (`TrainingArguments`, shared by GRPO/NFT/AWM)

| Field | Meaning |
|-------|---------|
| `dynamic_allocation` | Master switch (default `False`). |
| `dynamic_allocation_phase1_ratio` | `x` in `(0,1)`; `K1 = clamp(round(x*K), 2, K-1)`. |
| `dynamic_allocation_metric` | Per-prompt metric registry key: `std` (default), `mean`, `advantage` (mean absolute group-relative advantage `mean(\|r - group_mean\|)`). |
| `dynamic_allocation_metric_direction` | `higher` (default) or `lower` — which metric value earns more phase-2 budget. |
| `dynamic_allocation_strategy` | `proportional` (default) or `rank_linear`. |
| `dynamic_allocation_min_per_prompt` | Min phase-2 (extra) rollouts guaranteed per prompt. |

## How it works (per epoch)

`generate_samples_dynamicallocation(trainer, ...)` replaces the trainer's
`sample()` body:

1. Select `M` epoch prompt **dataset indices** deterministically
   (`randperm(seed + epoch)[:M]`, identical on every rank — mirrors the samplers).
2. Build phase-1 job list (each index × `K1`), slice **evenly** across ranks.
   Generate by indexing the dataset directly — `dataloader.dataset[i]` +
   `dataloader.collate_fn` (preprocessing is cached and index-addressable, so any
   rank rolls out any prompt) — via `trainer.sample_batch` (reuses metadata
   inject / CPU offload / reward-buffer feed).
3. Finalize phase-1 rewards, aggregate per sample (reward weights), then
   `gather_object` the `(dataset_index, reward)` pairs across ranks and reduce to
   one metric per prompt → **global** ranking.
4. Apply the allocation strategy → integer extra-budget per prompt
   (largest-remainder, summing exactly to `M*(K-K1)`).
5. Build phase-2 job list (each index × its extra), slice across ranks, generate.
6. Combine phase-1 + phase-2 samples; stash merged per-sample rewards on
   `trainer._dynamicallocation_rewards`. `prepare_feedback` reuses them (guarded
   branch) instead of re-finalizing — no double reward compute.

## Why the invariants still hold

- **Equal per-rank counts.** The phase split is balanced so every rank ends with
  exactly `M*K/W` samples regardless of allocation skew (`phase2_count[r] =
  per_rank_total - phase1_count[r]`). This preserves the `distributed_k_repeat`
  gather + `_to_local` reshape (`advantage_processor.py`) and the gradient
  accumulation step count.
- **Forced gather path.** Dynamic mode forces `data_args.sampler_type =
  distributed_k_repeat` (`Arguments._resolve_sampler_type`): a prompt's rollouts
  are scattered across ranks, so reward/advantage grouping must gather. The
  training DataLoader's sampler is built but **not iterated** (generation is
  index-addressed); the type only drives the advantage communication strategy.
- **Automatic grouping.** All rollouts of a prompt share a content-derived
  `unique_id` (`samples.py`), so variable group sizes are grouped by the existing
  reward/advantage `np.unique` + `np.bincount` path with no change (constraint #14).

## Per-prompt logging

Each epoch, `_log_prompt_allocation` (main process only) emits, for every
training prompt: the **prompt text**, the **number of rollouts allocated**
(`K1 + extra`), and the **average reward over all its rollouts** (phase 1 +
phase 2 — phase-2 rewards are gathered too, not just the phase-1 scoring slice),
plus the phase-1 metric value. Three sinks:

1. Scalar spreads `train/dynalloc_{phase1_k1, metric_mean, metric_std,
   budget_min, budget_max, budget_std}` (curves).
2. A per-prompt table `train/dynalloc_prompts` to the experiment tracker
   (`LogTable` → wandb Table / SwanLab columns).
3. A durable per-epoch JSONL at
   `{save_dir}/{run_name}/dynamic_allocation/epoch_{epoch:04d}.jsonl`
   (one record per prompt: `prompt`, `num_rollouts`, `avg_reward`,
   `phase1_avg_reward`, `phase1_<metric>`) — written when `save_dir` is set;
   ideal for offline analysis of whether the allocation tracks reward.

## Scope limits (v1, enforced with fail-fast errors)

- **Single training source** (one `data.datasets` train entry) — generation uses
  the one underlying DataLoader's `dataset` + `collate_fn`.
- **Pointwise rewards only** — the metric does the grouping; a prompt's rollouts
  are scattered across ranks, so groupwise reward models (which need
  group-complete ranks) are rejected.
- **`group_size >= 3`** — phase 1 needs `>= 2` rollouts/prompt for a metric and
  phase 2 needs a non-empty remainder.

## Extending metrics / strategies

Register at import time (no core edits):

```python
@register_dynamic_allocation_metric("entropy")
def _entropy(phase1_rewards: np.ndarray) -> float: ...

@register_dynamic_allocation_strategy("topk")
def _topk(metric, total_extra, direction, floor) -> np.ndarray: ...
```

then select via `dynamic_allocation_metric` / `dynamic_allocation_strategy`.

## Metric direction note

`std` with `higher` is well-motivated: a zero-variance group yields zero
advantage/gradient in GRPO/NFT/AWM (tracked as `reward_zero_std_ratio`), so
high-std prompts buy more useful signal. For `mean` the direction is a
hypothesis — high-mean prompts are often near the reward ceiling (low variance →
little to learn), so testing `lower` is reasonable.

## Files

- `trainers/dynamic_allocation.py` — all new logic (metric/strategy registries,
  two-phase driver). Isolated; baseline untouched.
- `trainers/{grpo,nft,awm}.py` — guarded dispatch in `sample()` +
  reward-source branch in `prepare_feedback()`.
- `hparams/training_args/_base.py` — config fields + range validation.
- `hparams/args.py` — `_resolve_sampler_type` override.
- `examples/{grpo,nft,awm}/lora/sd3_5/dynamic_allocation.yaml` — runnable configs.

## Cross-refs

- `topics/samplers.md` (sampler types, geometry, `distributed_k_repeat`)
- `topics/sample_lifecycle.md` (`sample()` → `prepare_feedback()` → `optimize()`)
- `architecture.md` "Advantage Computation"; `constraints.md` #9, #9a, #14
