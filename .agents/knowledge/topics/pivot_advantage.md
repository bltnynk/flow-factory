# Pivot-Advantage

**Read when**: Touching `trainers/pivot_advantage.py`, the `pivot_advantage` field
in `hparams/training_args/_base.py`, the `latents=` injection in
`models/stable_diffusion/sd3_5.py::inference`, or the `sample()` /
`prepare_feedback()` pivot dispatch in `trainers/nft.py`.

---

## What it does

Standard group-relative advantage centers a prompt's rewards on the group
**mean**, so ~half the samples are positive and half negative. Pivot-advantage
centers on the reward of a purpose-built **pivot** rollout instead:

1. **Sampling** — for each prompt group of size `N`, sample the `N` initial
   latents, then form one extra *pivot* latent `pivot = sqrt(N) * mean(latents)`.
   The `sqrt(N)` keeps it a valid `N(0, I)` draw (the mean of `N` i.i.d.
   standard normals has variance `1/N`). Roll out all `N + 1`.
2. **Reward** — compute rewards for all `N + 1`.
3. **Advantage** — `A_i = (reward_i - reward_pivot) / std`. The `/ std` is kept
   (reusing the `global_std` flag) because NFT/AWM map advantage into `[0, 1]`
   via `adv_clip_range`; an unscaled `reward_i - reward_pivot` (~0.05 for
   PickScore) would collapse the signal.
4. **Optimize** — train on the `N` samples only (the pivot is a reference and is
   dropped), so the returned per-epoch count is `M * N` — **identical geometry
   to uniform NFT** (gradient accumulation unchanged).

Opt-in via `training_args.pivot_advantage` (default `False`). Off ⇒ the standard
advantage path runs unchanged.

## Config

| Field | Meaning |
|-------|---------|
| `pivot_advantage` | Master switch (default `False`). |
| `global_std` | Reused: `False` ⇒ per-group std of the `N` normal rewards; `True` ⇒ global std (one all-reduce). Centering is always the per-group pivot reward. |

No other new fields — the pivot count is fixed at one per group.

## How it works (per epoch, NFT)

`generate_samples_pivotadvantage(trainer, ...)` replaces the trainer's `sample()`
body:

1. Select `M` epoch prompt indices deterministically; rank `r` owns
   `[r*M/W : (r+1)*M/W]` (group_contiguous ⇒ `M % W == 0`).
2. Per local group: `init = pipeline.prepare_latents(N, ...)`;
   `pivot = sqrt(N) * mean(init)`; build `N` normal + 1 pivot jobs.
3. Generate all jobs in `per_device_batch_size` chunks via `trainer.sample_batch`
   with an **injected** `latents=` (added to `SD3_5Adapter.inference`, default
   `None` → baseline unchanged); tag each sample with its prompt id + is-pivot.
4. Finalize rewards; group locally; `A_i = (r_i - r_pivot) / std`; store on the
   `N` normal samples' `extra_kwargs['advantage']`; drop the pivots; return `M*N/W`.
5. `prepare_feedback` (pivot branch) is a no-op — advantages are already set and
   logged (`train/pivot_adv_*`, incl. `pivot_adv_frac_positive`, the fraction of
   samples that beat the pivot).

## Why the invariants hold

- **Local groups.** `group_contiguous` is forced (`Arguments._resolve_sampler_type`),
  so every group's `N` latents, its pivot, and their rewards live on one rank —
  latent averaging and pivot-centering need no cross-rank comm (only `global_std`
  does one all-reduce).
- **Baseline geometry.** Dropping the `M` pivots before returning restores the
  `M*N` per-epoch count, so the optimize loop and gradient accumulation are
  identical to uniform NFT.
- **Advantage storage.** Per-sample scalar tensors in `extra_kwargs['advantage']`,
  the same contract `AdvantageProcessor` uses, so NFT `optimize()` is unchanged.

## Latent injection (`sd3_5.py`)

`inference(..., latents: Optional[Tensor] = None, ...)` forwards `latents` to
`pipeline.prepare_latents(..., latents)`. `None` (default) ⇒ fresh noise from
`generator` (baseline). Non-`None` ⇒ used verbatim (only moved to device/dtype).
The pivot path sends pre-sampled `init`/`pivot` latents through this. Backward
compatible; only SD3.5 is wired (AWM/GRPO + other models: same 1-line change per
adapter when extended).

## Scope limits (v1, fail-fast)

- **NFT trainer only** (`trainer_type == 'nft'`); AWM/GRPO planned.
- **Single training source**, **pointwise rewards** (the pivot is scored
  pointwise; groups must be complete on-rank).
- **`group_size >= 2`**; cannot be combined with `dynamic_allocation`.

## Cross-refs

- `advantage_processor.py` (`global_std` semantics; the baseline it replaces)
- `topics/train_inference_consistency.md` (rollout under sampling policy / EMA)
- `topics/samplers.md` (`group_contiguous`); `constraints.md` #14
