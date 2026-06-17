# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Authoritative docs — read first

This repo maintains a dedicated agent knowledge base; this file is a quick orientation that defers to it. **`AGENTS.md` is the full development guide** (operating principles, commit flow, skills index). On session start also load:

- `.agents/knowledge/philosophy.md` — design principles (the core invariant is **train-inference consistency**)
- `.agents/knowledge/constraints.md` — 29 hard constraints; **consult before any code change**
- `.agents/knowledge/architecture.md` — module graph, six-stage pipeline, registry tables (every model/algorithm/reward key is listed here)

`.agents/knowledge/README.md` maps a change-area → the Tier-2 topic doc to read (e.g. touching dtype → `topics/dtype_precision.md`; editing samplers → `topics/samplers.md`; adding an adapter → `topics/adapter_conventions.md`).

Skills live in `.agents/skills/` and are invokable as `/ff-develop`, `/ff-debug`, `/ff-review`, `/ff-new-model`, `/ff-new-reward`, `/ff-new-algorithm`. Decision guide: "add model X" → `/ff-new-model`; "add reward" → `/ff-new-reward`; "add algorithm" → `/ff-new-algorithm`; "fix/hang/wrong output" → `/ff-debug`; everything else / refactor → `/ff-develop`; before committing → `/ff-review`.

## Hard rules (always enforced, including sub-agents)

- **Scratch files only** — all temporary/intermediate files (analysis reports, investigation notes, checklists, diagrams) MUST go under `.scratch/` (git-ignored). NEVER write to the project root or any tracked directory (`src/`, `guidance/`, `.agents/`, `examples/`).

## Commands

```bash
# Install
pip install -e .                                          # core
pip install -e ".[deepspeed]"                             # + DeepSpeed ZeRO-1/2
pip install -e ".[bagel]"                                 # Bagel adapter (flash-attn) — intentionally NOT in [all]
pip install -e ".[wandb]"  /  ".[swanlab]"                # experiment trackers
git submodule update --init && pip install -e ./diffusers # bundled diffusers (required by LTX-2, etc.)

# Train — single entry point. Auto-detects single/multi-GPU/multi-node and wraps `accelerate launch`.
ff-train <config.yaml>                                    # e.g. ff-train examples/grpo/lora/flux1/default.yaml

# Format — the ONLY enforced checks; run before every commit
black --check src/ && isort --check src/                  # check
black src/ && isort src/                                  # apply (line-length 100, isort profile=black)
```

There is **no automated test suite** in the repository — `AGENTS.md` references `pytest`, but no test files exist yet. Verify changes by running a short example config end-to-end and, for adapter/numerics work, follow `.agents/knowledge/topics/parity_testing.md`.

## Entry point & launch flow

`ff-train` (= `flow-factory-train`) → `flow_factory.cli:train_cli`. It loads the YAML, merges launch params with priority **CLI args > env vars > YAML**, then either runs `python -m flow_factory.train <config>` directly (single process / external launcher like torchrun) or builds an `accelerate launch … -m flow_factory.train <config>` command. Multi-node is auto-detected from env vars (`MASTER_IP`, `NUM_MACHINES`, `MACHINE_RANK`, `GPUS_PER_NODE`, …) — the same YAML works single- and multi-node. Accelerate / DeepSpeed configs live in `config/`; multi-node examples in `multinode_examples/`.

## Architecture (big picture)

**Registry-based plugin system.** Trainers, model adapters, and reward models are independently pluggable and **model × algorithm are fully decoupled** — any combination works. Three registries (`trainers/registry.py`, `models/registry.py`, `rewards/registry.py`) map a lowercase string key → lazy import path, with fallback to a direct `pkg.module.Class` path. Renaming/moving a registered class requires updating its registry entry or you get a runtime `ImportError`.

**Config (Pydantic), in `hparams/`.** Top-level `Arguments` aggregates `ModelArguments`, `TrainingArguments` (algorithm-specific subclass resolved by `get_training_args_class()`), `SchedulerArguments`, `DataArguments`, reward args, and `LogArguments`. YAML keys must match field names exactly — typos fail silently to defaults. Renaming a field means updating the dataclass, every `examples/*.yaml`, and all `config.<field>` accesses.

**Six-stage per-epoch training pipeline (order is invariant):**
1. Data preprocessing (offline, cached) — `adapter.preprocess_func()` encodes text/image/video/audio → tensors
2. K-repeat sampling — distributed sampler keeps `group_size` copies grouped (sampler geometry constraints in `topics/samplers.md`)
3. Trajectory generation — `adapter.inference()` (full multi-step denoising)
4. Reward computation — `RewardProcessor` (pointwise / groupwise / multi-reward, optional async)
5. Advantage computation — `AdvantageProcessor.compute_advantages()` (communication-aware)
6. Policy optimization — `adapter.forward()` (single-step) + optimizer step

Trainer hooks per epoch: `sample()` (stages 2–3) → `prepare_feedback()` (4–5) → `optimize()` (6).

**Base classes — flat hierarchies (constraints #11–14):**
- `trainers/abc.py: BaseTrainer` — implement `start/prepare_feedback/optimize/evaluate`. New trainers inherit `BaseTrainer` **directly**; only sanctioned exception is `GRPOGuardTrainer → GRPOTrainer`.
- `models/abc.py: BaseAdapter` — 4 abstract methods: `load_pipeline` (returns a diffusers pipeline), `decode_latents`, `inference`, `forward`. Per-modality encoders (`encode_prompt/image/video/audio`) are **no-op by default** — override only what the model consumes. Adapters inherit `BaseAdapter` directly, never another adapter.
- `rewards/abc.py` — `PointwiseRewardModel` (returns `(batch_size,)`) vs `GroupwiseRewardModel` (receives a whole group, returns `(group_size,)`); `RewardProcessor` dispatches by type.
- `samples/samples.py` — two-layer sample dataclasses: task-level (`T2ISample`, `I2AVSample`, …) inherit `BaseSample`/condition mixins; model-specific (`LTX2Sample`, …) inherit the matching task-level sample, never another model-specific one.

**Highest-impact invariants (full list in `constraints.md`):**
- **Coupled vs decoupled paradigm:** GRPO / GRPO-Guard require SDE dynamics + log-probs; DPO / NFT / AWM / DGPO / CRD are decoupled and may use ODE. Mixing (e.g. ODE with GRPO) silently produces wrong gradients.
- Timesteps are `[0, 1000]` (scheduler scale); sigmas are `[0, 1]` (flow-matching noise level).
- DeepSpeed **ZeRO-3 is unsupported** (reward-model sharding is broken); use ZeRO-1/2 only.
- Text encoders / VAEs are loaded for stage 1, offloaded, then reloaded for sampling — don't assume they're always on-device.

## Conventions

- **Commits:** concise English. **PR title:** `[{modules}] {type}: {description}` where type ∈ `feat|fix|refactor|docs|test|chore` (e.g. `[trainer,reward] feat: add multi-reward weighting`).
- **Examples path:** `examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml`; model dirs use underscores matching the `model_type` field (`sd3-5` → `sd3_5`), baseline variant is `default.yaml`.
- **Code style:** top-level imports only (sanctioned exceptions: optional deps in `try/except ImportError`, backend-gated imports); relative imports within `flow_factory`; Apache-2.0 header `Copyright 2026 Jayce-Ping` on every source file; Google-style English docstrings; type annotations on public methods; **fail-fast over silent fallback** (`.cursor/rules/no-defensive-except.mdc`). Section-divider comments are allowed between methods but forbidden inside function bodies.
