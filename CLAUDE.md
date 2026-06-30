# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Authoritative docs — read these first

This repo carries a layered knowledge base; treat it as the source of truth and do **not** duplicate it here.

- **`AGENTS.md`** — full development guide (operating principles, commit/PR conventions, skills).
- **`.agents/knowledge/`** — read **Tier 1** at session start:
  - `philosophy.md` — design principles (the central invariant is **train-inference consistency**).
  - `constraints.md` — 29 hard constraints; consult before *any* code change.
  - `architecture.md` — module graph, 6-stage pipeline, the three registries.
- **`.agents/knowledge/topics/*.md`** — Tier 2, read by change area (trigger table in `.agents/knowledge/README.md`). E.g. touching `forward()`/`inference()`/`scheduler.step` → `train_inference_consistency.md`; editing samplers → `samplers.md`; debugging NaN/dtype → `dtype_precision.md`; adding an adapter → `adapter_conventions.md` + `parity_testing.md`.
- **`guidance/*.md`** — user-facing deep dives (`workflow.md`, `algorithms.md`, `rewards.md`, `new_model.md`).
- **`.cursor/rules/*.mdc`** — coding conventions enforced as rules (base-class contract, registry consistency, no defensive `try/except`, examples↔YAML sync, variable naming).

Invocable workflow skills live in `.agents/skills/`: `/ff-new-model`, `/ff-new-reward`, `/ff-new-algorithm`, `/ff-debug`, `/ff-develop`, `/ff-review`.

## Commands

```bash
# Install (editable). Core is enough for most work.
pip install -e "."                 # core
pip install -e ".[all]"            # + deepspeed + quantization
pip install -e ".[deepspeed]"      # deepspeed only
pip install -e ".[bagel]"          # flash-attn + opencv (NOT in [all]); needed for model_type: bagel
pip install -e ".[wandb]"          # or [swanlab] for experiment tracking

# Some models (LTX-2) need the bundled diffusers submodule:
git submodule update --init && pip install -e ./diffusers

# GenEval reward extra (needs mmcv/mmdet, Python 3.10):
bash scripts/install_geneval_deps.sh

# Train — this is the single entry point.
ff-train examples/grpo/lora/flux1/default.yaml      # ff-train == flow-factory-train
```

`ff-train` (`src/flow_factory/cli.py`) is a **launcher that auto-wraps `accelerate launch`**: it detects local GPU count and multi-node env vars (`MASTER_IP`, `NUM_MACHINES`, `MACHINE_RANK`, …) and builds the `accelerate` command for you. Single GPU / `RANK` already set → runs `python -m flow_factory.train` directly. You normally do **not** call `accelerate launch` yourself. Override launch params with CLI flags (`--num_processes`, `--num_machines`, …) or YAML keys (`config_file:` points at an accelerate config under `config/accelerate_configs/`). See `multinode_examples/`.

## Linting & tests

- Formatters: **black** and **isort**, both `line-length=100` (config in `pyproject.toml`). Comments/docstrings in English; type-annotate public methods.
- **Format only the files you changed** — run `black`/`isort` on those paths, not a repo-wide `black src/`. The tree is not uniformly black-clean, so a bulk run reformats unrelated files and pollutes the diff.
- **There is no unit-test suite** (no `tests/` dir, no pytest config; the only `test_*.py` live under git-ignored `.scratch/` and `.geneval_build/`). Correctness is verified by running training end-to-end and by **parity testing** — see `.agents/knowledge/topics/parity_testing.md`. The `pytest` line in `AGENTS.md` is aspirational.

## Architecture in one screen

Three independently pluggable axes, wired by string-keyed registries (lazy import paths), so any (model × algorithm × reward) combination works:

| Axis | Base class | Registry | Examples |
|------|-----------|----------|----------|
| **Trainer** (algorithm) | `trainers/abc.py::BaseTrainer` | `trainers/registry.py` | `grpo`, `grpo-guard`, `dpo`, `dgpo`, `nft`, `awm`, `crd`, `diffusion-opd` |
| **Adapter** (model) | `models/abc.py::BaseAdapter` | `models/registry.py` | `flux1`/`flux2`, `sd3-5`, `qwen-image`, `wan2_*`, `ltx2_*`, `z-image`, `bagel` |
| **Reward** | `rewards/abc.py::BaseRewardModel` | `rewards/registry.py` | `pickscore`, `clip`, `ocr`, `geneval`, `vllm_evaluate`, `rational_rewards_*` |

Config dataclasses (Pydantic) live in `hparams/`; top-level `Arguments` aggregates `Model/Training/Scheduler/Data/Reward/Log/Evaluation` args. YAML keys must match field names exactly (typos fail **silently** to defaults). `TrainingArguments` is subclassed per algorithm and resolved via `get_training_args_class()`.

**Six-stage per-epoch pipeline** (order is invariant — see constraint #6): data preprocessing (offline, cached) → K-repeat sampling (`group_size`) → trajectory generation (`adapter.inference`) → reward (`RewardProcessor`) → advantage (`AdvantageProcessor`, comm-aware) → policy optimization (`adapter.forward`). Trainer hooks map to stages: `sample()` = 2–3, `prepare_feedback()` = 4–5, `optimize()` = 6.

## Constraints that bite most often (full list in `constraints.md`)

- **Coupled vs decoupled (#7):** GRPO / GRPO-Guard are *coupled* — must use SDE dynamics (`Flow-SDE`/`Dance-SDE`/`CPS`) with log-probs. DPO/NFT/AWM are *decoupled* — any dynamics incl. `ODE`. Mixing (e.g. `ODE` + GRPO) silently produces wrong gradients.
- **Flat class hierarchy (#11, #12, #14):** new trainers/adapters inherit **directly** from their base class, never from a sibling. Only sanctioned exception: `GRPOGuardTrainer → GRPOTrainer`. Model-specific samples inherit from the *task-level* sample (`LTX2I2AVSample(I2AVSample)`), not another model sample. Share code via helpers/mixins.
- **Registry sync (#1):** moving/renaming a registered class without updating its registry path → runtime `ImportError`. Keys are lowercase (hyphens for adapters).
- **hparams ↔ YAML sync (#15):** a field change must be reflected in the dataclass, **every** YAML under `examples/`, and every `config.<field>` access site.
- **Adapter contract (#12):** subclasses implement exactly 4 abstract methods — `load_pipeline()` (returns a `DiffusionPipeline`), `decode_latents()`, `inference()`, `forward()`. The 4 per-modality encoders (`encode_prompt/image/video/audio`) are no-op by default — override only what the model consumes.
- **Distributed (#9, #10, #18, #19):** only trainable modules + optimizer go through `accelerator.prepare()` (the dataloader uses a custom sampler, never `prepare()`'d). DeepSpeed **ZeRO-3 is unsupported** (reward sharding broken) — only ZeRO-1/2. Keep `wait_for_everyone()` barriers and `_synchronize_frozen_components()`.
- **Fail-fast (#26):** raise with context on invalid state; do **not** add defensive `try/except` that silently recovers (`.cursor/rules/no-defensive-except.mdc`).
- **Examples path convention (#29):** `examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml`; baseline is `default.yaml`; model dirs use underscores (`sd3_5`, `flux1_kontext`).

## Hard rule (always enforced, including sub-agents)

- **Scratch files only** — all temporary/intermediate files (analysis reports, investigation notes, checklists, diagrams) MUST go under `.scratch/` (git-ignored). NEVER write to the project root or any tracked directory (`src/`, `guidance/`, `.agents/`, `examples/`).
