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

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import Any, Literal, Union, Optional, Tuple

from ..abc import ArgABC
from ...utils.dist import get_world_size
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__, rank_zero_only=True)


@dataclass
class EvaluationArguments(ArgABC):
    resolution: Union[int, tuple[int, int], list[int]] = field(
        default=(1024, 1024),
        metadata={"help": "Resolution for evaluation."},
    )
    height: Optional[int] = field(
        default=None,
        metadata={"help": "Height for evaluation. If None, use the first element of `resolution`."},
    )
    width: Optional[int] = field(
        default=None,
        metadata={"help": "Width for evaluation. If None, use the second element of `resolution`."},
    )
    per_device_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for evaluation."},
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed. Default to be the same as training."},
    )
    guidance_scale: float = field(
        default=3.5,
        metadata={"help": "Guidance scale for evaluation sampling."},
    )
    num_inference_steps: int = field(
        default=30,
        metadata={"help": "Number of timesteps for SDE."},
    )
    eval_freq: int = field(
        default=10,
        metadata={"help": "Evaluation frequency (in epochs). 0 for no evaluation."},
    )

    def __post_init__(self):
        if not self.resolution:
            logger.warning("`resolution` is not set, using default (512, 512).")
            self.resolution = (512, 512)
        elif isinstance(self.resolution, (list, tuple)):
            if len(self.resolution) == 1:
                self.resolution = (self.resolution[0], self.resolution[0])
            elif len(self.resolution) > 2:
                logger.warning(f"`resolution` has {len(self.resolution)} elements, only using the first two: ({self.resolution[0]}, {self.resolution[1]}).")
                self.resolution = (self.resolution[0], self.resolution[1])
            else:  # len == 2
                self.resolution = (self.resolution[0], self.resolution[1])
        else:  # int
            self.resolution = (self.resolution, self.resolution)

        # height/width override
        if self.height is not None and self.resolution[0] != self.height:
            logger.warning(
                f"Both `resolution={self.resolution}` and `height={self.height}` are set. "
                f"Using height to override: ({self.height}, {self.resolution[1]})."
            )
            self.resolution = (self.height, self.resolution[1])
        if self.width is not None and self.resolution[1] != self.width:
            logger.warning(
                f"Both `resolution={self.resolution}` and `width={self.width}` are set. "
                f"Using width to override: ({self.resolution[0]}, {self.width})."
            )
            self.resolution = (self.resolution[0], self.width)

        # Final assignment
        self.height, self.width = self.resolution

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()


# ============================================================================
# Training Arguments Base Class
# ============================================================================

@dataclass
class TrainingArguments(ArgABC):
    r"""Base training arguments shared across all algorithms."""

    # --- Trainer type ---
    trainer_type: str = field(
        default="grpo",
        metadata={"help": "Type of trainer to use."},
    )

    # --- Resolution ---
    resolution: Union[int, tuple[int, int], list[int]] = field(
        default=(512, 512),
        metadata={"help": "Resolution for sampling and training."},
    )
    height: Optional[int] = field(
        default=None,
        metadata={"help": "Height for sampling and training. If None, use the first element of `resolution`."},
    )
    width: Optional[int] = field(
        default=None,
        metadata={"help": "Width for sampling and training. If None, use the second element of `resolution`."},
    )

    # --- Sampling and training ---
    max_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of outer training epochs (counter `epoch` runs 0 .. max_epochs-1). "
                "None or a negative value means no limit (train until interrupted)."
            ),
        },
    )
    per_device_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for sampling and training."},
    )
    gradient_step_per_epoch: int = field(
        default=2,
        metadata={"help": "Number of gradient steps per epoch."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping."},
    )
    num_batches_per_epoch: int = field(init=False)
    gradient_accumulation_steps: Union[int, Literal["auto"]] = field(
        default="auto",
        metadata={
            "help": (
                "Number of backward passes before each optimizer step. "
                "'auto' derives from `gradient_step_per_epoch`. "
                "When set to an integer, `gradient_step_per_epoch` is ignored "
                "and this value is passed directly to Accelerator."
            )
        },
    )
    num_inner_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs for each inner loop optimization."},
    )
    group_size: int = field(
        default=1,
        metadata={"help": "Group size for GRPO sampling."},
    )
    unique_sample_num_per_epoch: int = field(
        default=8,
        metadata={"help": "Number of unique samples per group."},
    )
    # --- Sampling ---
    num_inference_steps: int = field(
        default=10,
        metadata={"help": "Number of timesteps for inference/SDE."},
    )
    guidance_scale: float = field(
        default=3.5,
        metadata={"help": "Guidance scale for sampling."},
    )

    # --- Seed ---
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )

    # --- Optimization ---
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Initial learning rate."},
    )
    adam_weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Weight decay for AdamW optimizer."},
    )
    adam_betas: tuple[float, float] = field(
        default=(0.9, 0.999),
        metadata={"help": "Betas for AdamW optimizer."},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for AdamW optimizer."},
    )
    enable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to enable gradient checkpointing."},
    )
    offload_samples_to_cpu: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, sample tensor fields are moved to CPU at the end of each "
                "sample() iteration and lazily reloaded per micro-batch in optimize(). "
                "Reduces sample()/optimize() GPU peak by ~num_batches_per_epoch x "
                "per_batch_size at the cost of one D2H per sample plus per-reward H2D "
                "(~100ms/epoch total). Required for large per-sample tensors (video "
                "models such as Wan); recommended for higher resolutions or larger "
                "batch sizes; safe to leave off for moderate-VRAM image models. "
                "See .agents/knowledge/topics/sample_lifecycle.md for details."
            ),
        },
    )

    # --- EMA (accessed by models/abc.py for all algorithms) ---
    ema_decay: float = field(
        default=0.995,
        metadata={"help": "Decay for EMA model. Set to 0 to disable EMA."},
    )
    ema_update_interval: int = field(
        default=10,
        metadata={"help": "Update EMA every N epochs."},
    )
    ema_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store EMA model."},
    )
    ema_decay_schedule: Literal["constant", "power", "linear", "piecewise_linear", "cosine", "warmup_cosine"] = field(
        default="power",
        metadata={"help": "Decay schedule for EMA."},
    )

    # --- Latent storage precision ---
    latent_storage_dtype: Optional[Literal['bf16', 'fp16', 'fp32']] = field(
        default='fp16',
        metadata={"help": (
            "Dtype for storing latents in trajectory. "
            "Default fp16 uses `float16`. It's recommended to use fp16 for both precision and memory efficiency. "
            "Options: bf16, fp16, fp32, None (use model-native dtype)."
        )},
    )

    # --- Optimize-loop sample ordering ---
    shuffle_samples: bool = field(
        default=True,
        metadata={"help": (
            "Shuffle samples before each inner optimize epoch. Keep True normally. "
            "Set False for adapters whose batched forward is pack-composition-dependent "
            "(e.g. Bagel/NaViT sequence packing): then each training micro-batch packs the "
            "same samples as its rollout pack, so the bf16 forward is bit-identical between "
            "rollout and training and the on-policy ratio stays 1. Requires matched sampling "
            "and training `per_device_batch_size`."
        )},
    )

    # --- Dynamic (adaptive) rollout allocation ---
    # Two-phase rollout that redistributes a FIXED epoch budget (M*K) across
    # prompts by a per-phase-1 metric. Default off -> uniform K-per-prompt
    # baseline (FlowGRPO / DiffusionNFT / AWM) runs unchanged. See
    # trainers/dynamic_allocation.py and
    # .agents/knowledge/topics/dynamic_allocation.md.
    dynamic_allocation: bool = field(
        default=False,
        metadata={"help": (
            "Enable two-phase adaptive rollout allocation. Phase 1 rolls out "
            "`dynamic_allocation_phase1_ratio` of the per-prompt budget uniformly, "
            "scores each prompt by `dynamic_allocation_metric`, then phase 2 "
            "allocates the remaining budget by `dynamic_allocation_strategy`. The "
            "epoch total (unique_sample_num_per_epoch * group_size) is held fixed. "
            "v1 supports a single training source and pointwise rewards only."
        )},
    )
    dynamic_allocation_phase1_ratio: float = field(
        default=0.5,
        metadata={"help": (
            "Fraction x of the per-prompt budget K rolled out in phase 1 "
            "(K1 = round(x * group_size), clamped to [2, group_size - 1]). "
            "Must be in (0, 1)."
        )},
    )
    dynamic_allocation_metric: str = field(
        default="std",
        metadata={"help": (
            "Per-prompt phase-1 scoring metric. Options: 'std' (reward spread; "
            "higher = stronger advantage signal, recommended), 'mean' (average "
            "reward), 'advantage' (mean absolute group-relative advantage, "
            "mean(|reward - group_mean|)). Extend via "
            "`register_dynamic_allocation_metric`."
        )},
    )
    dynamic_allocation_metric_direction: Literal["higher", "lower"] = field(
        default="higher",
        metadata={"help": (
            "Whether a higher or lower metric earns more phase-2 budget. "
            "Options: 'higher', 'lower'. ('std' is well-motivated with 'higher'; "
            "for 'mean' the direction is a hypothesis worth testing both ways.)"
        )},
    )
    dynamic_allocation_strategy: Literal["proportional", "rank_linear"] = field(
        default="proportional",
        metadata={"help": (
            "How the metric maps to phase-2 budget. Options: 'proportional' "
            "(budget proportional to the shifted metric), 'rank_linear' "
            "(budget linear in metric rank; magnitude-insensitive)."
        )},
    )
    dynamic_allocation_min_per_prompt: int = field(
        default=0,
        metadata={"help": (
            "Minimum phase-2 (extra) rollouts guaranteed to every prompt. 0 lets "
            "low-ranked prompts keep only their K1 phase-1 rollouts. Must satisfy "
            "min_per_prompt * unique_sample_num_per_epoch <= phase-2 budget."
        )},
    )

    # --- Pivot-advantage ---
    # Per group of N, generate one extra "pivot" rollout whose initial latent is
    # the (Gaussian-preserving) average of the N latents; center advantages on
    # the pivot's reward instead of the group mean. Optimize on the N samples
    # only. Default off -> baseline advantage. See
    # trainers/pivot_advantage.py and topics/pivot_advantage.md.
    pivot_advantage: bool = field(
        default=False,
        metadata={"help": (
            "Enable pivot-advantage. For each prompt group of size N, roll out one "
            "extra 'pivot' sample from the averaged initial latent "
            "(sqrt(N) * mean, preserving N(0,I)), then normalize advantages as "
            "(reward - pivot_reward) / std instead of (reward - group_mean) / std. "
            "Training optimizes on the N samples only. v1: NFT + single source + "
            "pointwise rewards; forces sampler_type=group_contiguous."
        )},
    )

    def __post_init__(self):
        # --- Resolution standardization ---
        if not self.resolution:
            logger.warning("`resolution` is not set, using default (512, 512).")
            self.resolution = (512, 512)
        elif isinstance(self.resolution, (list, tuple)):
            if len(self.resolution) == 1:
                self.resolution = (self.resolution[0], self.resolution[0])
            elif len(self.resolution) > 2:
                logger.warning(f"`resolution` has {len(self.resolution)} elements, only using the first two: ({self.resolution[0]}, {self.resolution[1]}).")
                self.resolution = (self.resolution[0], self.resolution[1])
            else:
                self.resolution = (self.resolution[0], self.resolution[1])
        else:
            self.resolution = (self.resolution, self.resolution)

        if self.height is not None and self.resolution[0] != self.height:
            logger.warning(
                f"Both `resolution={self.resolution}` and `height={self.height}` are set. "
                f"Using height to override: ({self.height}, {self.resolution[1]})."
            )
            self.resolution = (self.height, self.resolution[1])
        if self.width is not None and self.resolution[1] != self.width:
            logger.warning(
                f"Both `resolution={self.resolution}` and `width={self.width}` are set. "
                f"Using width to override: ({self.resolution[0]}, {self.width})."
            )
            self.resolution = (self.resolution[0], self.width)

        self.height, self.width = self.resolution

        # --- Batch size calculation ---
        # NOTE: M alignment and derived quantities (num_batches_per_epoch,
        # gradient_accumulation_steps) are computed in Arguments._align_batch_geometry()
        # because the correct alignment strategy depends on the resolved sampler type,
        # which requires cross-component information (data_args, reward_args) only
        # available at the Arguments level.
        # Placeholder values are set here so the fields exist; they will be
        # overwritten by _align_batch_geometry() before any consumer reads them.
        world_size = get_world_size()
        logger.info(f"World Size: {world_size}")

        sample_num_per_iteration = world_size * self.per_device_batch_size
        self.num_batches_per_epoch = (
            (self.unique_sample_num_per_epoch * self.group_size)
            // max(1, sample_num_per_iteration)
        )
        if self.gradient_accumulation_steps == "auto":
            self._manual_gradient_accumulation_steps = False
            self.gradient_accumulation_steps = self.compute_gradient_accumulation_steps(
                self.num_batches_per_epoch,
            )
        else:
            self._manual_gradient_accumulation_steps = True
            self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
            if self.gradient_accumulation_steps < 1:
                raise ValueError(
                    f"`gradient_accumulation_steps` must be >= 1, "
                    f"got {self.gradient_accumulation_steps}."
                )

        # --- Optimizer defaults ---
        # Explicit float() casts guard against scientific-notation values (e.g. 1e-4)
        # arriving as strings from non-standard config sources or future CLI overrides.
        self.adam_betas = (float(self.adam_betas[0]), float(self.adam_betas[1]))
        self.adam_weight_decay = float(self.adam_weight_decay)
        self.adam_epsilon = float(self.adam_epsilon)
        self.max_grad_norm = float(self.max_grad_norm)

        if self.learning_rate is None:
            if 'lora' in self.trainer_type.lower():
                self.learning_rate = 1e-4
            else:
                self.learning_rate = 1e-5
            logger.info(f"`learning_rate` is not set, using default {self.learning_rate} for `{self.trainer_type}` training.")
        else:
            self.learning_rate = float(self.learning_rate)

        # --- Dynamic allocation validation (range only; metric/strategy names
        # are validated at runtime in trainers/dynamic_allocation.py to avoid a
        # hparams -> trainers import cycle). ---
        if self.dynamic_allocation:
            if self.trainer_type.lower() not in {"grpo", "nft", "awm"}:
                raise ValueError(
                    "`dynamic_allocation` is only wired for trainer_type in "
                    f"{{grpo, nft, awm}}, got '{self.trainer_type}'. Other trainers "
                    "(e.g. grpo-guard, dpo, dgpo, crd) keep the uniform rollout path."
                )
            self.dynamic_allocation_phase1_ratio = float(self.dynamic_allocation_phase1_ratio)
            if not 0.0 < self.dynamic_allocation_phase1_ratio < 1.0:
                raise ValueError(
                    f"`dynamic_allocation_phase1_ratio` must be in (0, 1), got "
                    f"{self.dynamic_allocation_phase1_ratio}."
                )
            if self.dynamic_allocation_min_per_prompt < 0:
                raise ValueError(
                    f"`dynamic_allocation_min_per_prompt` must be >= 0, got "
                    f"{self.dynamic_allocation_min_per_prompt}."
                )
            if self.group_size < 3:
                raise ValueError(
                    f"`dynamic_allocation` requires `group_size` >= 3 (got "
                    f"{self.group_size}): phase 1 needs >= 2 rollouts per prompt and "
                    "phase 2 needs a non-empty remaining budget."
                )

        # --- Pivot-advantage validation ---
        if self.pivot_advantage:
            if self.dynamic_allocation:
                raise ValueError(
                    "`pivot_advantage` and `dynamic_allocation` cannot be enabled "
                    "together (they use different samplers and rollout paths). "
                    "Enable one at a time."
                )
            if self.trainer_type.lower() != "nft":
                raise ValueError(
                    "`pivot_advantage` is currently wired for trainer_type='nft' "
                    f"only, got '{self.trainer_type}'. AWM/GRPO support is planned."
                )
            if self.group_size < 2:
                raise ValueError(
                    f"`pivot_advantage` requires `group_size` >= 2 (got "
                    f"{self.group_size}): the pivot latent is the average of the "
                    "group's N initial latents."
                )

    def compute_gradient_accumulation_steps(
        self, num_batches_per_epoch: int,
    ) -> int:
        """Compute gradient accumulation steps (before x num_train_timesteps).

        Default: the optimize loop iterates over all ``num_batches_per_epoch``
        sample batches, so ``GAS = num_batches_per_epoch / gradient_step_per_epoch``.

        Subclasses may override when their optimize loop iterates over a
        different number of batches than the sampling loop (e.g. DPO consumes
        K during pair formation, reducing the batch count).
        """
        return max(1, num_batches_per_epoch // self.gradient_step_per_epoch)

    def get_num_train_timesteps(self, args: Any) -> int:
        """Return the gradient accumulation multiplier for per-timestep losses.

        Subclasses override this to provide algorithm-specific values.
        The `args` parameter is the parent `Arguments` object, giving access
        to sibling config groups like `scheduler_args` if needed.
        """
        return 1

    @property
    def requires_ref_model(self) -> bool:
        """Whether the algorithm requires maintaining reference model parameters.

        Defaults to True when ``kl_beta`` exists and is positive.
        Subclasses may override for custom semantics (e.g. always False for
        algorithms that never use a reference model, or always True for
        algorithms that need one regardless of KL).
        """
        return getattr(self, 'kl_beta', 0) > 0.0

    def get_preprocess_guidance_scale(self) -> float:
        """Return the guidance_scale for data preprocessing.

        The preprocessing stage uses this to decide whether to encode
        negative prompts.  Base implementation returns ``self.guidance_scale``.
        Subclasses may override to account for optimizer-time CFG needs
        (e.g., DGPO ``kl_cfg``), ensuring negative prompts are always
        encoded when any stage might require them.
        """
        return self.guidance_scale

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()


# ============================================================================
# Shared Utilities
# ============================================================================

def _standardize_clip_range(value, name: str) -> tuple[float, float]:
    """Convert a scalar or sequence to a symmetric (lo, hi) tuple.

    Handles values that may arrive as strings (e.g. "1e-4" from YAML parsing)
    by explicitly casting to float.
    """
    if not isinstance(value, (tuple, list)):
        v = float(value)
        return (-abs(v), abs(v))
    lo, hi = float(value[0]), float(value[1])
    assert lo < hi, f"`{name}` lower bound must be less than upper bound, got ({lo}, {hi})."
    return (lo, hi)


def _standardize_timestep_range(value: Union[float, Tuple[float, float]]) -> Tuple[float, float]:
    """Convert float or tuple to ``(frac_lo, frac_hi)`` along denoising 1000->0.

    Fraction ``f`` maps to scheduler time ``1000 * (1 - f)``. Thus ``(0, 0.99)``
    corresponds to times from ``1000`` down to ``10``.
    """
    if not isinstance(value, (list, tuple)):
        result = (0.0, float(value))
    else:
        result = (float(value[0]), float(value[1]))
    assert 0 <= result[0] < result[1] <= 1.0, (
        f"`timestep_range` must satisfy 0 <= start < end <= 1, got {result}"
    )
    return result
