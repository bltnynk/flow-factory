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

"""Training arguments for DiffusionNFT."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union, Tuple

from ._base import (
    TrainingArguments,
    _standardize_clip_range,
    _standardize_spatial_shaping,
    _standardize_timestep_range,
)


@dataclass
class NFTTrainingArguments(TrainingArguments):
    r"""Training arguments for DiffusionNFT."""

    # Group-wise advantage normalization
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal['sum', 'gdpo'] = field(
        default='gdpo',
        metadata={"help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."},
    )
    # NFT core
    nft_beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for NFT trainer."},
    )
    off_policy: bool = field(
        default=False,
        metadata={"help": "Whether to use EMA parameters for sampling off-policy data."},
    )

    # Clipping / KL
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal['v-based'] = field(
        default='v-based',
        metadata={"help": "Type of KL divergence. NFT defaults to 'v-based'."},
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={"help": "Total number of training timesteps. 0 or None defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."},
    )
    time_sampling_strategy: Literal['uniform', 'logit_normal', 'discrete', 'discrete_with_init', 'discrete_wo_init'] = field(
        default='discrete',
        metadata={"help": "Time sampling strategy for training."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Time shift for logit normal time sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.9,
        metadata={
            "help": "Fraction range along denoise axis 1000->0; maps to scheduler times "
            "[1000*(1-end), 1000*(1-start)]. Float means [0, value]."
        },
    )

    # --- Spatial-aware advantage shaping (score dynamics; arXiv:2503.16218) ---
    spatial_advantage_shaping: bool = field(
        default=False,
        metadata={"help": (
            "Enable spatial-aware advantage shaping: derive a per-pixel saliency map from "
            "flow-matching score dynamics during rollout and use it to focus the matching loss "
            "on salient (subject) regions during optimization. Off by default (byte-identical to baseline)."
        )},
    )
    spatial_shaping_strength: float = field(
        default=1.0,
        metadata={"help": "Modulation strength gamma (>= 0). 0 recovers uniform spatial weighting."},
    )
    spatial_shaping_weight_clip: Tuple[float, float] = field(
        default=(0.1, 3.0),
        metadata={"help": "(min, max) clamp on the per-pixel weight before spatial-mean-1 renormalization."},
    )
    saliency_sigma_window: Tuple[float, float] = field(
        default=(0.2, 0.8),
        metadata={"help": (
            "Noise-level sigma window over which score dynamics are accumulated (sigma runs 1->0; "
            "a mid window targets the content-revealing 'mutation' phase)."
        )},
    )
    saliency_score_type: Literal['pred_x0', 'weighted_score'] = field(
        default='pred_x0',
        metadata={"help": (
            "Per-step score-dynamics signal. 'pred_x0' (default): predicted-clean latent, tracks "
            "subject content most directly. 'weighted_score': ASCED paper's weighted score."
        )},
    )
    saliency_aggregation: Literal['mean', 'max'] = field(
        default='mean',
        metadata={"help": (
            "How per-step robust-z responses are aggregated over the sigma window. 'mean' (default): "
            "persistence — salient where the model consistently refines content; step-count invariant and "
            "robust to transient background spikes. 'max': ASCED's artifact union; sparser, step-count "
            "dependent (pair with a higher saliency_mad_scale, ~3)."
        )},
    )
    saliency_mad_scale: float = field(
        default=1.0,
        metadata={"help": (
            "Sigmoid centering k on the aggregated robust-z (median + k*1.4826*MAD); the saliency map "
            "crosses 0.5 where the aggregate reaches k. Lower k marks a broader salient region. Default 1 "
            "suits 'mean' aggregation; use ~3 for 'max'."
        )},
    )
    saliency_smooth_sigma: float = field(
        default=1.0,
        metadata={"help": "Per-step Gaussian-blur sigma (latent pixels) for the difference map; 0 disables. Unpacked 2D latents only (no-op for packed)."},
    )
    saliency_temperature: float = field(
        default=1.0,
        metadata={"help": "Sigmoid softness of the saliency map around the centering threshold; larger is softer."},
    )

    def __post_init__(self):
        super().__post_init__()

        # Guard kl_beta against scientific-notation strings (e.g. "1e-3" from CLI overrides).
        self.kl_beta = float(self.kl_beta)

        self.timestep_range = _standardize_timestep_range(self.timestep_range)

        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0])))

        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, 'adv_clip_range')
        if self.kl_type not in ['v-based']:
            raise ValueError(f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based'].")
        (
            self.spatial_shaping_strength,
            self.spatial_shaping_weight_clip,
            self.saliency_sigma_window,
            self.saliency_score_type,
            self.saliency_aggregation,
            self.saliency_mad_scale,
            self.saliency_temperature,
        ) = _standardize_spatial_shaping(
            self.spatial_shaping_strength,
            self.spatial_shaping_weight_clip,
            self.saliency_sigma_window,
            self.saliency_score_type,
            self.saliency_aggregation,
            self.saliency_mad_scale,
            self.saliency_temperature,
        )
        self.saliency_smooth_sigma = float(self.saliency_smooth_sigma)

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps
