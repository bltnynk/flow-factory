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

# src/flow_factory/utils/score_dynamics.py
"""
Score-Dynamics utilities for spatial-aware advantage shaping.

Flow-matching adaptation of the "Temporal Score Analysis" signal from ASCED
(arXiv:2503.16218). ASCED uses **abnormal score dynamics** across denoising steps to
flag artifacts; here we repurpose the *same* per-step signal as a **saliency** proxy.
Regions where the predicted clean image keeps changing across the mid-noise window are
where the model is actively constructing content — i.e. the subject / important
foreground — whereas flat low-frequency background settles early and barely moves. We
therefore read a high score-dynamics response as **high saliency** and use it to focus
optimization on those regions (rather than to down-weight them as ASCED does).

The per-step detector is kept faithful to ASCED Eq. 4 / Algorithm 1 (per-step robust
z-score against an adaptive MAD threshold), but the per-step responses are aggregated
over the window by their **mean** (persistence; the default) rather than ASCED's
running-**max** union — saliency is where the model *consistently* refines content, and
the mean is robust to transient spikes and invariant to the step count. In flow-matching
form
(``x_s = (1-s)x0 + s*eps``, velocity ``v = eps - x0``, ``eps_hat = x_s + (1-s)*v``) the
per-step signal is one of:

    pred_x0:        x0_hat = x_s - s * v                              # default (subject saliency)
    weighted_score: w*s = x0_hat - x_s/(1-s) = -(s/(1-s)) * eps_hat   # ASCED paper's weighted score

``score_type`` selects between them. ``pred_x0`` (the predicted-clean latent, the
official ASCED code's simplification) is the **default** because its temporal change
tracks subject content most directly. Both ``latents`` and ``noise_pred`` are available
at every rollout step, so the signal is formed online; the saliency map is the per-step
robust z-score aggregated over the window (mean by default) and is used later (during
optimization) to concentrate the scalar per-sample advantage on salient space.

This module is model- and algorithm-agnostic. It operates on whatever latent layout
the adapter uses (unpacked ``(B, C, H, W)`` or packed ``(B, seq, C)``): the channel
dimension is collapsed to a singleton, leaving a broadcastable per-position map.
"""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .logger_utils import setup_logger

logger = setup_logger(__name__)

_EPS = 1e-8


def _gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a separable 2D Gaussian blur to a ``(B, 1, H, W)`` tensor.

    Args:
        x: Input tensor of shape ``(B, 1, H, W)``.
        sigma: Standard deviation of the Gaussian kernel (in latent pixels).

    Returns:
        Blurred tensor of the same shape as ``x``.
    """
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_x = kernel_1d.view(1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1)
    x = F.conv2d(x, kernel_x, padding=(0, radius))
    x = F.conv2d(x, kernel_y, padding=(radius, 0))
    return x


def _robust_z(score: torch.Tensor) -> torch.Tensor:
    """Per-sample robust z-score of a per-position map.

    ``z = (score - median) / (1.4826 * MAD)`` computed per sample over all non-batch
    dims. This makes ASCED's per-step threshold ``median + k*1.4826*MAD`` equivalent
    to the dimensionless test ``z > k``: a position is anomalous iff its ``z`` exceeds
    the multiplier ``mad_scale``.

    Args:
        score: Non-negative per-position map, shape ``(B, ...)`` with the channel
            dimension already collapsed to a singleton.

    Returns:
        Per-sample robust z-score, same shape as ``score``.
    """
    flat = score.flatten(1)
    median = flat.median(dim=1, keepdim=True).values
    mad = (flat - median).abs().median(dim=1, keepdim=True).values
    z = (flat - median) / (1.4826 * mad + _EPS)
    return z.view_as(score)


class ScoreDynamicsTracker:
    """Incremental flow-matching score-dynamics → **saliency map** during a rollout.

    Reuses ASCED's per-step detector (arXiv:2503.16218, Eq. 4 / Algorithm 1): at each
    in-window step it forms the per-step signal (selected by ``score_type``), takes its
    temporal difference against the previous in-window step, and normalizes that
    difference by **per-step** robust statistics. ASCED reads the response as an
    artifact; here it is read as **saliency** (the subject region the model is actively
    constructing), so the per-step responses are aggregated across the window by their
    **mean** (default) rather than ASCED's running-**max** union.

    Per-step signal (flow matching, ``x_s = (1-s)x0 + s*eps``, ``v = eps - x0``,
    ``eps_hat = x_s + (1-s)*v``)::

        pred_x0:        x0_hat = x_s - s * v                              # default (subject saliency)
        weighted_score: w*s = x0_hat - x_s/(1-s) = -(s/(1-s)) * eps_hat   # ASCED paper's weighted score

    The accumulator is the per-step robust z-score of ``|Delta(signal)|`` aggregated over
    steps — the **mean** (default) or running **max**. The mean reads as *persistence*: a
    position is salient when it is *consistently* refined across the window, so one-off
    background spikes are averaged down and the result is step-count invariant (the rollout
    uses few steps; eval many). The max is ASCED's artifact **union** ``U_k {z_k > k}`` and
    grows with the number of steps. Memory is ``O(latent)`` either way (previous signal +
    a running max, or a running sum and count; never the full sequence).

    Args:
        sigma_window: ``(low, high)`` sigma range to accumulate over. sigma runs
            ``1 -> 0`` across the trajectory; a mid-window such as ``(0.2, 0.8)``
            targets the "mutation" phase where the signal is strongest.
        channel_dim: Latent channel dimension (``1`` for unpacked ``(B, C, H, W)``,
            ``-1`` for packed ``(B, seq, C)``); collapsed to a singleton.
        score_type: ``"pred_x0"`` (the official ASCED code's predicted-clean latent,
            default — tracks subject content most directly) or ``"weighted_score"``
            (ASCED paper's weighted score).
        aggregation: ``"mean"`` (default — persistence; step-count invariant, robust to
            transient background spikes) or ``"max"`` (ASCED's artifact union; sparser
            and step-count dependent). ``mad_scale`` is interpreted against whichever is
            chosen, so its sensible default differs (≈1 for mean, ≈3 for max).
        mad_scale: Sigmoid centering ``k`` on the aggregated robust z-score; the saliency
            map crosses ``0.5`` where the aggregate reaches ``k`` (lower ``k`` marks a
            broader region as salient). Default ``1`` suits the ``mean`` aggregation.
        temperature: Softness of the sigmoid relaxation around the threshold.
        smooth_sigma: Per-step 2D Gaussian-blur sigma applied to the difference map
            (unpacked latents only); ``0`` disables.
    """

    _SCORE_TYPES = ("pred_x0", "weighted_score")
    _AGGREGATIONS = ("mean", "max")

    def __init__(
        self,
        sigma_window: Tuple[float, float],
        channel_dim: int,
        score_type: str = "pred_x0",
        aggregation: str = "mean",
        mad_scale: float = 1.0,
        temperature: float = 1.0,
        smooth_sigma: float = 0.0,
    ) -> None:
        self.sigma_low, self.sigma_high = float(sigma_window[0]), float(sigma_window[1])
        if not 0.0 <= self.sigma_low < self.sigma_high <= 1.0:
            raise ValueError(
                f"`saliency_sigma_window` must satisfy 0 <= low < high <= 1, "
                f"got ({self.sigma_low}, {self.sigma_high})."
            )
        if score_type not in self._SCORE_TYPES:
            raise ValueError(
                f"`saliency_score_type` must be one of {self._SCORE_TYPES}, got '{score_type}'."
            )
        if aggregation not in self._AGGREGATIONS:
            raise ValueError(
                f"`saliency_aggregation` must be one of {self._AGGREGATIONS}, got '{aggregation}'."
            )
        self.channel_dim = channel_dim
        self.score_type = score_type
        self.aggregation = aggregation
        self.mad_scale = float(mad_scale)
        self.temperature = max(float(temperature), _EPS)
        self.smooth_sigma = float(smooth_sigma)
        self._prev_score_field: Optional[torch.Tensor] = None
        # 'max' keeps the running maximum (ASCED union); 'mean' keeps a running sum + count.
        self._running_max_z: Optional[torch.Tensor] = None
        self._running_sum_z: Optional[torch.Tensor] = None
        self._count: int = 0

    def _score_field(
        self, latents: torch.Tensor, noise_pred: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """Per-step score-dynamics signal selected by ``score_type``.

        ``weighted_score`` returns ASCED's weighted score ``-(s/(1-s))*eps_hat``
        (paper); ``pred_x0`` returns the predicted clean latent ``x0_hat = x_s - s*v``
        (official code).
        """
        if self.score_type == "weighted_score":
            one_minus_sigma = max(1.0 - sigma, _EPS)
            eps_hat = latents + one_minus_sigma * noise_pred
            return -(sigma / one_minus_sigma) * eps_hat
        return latents - sigma * noise_pred

    def update(self, latents: torch.Tensor, noise_pred: torch.Tensor, sigma: float) -> None:
        """Feed one denoising step (no-op outside the sigma window).

        Args:
            latents: Current latents ``x_s`` (pre-step), shape ``(B, ...)``.
            noise_pred: Velocity prediction at this step, same shape as ``latents``.
            sigma: Scalar noise level ``sigma = t / 1000`` for this step.
        """
        if not (self.sigma_low <= sigma <= self.sigma_high):
            return
        score_field = self._score_field(latents.float(), noise_pred.float(), sigma)

        if self._prev_score_field is not None:
            diff = (score_field - self._prev_score_field).abs()
            score = diff.mean(dim=self.channel_dim, keepdim=True)
            # Per-step spatial smoothing (2D grid only; packed latents have no grid).
            if self.smooth_sigma > 0.0 and score.ndim == 4:
                score = _gaussian_blur_2d(score, self.smooth_sigma)
            # Per-step robust z-score, then incremental aggregation across the window.
            z = _robust_z(score)
            if self.aggregation == "max":
                self._running_max_z = (
                    z if self._running_max_z is None else torch.maximum(self._running_max_z, z)
                )
            else:  # "mean": running sum + count (persistence; O(latent) memory).
                self._running_sum_z = z if self._running_sum_z is None else self._running_sum_z + z
                self._count += 1

        self._prev_score_field = score_field

    def finalize(self) -> Optional[torch.Tensor]:
        """Produce the per-sample saliency map from the accumulated per-step responses.

        Returns:
            Saliency map in ``(0, 1)`` of shape ``(B, ...)`` with the channel dimension
            collapsed to size ``1`` (high = salient subject region). It is
            ``sigmoid((A_k - mad_scale) / temperature)`` where ``A_k`` is the per-step
            robust z-score aggregated across the window: the **mean** (default;
            persistence — a position is salient when it is *consistently* refined) or the
            running **max** (ASCED's artifact union). Returns ``None`` when fewer than two
            in-window steps were seen.
        """
        if self.aggregation == "max":
            agg = self._running_max_z
        else:
            agg = self._running_sum_z / self._count if self._count > 0 else None
        if agg is None:
            logger.warning(
                "ScoreDynamicsTracker saw fewer than two steps inside "
                f"saliency_sigma_window=({self.sigma_low}, {self.sigma_high}); "
                "no saliency map produced. Widen the window or increase num_inference_steps."
            )
            return None
        return torch.sigmoid((agg - self.mad_scale) / self.temperature)


def compute_spatial_weight(
    saliency_map: torch.Tensor,
    strength: float = 1.0,
    weight_clip: Tuple[float, float] = (0.1, 3.0),
) -> torch.Tensor:
    """Convert a saliency map into a per-position loss weight that focuses on salient space.

    The weight redistributes the (otherwise spatially uniform) per-sample advantage
    toward salient regions and is normalized to spatial-mean ``1`` per sample, so the
    overall loss scale is preserved::

        W = 1 + strength * (S - S_bar)

    where ``S`` is the saliency map and ``S_bar`` its per-sample spatial mean. Salient
    regions (high ``S`` — the subject the model is constructing) are upweighted and flat
    background (low ``S``) is downweighted, so optimization concentrates on content that
    matters. The weight is **independent of advantage sign and of the policy** (it is
    fixed from the rollout saliency map), so it never perturbs the on-policy ratio or
    train-inference consistency — it applies identically to good and bad samples.

    Args:
        saliency_map: Map in ``[0, 1]`` of shape ``(B, ...)`` with a singleton channel
            dimension (high = salient subject region).
        strength: Modulation strength ``gamma >= 0`` (``0`` recovers uniform weighting).
        weight_clip: ``(min, max)`` clamp applied before renormalization to keep
            weights positive and bounded.

    Returns:
        Per-position weight of the same shape as ``saliency_map``, broadcastable
        against the per-pixel loss and normalized to spatial-mean ``1`` per sample.
    """
    saliency_map = saliency_map.float()
    spatial_dims = tuple(range(1, saliency_map.ndim))

    centered = saliency_map - saliency_map.mean(dim=spatial_dims, keepdim=True)
    weight = 1.0 + strength * centered

    weight = weight.clamp(min=weight_clip[0], max=weight_clip[1])
    weight = weight / weight.mean(dim=spatial_dims, keepdim=True).clamp_min(_EPS)
    return weight


def maybe_compute_spatial_weight(
    training_args: Any,
    batch: Dict[str, Any],
) -> Optional[torch.Tensor]:
    """Build the per-position loss weight for a batch, or ``None`` when inactive.

    Single entry point shared by all trainers. Returns ``None`` (so callers fall back
    to the unweighted reduction) when spatial shaping is disabled or the batch carries
    no saliency map (e.g. an adapter that did not compute one).

    Args:
        training_args: The active ``TrainingArguments`` (read for the shaping config).
        batch: Stacked batch dict; must expose ``saliency_map``.

    Returns:
        Per-position weight broadcastable against the per-pixel loss, or ``None``.
    """
    if not getattr(training_args, "spatial_advantage_shaping", False):
        return None
    saliency_map = batch.get("saliency_map")
    if saliency_map is None:
        return None
    return compute_spatial_weight(
        saliency_map=saliency_map,
        strength=training_args.spatial_shaping_strength,
        weight_clip=training_args.spatial_shaping_weight_clip,
    )


def spatial_weighted_mean(per_pixel: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reduce a per-pixel loss to per-sample with a broadcastable spatial weight.

    Because ``weight`` is normalized to spatial-mean ``1``, this is scale-comparable
    to the unweighted ``per_pixel.mean(dim=1..)`` it replaces.

    Args:
        per_pixel: Per-element loss, shape ``(B, ...)`` (full latent layout).
        weight: Per-position weight broadcastable against ``per_pixel`` (singleton
            channel dimension), as returned by :func:`compute_spatial_weight`.

    Returns:
        Per-sample reduced loss of shape ``(B,)``.
    """
    return (per_pixel * weight).mean(dim=tuple(range(1, per_pixel.ndim)))
