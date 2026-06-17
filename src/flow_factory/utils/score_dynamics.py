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

Flow-matching adaptation of "Temporal Score Analysis" (ASCED, arXiv:2503.16218).
ASCED detects visual artifacts/hallucinations by monitoring the per-pixel temporal
variation of the running predicted-clean image ``x̂₀`` across denoising steps —
artifact regions exhibit anomalously large variation in a mid-noise window.

For flow matching with ``x_σ = (1 - σ) x₀ + σ ε`` and velocity prediction
``v = ε - x₀`` (the AWM target), the running clean estimate is

    x̂₀(σ) = x_σ - σ · v_θ(x_σ, σ) = latents - σ · noise_pred

Both ``latents`` and ``noise_pred`` are available at every rollout step, so the
score-dynamics map can be accumulated online during trajectory generation and used
later (during optimization) to redistribute the scalar per-sample advantage across
space — upweighting clean regions and downweighting artifact regions.

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


def _robust_normalize(score: torch.Tensor, temperature: float) -> torch.Tensor:
    """Map a raw per-position score to an artifact intensity in ``[0, 1]``.

    Uses per-sample robust statistics (median / MAD) so the result is invariant
    to per-sample scale, then a sigmoid so the median maps to ``0.5`` and large
    deviations (candidate artifacts) saturate towards ``1``.

    Args:
        score: Raw non-negative per-position score, shape ``(B, ...)`` with the
            channel dimension already collapsed to a singleton.
        temperature: Sigmoid temperature; larger values produce a softer map.

    Returns:
        Artifact intensity map in ``(0, 1)`` with the same shape as ``score``.
    """
    flat = score.flatten(1)
    median = flat.median(dim=1, keepdim=True).values
    mad = (flat - median).abs().median(dim=1, keepdim=True).values
    z = (flat - median) / (1.4826 * mad + _EPS)
    artifact = torch.sigmoid(z / max(temperature, _EPS))
    return artifact.view_as(score)


class ScoreDynamicsTracker:
    """Online accumulator of flow-matching score dynamics during a rollout.

    Maintains the running predicted-clean estimate ``x̂₀ = latents - σ · noise_pred``
    and accumulates the per-element temporal variation ``|x̂₀(σᵢ) - x̂₀(σᵢ₋₁)|`` over
    consecutive denoising steps whose noise level ``σ`` falls inside a configurable
    mid-trajectory window. Memory is ``O(latent)`` — only the previous estimate and a
    running accumulator are kept, never the full sequence.

    Args:
        sigma_window: ``(low, high)`` σ range over which to accumulate temporal
            variation. σ runs ``1 → 0`` across the trajectory (high noise → clean),
            so a mid-window such as ``(0.2, 0.8)`` targets the "mutation" phase where
            ASCED reports the artifact signal is most informative.
    """

    def __init__(self, sigma_window: Tuple[float, float]) -> None:
        self.sigma_low, self.sigma_high = float(sigma_window[0]), float(sigma_window[1])
        if not 0.0 <= self.sigma_low < self.sigma_high <= 1.0:
            raise ValueError(
                f"`artifact_sigma_window` must satisfy 0 <= low < high <= 1, "
                f"got ({self.sigma_low}, {self.sigma_high})."
            )
        self._accum: Optional[torch.Tensor] = None
        self._prev_x0: Optional[torch.Tensor] = None
        self._count: int = 0

    def update(self, latents: torch.Tensor, noise_pred: torch.Tensor, sigma: float) -> None:
        """Feed one denoising step.

        Args:
            latents: Current latents ``x_σ`` (pre-step), shape ``(B, ...)``.
            noise_pred: Velocity prediction at this step, same shape as ``latents``.
            sigma: Scalar noise level ``σ = t / 1000`` for this step.
        """
        if not (self.sigma_low <= sigma <= self.sigma_high):
            return
        x0 = latents.float() - sigma * noise_pred.float()
        if self._prev_x0 is not None:
            diff = (x0 - self._prev_x0).abs()
            self._accum = diff if self._accum is None else self._accum + diff
            self._count += 1
        self._prev_x0 = x0

    def finalize(
        self,
        channel_dim: int,
        temperature: float = 1.0,
        smooth_sigma: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """Produce the per-sample artifact map from the accumulated dynamics.

        Args:
            channel_dim: Dimension index of the latent channels (e.g. ``1`` for
                unpacked ``(B, C, H, W)``, ``-1`` for packed ``(B, seq, C)``).
                Collapsed to a singleton so the map broadcasts over channels.
            temperature: Sigmoid temperature for robust normalization.
            smooth_sigma: If ``> 0`` and the map has two spatial dimensions
                (unpacked latents), apply a 2D Gaussian blur of this σ.

        Returns:
            Artifact map in ``[0, 1]`` of shape ``(B, ...)`` with the channel
            dimension collapsed to size ``1`` (high = artifact), or ``None`` when
            fewer than two in-window steps were seen.
        """
        if self._accum is None or self._count == 0:
            logger.warning(
                "ScoreDynamicsTracker saw fewer than two steps inside "
                f"artifact_sigma_window=({self.sigma_low}, {self.sigma_high}); "
                "no artifact map produced. Widen the window or increase num_inference_steps."
            )
            return None

        score = self._accum / self._count
        score = score.mean(dim=channel_dim, keepdim=True)

        if smooth_sigma > 0.0:
            if score.ndim == 4:
                score = _gaussian_blur_2d(score, smooth_sigma)
            else:
                logger.warning(
                    f"artifact_smooth_sigma={smooth_sigma} requested but the latent map "
                    f"has ndim={score.ndim} (no 2D spatial grid); skipping spatial smoothing."
                )

        return _robust_normalize(score, temperature)


def compute_spatial_weight(
    artifact_map: torch.Tensor,
    advantage: torch.Tensor,
    mode: str = "sign_aware",
    strength: float = 1.0,
    weight_clip: Tuple[float, float] = (0.1, 3.0),
) -> torch.Tensor:
    """Convert an artifact map + scalar advantage into a per-position loss weight.

    The weight redistributes the (otherwise spatially uniform) per-sample advantage
    across space and is normalized to spatial-mean ``1`` per sample, so the overall
    loss scale is preserved.

    Modes (``g = 1 - artifact_map`` is the per-position goodness, ``ḡ`` its spatial
    mean):

    - ``"sign_aware"`` (default): ``W = 1 + strength · sign(advantage) · (g - ḡ)``.
      For positive-advantage samples this upweights clean regions and downweights
      artifacts (reinforce what is good); for negative-advantage samples it flips —
      upweighting artifact regions so the penalty concentrates on what is broken.
    - ``"favor_clean"``: ``W = 1 + strength · (g - ḡ)`` — always favors clean regions
      regardless of advantage sign.

    Args:
        artifact_map: Map in ``[0, 1]`` of shape ``(B, ...)`` with a singleton
            channel dimension (high = artifact).
        advantage: Per-sample advantage, shape ``(B,)``.
        mode: ``"sign_aware"`` or ``"favor_clean"``.
        strength: Modulation strength ``γ >= 0`` (``0`` recovers uniform weighting).
        weight_clip: ``(min, max)`` clamp applied before renormalization to keep
            weights positive and bounded.

    Returns:
        Per-position weight of the same shape as ``artifact_map``, broadcastable
        against the per-pixel loss and normalized to spatial-mean ``1`` per sample.
    """
    artifact_map = artifact_map.float()
    spatial_dims = tuple(range(1, artifact_map.ndim))

    goodness = 1.0 - artifact_map
    centered = goodness - goodness.mean(dim=spatial_dims, keepdim=True)

    if mode == "sign_aware":
        broadcast_shape = (-1, *([1] * (artifact_map.ndim - 1)))
        sign = torch.sign(advantage).to(artifact_map.dtype).view(*broadcast_shape)
        weight = 1.0 + strength * sign * centered
    elif mode == "favor_clean":
        weight = 1.0 + strength * centered
    else:
        raise ValueError(
            f"Unknown spatial_shaping_mode='{mode}'. Valid options: ['sign_aware', 'favor_clean']."
        )

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
    no artifact map (e.g. an adapter that did not compute one).

    Args:
        training_args: The active ``TrainingArguments`` (read for the shaping config).
        batch: Stacked batch dict; must expose ``artifact_map`` and ``advantage``.

    Returns:
        Per-position weight broadcastable against the per-pixel loss, or ``None``.
    """
    if not getattr(training_args, "spatial_advantage_shaping", False):
        return None
    artifact_map = batch.get("artifact_map")
    if artifact_map is None:
        return None
    return compute_spatial_weight(
        artifact_map=artifact_map,
        advantage=batch["advantage"],
        mode=training_args.spatial_shaping_mode,
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
