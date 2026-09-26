"""Activity conditioning of the *same* Mixer weight operator for BP and DFA.

This adapter keeps the existing Mixer credit conventions and captures the
presynaptic samples from the very forward pass used to compute each update.
It does not infer that high-variance activity is noise. Its position-pooled
moment is not a Fisher matrix for the tied parameters.
"""

from __future__ import annotations

import math

import torch

from infogeo.mixer_dfa import ManualMixer, MixerGradients
from infogeo.paired_conditioning import right_condition_from_samples
from infogeo.task_conditioning import task_conditioning_diagnostics, task_conditioning_samples


class ActivityTrackingMixer(ManualMixer):
    """Capture detached activities only during an explicitly requested update."""

    def forward_full(self, x):
        full = super().forward_full(x)
        if getattr(self, "_capture_activity", False):
            self._step_activities = tuple(value.detach() for value in full.presyn)
        return full

    def gradients_with_activities(self, x, y, *, credit, feedback=None):
        if credit not in {"bp", "dfa"}:
            raise ValueError("credit must be bp or dfa")
        if credit == "dfa" and feedback is None:
            raise ValueError("dfa requires fixed feedback")
        self._capture_activity = True
        self._step_activities = None
        try:
            gradients = self.bp_gradients(x, y) if credit == "bp" else self.dfa_gradients(x, y, feedback)
            activities = self._step_activities
            if activities is None:
                raise RuntimeError("the gradient forward did not produce activities")
            return gradients, activities
        finally:
            self._capture_activity = False
            self._step_activities = None


def mixer_task_groups(labels: torch.Tensor, n_samples: int, *, position_only=False):
    """Map image labels to class × broadcast-position groups for a tied layer."""
    if labels.ndim != 1 or labels.numel() < 1 or labels.dtype not in {torch.int32, torch.int64}:
        raise ValueError("labels must be a nonempty integer vector")
    if n_samples < len(labels) or n_samples % len(labels):
        raise ValueError("position samples must be an exact multiple of image labels")
    broadcast = n_samples // len(labels)
    positions = torch.arange(broadcast, device=labels.device)
    if position_only:
        return positions.repeat(len(labels))
    _, classes = torch.unique(labels, sorted=True, return_inverse=True)
    return (classes[:, None] * broadcast + positions[None, :]).flatten()


@torch.no_grad()
def condition_mixer_activity(
    gradients: MixerGradients,
    activities,
    *,
    relative_damping: float,
    damping_floor: float = 1e-6,
    backend: str = "auto",
    norm_match: bool = True,
    diagonal: bool = False,
    task_labels: torch.Tensor | None = None,
    task_penalty: float = 0.,
    task_grouping: str = "task",
) -> MixerGradients:
    """Apply G(C_A + lambda I)^-1 to every local weight, for either credit rule.

    lambda = relative_damping * mean(diag(C_A)) + damping_floor. Norm matching,
    when enabled, matches each conditioned weight to its own raw weight-update
    norm. An optional task penalty uses within-class activity residuals while
    preserving the broadcast position/channel as a separate grouping variable.
    The position-only control matches the true class metric's degrees-of-freedom
    scale but removes class information. The optimizer consumes the gradients afterwards; its actual
    step can therefore have a different norm. Biases and classifier are kept.
    """
    if len(activities) != len(gradients.weights) - 1:
        raise ValueError("activities must cover each local weight and exclude the head")
    if not math.isfinite(relative_damping) or relative_damping <= 0:
        raise ValueError("relative_damping must be positive and finite")
    if not math.isfinite(damping_floor) or damping_floor <= 0:
        raise ValueError("damping_floor must be positive and finite")
    if backend not in {"auto", "feature", "sample"}:
        raise ValueError("backend must be auto, feature, or sample")
    if not math.isfinite(task_penalty) or task_penalty < 0:
        raise ValueError("task_penalty must be nonnegative and finite")
    if task_grouping not in {"task", "position"}:
        raise ValueError("task_grouping must be task or position")
    if task_penalty and task_labels is None:
        raise ValueError("a positive task penalty requires image task_labels")
    weights = []
    for gradient, activity in zip(gradients.weights[:-1], activities):
        diagonal_moment = activity.square().mean(dim=0)
        damping = relative_damping * float(diagonal_moment.mean()) + damping_floor
        samples = activity
        if task_penalty:
            if task_labels.device != activity.device:
                raise ValueError("task labels and activities must share a device")
            groups = mixer_task_groups(task_labels, len(activity))
            if task_grouping == "task":
                samples = task_conditioning_samples(activity, groups, penalty=task_penalty)
            else:
                diagnostics = task_conditioning_diagnostics(groups)
                if diagnostics["penalty_available"]:
                    positions = mixer_task_groups(task_labels, len(activity), position_only=True)
                    samples = task_conditioning_samples(
                        activity, positions, penalty=task_penalty * diagnostics["residual_scale"], df_correction=False,
                    )
        if diagonal:
            conditioned = gradient / (samples.square().mean(0) + damping)
        else:
            conditioned = right_condition_from_samples(gradient, samples, damping=damping, backend=backend)
        if norm_match:
            conditioned = conditioned * (gradient.norm() / conditioned.norm().clamp_min(torch.finfo(gradient.dtype).tiny))
        weights.append(conditioned)
    return MixerGradients(weights + [gradients.weights[-1]], list(gradients.biases), gradients.deltas, gradients.loss)
