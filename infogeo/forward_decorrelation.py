"""Dense forward decorrelation following Ahmad's pinned reference code.

Primary source: ``nasiryahm/CorrelationsRuinGD``, revision
``00cf47050bbd20e6a153e10bd86e1651524f9779``, ``crgd/decor.py:6-89``:
https://github.com/nasiryahm/CorrelationsRuinGD/blob/00cf47050bbd20e6a153e10bd86e1651524f9779/crgd/decor.py#L6-L89

For row-batched activity A, the source computes U = (A - mean(A)) D and
updates D as g (D - lr C D), where C is the output moment. We deliberately
preserve the reference code's *left* multiplication by C; transposing the
paper's column-vector convention would produce a different dense update.

The forward map and batch centering support ordinary autograd. Running state
is updated separately, without differentiating through update history. The
returned cache also supplies the exact local derivative to manual learners.
This primitive does not implement a complete BP/FA/DFA network or experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


UPSTREAM_REVISION = "00cf47050bbd20e6a153e10bd86e1651524f9779"
UPSTREAM_URL = (
    "https://github.com/nasiryahm/CorrelationsRuinGD/blob/"
    f"{UPSTREAM_REVISION}/crgd/decor.py#L6-L89"
)


@dataclass(frozen=True)
class DecorrelationCache:
    """The actual matrix and mean used for one forward, before state updates.

    Tensors are retained by reference, not copied; callers must treat them as
    read-only. The layer replaces its state tensors rather than mutating them,
    so a later forward cannot invalidate an earlier cache. An explicit cache
    avoids any dependency on the layer's most recent forward or current mode.
    """

    matrix: torch.Tensor
    mean: torch.Tensor
    training: bool
    input_shape: tuple[int, int]


class DenseForwardDecorrelator(torch.nn.Module):
    """Online dense decorator with an explicit optional train/eval override.

    ``forward(A, training=True)`` returns ``(U, cache)`` and updates the matrix
    and running mean exactly once. ``training=False`` uses the running mean
    without updating state. If omitted, the standard module mode is used.

    The gain is the mean of original, *uncentered* sample norms divided by
    ``sqrt(decorrelated_norm_squared + 1e-8)``. The first
    ``min(n, int(sample_fraction * n) + 1)`` rows estimate both gain and output
    moment. ``low_rank`` associates C D as U.T (U D) / k; it changes neither
    the estimator nor damping (this algorithm has no damping parameter).

    ``lr=0`` still centers and rescales the matrix, as in the upstream layer;
    it does not disable the decorator. There is no gain clipping, matrix
    symmetrization, or fallback that silently changes the reference rule.
    """

    def __init__(
        self,
        num_features: int,
        *,
        lr: float = 1e-5,
        mean_momentum: float = .1,
        sample_fraction: float = .1,
        backend: str = "low_rank",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if isinstance(num_features, bool) or not isinstance(num_features, int) or num_features < 1:
            raise ValueError("num_features must be a positive integer")
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be nonnegative and finite")
        if not math.isfinite(mean_momentum) or not 0 <= mean_momentum <= 1:
            raise ValueError("mean_momentum must lie in [0, 1]")
        if not math.isfinite(sample_fraction) or not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must lie in (0, 1]")
        if backend not in {"dense", "low_rank"}:
            raise ValueError("backend must be dense or low_rank")
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("supported dtypes are float32 and float64")
        self.num_features = num_features
        self.lr = float(lr)
        self.mean_momentum = float(mean_momentum)
        self.sample_fraction = float(sample_fraction)
        self.backend = backend
        self.register_buffer("decor_weight", torch.eye(num_features, dtype=dtype, device=device))
        self.register_buffer("running_mean", torch.zeros(num_features, dtype=dtype, device=device))

    def reset_parameters(self):
        """Restore initial state without modifying any outstanding cache."""
        self.decor_weight = torch.eye(
            self.num_features, dtype=self.decor_weight.dtype, device=self.decor_weight.device)
        self.running_mean = torch.zeros(
            self.num_features, dtype=self.running_mean.dtype, device=self.running_mean.device)

    def forward(self, activity: torch.Tensor, *, training: bool | None = None):
        if activity.ndim != 2 or activity.shape[0] < 1 or activity.shape[1] != self.num_features:
            raise ValueError("activity must be a nonempty batch with num_features columns")
        if activity.dtype not in {torch.float32, torch.float64} or activity.dtype != self.decor_weight.dtype:
            raise ValueError("activity and state must share float32 or float64 dtype")
        if activity.device != self.decor_weight.device:
            raise ValueError("activity and state must share a device")
        if training is None:
            training = self.training
        if not isinstance(training, bool):
            raise ValueError("training must be a bool or None")
        if not torch.isfinite(activity).all():
            raise ValueError("activity must be finite")

        # Keep the batch mean differentiable. The saved old matrix remains
        # valid for autograd and manual backward after the buffer is replaced.
        mean = activity.mean(0) if training else self.running_mean
        matrix = self.decor_weight
        output = (activity - mean) @ matrix
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite decorated output; inspect input and decorator state")
        cache = DecorrelationCache(matrix, mean, training, tuple(activity.shape))
        if training:
            self._update_state(activity, output, mean, matrix)
        return output, cache

    @torch.no_grad()
    def _update_state(self, activity, output, batch_mean, old_matrix):
        count = min(len(activity), int(self.sample_fraction * len(activity)) + 1)
        original, transformed = activity[:count], output[:count]
        gain = (original.square().sum(1).sqrt()
                / (transformed.square().sum(1) + 1e-8).sqrt()).mean()
        scaled_matrix = gain * old_matrix
        if self.backend == "dense":
            moment = (transformed.T @ transformed) / count
            change = moment @ scaled_matrix
        else:
            # C @ (g D), with no d-by-d times d-by-d multiplication.
            change = (transformed.T @ (transformed @ scaled_matrix)) / count
        new_matrix = scaled_matrix - self.lr * change
        new_mean = ((1 - self.mean_momentum) * self.running_mean
                    + self.mean_momentum * batch_mean)
        if not torch.isfinite(new_matrix).all() or not torch.isfinite(new_mean).all():
            raise FloatingPointError("nonfinite decorator update; no state update was committed")
        # Assign both only after validation. No tensor saved by the forward or
        # an earlier cache is modified, and state carries no autograd history.
        self.decor_weight = new_matrix
        self.running_mean = new_mean

    @staticmethod
    @torch.no_grad()
    def backward(signal: torch.Tensor, cache: DecorrelationCache) -> torch.Tensor:
        """Pull back a local signal through the exact cached forward map.

        For training, dL/dA = H (dL/dU) D_old.T, where H subtracts the batch
        mean. For evaluation, the running mean is constant and H is absent.
        This operation is first-order manual differentiation; it neither
        updates state nor differentiates through the matrix-learning history.
        """
        if not isinstance(cache, DecorrelationCache):
            raise TypeError("cache must be returned by DenseForwardDecorrelator.forward")
        if tuple(signal.shape) != cache.input_shape:
            raise ValueError("signal must have the cached forward's output shape")
        if signal.dtype != cache.matrix.dtype or signal.device != cache.matrix.device:
            raise ValueError("signal must share the cached matrix's dtype and device")
        if not torch.isfinite(signal).all():
            raise ValueError("signal must be finite")
        pulled_back = signal @ cache.matrix.T
        if cache.training:
            pulled_back = pulled_back - pulled_back.mean(0)
        if not torch.isfinite(pulled_back).all():
            raise FloatingPointError("nonfinite local pullback")
        return pulled_back
