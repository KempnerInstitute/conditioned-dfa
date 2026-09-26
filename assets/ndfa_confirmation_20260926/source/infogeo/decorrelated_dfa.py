"""Manual BP and gated DFA with Ahmad's forward-decorrelation mechanism.

The decorator is pinned to CorrelationsRuinGD revision
00cf47050bbd20e6a153e10bd86e1651524f9779; see ``forward_decorrelation.py``
and its primary-source URL. This adapter preserves ManualMLP's actual gated
direct feedback, not the upstream option named DFA (FA with straight-through
activations). Its hidden preactivation delta is (output_delta @ B) * ReLU'.
Feedback is not injected into decorated coordinates and is not multiplied by
an outgoing decorrelation matrix. Exact BP, in contrast, pulls back through
every saved forward decorator and the differentiable training batch mean.

All linear inputs are decorated, including the classifier input. The model
otherwise shares ManualMLP's seeded weights, biases, ReLU architecture, and
mean cross-entropy objective. This is a reusable adapter, not an experiment
or a reproduction of the prior paper's complete optimizer/architecture setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F

from infogeo.dfa import Gradients, ManualMLP, _output_delta
from infogeo.forward_decorrelation import DecorrelationCache, DenseForwardDecorrelator


@dataclass(frozen=True)
class DecorrelatedForwardCache:
    """One forward's tensors, retained independently of later forward passes.

    Treat the tensors as read-only. The manual optimizer and decorators replace
    tensors rather than modifying them in place, so saved weights and matrices
    remain valid after subsequent updates. ``linear_inputs`` are the actual
    transformed presynaptic activities required by local weight updates.
    """

    owner: object = field(repr=False, compare=False)
    logits: torch.Tensor
    linear_inputs: tuple[torch.Tensor, ...]
    hidden_activities: tuple[torch.Tensor, ...]
    preactivations: tuple[torch.Tensor, ...]
    decorator_caches: tuple[DecorrelationCache, ...]
    weights: tuple[torch.Tensor, ...]
    training: bool


@dataclass
class DecorrelatedGradients(Gradients):
    forward_cache: DecorrelatedForwardCache = field(kw_only=True)


class DecorrelatedManualMLP(ManualMLP):
    """ManualMLP-compatible parameters and gradients, with explicit caches.

    Set ``model.training`` exactly as for ManualMLP. A training forward updates
    each decorator once; evaluation freezes all decorator state. The inherited
    ``apply_gradients`` changes supervised weights/biases only.

    ``bp_gradients`` and ``dfa_gradients`` attach the original forward cache to
    their result. Retrieve presynaptic moments using
    ``model.conditioning_activities(gradients)``; do not replay a forward after
    the decorator has updated. Use the ``*_gradients_from_cache`` methods to
    compare BP and DFA at the same weights, activities, and decorator state.

    The usual three-item ``forward`` interface is retained. Its activity list
    contains every transformed linear input followed by logits; raw hidden
    ReLU outputs are available through the cache or ``hidden_activations``.
    BatchNorm composition, layerwise FA, and target projection are intentionally
    unsupported here rather than inheriting an incorrect local derivative.
    """

    feedback_injection = "gated_hidden_preactivation_before_outgoing_decorator"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        seed: int = 0,
        device: str = "cpu",
        batchnorm: bool = False,
        bn_eps: float = 1e-5,
        bn_momentum: float = .1,
        decor_lr: float = 1e-5,
        decor_mean_momentum: float = .1,
        decor_sample_fraction: float = .1,
        decor_backend: str = "low_rank",
        dtype: torch.dtype = torch.float32,
    ):
        if batchnorm:
            raise NotImplementedError("BatchNorm plus forward decorrelation is not implemented in this adapter")
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("supported dtypes are float32 and float64")
        super().__init__(input_dim, hidden_dims, output_dim, seed=seed, device=device,
                         batchnorm=False, bn_eps=bn_eps, bn_momentum=bn_momentum)
        if dtype != torch.float32:
            self.weights = [weight.to(dtype=dtype) for weight in self.weights]
            self.biases = [bias.to(dtype=dtype) for bias in self.biases]
        self.decorators = [
            DenseForwardDecorrelator(
                weight.shape[1], lr=decor_lr, mean_momentum=decor_mean_momentum,
                sample_fraction=decor_sample_fraction, backend=decor_backend,
                dtype=dtype, device=self.device)
            for weight in self.weights
        ]
        self._cache_owner = object()

    @property
    def decorrelation_state_numel(self):
        """Persistent unsupervised matrix/mean entries, excluding caches."""
        return sum(buffer.numel() for decorator in self.decorators for buffer in decorator.buffers())

    def forward_with_cache(self, x: torch.Tensor) -> DecorrelatedForwardCache:
        activity = x.to(self.device)
        if activity.dtype != self.weights[0].dtype:
            raise ValueError("input and model parameters must share a dtype")
        inputs, hidden, preactivations, decorator_caches = [], [], [], []
        weights = tuple(self.weights)
        for index, (weight, bias, decorator) in enumerate(zip(weights, self.biases, self.decorators)):
            transformed, local_cache = decorator(activity, training=self.training)
            inputs.append(transformed)
            decorator_caches.append(local_cache)
            preactivation = transformed @ weight.T + bias
            preactivations.append(preactivation)
            if index < self.n_hidden_layers:
                activity = torch.relu(preactivation)
                hidden.append(activity)
            else:
                activity = preactivation
        return DecorrelatedForwardCache(
            owner=self._cache_owner, logits=activity, linear_inputs=tuple(inputs),
            hidden_activities=tuple(hidden), preactivations=tuple(preactivations),
            decorator_caches=tuple(decorator_caches), weights=weights, training=self.training)

    def forward(self, x):
        cache = self.forward_with_cache(x)
        return cache.logits, [*cache.linear_inputs, cache.logits], list(cache.preactivations)

    def hidden_activations(self, x):
        return list(self.forward_with_cache(x).hidden_activities)

    def _check_cache(self, cache):
        if not isinstance(cache, DecorrelatedForwardCache) or cache.owner is not self._cache_owner:
            raise ValueError("cache must come from this model's forward_with_cache or gradient result")

    def conditioning_activities(self, gradients: DecorrelatedGradients):
        """Return the original transformed inputs without any forward replay."""
        if not isinstance(gradients, DecorrelatedGradients):
            raise TypeError("conditioning requires gradients carrying this model's forward cache")
        self._check_cache(gradients.forward_cache)
        return gradients.forward_cache.linear_inputs

    def _loss_and_output_delta(self, cache, y):
        self._check_cache(cache)
        labels = y.to(self.device)
        if labels.shape != (len(cache.logits),):
            raise ValueError("labels must contain one class index per cached observation")
        loss = float(F.cross_entropy(cache.logits, labels).item())
        return loss, _output_delta(cache.logits, labels)

    @torch.no_grad()
    def bp_gradients(self, x, y):
        return self.bp_gradients_from_cache(self.forward_with_cache(x), y)

    @torch.no_grad()
    def bp_gradients_from_cache(self, cache: DecorrelatedForwardCache, y):
        loss, delta = self._loss_and_output_delta(cache, y)
        weight_grads, bias_grads, deltas = [], [], []
        # Each backward step uses the next layer's *old* decorator, because
        # that map transforms the current hidden activity in the forward pass.
        for index in range(len(cache.weights) - 1, -1, -1):
            if index < self.n_hidden_layers:
                incoming = delta @ cache.weights[index + 1]
                delta = DenseForwardDecorrelator.backward(incoming, cache.decorator_caches[index + 1])
                delta = delta * (cache.preactivations[index] > 0).to(delta.dtype)
            deltas.append(delta)
            weight_grads.append(delta.T @ cache.linear_inputs[index])
            bias_grads.append(delta.sum(0))
        return DecorrelatedGradients(
            list(reversed(weight_grads)), list(reversed(bias_grads)), list(reversed(deltas)),
            loss, forward_cache=cache)

    def _check_feedback(self, feedback):
        if len(feedback) != self.n_hidden_layers:
            raise ValueError("feedback must contain one direct matrix per hidden layer")
        for width, matrix in zip(self.hidden_dims, feedback):
            if matrix.shape != (self.output_dim, width) or not torch.isfinite(matrix).all():
                raise ValueError("feedback must be finite with shape (output classes, hidden width)")

    @torch.no_grad()
    def dfa_gradients(self, x, y, feedback):
        self._check_feedback(feedback)
        return self.dfa_gradients_from_cache(self.forward_with_cache(x), y, feedback)

    @torch.no_grad()
    def dfa_gradients_from_cache(self, cache: DecorrelatedForwardCache, y, feedback):
        self._check_feedback(feedback)
        loss, output_delta = self._loss_and_output_delta(cache, y)
        weight_grads, bias_grads, deltas = [], [], []
        for index in range(len(cache.weights)):
            if index < self.n_hidden_layers:
                # The fixed direct teacher has the same preactivation target
                # as ManualMLP. No outgoing decorator pullback belongs here.
                direct = feedback[index].to(device=self.device, dtype=output_delta.dtype)
                delta = (output_delta @ direct) * (cache.preactivations[index] > 0).to(output_delta.dtype)
            else:
                delta = output_delta
            deltas.append(delta)
            weight_grads.append(delta.T @ cache.linear_inputs[index])
            bias_grads.append(delta.sum(0))
        return DecorrelatedGradients(weight_grads, bias_grads, deltas, loss, forward_cache=cache)

    def fa_gradients(self, *args, **kwargs):
        raise NotImplementedError("Layerwise FA requires decorator pullbacks and is not implemented here")

    def target_projection_gradients(self, *args, **kwargs):
        raise NotImplementedError("Target projection has not been implemented or validated for this adapter")
