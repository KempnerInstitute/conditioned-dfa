"""Matched direct-feedback adaptation of Ahmad's pinned forward decorator.

The inherited DFA injects at each hidden activity BEFORE its outgoing
decorator. Its local ReLU and BN pullbacks are therefore unchanged; incoming
weight gradients use the decorated presynaptic activity. The exact BP path
below is provided for derivative verification, not as an additional study arm.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from infogeo.dfa import ManualMLP, Gradients, _output_delta
from infogeo.forward_decorrelation import DenseForwardDecorrelator


class DecorrelatedMLP(ManualMLP):
    @classmethod
    def from_base(cls, base, decorrelation_lr, *, sample_fraction=.1, mean_momentum=.1):
        model = cls.__new__(cls)
        model.__dict__ = base.__dict__.copy()
        model.decorrelation_lr = float(decorrelation_lr)
        model.decorators = [DenseForwardDecorrelator(
            weight.shape[1], lr=decorrelation_lr, sample_fraction=sample_fraction,
            mean_momentum=mean_momentum, backend="low_rank", device=weight.device,
            dtype=weight.dtype) for weight in model.weights]
        model.decorrelation_updates = [0 for _ in model.weights]
        model._decor_cache = []
        return model

    @property
    def decor_weights(self):
        return [layer.decor_weight for layer in self.decorators]

    @property
    def decor_means(self):
        return [layer.running_mean for layer in self.decorators]

    def decorrelation_spec(self):
        return {"learning_rate": self.decorrelation_lr, "sample_fraction": .1, "mean_momentum": .1,
                "backend": "low_rank", "injection": "before outgoing decorator",
                "upstream_revision": "00cf47050bbd20e6a153e10bd86e1651524f9779"}

    def _forward_with_cache(self, x):
        h = x.to(self.device)
        presynaptic, preactivations, caches = [], [], []
        for i, (weight, bias, decorator) in enumerate(zip(self.weights, self.biases, self.decorators)):
            decorated, cache = decorator(h, training=self.training)
            if self.training:
                self.decorrelation_updates[i] += 1
            presynaptic.append(decorated)
            caches.append(cache)
            z = decorated @ weight.T + bias
            if self.batchnorm and i < self.n_hidden_layers:
                z = self._bn_forward(i, z)
            preactivations.append(z)
            h = torch.relu(z) if i < self.n_hidden_layers else z
        # Diagnostics/inference must not replace the training forward cache.
        if self.training:
            self._decor_cache = caches
        return h, [*presynaptic, h], preactivations, caches

    def forward(self, x):
        logits, activities, preactivations, _ = self._forward_with_cache(x)
        return logits, activities, preactivations

    def bp_gradients(self, x, y):
        logits, activities, preactivations, caches = self._forward_with_cache(x)
        y = y.to(self.device)
        loss = float(F.cross_entropy(logits, y).item())
        delta = _output_delta(logits, y)
        weights = [torch.empty_like(w) for w in self.weights]
        biases = [torch.empty_like(b) for b in self.biases]
        deltas = [torch.empty(0, device=self.device) for _ in self.weights]
        gammas, betas = self._empty_bn_grads()
        for i in range(len(self.weights)-1, -1, -1):
            if i < self.n_hidden_layers:
                # Derivative of U_i=(A_i-mean(A_i)) D_old, then ReLU and BN.
                signal = delta @ self.weights[i+1]
                pulled = self.decorators[i+1].backward(signal, caches[i+1])
                delta = pulled * (preactivations[i] > 0).to(pulled.dtype)
                if self.batchnorm:
                    delta, gammas[i], betas[i] = self._bn_backward(i, delta)
            deltas[i] = delta
            weights[i] = delta.T @ activities[i]
            biases[i] = delta.sum(0)
        return Gradients(weights, biases, deltas, loss, gammas, betas)


def decorator_state(model):
    if not isinstance(model, DecorrelatedMLP):
        return []
    return [value for layer in model.decorators for value in (layer.decor_weight, layer.running_mean)]


def learned_state_count(model):
    return sum(value.numel() for value in decorator_state(model))
