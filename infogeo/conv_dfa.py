"""Manual convolutional networks for DFA and K-nDFA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from infogeo.dfa import Gradients, _torch_cosine


@dataclass(frozen=True)
class ConvGradients:
    conv_weights: list[torch.Tensor]
    conv_biases: list[torch.Tensor]
    fc_weights: list[torch.Tensor]
    fc_biases: list[torch.Tensor]
    deltas: list[torch.Tensor]
    loss: float

    @property
    def fc_weight(self) -> torch.Tensor:
        return self.fc_weights[-1]

    @property
    def fc_bias(self) -> torch.Tensor:
        return self.fc_biases[-1]


@dataclass(frozen=True)
class ConvForward:
    logits: torch.Tensor
    conv_activations: list[torch.Tensor]
    conv_preactivations: list[torch.Tensor]
    fc_activations: list[torch.Tensor]
    fc_preactivations: list[torch.Tensor]

    @property
    def flat(self) -> torch.Tensor:
        return self.fc_activations[0]


@dataclass
class ConvLocalHeads:
    conv_weights: list[torch.Tensor]
    conv_biases: list[torch.Tensor]
    fc_weights: list[torch.Tensor]
    fc_biases: list[torch.Tensor]


@dataclass
class ConvLocalHeadGradients:
    conv_weights: list[torch.Tensor]
    conv_biases: list[torch.Tensor]
    fc_weights: list[torch.Tensor]
    fc_biases: list[torch.Tensor]


class ManualConvNet:
    """Small stride-conv network with manual BP/DFA updates."""

    def __init__(
        self,
        input_shape: Sequence[int],
        output_dim: int,
        *,
        channels: Sequence[int] = (32, 64, 128),
        strides: Sequence[int] = (1, 2, 2),
        kernels: Sequence[int] | None = None,
        paddings: Sequence[int] | None = None,
        fc_hidden: Sequence[int] = (),
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.output_dim = int(output_dim)
        self.channels = [int(v) for v in channels]
        self.strides = [int(v) for v in strides]
        self.kernels = [int(v) for v in (kernels if kernels is not None else [3] * len(self.channels))]
        self.paddings = [int(v) for v in (paddings if paddings is not None else [1] * len(self.channels))]
        self.fc_hidden = [int(v) for v in fc_hidden]
        if len(self.channels) != len(self.strides):
            raise ValueError("channels and strides must have the same length")
        if len(self.channels) != len(self.kernels) or len(self.channels) != len(self.paddings):
            raise ValueError("channels, strides, kernels, and paddings must have the same length")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        in_channels = [self.input_shape[0], *self.channels[:-1]]
        self.conv_weights: list[torch.Tensor] = []
        self.conv_biases: list[torch.Tensor] = []
        for in_ch, out_ch, kernel in zip(in_channels, self.channels, self.kernels):
            scale = np.sqrt(2.0 / max(kernel * kernel * in_ch, 1))
            weight = torch.randn(out_ch, in_ch, kernel, kernel, generator=generator, device=self.device) * scale
            bias = torch.zeros(out_ch, device=self.device)
            self.conv_weights.append(weight)
            self.conv_biases.append(bias)

        flat_dim = int(np.prod(self.hidden_shapes(batch_size=1)[-1][1:]))
        fc_dims = [flat_dim, *self.fc_hidden, self.output_dim]
        self.fc_weights: list[torch.Tensor] = []
        self.fc_biases: list[torch.Tensor] = []
        for in_dim, out_dim in zip(fc_dims[:-1], fc_dims[1:]):
            self.fc_weights.append(torch.randn(out_dim, in_dim, generator=generator, device=self.device) * np.sqrt(2.0 / max(in_dim, 1)))
            self.fc_biases.append(torch.zeros(out_dim, device=self.device))

    @property
    def n_fc_hidden_layers(self) -> int:
        return len(self.fc_hidden)

    @property
    def fc_weight(self) -> torch.Tensor:
        return self.fc_weights[-1]

    @fc_weight.setter
    def fc_weight(self, value: torch.Tensor) -> None:
        self.fc_weights[-1] = value

    @property
    def fc_bias(self) -> torch.Tensor:
        return self.fc_biases[-1]

    @fc_bias.setter
    def fc_bias(self, value: torch.Tensor) -> None:
        self.fc_biases[-1] = value

    @property
    def n_hidden_layers(self) -> int:
        return len(self.conv_weights)

    def hidden_shapes(self, *, batch_size: int) -> list[tuple[int, int, int, int]]:
        _, height, width = self.input_shape
        shapes = []
        for channels, stride, kernel, padding in zip(self.channels, self.strides, self.kernels, self.paddings):
            height = int(np.floor((height + 2 * padding - kernel) / stride + 1))
            width = int(np.floor((width + 2 * padding - kernel) / stride + 1))
            shapes.append((batch_size, channels, height, width))
        return shapes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        full = self.forward_full(x)
        return full.logits, full.conv_activations, full.conv_preactivations, full.flat

    def forward_full(self, x: torch.Tensor) -> ConvForward:
        h = x.to(self.device)
        conv_activations = [h]
        conv_preactivations: list[torch.Tensor] = []
        for weight, bias, stride, padding in zip(self.conv_weights, self.conv_biases, self.strides, self.paddings):
            a = F.conv2d(h, weight, bias, stride=stride, padding=padding)
            conv_preactivations.append(a)
            h = torch.relu(a)
            conv_activations.append(h)
        h = conv_activations[-1].reshape(conv_activations[-1].shape[0], -1)
        fc_activations = [h]
        fc_preactivations: list[torch.Tensor] = []
        for weight, bias in zip(self.fc_weights[:-1], self.fc_biases[:-1]):
            a = h @ weight.T + bias
            fc_preactivations.append(a)
            h = torch.relu(a)
            fc_activations.append(h)
        logits = h @ self.fc_weights[-1].T + self.fc_biases[-1]
        return ConvForward(logits, conv_activations, conv_preactivations, fc_activations, fc_preactivations)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _, _ = self.forward(x)
        return torch.argmax(logits, dim=1)

    def accuracy(self, x: torch.Tensor, y: torch.Tensor, *, batch_size: int | None = None) -> float:
        if batch_size is None or batch_size <= 0 or len(x) <= batch_size:
            with torch.no_grad():
                pred = self.predict(x)
                return float((pred == y.to(self.device)).float().mean().item())
        correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                stop = min(start + batch_size, len(x))
                xb = x[start:stop]
                yb = y[start:stop].to(self.device)
                pred = self.predict(xb)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        return float(correct / max(total, 1))

    def bp_gradients(self, x: torch.Tensor, y: torch.Tensor) -> ConvGradients:
        full = self.forward_full(x)
        logits = full.logits
        activations = full.conv_activations
        preactivations = full.conv_preactivations
        y = y.to(self.device)
        loss = float(F.cross_entropy(logits, y).item())
        output_delta = output_delta_from_logits(logits, y)

        fc_weight_grads: list[torch.Tensor] = [torch.empty_like(w) for w in self.fc_weights]
        fc_bias_grads: list[torch.Tensor] = [torch.empty_like(b) for b in self.fc_biases]
        delta = output_delta
        for fc_idx in range(len(self.fc_weights) - 1, -1, -1):
            fc_weight_grads[fc_idx] = delta.T @ full.fc_activations[fc_idx]
            fc_bias_grads[fc_idx] = delta.sum(dim=0)
            delta = delta @ self.fc_weights[fc_idx]
            if fc_idx > 0:
                delta = delta * (full.fc_preactivations[fc_idx - 1] > 0).float()

        delta = delta.reshape_as(activations[-1])
        delta = delta * (preactivations[-1] > 0).float()
        conv_weight_grads: list[torch.Tensor] = [torch.empty_like(w) for w in self.conv_weights]
        conv_bias_grads: list[torch.Tensor] = [torch.empty_like(b) for b in self.conv_biases]
        deltas: list[torch.Tensor] = [torch.empty(0, device=self.device) for _ in range(self.n_hidden_layers + self.n_fc_hidden_layers + 1)]

        for layer_idx in range(self.n_hidden_layers - 1, -1, -1):
            deltas[layer_idx] = delta
            conv_weight_grads[layer_idx] = torch.nn.grad.conv2d_weight(
                activations[layer_idx],
                self.conv_weights[layer_idx].shape,
                delta,
                stride=self.strides[layer_idx],
                padding=self.paddings[layer_idx],
            )
            conv_bias_grads[layer_idx] = delta.sum(dim=(0, 2, 3))
            if layer_idx > 0:
                delta = torch.nn.grad.conv2d_input(
                    activations[layer_idx].shape,
                    self.conv_weights[layer_idx],
                    delta,
                    stride=self.strides[layer_idx],
                    padding=self.paddings[layer_idx],
                )
                delta = delta * (preactivations[layer_idx - 1] > 0).float()
        # Store BP hidden deltas for activity-level diagnostics.
        # Recompute cheaply from the recursive pass above.
        bp_delta = output_delta
        fc_hidden_deltas: list[torch.Tensor] = []
        for fc_idx in range(len(self.fc_weights) - 1, 0, -1):
            bp_delta = bp_delta @ self.fc_weights[fc_idx]
            bp_delta = bp_delta * (full.fc_preactivations[fc_idx - 1] > 0).float()
            fc_hidden_deltas.append(bp_delta)
        for offset, hidden_delta in enumerate(reversed(fc_hidden_deltas)):
            deltas[self.n_hidden_layers + offset] = hidden_delta
        deltas[-1] = output_delta
        return ConvGradients(conv_weight_grads, conv_bias_grads, fc_weight_grads, fc_bias_grads, deltas, loss)

    def dfa_gradients(self, x: torch.Tensor, y: torch.Tensor, feedback: Sequence[torch.Tensor]) -> ConvGradients:
        expected_feedback = self.n_hidden_layers + self.n_fc_hidden_layers
        if len(feedback) != expected_feedback:
            raise ValueError(f"feedback must contain {expected_feedback} matrices")

        full = self.forward_full(x)
        logits = full.logits
        activations = full.conv_activations
        preactivations = full.conv_preactivations
        y = y.to(self.device)
        loss = float(F.cross_entropy(logits, y).item())
        output_delta = output_delta_from_logits(logits, y)
        fc_weight_grads: list[torch.Tensor] = [torch.empty_like(w) for w in self.fc_weights]
        fc_bias_grads: list[torch.Tensor] = [torch.empty_like(b) for b in self.fc_biases]
        fc_weight_grads[-1] = output_delta.T @ full.fc_activations[-1]
        fc_bias_grads[-1] = output_delta.sum(dim=0)

        conv_weight_grads: list[torch.Tensor] = []
        conv_bias_grads: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        for layer_idx in range(self.n_hidden_layers):
            hidden_shape = activations[layer_idx + 1].shape
            delta = conv_feedback_delta(output_delta, feedback[layer_idx].to(self.device), hidden_shape)
            delta = delta * (preactivations[layer_idx] > 0).float()
            deltas.append(delta)
            conv_weight_grads.append(
                torch.nn.grad.conv2d_weight(
                    activations[layer_idx],
                    self.conv_weights[layer_idx].shape,
                    delta,
                    stride=self.strides[layer_idx],
                    padding=self.paddings[layer_idx],
                )
            )
            conv_bias_grads.append(delta.sum(dim=(0, 2, 3)))
        fc_feedback = feedback[self.n_hidden_layers :]
        for fc_idx in range(self.n_fc_hidden_layers):
            delta = output_delta @ fc_feedback[fc_idx].to(self.device)
            delta = delta * (full.fc_preactivations[fc_idx] > 0).float()
            deltas.append(delta)
            fc_weight_grads[fc_idx] = delta.T @ full.fc_activations[fc_idx]
            fc_bias_grads[fc_idx] = delta.sum(dim=0)
        deltas.append(output_delta)
        return ConvGradients(conv_weight_grads, conv_bias_grads, fc_weight_grads, fc_bias_grads, deltas, loss)

    def target_projection_gradients(self, x: torch.Tensor, y: torch.Tensor, feedback: Sequence[torch.Tensor]) -> ConvGradients:
        """Direct random target-projection gradients for conv activations."""

        expected_feedback = self.n_hidden_layers + self.n_fc_hidden_layers
        if len(feedback) != expected_feedback:
            raise ValueError(f"feedback must contain {expected_feedback} matrices")

        full = self.forward_full(x)
        logits = full.logits
        activations = full.conv_activations
        preactivations = full.conv_preactivations
        y = y.to(self.device)
        loss = float(F.cross_entropy(logits, y).item())
        output_delta = output_delta_from_logits(logits, y)
        targets = F.one_hot(y, num_classes=logits.shape[1]).float() / logits.shape[0]
        fc_weight_grads: list[torch.Tensor] = [torch.empty_like(w) for w in self.fc_weights]
        fc_bias_grads: list[torch.Tensor] = [torch.empty_like(b) for b in self.fc_biases]
        fc_weight_grads[-1] = output_delta.T @ full.fc_activations[-1]
        fc_bias_grads[-1] = output_delta.sum(dim=0)

        conv_weight_grads: list[torch.Tensor] = []
        conv_bias_grads: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        for layer_idx in range(self.n_hidden_layers):
            hidden_shape = activations[layer_idx + 1].shape
            delta = conv_feedback_delta(targets, feedback[layer_idx].to(self.device), hidden_shape)
            delta = delta * (preactivations[layer_idx] > 0).float()
            deltas.append(delta)
            conv_weight_grads.append(
                torch.nn.grad.conv2d_weight(
                    activations[layer_idx],
                    self.conv_weights[layer_idx].shape,
                    delta,
                    stride=self.strides[layer_idx],
                    padding=self.paddings[layer_idx],
                )
            )
            conv_bias_grads.append(delta.sum(dim=(0, 2, 3)))
        fc_feedback = feedback[self.n_hidden_layers :]
        for fc_idx in range(self.n_fc_hidden_layers):
            delta = targets @ fc_feedback[fc_idx].to(self.device)
            delta = delta * (full.fc_preactivations[fc_idx] > 0).float()
            deltas.append(delta)
            fc_weight_grads[fc_idx] = delta.T @ full.fc_activations[fc_idx]
            fc_bias_grads[fc_idx] = delta.sum(dim=0)
        deltas.append(output_delta)
        return ConvGradients(conv_weight_grads, conv_bias_grads, fc_weight_grads, fc_bias_grads, deltas, loss)

    def apply_gradients(self, gradients: ConvGradients, *, lr: float) -> None:
        for idx in range(self.n_hidden_layers):
            self.conv_weights[idx] = self.conv_weights[idx] - lr * gradients.conv_weights[idx]
            self.conv_biases[idx] = self.conv_biases[idx] - lr * gradients.conv_biases[idx]
        for idx in range(len(self.fc_weights)):
            self.fc_weights[idx] = self.fc_weights[idx] - lr * gradients.fc_weights[idx]
            self.fc_biases[idx] = self.fc_biases[idx] - lr * gradients.fc_biases[idx]


def output_delta_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
    return (probs - one_hot) / logits.shape[0]


def init_conv_feedback(
    model: ManualConvNet,
    *,
    batch_size: int = 1,
    seed: int = 0,
    scale: float = 1.0,
    rank: int | None = None,
    mode: str = "flat",
    normalization: str = "none",
) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    feedback = []
    for shape in model.hidden_shapes(batch_size=batch_size):
        _, channels, height, width = shape
        flat_dim = int(channels * height * width)
        if mode == "flat":
            target_dim = flat_dim
            raw = rng.normal(size=(model.output_dim, flat_dim))
        elif mode == "channel":
            # Translation-tied feedback: one coefficient per output class and
            # feature channel, broadcast over spatial positions. This respects
            # convolutional weight sharing and avoids IID random signs for every
            # pixel in the same filter.
            target_dim = int(channels)
            raw = rng.normal(size=(model.output_dim, target_dim))
        else:
            raise ValueError(f"Unknown conv feedback mode: {mode}")
        original_norm = np.linalg.norm(raw)
        if rank is not None and rank > 0:
            raw = truncate_rank(raw, rank)
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw * (scale * original_norm / norm)
        if normalization == "none":
            pass
        elif normalization == "sqrt_target":
            raw = raw / np.sqrt(max(target_dim, 1))
        elif normalization == "sqrt_flat":
            raw = raw / np.sqrt(max(flat_dim, 1))
        else:
            raise ValueError(f"Unknown conv feedback normalization: {normalization}")
        feedback.append(torch.tensor(raw, dtype=torch.float32, device=model.device))
    for hidden_dim in model.fc_hidden:
        raw = rng.normal(size=(model.output_dim, int(hidden_dim)))
        original_norm = np.linalg.norm(raw)
        if rank is not None and rank > 0:
            raw = truncate_rank(raw, rank)
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw * (scale * original_norm / norm)
        if normalization == "sqrt_target":
            raw = raw / np.sqrt(max(int(hidden_dim), 1))
        elif normalization == "sqrt_flat":
            raw = raw / np.sqrt(max(int(hidden_dim), 1))
        elif normalization != "none":
            raise ValueError(f"Unknown conv feedback normalization: {normalization}")
        feedback.append(torch.tensor(raw, dtype=torch.float32, device=model.device))
    return feedback


def init_conv_local_heads(model: ManualConvNet, *, seed: int = 0) -> ConvLocalHeads:
    """Initialize trainable local auxiliary classifiers for hidden activations."""

    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    conv_weights: list[torch.Tensor] = []
    conv_biases: list[torch.Tensor] = []
    for channels in model.channels:
        conv_weights.append(
            torch.randn(model.output_dim, int(channels), generator=generator, device=model.device)
            * np.sqrt(2.0 / max(int(channels), 1))
        )
        conv_biases.append(torch.zeros(model.output_dim, device=model.device))
    fc_weights: list[torch.Tensor] = []
    fc_biases: list[torch.Tensor] = []
    for hidden_dim in model.fc_hidden:
        fc_weights.append(
            torch.randn(model.output_dim, int(hidden_dim), generator=generator, device=model.device)
            * np.sqrt(2.0 / max(int(hidden_dim), 1))
        )
        fc_biases.append(torch.zeros(model.output_dim, device=model.device))
    return ConvLocalHeads(conv_weights, conv_biases, fc_weights, fc_biases)


def conv_local_loss_gradients(
    model: ManualConvNet,
    x: torch.Tensor,
    y: torch.Tensor,
    heads: ConvLocalHeads,
) -> tuple[ConvGradients, ConvLocalHeadGradients]:
    """Local auxiliary-loss baseline.

    Each hidden activation trains from its own linear classifier. Conv heads
    read global-average-pooled channels; FC heads read the hidden activity. The
    final classifier is still trained on the global output loss so the baseline
    stays comparable to DFA-style methods that use the same output layer.
    """

    full = model.forward_full(x)
    y = y.to(model.device)
    output_delta = output_delta_from_logits(full.logits, y)
    losses = [float(F.cross_entropy(full.logits, y).item())]

    conv_weight_grads: list[torch.Tensor] = []
    conv_bias_grads: list[torch.Tensor] = []
    conv_head_weight_grads: list[torch.Tensor] = []
    conv_head_bias_grads: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []

    for layer_idx in range(model.n_hidden_layers):
        activation = full.conv_activations[layer_idx + 1]
        pooled = activation.mean(dim=(2, 3))
        local_logits = pooled @ heads.conv_weights[layer_idx].T + heads.conv_biases[layer_idx]
        losses.append(float(F.cross_entropy(local_logits, y).item()))
        local_delta = output_delta_from_logits(local_logits, y)
        conv_head_weight_grads.append(local_delta.T @ pooled)
        conv_head_bias_grads.append(local_delta.sum(dim=0))
        delta_channels = local_delta @ heads.conv_weights[layer_idx]
        delta = delta_channels.reshape(delta_channels.shape[0], delta_channels.shape[1], 1, 1).expand_as(activation)
        delta = delta / max(int(activation.shape[2] * activation.shape[3]), 1)
        delta = delta * (full.conv_preactivations[layer_idx] > 0).float()
        deltas.append(delta)
        conv_weight_grads.append(
            torch.nn.grad.conv2d_weight(
                full.conv_activations[layer_idx],
                model.conv_weights[layer_idx].shape,
                delta,
                stride=model.strides[layer_idx],
                padding=model.paddings[layer_idx],
            )
        )
        conv_bias_grads.append(delta.sum(dim=(0, 2, 3)))

    fc_weight_grads: list[torch.Tensor] = [torch.empty_like(w) for w in model.fc_weights]
    fc_bias_grads: list[torch.Tensor] = [torch.empty_like(b) for b in model.fc_biases]
    fc_head_weight_grads: list[torch.Tensor] = []
    fc_head_bias_grads: list[torch.Tensor] = []
    for fc_idx in range(model.n_fc_hidden_layers):
        activation = full.fc_activations[fc_idx + 1]
        local_logits = activation @ heads.fc_weights[fc_idx].T + heads.fc_biases[fc_idx]
        losses.append(float(F.cross_entropy(local_logits, y).item()))
        local_delta = output_delta_from_logits(local_logits, y)
        fc_head_weight_grads.append(local_delta.T @ activation)
        fc_head_bias_grads.append(local_delta.sum(dim=0))
        delta = (local_delta @ heads.fc_weights[fc_idx]) * (full.fc_preactivations[fc_idx] > 0).float()
        deltas.append(delta)
        fc_weight_grads[fc_idx] = delta.T @ full.fc_activations[fc_idx]
        fc_bias_grads[fc_idx] = delta.sum(dim=0)

    fc_weight_grads[-1] = output_delta.T @ full.fc_activations[-1]
    fc_bias_grads[-1] = output_delta.sum(dim=0)
    deltas.append(output_delta)
    gradients = ConvGradients(
        conv_weight_grads,
        conv_bias_grads,
        fc_weight_grads,
        fc_bias_grads,
        deltas,
        float(np.mean(losses)),
    )
    head_gradients = ConvLocalHeadGradients(
        conv_head_weight_grads,
        conv_head_bias_grads,
        fc_head_weight_grads,
        fc_head_bias_grads,
    )
    return gradients, head_gradients


def apply_conv_local_head_gradients(heads: ConvLocalHeads, gradients: ConvLocalHeadGradients, *, lr: float) -> None:
    for idx in range(len(heads.conv_weights)):
        heads.conv_weights[idx] = heads.conv_weights[idx] - lr * gradients.conv_weights[idx]
        heads.conv_biases[idx] = heads.conv_biases[idx] - lr * gradients.conv_biases[idx]
    for idx in range(len(heads.fc_weights)):
        heads.fc_weights[idx] = heads.fc_weights[idx] - lr * gradients.fc_weights[idx]
        heads.fc_biases[idx] = heads.fc_biases[idx] - lr * gradients.fc_biases[idx]


def conv_feedback_delta(output_delta: torch.Tensor, feedback: torch.Tensor, hidden_shape: torch.Size) -> torch.Tensor:
    """Project output error into a convolutional activation tensor.

    ``feedback`` can either be a dense flattened DFA matrix with shape
    ``(output_dim, channels * height * width)`` or a translation-tied channel
    matrix with shape ``(output_dim, channels)``. The latter is closer to the
    convolutional DFA implementations in which feedback respects spatial weight
    sharing.
    """
    _, channels, height, width = hidden_shape
    if feedback.ndim != 2:
        raise ValueError("conv feedback must be a 2-D matrix")
    projected = output_delta @ feedback
    flat_dim = int(channels * height * width)
    if feedback.shape[1] == flat_dim:
        return projected.reshape(hidden_shape)
    if feedback.shape[1] == channels:
        return projected.reshape(output_delta.shape[0], channels, 1, 1).expand(hidden_shape)
    raise ValueError(
        f"Feedback second dimension {feedback.shape[1]} does not match flat dim {flat_dim} or channels {channels}"
    )


def natural_precondition_conv_gradients(
    model: ManualConvNet,
    gradients: ConvGradients,
    x: torch.Tensor,
    *,
    damping: float,
    mode: str = "activity",
) -> ConvGradients:
    full = model.forward_full(x)
    activations = full.conv_activations
    conv_weights = [grad.clone() for grad in gradients.conv_weights]
    for layer_idx in range(model.n_hidden_layers):
        grad = conv_weights[layer_idx]
        if mode in {"activity", "kronecker"}:
            cov = channel_covariance(activations[layer_idx].detach())
            grad = solve_input_channel(cov, grad, damping=damping)
        if mode in {"error", "kronecker"}:
            cov = channel_covariance(gradients.deltas[layer_idx].detach())
            grad = solve_output_channel(cov, grad, damping=damping)
        conv_weights[layer_idx] = grad
    fc_weights = [grad.clone() for grad in gradients.fc_weights]
    for fc_idx in range(model.n_fc_hidden_layers):
        grad = fc_weights[fc_idx]
        if mode in {"activity", "kronecker"}:
            cov = feature_covariance(full.fc_activations[fc_idx].detach())
            grad = solve_linear_input(cov, grad, damping=damping)
        if mode in {"error", "kronecker"}:
            cov = feature_covariance(gradients.deltas[model.n_hidden_layers + fc_idx].detach())
            grad = solve_linear_output(cov, grad, damping=damping)
        fc_weights[fc_idx] = grad
    # The classifier is trained with the exact output error. Full Kronecker
    # preconditioning of the flattened conv representation would require an
    # enormous spatial covariance block, so the first conv K-nDFA version
    # preconditions only local convolutional filters.
    return ConvGradients(conv_weights, gradients.conv_biases, fc_weights, gradients.fc_biases, gradients.deltas, gradients.loss)


def kernel_spatial_covariance(
    activation: torch.Tensor,
    kernel_size: tuple[int, int],
    stride: int,
    padding: int,
) -> torch.Tensor:
    """Covariance of the input over the kH*kW kernel-spatial positions.

    Unfolds the layer input into im2col patches and forms the (kH*kW)x(kH*kW)
    covariance across batch, input channels, and output positions. This is the
    spatial block of the Kronecker factorization C^ch (x) C^sp of the conv input
    covariance: the dimension that actually multiplies the kernel gradient (a
    feature-map HW x HW factor would mix spatial positions and break the
    convolution's locality).
    """
    k_h, k_w = kernel_size
    in_ch = activation.shape[1]
    patches = F.unfold(activation, (k_h, k_w), stride=stride, padding=padding)
    batch, _, positions = patches.shape
    patches = patches.reshape(batch, in_ch, k_h * k_w, positions)
    samples = patches.permute(0, 1, 3, 2).reshape(-1, k_h * k_w)
    return samples.T @ samples / max(samples.shape[0], 1)


def solve_kernel_spatial(cov: torch.Tensor, grad: torch.Tensor, *, damping: float) -> torch.Tensor:
    """Condition the kH*kW kernel-spatial axis of a conv kernel gradient."""
    out_ch, in_ch, k_h, k_w = grad.shape
    rhs = grad.reshape(out_ch * in_ch, k_h * k_w).T
    solved = damped_solve(cov, rhs, damping=damping)
    return solved.T.reshape(out_ch, in_ch, k_h, k_w).contiguous()


def spatial_kronecker_conv_gradients(
    model: ManualConvNet,
    gradients: ConvGradients,
    x: torch.Tensor,
    *,
    damping: float,
) -> ConvGradients:
    """Spatial-Kronecker conditioning of the conv update (Eq. spatial_kron).

    Extends channel-only nDFA by adding a kernel-spatial covariance factor: the
    DFA kernel gradient is conditioned by the damped inverse input-channel
    covariance (as in nDFA) *and* by the damped inverse kernel-spatial
    covariance, approximating C_in^{-1} for the Kronecker-factored conv input
    covariance C^ch (x) C^sp. Cost is in_ch^2 + (kH kW)^2 rather than
    (in_ch kH kW)^2. The channel factor matches nDFA exactly, so the comparison
    isolates the spatial term identified by the spatial-routing diagnostics.
    """
    full = model.forward_full(x)
    activations = full.conv_activations
    conv_weights = [grad.clone() for grad in gradients.conv_weights]
    for layer_idx in range(model.n_hidden_layers):
        grad = conv_weights[layer_idx]
        channel_cov = channel_covariance(activations[layer_idx].detach())
        grad = solve_input_channel(channel_cov, grad, damping=damping)
        k_h, k_w = model.conv_weights[layer_idx].shape[2:]
        spatial_cov = kernel_spatial_covariance(
            activations[layer_idx].detach(),
            (k_h, k_w),
            model.strides[layer_idx],
            model.paddings[layer_idx],
        )
        grad = solve_kernel_spatial(spatial_cov, grad, damping=damping)
        conv_weights[layer_idx] = grad
    # FC layers keep the channel-only (activity) feature conditioning of nDFA.
    fc_weights = [grad.clone() for grad in gradients.fc_weights]
    for fc_idx in range(model.n_fc_hidden_layers):
        cov = feature_covariance(full.fc_activations[fc_idx].detach())
        fc_weights[fc_idx] = solve_linear_input(cov, fc_weights[fc_idx], damping=damping)
    return ConvGradients(
        conv_weights, gradients.conv_biases, fc_weights, gradients.fc_biases, gradients.deltas, gradients.loss
    )


def norm_match_conv_gradients(
    model: ManualConvNet,
    gradients: ConvGradients,
    x: torch.Tensor,
    y: torch.Tensor,
) -> ConvGradients:
    """Rescale each hidden DFA weight gradient to the BP gradient norm.

    Controls for per-layer gradient *scale* without whitening the *direction*.
    Comparing this against nDFA isolates the contribution of covariance whitening
    beyond simple per-layer norm matching (the reviewer's novelty crux).
    """
    bp = model.bp_gradients(x, y)

    def matched(dfa_list, bp_list, n):
        out = [g.clone() for g in dfa_list]
        for i in range(n):
            gn = out[i].norm().clamp_min(1e-12)
            out[i] = out[i] * (bp_list[i].norm() / gn)
        return out

    conv_w = matched(gradients.conv_weights, bp.conv_weights, model.n_hidden_layers)
    fc_w = matched(gradients.fc_weights, bp.fc_weights, model.n_fc_hidden_layers)
    return ConvGradients(
        conv_w, gradients.conv_biases, fc_w, gradients.fc_biases, gradients.deltas, gradients.loss
    )


def channel_covariance(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError("channel covariance expects BCHW tensor")
    samples = values.permute(0, 2, 3, 1).reshape(-1, values.shape[1])
    return samples.T @ samples / max(samples.shape[0], 1)


def feature_covariance(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("feature covariance expects BF tensor")
    return values.T @ values / max(values.shape[0], 1)


def solve_input_channel(cov: torch.Tensor, grad: torch.Tensor, *, damping: float) -> torch.Tensor:
    out_ch, in_ch, k_h, k_w = grad.shape
    rhs = grad.permute(1, 0, 2, 3).reshape(in_ch, -1)
    solved = damped_solve(cov, rhs, damping=damping)
    return solved.reshape(in_ch, out_ch, k_h, k_w).permute(1, 0, 2, 3).contiguous()


def solve_output_channel(cov: torch.Tensor, grad: torch.Tensor, *, damping: float) -> torch.Tensor:
    out_ch = grad.shape[0]
    rhs = grad.reshape(out_ch, -1)
    solved = damped_solve(cov, rhs, damping=damping)
    return solved.reshape_as(grad)


def solve_linear_input(cov: torch.Tensor, grad: torch.Tensor, *, damping: float) -> torch.Tensor:
    if cov.shape[0] > 512:
        return grad / torch.diag(cov).add(float(damping)).clamp_min(1e-6).reshape(1, -1)
    rhs = grad.T
    solved = damped_solve(cov, rhs, damping=damping)
    return solved.T.contiguous()


def solve_linear_output(cov: torch.Tensor, grad: torch.Tensor, *, damping: float) -> torch.Tensor:
    if cov.shape[0] > 512:
        return grad / torch.diag(cov).add(float(damping)).clamp_min(1e-6).reshape(-1, 1)
    solved = damped_solve(cov, grad, damping=damping)
    return solved.contiguous()


def damped_solve(cov: torch.Tensor, rhs: torch.Tensor, *, damping: float) -> torch.Tensor:
    cov = 0.5 * (cov + cov.T)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    base = max(float(damping), 1e-6)
    for multiplier in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        try:
            solution = torch.linalg.solve(cov + (base * multiplier) * eye, rhs)
        except RuntimeError:
            continue
        if torch.isfinite(solution).all():
            return solution
    return torch.linalg.lstsq(cov + (base * 10000.0) * eye, rhs).solution


def conv_gradient_cosines(reference: ConvGradients, estimate: ConvGradients) -> dict[str, float]:
    hidden_ref_parts = [w.flatten() for w in reference.conv_weights] + [w.flatten() for w in reference.fc_weights[:-1]]
    hidden_est_parts = [w.flatten() for w in estimate.conv_weights] + [w.flatten() for w in estimate.fc_weights[:-1]]
    output_ref = reference.fc_weights[-1].flatten()
    output_est = estimate.fc_weights[-1].flatten()
    ref = torch.cat([*hidden_ref_parts, output_ref])
    est = torch.cat([*hidden_est_parts, output_est])
    hidden_ref = torch.cat(hidden_ref_parts) if hidden_ref_parts else torch.empty(0, device=ref.device)
    hidden_est = torch.cat(hidden_est_parts) if hidden_est_parts else torch.empty(0, device=est.device)
    activity = []
    for ref_delta, est_delta in zip(reference.deltas[:-1], estimate.deltas[:-1]):
        activity.append(_torch_cosine(ref_delta.flatten(), est_delta.flatten()))
    projected_step = float((torch.sum(ref * est) / torch.sum(ref * ref).clamp_min(1e-12)).detach().cpu().item())
    hidden_projected_step = float("nan")
    hidden_norm_ratio = float("nan")
    if hidden_ref.numel() and hidden_est.numel():
        hidden_projected_step = float((torch.sum(hidden_ref * hidden_est) / torch.sum(hidden_ref * hidden_ref).clamp_min(1e-12)).detach().cpu().item())
        hidden_norm_ratio = float((torch.linalg.vector_norm(hidden_est) / torch.linalg.vector_norm(hidden_ref).clamp_min(1e-12)).detach().cpu().item())
    output_projected_step = float((torch.sum(output_ref * output_est) / torch.sum(output_ref * output_ref).clamp_min(1e-12)).detach().cpu().item())
    return {
        "param_cosine": _torch_cosine(ref, est),
        "hidden_param_cosine": _torch_cosine(hidden_ref, hidden_est) if hidden_ref.numel() and hidden_est.numel() else float("nan"),
        "output_param_cosine": _torch_cosine(output_ref, output_est),
        "activity_cosine_mean": float(np.nanmean(activity)) if activity else float("nan"),
        "projected_weight_step": projected_step,
        "hidden_projected_weight_step": hidden_projected_step,
        "output_projected_weight_step": output_projected_step,
        "hidden_weight_norm_ratio": hidden_norm_ratio,
    }


def conv_weight_norm_ratio(numerator: ConvGradients, denominator: ConvGradients) -> float:
    num = sum(float(torch.sum(w * w).item()) for w in numerator.conv_weights)
    num += sum(float(torch.sum(w * w).item()) for w in numerator.fc_weights)
    den = sum(float(torch.sum(w * w).item()) for w in denominator.conv_weights)
    den += sum(float(torch.sum(w * w).item()) for w in denominator.fc_weights)
    return float(np.sqrt(num / max(den, 1e-12)))


def conv_hidden_weight_norm_ratio(numerator: ConvGradients, denominator: ConvGradients) -> float:
    num = sum(float(torch.sum(w * w).item()) for w in numerator.conv_weights)
    num += sum(float(torch.sum(w * w).item()) for w in numerator.fc_weights[:-1])
    den = sum(float(torch.sum(w * w).item()) for w in denominator.conv_weights)
    den += sum(float(torch.sum(w * w).item()) for w in denominator.fc_weights[:-1])
    return float(np.sqrt(num / max(den, 1e-12)))


def truncate_rank(matrix: np.ndarray, rank: int) -> np.ndarray:
    max_rank = min(matrix.shape)
    rank = int(min(max(rank, 1), max_rank))
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank]
