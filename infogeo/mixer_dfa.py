"""Small manual MLP-Mixer with BP/DFA/nDFA updates.

Every mixer sublayer (patch embedding, token-mixing MLP, channel-mixing MLP)
receives direct feedback from the output error (DFA): a fixed random matrix
projects the ``(batch, n_classes)`` output delta to the sublayer's output
dimension, the projected error is broadcast over the sublayer's other axis
(channels for token-mixing weights, tokens for channel-mixing weights and the
patch embedding), and gated by the sublayer ReLU where one follows — the exact
mixer analogue of the translation-tied channel feedback in
``infogeo.conv_dfa``. nDFA right-multiplies each local weight update by the
damped inverse second moment of that layer's presynaptic activity (token
profiles per channel for token-mixing layers, channel vectors per token for
channel-mixing layers), estimated over batch x the broadcast axis; K-nDFA adds
the error-side factor. The classifier head always trains on the exact output
error. BP gradients use autograd on the same forward pass, so BP is exact
through LayerNorm and the residual paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from infogeo.conv_dfa import damped_solve, output_delta_from_logits, truncate_rank
from infogeo.dfa import _torch_cosine, error_second_moment


@dataclass
class MixerLayerSpec:
    """One locally trained mixer sublayer (excludes the exact-gradient head)."""

    name: str
    out_dim: int
    gated: bool


@dataclass
class MixerForward:
    logits: torch.Tensor
    presyn: list[torch.Tensor]
    preact: list[torch.Tensor | None]
    pooled: torch.Tensor


@dataclass
class MixerGradients:
    weights: list[torch.Tensor]
    biases: list[torch.Tensor]
    deltas: list[torch.Tensor]
    loss: float


class ManualMixer:
    """A minimal ReLU MLP-Mixer with manual BP/DFA updates.

    Patch embedding (no nonlinearity) -> ``depth`` mixer blocks of pre-LN
    token-mixing MLP + pre-LN channel-mixing MLP with residual connections ->
    global average pool over tokens -> linear classifier. ReLU replaces the
    standard GELU so the repo's binary nonlinearity gating conventions apply
    unchanged; LayerNorm is non-affine (pure standardization) so no
    normalization parameters need local credit assignment. ``layernorm=False``
    ablates both LayerNorms.
    """

    def __init__(
        self,
        image_shape: Sequence[int],
        output_dim: int,
        *,
        patch_size: int = 4,
        dim: int = 128,
        depth: int = 4,
        expansion: int = 2,
        layernorm: bool = True,
        seed: int = 0,
        device: str = "cpu",
        ln_eps: float = 1e-5,
    ) -> None:
        self.device = torch.device(device)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.output_dim = int(output_dim)
        self.patch_size = int(patch_size)
        self.dim = int(dim)
        self.depth = int(depth)
        self.expansion = int(expansion)
        self.layernorm = bool(layernorm)
        self.ln_eps = float(ln_eps)
        channels, height, width = self.image_shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image height/width must be divisible by patch_size")
        self.n_tokens = (height // self.patch_size) * (width // self.patch_size)
        self.patch_dim = channels * self.patch_size * self.patch_size
        self.token_hidden = self.expansion * self.n_tokens
        self.channel_hidden = self.expansion * self.dim

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        self.weights: list[torch.Tensor] = []
        self.biases: list[torch.Tensor] = []
        self.layer_specs: list[MixerLayerSpec] = []

        def add_layer(name: str, out_dim: int, in_dim: int, *, gated: bool, scale_mult: float = 1.0) -> None:
            scale = scale_mult * np.sqrt(2.0 / max(in_dim, 1))
            weight = torch.randn(out_dim, in_dim, generator=generator, device=self.device) * scale
            self.weights.append(weight)
            self.biases.append(torch.zeros(out_dim, device=self.device))
            self.layer_specs.append(MixerLayerSpec(name, int(out_dim), bool(gated)))

        # Residual-output init scaling (GPT-2 style): shrink each sublayer's
        # second linear by 1/sqrt(2*depth) so the residual stream's variance
        # stays bounded even with LayerNorm ablated. Applied identically for
        # every learning rule, so no method-specific confound.
        residual_scale = 1.0 / np.sqrt(2.0 * max(self.depth, 1))
        add_layer("patch_embed", self.dim, self.patch_dim, gated=False)
        for block in range(self.depth):
            add_layer(f"block{block}_token_fc1", self.token_hidden, self.n_tokens, gated=True)
            add_layer(f"block{block}_token_fc2", self.n_tokens, self.token_hidden, gated=False, scale_mult=residual_scale)
            add_layer(f"block{block}_channel_fc1", self.channel_hidden, self.dim, gated=True)
            add_layer(f"block{block}_channel_fc2", self.dim, self.channel_hidden, gated=False, scale_mult=residual_scale)
        # Classifier head (exact output-error gradient, never DFA/preconditioned).
        head_scale = np.sqrt(2.0 / max(self.dim, 1))
        self.weights.append(torch.randn(self.output_dim, self.dim, generator=generator, device=self.device) * head_scale)
        self.biases.append(torch.zeros(self.output_dim, device=self.device))
        for weight, bias in zip(self.weights, self.biases):
            weight.requires_grad_(True)
            bias.requires_grad_(True)

    @property
    def n_local_layers(self) -> int:
        return len(self.layer_specs)

    def _maybe_layernorm(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.layernorm:
            return tokens
        return F.layer_norm(tokens, (self.dim,), eps=self.ln_eps)

    def forward_full(self, x: torch.Tensor) -> MixerForward:
        x = x.to(self.device)
        if x.ndim == 2:
            x = x.reshape(-1, *self.image_shape)
        batch = x.shape[0]
        patches = F.unfold(x, self.patch_size, stride=self.patch_size).transpose(1, 2)  # (B, T, P)
        presyn: list[torch.Tensor] = [patches.reshape(-1, self.patch_dim)]
        preact: list[torch.Tensor | None] = [None]
        idx = 0
        tokens = patches @ self.weights[idx].T + self.biases[idx]  # (B, T, C)
        for block in range(self.depth):
            # Token-mixing sublayer: mixes across the token axis per channel.
            u = self._maybe_layernorm(tokens).transpose(1, 2)  # (B, C, T)
            presyn.append(u.reshape(-1, self.n_tokens))
            a1 = u @ self.weights[idx + 1].T + self.biases[idx + 1]  # (B, C, T_h)
            preact.append(a1.reshape(-1, self.token_hidden))
            h1 = torch.relu(a1)
            presyn.append(h1.reshape(-1, self.token_hidden))
            preact.append(None)
            z1 = h1 @ self.weights[idx + 2].T + self.biases[idx + 2]  # (B, C, T)
            tokens = tokens + z1.transpose(1, 2)
            # Channel-mixing sublayer: mixes across the channel axis per token.
            u2 = self._maybe_layernorm(tokens)  # (B, T, C)
            presyn.append(u2.reshape(-1, self.dim))
            a2 = u2 @ self.weights[idx + 3].T + self.biases[idx + 3]  # (B, T, C_h)
            preact.append(a2.reshape(-1, self.channel_hidden))
            h2 = torch.relu(a2)
            presyn.append(h2.reshape(-1, self.channel_hidden))
            preact.append(None)
            z2 = h2 @ self.weights[idx + 4].T + self.biases[idx + 4]  # (B, T, C)
            tokens = tokens + z2
            idx += 4
        pooled = tokens.mean(dim=1)  # (B, C)
        logits = pooled @ self.weights[-1].T + self.biases[-1]
        assert len(presyn) == self.n_local_layers and batch == x.shape[0]
        return MixerForward(logits, presyn, preact, pooled)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.argmax(self.forward_full(x).logits, dim=1)

    def accuracy(self, x: torch.Tensor, y: torch.Tensor, *, batch_size: int | None = None) -> float:
        if batch_size is None or batch_size <= 0 or len(x) <= batch_size:
            pred = self.predict(x)
            return float((pred == y.to(self.device)).float().mean().item())
        correct = 0
        total = 0
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            pred = self.predict(x[start:stop])
            yb = y[start:stop].to(self.device)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
        return float(correct / max(total, 1))

    def bp_gradients(self, x: torch.Tensor, y: torch.Tensor) -> MixerGradients:
        """Exact gradients via autograd on the manual forward pass."""

        full = self.forward_full(x)
        y = y.to(self.device)
        loss_t = F.cross_entropy(full.logits, y)
        params = [*self.weights, *self.biases]
        grads = torch.autograd.grad(loss_t, params)
        n_weights = len(self.weights)
        grad_w = [g.detach() for g in grads[:n_weights]]
        grad_b = [g.detach() for g in grads[n_weights:]]
        deltas = [torch.empty(0, device=self.device) for _ in range(self.n_local_layers)]
        deltas.append(output_delta_from_logits(full.logits.detach(), y))
        return MixerGradients(grad_w, grad_b, deltas, float(loss_t.item()))

    def dfa_gradients(self, x: torch.Tensor, y: torch.Tensor, feedback: Sequence[torch.Tensor]) -> MixerGradients:
        if len(feedback) != self.n_local_layers:
            raise ValueError(f"feedback must contain {self.n_local_layers} matrices")

        with torch.no_grad():
            full = self.forward_full(x)
        y = y.to(self.device)
        loss = float(F.cross_entropy(full.logits, y).item())
        output_delta = output_delta_from_logits(full.logits, y)
        batch = output_delta.shape[0]

        grad_w: list[torch.Tensor] = []
        grad_b: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        for layer_idx, spec in enumerate(self.layer_specs):
            presyn = full.presyn[layer_idx]
            n_samples = presyn.shape[0]
            broadcast = n_samples // batch
            # Divide by the broadcast multiplicity so the weight update is a
            # mean over the non-mixing axis: the repo's (probs - onehot)/batch
            # output-delta normalization extended to the batch x axis samples.
            delta = output_delta @ feedback[layer_idx].to(self.device) / max(broadcast, 1)  # (B, out_dim)
            delta = delta.unsqueeze(1).expand(batch, broadcast, spec.out_dim).reshape(n_samples, spec.out_dim)
            if spec.gated:
                delta = delta * (full.preact[layer_idx] > 0).float()
            deltas.append(delta)
            grad_w.append(delta.T @ presyn)
            grad_b.append(delta.sum(dim=0))
        grad_w.append(output_delta.T @ full.pooled)
        grad_b.append(output_delta.sum(dim=0))
        deltas.append(output_delta)
        return MixerGradients(grad_w, grad_b, deltas, loss)

    def apply_gradients(self, gradients: MixerGradients, *, lr: float) -> None:
        with torch.no_grad():
            for layer_idx in range(len(self.weights)):
                self.weights[layer_idx].sub_(lr * gradients.weights[layer_idx])
                self.biases[layer_idx].sub_(lr * gradients.biases[layer_idx])


def init_mixer_feedback(
    model: ManualMixer,
    *,
    seed: int = 0,
    scale: float = 1.0,
    rank: int | None = None,
) -> list[torch.Tensor]:
    """Fixed random output-to-sublayer feedback matrices (one per sublayer)."""

    rng = np.random.default_rng(seed)
    feedback: list[torch.Tensor] = []
    for spec in model.layer_specs:
        raw = rng.normal(size=(model.output_dim, spec.out_dim))
        original_norm = np.linalg.norm(raw)
        if rank is not None and rank > 0:
            raw = truncate_rank(raw, rank)
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw * (scale * original_norm / norm)
        feedback.append(torch.tensor(raw, dtype=torch.float32, device=model.device))
    return feedback


def natural_precondition_mixer_gradients(
    model: ManualMixer,
    gradients: MixerGradients,
    x: torch.Tensor,
    *,
    damping: float,
    error_damping: float | None = None,
    mode: str = "activity",
    cache: dict | None = None,
    refresh: bool = True,
) -> MixerGradients:
    """Local natural-DFA preconditioning of mixer sublayer updates.

    ``mode="activity"`` (nDFA) right-multiplies each sublayer weight update by
    the damped inverse presynaptic second moment, estimated over
    batch x the broadcast axis (channels for token-mixing weights, tokens for
    channel-mixing weights). ``mode="kronecker"`` (K-nDFA) additionally
    left-multiplies by the damped inverse second moment of the layer's own
    gated feedback error. The classifier head is never preconditioned.
    ``cache``/``refresh`` implement amortized covariance refresh exactly as in
    ``experiments.run_dfa_synthetic.natural_precondition_gradients``.
    """

    left_damping = damping if error_damping is None else float(error_damping)
    if cache is not None and not cache:
        refresh = True  # never apply an uninitialized cache
    need_forward = cache is None or refresh
    full = None
    if need_forward:
        with torch.no_grad():
            full = model.forward_full(x)
    new_weights = [grad.clone() for grad in gradients.weights]
    new_biases = [grad.clone() for grad in gradients.biases]
    for layer_idx in range(model.n_local_layers):
        grad = new_weights[layer_idx]
        if mode in {"activity", "kronecker"}:
            if cache is None:
                activity = full.presyn[layer_idx].detach()
                cov = activity.T @ activity / max(activity.shape[0], 1)
                grad = damped_solve(cov, grad.T, damping=damping).T
            else:
                key = ("activity", layer_idx)
                if refresh:
                    activity = full.presyn[layer_idx].detach()
                    cov = activity.T @ activity / max(activity.shape[0], 1)
                    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
                    cache[key] = damped_solve(cov, eye, damping=damping)
                grad = (cache[key] @ grad.T).T
        if mode in {"error", "kronecker"}:
            if cache is None:
                delta = gradients.deltas[layer_idx].detach()
                cov = error_second_moment(delta, normalization_count=delta.shape[0])
                grad = damped_solve(cov, grad, damping=left_damping)
            else:
                key = ("error", layer_idx)
                if refresh:
                    delta = gradients.deltas[layer_idx].detach()
                    cov = error_second_moment(delta, normalization_count=delta.shape[0])
                    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
                    cache[key] = damped_solve(cov, eye, damping=left_damping)
                grad = cache[key] @ grad
        new_weights[layer_idx] = grad
    return MixerGradients(new_weights, new_biases, gradients.deltas, gradients.loss)


def mixer_gradient_cosines(reference: MixerGradients, estimate: MixerGradients) -> dict[str, float]:
    """Hidden-parameter alignment between two mixer gradient estimates."""

    hidden_ref = torch.cat([grad.flatten() for grad in reference.weights[:-1]])
    hidden_est = torch.cat([grad.flatten() for grad in estimate.weights[:-1]])
    return {"param_cosine": _torch_cosine(hidden_ref, hidden_est)}
