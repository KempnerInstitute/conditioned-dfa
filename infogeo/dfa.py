"""Small manual MLP trainer for backpropagation and DFA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


STABLE_SOLVE_STATS = {
    "stable_solve_damping_escalations": 0,
    "stable_solve_lstsq_fallbacks": 0,
    "stable_solve_max_damping_multiplier": 1.0,
}


def reset_stable_solve_stats() -> None:
    STABLE_SOLVE_STATS["stable_solve_damping_escalations"] = 0
    STABLE_SOLVE_STATS["stable_solve_lstsq_fallbacks"] = 0
    STABLE_SOLVE_STATS["stable_solve_max_damping_multiplier"] = 1.0


def stable_solve_stats_dict() -> dict[str, float]:
    return {key: float(value) for key, value in STABLE_SOLVE_STATS.items()}


def _record_stable_solve(multiplier: float, *, used_lstsq: bool = False) -> None:
    if multiplier > 1.0:
        STABLE_SOLVE_STATS["stable_solve_damping_escalations"] += 1
    if used_lstsq:
        STABLE_SOLVE_STATS["stable_solve_lstsq_fallbacks"] += 1
    STABLE_SOLVE_STATS["stable_solve_max_damping_multiplier"] = max(
        float(STABLE_SOLVE_STATS["stable_solve_max_damping_multiplier"]),
        float(multiplier),
    )


@dataclass
class Gradients:
    weights: list[torch.Tensor]
    biases: list[torch.Tensor]
    deltas: list[torch.Tensor]
    loss: float
    bn_gammas: list[torch.Tensor] | None = None
    bn_betas: list[torch.Tensor] | None = None


class ManualMLP:
    """A minimal ReLU MLP with manual BP/DFA updates."""

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
        bn_momentum: float = 0.1,
    ) -> None:
        self.device = torch.device(device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        dims = [input_dim, *hidden_dims, output_dim]
        self.weights: list[torch.Tensor] = []
        self.biases: list[torch.Tensor] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / max(in_dim, 1))
            weight = torch.randn(out_dim, in_dim, generator=generator, device=self.device) * scale
            bias = torch.zeros(out_dim, device=self.device)
            self.weights.append(weight)
            self.biases.append(bias)
        # Optional BatchNorm1d after each hidden linear layer (pre-activation).
        # Off by default; without it every code path is unchanged.
        self.batchnorm = bool(batchnorm)
        self.bn_eps = float(bn_eps)
        self.bn_momentum = float(bn_momentum)
        self.training = False
        self.bn_gamma: list[torch.Tensor] = []
        self.bn_beta: list[torch.Tensor] = []
        self.bn_running_mean: list[torch.Tensor] = []
        self.bn_running_var: list[torch.Tensor] = []
        self._bn_cache: list[tuple[torch.Tensor, torch.Tensor, bool] | None] = []
        if self.batchnorm:
            for dim in hidden_dims:
                self.bn_gamma.append(torch.ones(dim, device=self.device))
                self.bn_beta.append(torch.zeros(dim, device=self.device))
                self.bn_running_mean.append(torch.zeros(dim, device=self.device))
                self.bn_running_var.append(torch.ones(dim, device=self.device))
                self._bn_cache.append(None)

    @property
    def n_hidden_layers(self) -> int:
        return len(self.weights) - 1

    @property
    def hidden_dims(self) -> list[int]:
        return [weight.shape[0] for weight in self.weights[:-1]]

    @property
    def output_dim(self) -> int:
        return int(self.weights[-1].shape[0])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        h = x.to(self.device)
        activations = [h]
        preactivations: list[torch.Tensor] = []
        for layer_idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            a = h @ weight.T + bias
            if self.batchnorm and layer_idx < len(self.weights) - 1:
                a = self._bn_forward(layer_idx, a)
            preactivations.append(a)
            if layer_idx < len(self.weights) - 1:
                h = torch.relu(a)
            else:
                h = a
            activations.append(h)
        return h, activations, preactivations

    def _bn_forward(self, layer_idx: int, a: torch.Tensor) -> torch.Tensor:
        """BatchNorm1d after the hidden linear map (pre-activation).

        Batch statistics (and running-stat updates) are used only while
        ``self.training`` is True; evaluation uses the running statistics.
        """

        if self.training and a.shape[0] > 1:
            mean = a.mean(dim=0)
            var = a.var(dim=0, unbiased=False)
            n = a.shape[0]
            momentum = self.bn_momentum
            unbiased_var = var * (n / max(n - 1, 1))
            self.bn_running_mean[layer_idx] = (1.0 - momentum) * self.bn_running_mean[layer_idx] + momentum * mean
            self.bn_running_var[layer_idx] = (1.0 - momentum) * self.bn_running_var[layer_idx] + momentum * unbiased_var
            was_training = True
        else:
            mean = self.bn_running_mean[layer_idx]
            var = self.bn_running_var[layer_idx]
            was_training = False
        inv_std = torch.rsqrt(var + self.bn_eps)
        x_hat = (a - mean) * inv_std
        self._bn_cache[layer_idx] = (x_hat, inv_std, was_training)
        return self.bn_gamma[layer_idx] * x_hat + self.bn_beta[layer_idx]

    def _bn_backward(self, layer_idx: int, dz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward through the layer-local BN given the grad w.r.t. its output."""

        cache = self._bn_cache[layer_idx]
        if cache is None:
            raise RuntimeError("BatchNorm backward called before forward")
        x_hat, inv_std, was_training = cache
        dgamma = (dz * x_hat).sum(dim=0)
        dbeta = dz.sum(dim=0)
        dx_hat = dz * self.bn_gamma[layer_idx]
        if was_training:
            da = inv_std * (dx_hat - dx_hat.mean(dim=0) - x_hat * (dx_hat * x_hat).mean(dim=0))
        else:
            da = dx_hat * inv_std
        return da, dgamma, dbeta

    def _empty_bn_grads(self) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        if not self.batchnorm:
            return None, None
        zeros_g = [torch.zeros_like(g) for g in self.bn_gamma]
        zeros_b = [torch.zeros_like(b) for b in self.bn_beta]
        return zeros_g, zeros_b

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward(x)
        return torch.argmax(logits, dim=1)

    def accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        pred = self.predict(x)
        return float((pred == y.to(self.device)).float().mean().item())

    def bp_gradients(self, x: torch.Tensor, y: torch.Tensor) -> Gradients:
        logits, activations, preactivations = self.forward(x)
        y = y.to(self.device)
        loss = float(F.cross_entropy(logits, y).item())
        delta = _output_delta(logits, y)

        grad_w: list[torch.Tensor] = [torch.empty_like(weight) for weight in self.weights]
        grad_b: list[torch.Tensor] = [torch.empty_like(bias) for bias in self.biases]
        deltas: list[torch.Tensor] = [torch.empty(0, device=self.device) for _ in self.weights]

        last = len(self.weights) - 1
        deltas[last] = delta
        grad_w[last] = delta.T @ activations[last]
        grad_b[last] = delta.sum(dim=0)

        bn_gammas, bn_betas = self._empty_bn_grads()
        for layer_idx in range(last - 1, -1, -1):
            delta = (delta @ self.weights[layer_idx + 1]) * (preactivations[layer_idx] > 0).float()
            if self.batchnorm:
                delta, bn_gammas[layer_idx], bn_betas[layer_idx] = self._bn_backward(layer_idx, delta)
            deltas[layer_idx] = delta
            grad_w[layer_idx] = delta.T @ activations[layer_idx]
            grad_b[layer_idx] = delta.sum(dim=0)

        return Gradients(grad_w, grad_b, deltas, loss, bn_gammas, bn_betas)

    def dfa_gradients(self, x: torch.Tensor, y: torch.Tensor, feedback: Sequence[torch.Tensor]) -> Gradients:
        if len(feedback) != self.n_hidden_layers:
            raise ValueError("feedback must contain one matrix per hidden layer")

        logits, activations, preactivations = self.forward(x)
        y = y.to(self.device)
        loss = float(F.cross_entropy(logits, y).item())
        output_delta = _output_delta(logits, y)

        grad_w: list[torch.Tensor] = [torch.empty_like(weight) for weight in self.weights]
        grad_b: list[torch.Tensor] = [torch.empty_like(bias) for bias in self.biases]
        deltas: list[torch.Tensor] = [torch.empty(0, device=self.device) for _ in self.weights]

        bn_gammas, bn_betas = self._empty_bn_grads()
        for layer_idx in range(self.n_hidden_layers):
            delta = (output_delta @ feedback[layer_idx].to(self.device)) * (preactivations[layer_idx] > 0).float()
            if self.batchnorm:
                delta, bn_gammas[layer_idx], bn_betas[layer_idx] = self._bn_backward(layer_idx, delta)
            deltas[layer_idx] = delta
            grad_w[layer_idx] = delta.T @ activations[layer_idx]
            grad_b[layer_idx] = delta.sum(dim=0)

        last = len(self.weights) - 1
        deltas[last] = output_delta
        grad_w[last] = output_delta.T @ activations[last]
        grad_b[last] = output_delta.sum(dim=0)
        return Gradients(grad_w, grad_b, deltas, loss, bn_gammas, bn_betas)

    def fa_gradients(self, x: torch.Tensor, y: torch.Tensor, feedback: Sequence[torch.Tensor]) -> Gradients:
        """Layerwise feedback-alignment gradients.

        Unlike DFA, each hidden layer receives the error from the next layer
        through a fixed random feedback matrix with the same shape as the
        corresponding backpropagated weight transpose.
        """

        if self.batchnorm:
            raise NotImplementedError("fa_gradients does not support --batchnorm")
        if len(feedback) != self.n_hidden_layers:
            raise ValueError("feedback must contain one matrix per hidden layer")

        logits, activations, preactivations = self.forward(x)
        y = y.to(self.device)
        loss = float(F.cross_entropy(logits, y).item())
        delta_next = _output_delta(logits, y)

        grad_w: list[torch.Tensor] = [torch.empty_like(weight) for weight in self.weights]
        grad_b: list[torch.Tensor] = [torch.empty_like(bias) for bias in self.biases]
        deltas: list[torch.Tensor] = [torch.empty(0, device=self.device) for _ in self.weights]

        last = len(self.weights) - 1
        deltas[last] = delta_next
        grad_w[last] = delta_next.T @ activations[last]
        grad_b[last] = delta_next.sum(dim=0)

        for layer_idx in range(last - 1, -1, -1):
            delta = (delta_next @ feedback[layer_idx].to(self.device)) * (preactivations[layer_idx] > 0).float()
            deltas[layer_idx] = delta
            grad_w[layer_idx] = delta.T @ activations[layer_idx]
            grad_b[layer_idx] = delta.sum(dim=0)
            delta_next = delta

        return Gradients(grad_w, grad_b, deltas, loss)

    def target_projection_gradients(self, x: torch.Tensor, y: torch.Tensor, feedback: Sequence[torch.Tensor]) -> Gradients:
        """Direct random target projection gradients.

        This baseline projects class targets rather than output errors to hidden
        layers, following the spirit of direct random target projection.
        """

        if self.batchnorm:
            raise NotImplementedError("target_projection_gradients does not support --batchnorm")
        if len(feedback) != self.n_hidden_layers:
            raise ValueError("feedback must contain one matrix per hidden layer")

        logits, activations, preactivations = self.forward(x)
        y = y.to(self.device)
        loss = float(F.cross_entropy(logits, y).item())
        targets = F.one_hot(y, num_classes=logits.shape[1]).float() / logits.shape[0]
        output_delta = _output_delta(logits, y)

        grad_w: list[torch.Tensor] = [torch.empty_like(weight) for weight in self.weights]
        grad_b: list[torch.Tensor] = [torch.empty_like(bias) for bias in self.biases]
        deltas: list[torch.Tensor] = [torch.empty(0, device=self.device) for _ in self.weights]

        for layer_idx in range(self.n_hidden_layers):
            delta = (targets @ feedback[layer_idx].to(self.device)) * (preactivations[layer_idx] > 0).float()
            deltas[layer_idx] = delta
            grad_w[layer_idx] = delta.T @ activations[layer_idx]
            grad_b[layer_idx] = delta.sum(dim=0)

        last = len(self.weights) - 1
        deltas[last] = output_delta
        grad_w[last] = output_delta.T @ activations[last]
        grad_b[last] = output_delta.sum(dim=0)
        return Gradients(grad_w, grad_b, deltas, loss)

    def apply_gradients(self, gradients: Gradients, *, lr: float) -> None:
        for layer_idx in range(len(self.weights)):
            self.weights[layer_idx] = self.weights[layer_idx] - lr * gradients.weights[layer_idx]
            self.biases[layer_idx] = self.biases[layer_idx] - lr * gradients.biases[layer_idx]
        if self.batchnorm and gradients.bn_gammas is not None and gradients.bn_betas is not None:
            for layer_idx in range(self.n_hidden_layers):
                self.bn_gamma[layer_idx] = self.bn_gamma[layer_idx] - lr * gradients.bn_gammas[layer_idx]
                self.bn_beta[layer_idx] = self.bn_beta[layer_idx] - lr * gradients.bn_betas[layer_idx]

    def hidden_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        _, activations, _ = self.forward(x)
        return activations[1:-1]


def init_feedback(
    model: ManualMLP,
    *,
    mode: str = "random",
    tangent_bases: Sequence[np.ndarray] | None = None,
    seed: int = 0,
    scale: float = 1.0,
    rank: int | None = None,
) -> list[torch.Tensor]:
    """Initialize DFA feedback matrices, optionally projected relative to tangents."""

    rng = np.random.default_rng(seed)
    feedback: list[torch.Tensor] = []
    for layer_idx, hidden_dim in enumerate(model.hidden_dims):
        raw = rng.normal(size=(model.output_dim, hidden_dim))
        original_norm = np.linalg.norm(raw)

        if mode in {"bp_aligned", "bp_orthogonal", "negative_bp"}:
            bp_matrix = _linear_backprop_matrix(model, layer_idx)
            if mode == "bp_aligned":
                raw = bp_matrix
            elif mode == "negative_bp":
                raw = -bp_matrix
            else:
                denom = float(np.sum(bp_matrix * bp_matrix))
                if denom > 1e-12:
                    raw = raw - (float(np.sum(raw * bp_matrix)) / denom) * bp_matrix

        if mode in {"tangent_biased", "tangent_orthogonal"}:
            if tangent_bases is None:
                raise ValueError("tangent_bases are required for tangent feedback modes")
            basis = np.asarray(tangent_bases[layer_idx], dtype=float)
            projector = basis @ basis.T if basis.size else np.zeros((hidden_dim, hidden_dim))
            if mode == "tangent_biased":
                raw = raw @ projector
            else:
                raw = raw @ (np.eye(hidden_dim) - projector)

        if rank is not None and rank > 0:
            raw = _truncate_rank(raw, rank)

        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw * (scale * original_norm / norm)
        feedback.append(torch.tensor(raw, dtype=torch.float32, device=model.device))
    return feedback


def init_fa_feedback(
    model: ManualMLP,
    *,
    seed: int = 0,
    scale: float = 1.0,
    rank: int | None = None,
) -> list[torch.Tensor]:
    """Initialize layerwise feedback-alignment matrices.

    Matrix ``l`` maps the next-layer error into hidden layer ``l``. Its shape is
    ``(dim_{l+1}, dim_l)``, matching the multiplication used by exact BP but
    without tying to the forward weights.
    """

    rng = np.random.default_rng(seed)
    hidden_dims = model.hidden_dims
    next_dims = [*hidden_dims[1:], model.output_dim]
    feedback: list[torch.Tensor] = []
    for hidden_dim, next_dim in zip(hidden_dims, next_dims):
        raw = rng.normal(size=(int(next_dim), int(hidden_dim)))
        original_norm = np.linalg.norm(raw)
        if rank is not None and rank > 0:
            raw = _truncate_rank(raw, rank)
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw * (scale * original_norm / norm)
        feedback.append(torch.tensor(raw, dtype=torch.float32, device=model.device))
    return feedback


def _linear_backprop_matrix(model: ManualMLP, layer_idx: int) -> np.ndarray:
    """Direct output-to-hidden matrix induced by current forward weights."""

    mat = model.weights[-1].detach().cpu().numpy()
    for weight_idx in range(model.n_hidden_layers - 2, layer_idx - 1, -1):
        mat = mat @ model.weights[weight_idx + 1].detach().cpu().numpy()
    return mat


def finite_difference_hidden_tangents(
    model: ManualMLP,
    z: np.ndarray,
    feature_fn,
    *,
    eps: float = 1e-3,
) -> list[np.ndarray]:
    """Estimate dh_l/dz for each hidden layer by finite differences."""

    z = np.asarray(z, dtype=np.float32)
    if z.ndim == 2:
        return finite_difference_hidden_jacobians(model, z, feature_fn, eps=eps)
    x_plus = torch.tensor(feature_fn(z + eps), dtype=torch.float32, device=model.device)
    x_minus = torch.tensor(feature_fn(z - eps), dtype=torch.float32, device=model.device)
    h_plus = model.hidden_activations(x_plus)
    h_minus = model.hidden_activations(x_minus)
    return [((hp - hm) / (2.0 * eps)).detach().cpu().numpy() for hp, hm in zip(h_plus, h_minus)]


def finite_difference_hidden_jacobians(
    model: ManualMLP,
    z: np.ndarray,
    feature_fn,
    *,
    eps: float = 1e-3,
) -> list[np.ndarray]:
    """Estimate hidden Jacobians for multidimensional latent coordinates.

    Returns one array per hidden layer with shape
    ``(n_samples, latent_dim, hidden_dim)``.
    """

    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError("z must have shape (n_samples, latent_dim)")
    n_samples, latent_dim = z.shape
    per_layer: list[list[np.ndarray]] | None = None
    for dim in range(latent_dim):
        step = np.zeros_like(z)
        step[:, dim] = eps
        x_plus = torch.tensor(feature_fn(z + step), dtype=torch.float32, device=model.device)
        x_minus = torch.tensor(feature_fn(z - step), dtype=torch.float32, device=model.device)
        h_plus = model.hidden_activations(x_plus)
        h_minus = model.hidden_activations(x_minus)
        layer_derivs = [((hp - hm) / (2.0 * eps)).detach().cpu().numpy() for hp, hm in zip(h_plus, h_minus)]
        if per_layer is None:
            per_layer = [[] for _ in layer_derivs]
        for layer_idx, deriv in enumerate(layer_derivs):
            if deriv.shape[0] != n_samples:
                raise ValueError("feature_fn returned inconsistent sample count")
            per_layer[layer_idx].append(deriv)
    assert per_layer is not None
    return [np.stack(parts, axis=1) for parts in per_layer]


def local_pca_tangent_spaces(
    hidden: np.ndarray,
    *,
    n_neighbors: int = 16,
    rank: int = 2,
) -> np.ndarray:
    """Estimate sample-wise tangent bases from nearby hidden activations.

    This is intended for datasets such as MNIST/Fashion-MNIST where there is no
    known latent coordinate. It returns an array with shape
    ``(n_samples, rank, hidden_dim)`` compatible with
    ``tangent_projected_cosines``.
    """

    hidden = np.asarray(hidden, dtype=float)
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape (n_samples, hidden_dim)")
    n_samples, hidden_dim = hidden.shape
    rank = int(min(max(rank, 1), hidden_dim))
    n_neighbors = int(min(max(n_neighbors, rank + 1), max(n_samples - 1, 1)))
    tangents = np.zeros((n_samples, rank, hidden_dim), dtype=float)
    if n_samples <= 1:
        return tangents

    sq_norm = np.sum(hidden * hidden, axis=1, keepdims=True)
    distances = sq_norm + sq_norm.T - 2.0 * hidden @ hidden.T
    np.fill_diagonal(distances, np.inf)
    for idx in range(n_samples):
        neighbors = np.argpartition(distances[idx], kth=n_neighbors - 1)[:n_neighbors]
        diffs = hidden[neighbors] - hidden[idx]
        diffs = diffs - diffs.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(diffs, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        n_components = min(rank, vt.shape[0])
        tangents[idx, :n_components] = vt[:n_components]
    return tangents


def rayleigh_quotient_scores(
    inputs: np.ndarray,
    weights: np.ndarray,
    *,
    relative: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Rayleigh quotient alignment of row weights with input covariance.

    For each row vector ``w``, this computes ``w^T C w / w^T w``. With
    ``relative=True`` the score is divided by ``trace(C)``, matching the
    normalized alignment-style diagnostic used in the neighboring alignment
    codebase.
    """

    x = np.asarray(inputs, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("inputs and weights must be two-dimensional")
    if x.shape[1] != w.shape[1]:
        raise ValueError("input feature dimension must match weight columns")
    if x.shape[0] < 2:
        return np.zeros(w.shape[0], dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    cov = 0.5 * (cov + cov.T) + eps * np.eye(cov.shape[0])
    numerator = np.einsum("oi,ij,oj->o", w, cov, w)
    denominator = np.sum(w * w, axis=1) + eps
    scores = numerator / denominator
    if relative:
        scores = scores / max(float(np.trace(cov)), eps)
    return scores.astype(float)


def gradient_cosines(bp: Gradients, dfa: Gradients) -> dict[str, float]:
    """Full hidden-parameter and hidden-activity gradient alignment."""

    hidden_weight_bp = torch.cat([grad.flatten() for grad in bp.weights[:-1]])
    hidden_weight_dfa = torch.cat([grad.flatten() for grad in dfa.weights[:-1]])
    out = {"param_cosine": _torch_cosine(hidden_weight_bp, hidden_weight_dfa)}
    for layer_idx in range(len(bp.deltas) - 1):
        out[f"activity_cosine_l{layer_idx + 1}"] = _torch_cosine(
            bp.deltas[layer_idx].flatten(),
            dfa.deltas[layer_idx].flatten(),
        )
    return out


def tangent_projected_cosines(
    bp: Gradients,
    dfa: Gradients,
    tangents: Sequence[np.ndarray],
    *,
    sample_weights: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Cosine after projecting each sample's field onto its latent tangent space."""

    out: dict[str, float] = {}
    weights_tensor = None
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float32)
        sample_weights = sample_weights / max(float(np.mean(sample_weights)), eps)
    for layer_idx, tangent_np in enumerate(tangents):
        tangent = torch.tensor(tangent_np, dtype=torch.float32, device=bp.deltas[layer_idx].device)
        bp_proj = _project_rows_onto_sample_subspaces(bp.deltas[layer_idx], tangent, eps=eps)
        dfa_proj = _project_rows_onto_sample_subspaces(dfa.deltas[layer_idx], tangent, eps=eps)
        cos = _row_cosine(bp_proj, dfa_proj, eps=eps)
        finite = cos[torch.isfinite(cos)]
        if sample_weights is not None and finite.numel():
            weights_tensor = torch.tensor(sample_weights, dtype=torch.float32, device=cos.device)
            finite_mask = torch.isfinite(cos)
            finite_weights = weights_tensor[finite_mask]
            denom = finite_weights.sum().clamp_min(eps)
            out[f"tangent_cosine_l{layer_idx + 1}"] = float((cos[finite_mask] * finite_weights).sum().div(denom).item())
        else:
            out[f"tangent_cosine_l{layer_idx + 1}"] = float(finite.mean().item()) if finite.numel() else float("nan")
    return out


def _output_delta(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
    return (probs - one_hot) / logits.shape[0]


def error_second_moment(
    loss_scaled_deltas: torch.Tensor,
    *,
    normalization_count: int,
) -> torch.Tensor:
    """Return the per-sample second moment of mean-loss error signals.

    Manual BP/DFA gradients store each row of ``deltas`` with the normalization
    used by the mean loss so that summing outer products produces an averaged
    weight gradient.  That normalization must be undone before estimating the
    error-side Kronecker factor; otherwise the factor is smaller by
    ``normalization_count**2`` and damping reduces it to an almost scalar
    rescaling.

    ``normalization_count`` is the number by which the local error was divided:
    the minibatch size for ordinary MLP/conv deltas, and batch times broadcast
    multiplicity for flattened mixer deltas.
    """

    if loss_scaled_deltas.ndim != 2:
        raise ValueError("error second moment expects a 2-D sample-by-feature tensor")
    count = int(normalization_count)
    if count <= 0:
        raise ValueError("normalization_count must be positive")
    per_sample = loss_scaled_deltas.detach() * count
    return per_sample.T @ per_sample / max(int(per_sample.shape[0]), 1)


def _truncate_rank(matrix: np.ndarray, rank: int) -> np.ndarray:
    max_rank = min(matrix.shape)
    rank = int(min(max(rank, 1), max_rank))
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank]


def _torch_cosine(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-12) -> float:
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    return float((torch.dot(a, b) / torch.clamp(denom, min=eps)).detach().cpu().item())


def _project_rows_onto_rows(values: torch.Tensor, basis_rows: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    coeff = torch.sum(values * basis_rows, dim=1, keepdim=True)
    denom = torch.sum(basis_rows * basis_rows, dim=1, keepdim=True).clamp_min(eps)
    return (coeff / denom) * basis_rows


def _project_rows_onto_sample_subspaces(
    values: torch.Tensor,
    basis: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    if basis.ndim == 2:
        return _project_rows_onto_rows(values, basis, eps=eps)
    if basis.ndim != 3:
        raise ValueError("basis must have shape (n, h) or (n, k, h)")
    if basis.shape[0] != values.shape[0] or basis.shape[2] != values.shape[1]:
        raise ValueError("basis shape is incompatible with values")
    projected = torch.zeros_like(values)
    eye_cache: dict[int, torch.Tensor] = {}
    for idx in range(values.shape[0]):
        b = basis[idx]
        valid = torch.linalg.norm(b, dim=1) > eps
        if not torch.any(valid):
            continue
        b = b[valid]
        gram = b @ b.T
        dim = int(gram.shape[0])
        if dim not in eye_cache:
            eye_cache[dim] = torch.eye(dim, dtype=gram.dtype, device=gram.device)
        coeff = _stable_solve(gram, b @ values[idx], eye_cache[dim], eps=eps)
        projected[idx] = coeff @ b
    return projected


def _stable_solve(gram: torch.Tensor, rhs: torch.Tensor, eye: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Solve tiny Gram systems robustly when sample tangents are rank-deficient."""

    base = max(float(eps), 1e-6)
    last_error: RuntimeError | None = None
    for multiplier in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        matrix = gram + (base * multiplier) * eye
        try:
            coeff = torch.linalg.solve(matrix, rhs)
        except RuntimeError as exc:
            last_error = exc
            continue
        if torch.isfinite(coeff).all():
            _record_stable_solve(multiplier)
            return coeff
    try:
        _record_stable_solve(10000.0, used_lstsq=True)
        return torch.linalg.lstsq(gram + (base * 10000.0) * eye, rhs).solution
    except RuntimeError:
        if last_error is not None:
            raise last_error
        raise


def _row_cosine(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    numerator = torch.sum(a * b, dim=1)
    denominator = torch.linalg.norm(a, dim=1) * torch.linalg.norm(b, dim=1)
    return numerator / denominator.clamp_min(eps)
