"""Synthetic data generators for the Conditioned DFA project (latent-manifold tasks)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np



@dataclass(frozen=True)
class CircleDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    z_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    z_test: np.ndarray
    projection: np.ndarray
    manifold: str = "circle"

def make_circle_split(
    *,
    n_train: int = 1024,
    n_test: int = 1024,
    input_dim: int = 8,
    noise: float = 0.05,
    task_frequency: int = 2,
    seed: int = 0,
) -> CircleDataset:
    """Noisy samples from S^1 with labels defined by a latent task direction."""

    rng = np.random.default_rng(seed)
    if input_dim < 2:
        raise ValueError("input_dim must be at least 2")
    projection = rng.normal(size=(2, input_dim))
    projection /= np.maximum(np.linalg.norm(projection, axis=1, keepdims=True), 1e-12)

    z_train = rng.uniform(0.0, 2.0 * np.pi, size=n_train)
    z_test = rng.uniform(0.0, 2.0 * np.pi, size=n_test)
    x_train = circle_features(z_train, projection, noise=noise, rng=rng)
    x_test = circle_features(z_test, projection, noise=noise, rng=rng)
    y_train = circle_labels(z_train, task_frequency=task_frequency)
    y_test = circle_labels(z_test, task_frequency=task_frequency)
    return CircleDataset(
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        z_train=z_train.astype(np.float32),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        z_test=z_test.astype(np.float32),
        projection=projection.astype(np.float32),
        manifold="circle",
    )


def make_manifold_split(
    *,
    manifold: str = "circle",
    n_train: int = 1024,
    n_test: int = 1024,
    input_dim: int = 8,
    noise: float = 0.05,
    task_frequency: int = 2,
    seed: int = 0,
) -> CircleDataset:
    """Noisy samples from a latent manifold with binary labels.

    Supported manifolds are:

    - ``circle``: 1D latent circle, the original DFA benchmark.
    - ``torus``: 2D latent torus with task and nuisance directions.
    - ``swiss_roll``: 2D curved sheet embedded through a random projection.
    - ``low_rank``: Gaussian latent factors with nuisance dimensions.
    """

    manifold = manifold.lower().replace("-", "_")
    if manifold == "circle":
        return make_circle_split(
            n_train=n_train,
            n_test=n_test,
            input_dim=input_dim,
            noise=noise,
            task_frequency=task_frequency,
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    base_dim = manifold_base_dim(manifold)
    projection = rng.normal(size=(base_dim, input_dim))
    projection /= np.maximum(np.linalg.norm(projection, axis=1, keepdims=True), 1e-12)
    z_train = sample_latent(manifold, n_train, rng=rng)
    z_test = sample_latent(manifold, n_test, rng=rng)
    x_train = manifold_features(z_train, projection, manifold=manifold, noise=noise, rng=rng)
    x_test = manifold_features(z_test, projection, manifold=manifold, noise=noise, rng=rng)
    y_train = manifold_labels(z_train, manifold=manifold, task_frequency=task_frequency)
    y_test = manifold_labels(z_test, manifold=manifold, task_frequency=task_frequency)
    return CircleDataset(
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        z_train=z_train.astype(np.float32),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        z_test=z_test.astype(np.float32),
        projection=projection.astype(np.float32),
        manifold=manifold,
    )


def circle_features(
    z: np.ndarray,
    projection: np.ndarray,
    *,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Embed latent circle coordinates into an input space."""

    z = np.asarray(z, dtype=float)
    base = np.column_stack([np.cos(z), np.sin(z)])
    x = base @ np.asarray(projection, dtype=float)
    if noise > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        x = x + rng.normal(scale=noise, size=x.shape)
    return x


def manifold_base_dim(manifold: str) -> int:
    """Feature dimension before random projection for each latent manifold."""

    manifold = manifold.lower().replace("-", "_")
    if manifold == "circle":
        return 2
    if manifold == "torus":
        return 4
    if manifold == "swiss_roll":
        return 3
    if manifold == "low_rank":
        return 5
    raise ValueError(f"Unknown manifold: {manifold}")


def sample_latent(manifold: str, n_samples: int, *, rng: np.random.Generator) -> np.ndarray:
    """Sample latent coordinates for supported manifolds."""

    manifold = manifold.lower().replace("-", "_")
    if manifold == "circle":
        return rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    if manifold == "torus":
        return rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, 2))
    if manifold == "swiss_roll":
        t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n_samples)
        height = rng.uniform(-1.0, 1.0, size=n_samples)
        return np.column_stack([t, height])
    if manifold == "low_rank":
        return rng.normal(size=(n_samples, 5))
    raise ValueError(f"Unknown manifold: {manifold}")


def manifold_features(
    z: np.ndarray,
    projection: np.ndarray,
    *,
    manifold: str,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Embed latent coordinates into input space."""

    manifold = manifold.lower().replace("-", "_")
    if manifold == "circle":
        return circle_features(np.asarray(z).reshape(-1), projection, noise=noise, rng=rng)

    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    if manifold == "torus":
        theta = z[:, 0]
        phi = z[:, 1]
        base = np.column_stack([np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)])
    elif manifold == "swiss_roll":
        t = z[:, 0]
        height = z[:, 1]
        scale = 1.0 / (4.5 * np.pi)
        base = np.column_stack([scale * t * np.cos(t), height, scale * t * np.sin(t)])
    elif manifold == "low_rank":
        base = z[:, : manifold_base_dim(manifold)]
    else:
        raise ValueError(f"Unknown manifold: {manifold}")

    x = base @ np.asarray(projection, dtype=float)
    if noise > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        x = x + rng.normal(scale=noise, size=x.shape)
    return x


def circle_labels(z: np.ndarray, *, task_frequency: int = 2) -> np.ndarray:
    """Binary labels that create multiple arcs on the circle."""

    z = np.asarray(z, dtype=float)
    return (np.sin(task_frequency * z) > 0.0).astype(np.int64)


def manifold_labels(z: np.ndarray, *, manifold: str, task_frequency: int = 2) -> np.ndarray:
    """Binary labels for supported latent manifolds."""

    manifold = manifold.lower().replace("-", "_")
    if manifold == "circle":
        return circle_labels(np.asarray(z).reshape(-1), task_frequency=task_frequency)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    if manifold == "torus":
        score = np.sin(task_frequency * z[:, 0]) + 0.65 * np.sin((task_frequency + 1) * z[:, 1])
    elif manifold == "swiss_roll":
        score = np.sin(0.75 * task_frequency * z[:, 0]) + 0.8 * z[:, 1]
    elif manifold == "low_rank":
        score = z[:, 0] + 0.75 * z[:, 1] - 0.35 * z[:, 2] * z[:, 3]
    else:
        raise ValueError(f"Unknown manifold: {manifold}")
    return (score > np.median(score)).astype(np.int64)


def task_boundary_weights(z: np.ndarray, *, manifold: str, task_frequency: int = 2) -> np.ndarray:
    """Weights that emphasize latent samples near task boundaries."""

    manifold = manifold.lower().replace("-", "_")
    z = np.asarray(z, dtype=float)
    if manifold == "circle":
        score = np.sin(task_frequency * z.reshape(-1))
    elif manifold == "torus":
        score = np.sin(task_frequency * z[:, 0]) + 0.65 * np.sin((task_frequency + 1) * z[:, 1])
    elif manifold == "swiss_roll":
        score = np.sin(0.75 * task_frequency * z[:, 0]) + 0.8 * z[:, 1]
    elif manifold == "low_rank":
        score = z[:, 0] + 0.75 * z[:, 1] - 0.35 * z[:, 2] * z[:, 3]
    else:
        raise ValueError(f"Unknown manifold: {manifold}")
    scale = np.std(score) + 1e-8
    return 0.05 + np.exp(-0.5 * (score / scale) ** 2)

