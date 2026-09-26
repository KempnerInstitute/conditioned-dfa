"""CIFAR-10N / CIFAR-100N human annotation noise labels.

Loads the re-annotation label files released by Wei et al. (2022),
"Learning with Noisy Labels Revisited" (https://github.com/UCSC-REAL/cifar-10-100n),
and swaps them into the standard torchvision CIFAR train split. The label files
are small .pt dictionaries aligned index-for-index with the torchvision train
set ordering:

- ``CIFAR-10_human.pt``: clean_label, aggre_label (~9% noise),
  random_label{1,2,3} (~17-18%), worse_label (~40%).
- ``CIFAR-100_human.pt``: clean_label, noisy_label (~40%), plus coarse labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

CIFAR10N_KEYS = {
    "clean": "clean_label",
    "aggre": "aggre_label",
    "random1": "random_label1",
    "random2": "random_label2",
    "random3": "random_label3",
    "worse": "worse_label",
}
CIFAR100N_KEYS = {"clean": "clean_label", "noisy": "noisy_label"}


def load_cifar_n_labels(data_dir: str | Path, dataset: str, noise_type: str) -> np.ndarray:
    """Return the (50000,) human-annotated train labels for the requested noise type.

    ``dataset`` is ``cifar10`` or ``cifar100``; ``noise_type`` is a key of
    :data:`CIFAR10N_KEYS` or :data:`CIFAR100N_KEYS`.
    """

    data_dir = Path(data_dir)
    if dataset == "cifar10":
        path = data_dir / "cifar_n" / "CIFAR-10_human.pt"
        keys = CIFAR10N_KEYS
    elif dataset == "cifar100":
        path = data_dir / "cifar_n" / "CIFAR-100_human.pt"
        keys = CIFAR100N_KEYS
    else:
        raise ValueError(f"CIFAR-N labels are defined for cifar10/cifar100, not {dataset!r}")
    if noise_type not in keys:
        raise ValueError(f"noise_type {noise_type!r} not in {sorted(keys)} for {dataset}")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download CIFAR-N label files from "
            "https://github.com/UCSC-REAL/cifar-10-100n into data/cifar_n/."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    labels = np.asarray(payload[keys[noise_type]], dtype=np.int64)
    if labels.shape != (50_000,):
        raise ValueError(f"Unexpected CIFAR-N label shape {labels.shape} in {path}")
    return labels


def apply_cifar_n_labels(train_targets, data_dir: str | Path, dataset: str, noise_type: str) -> np.ndarray:
    """Validate alignment against the clean labels, then return the noisy labels.

    ``train_targets`` are the torchvision train-set targets; the CIFAR-N files
    include the clean labels, so alignment with the torchvision ordering is
    checked exactly rather than assumed.
    """

    clean = load_cifar_n_labels(data_dir, dataset, "clean")
    given = np.asarray(train_targets, dtype=np.int64)
    if not np.array_equal(clean, given):
        raise ValueError(
            "CIFAR-N clean labels do not match the torchvision train targets; "
            "the dataset ordering differs and noisy labels cannot be applied."
        )
    return load_cifar_n_labels(data_dir, dataset, noise_type)
