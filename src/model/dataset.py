"""Dataset PyTorch con normalización consistente, SpecAugment y muestreo ponderado."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.config import PROJECT_ROOT


def normalize_spectrogram(spec_db: np.ndarray) -> np.ndarray:
    """Estandariza un espectrograma Mel (media 0, desviación ~1) de forma idéntica en train/infer."""
    arr = spec_db.astype(np.float32, copy=False)
    mean = float(arr.mean())
    std = float(arr.std())
    return (arr - mean) / (std + 1e-6)


def apply_spectrogram_augmentations(spec_norm: np.ndarray) -> np.ndarray:
    """Aplica aumentos de datos (solo en entrenamiento) sobre el espectrograma normalizado."""
    aug = spec_norm.copy()
    n_mels, n_frames = aug.shape

    # 1. Desplazamiento temporal circular (invarianza frente al inicio del canto)
    if random.random() < 0.70:
        shift = random.randint(-30, 30)
        if shift != 0:
            aug = np.roll(aug, shift=shift, axis=1)

    # 2. Contraste / ganancia leve
    if random.random() < 0.60:
        gain = random.uniform(0.85, 1.15)
        aug = aug * gain

    # 3. SpecAugment: enmascaramiento de bandas de frecuencia
    num_freq_masks = random.choice([1, 2])
    for _ in range(num_freq_masks):
        if random.random() < 0.75:
            f_width = random.randint(2, min(8, n_mels // 4))
            f_start = random.randint(0, max(0, n_mels - f_width))
            aug[f_start : f_start + f_width, :] = 0.0

    # 4. SpecAugment: enmascaramiento temporal
    num_time_masks = random.choice([1, 2])
    for _ in range(num_time_masks):
        if random.random() < 0.75:
            t_width = random.randint(4, min(22, n_frames // 4))
            t_start = random.randint(0, max(0, n_frames - t_width))
            aug[:, t_start : t_start + t_width] = 0.0

    # 5. Ruido gaussiano moderado (simula ruido ambiental de campo)
    if random.random() < 0.60:
        sigma = random.uniform(0.02, 0.12)
        noise = np.random.normal(0.0, sigma, size=aug.shape).astype(np.float32)
        aug = aug + noise

    return aug.astype(np.float32, copy=False)


class BirdSpectrogramDataset(Dataset):
    """Dataset de ventanas de espectrogramas Mel en memoria para PyTorch."""

    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: dict[str, int],
        augment: bool = False,
        cache_in_memory: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.augment = augment
        self.cache_in_memory = cache_in_memory

        self.spec_paths = [
            PROJECT_ROOT / Path(rel_path) for rel_path in self.df["spec_rel_path"].tolist()
        ]
        self.labels = [class_to_idx[sp] for sp in self.df["especie"].tolist()]
        self.recording_ids = self.df["id"].astype(str).tolist()
        self.qualities = self.df["calidad"].astype(str).tolist()
        self.vocalizations = self.df["tipo_vocalizacion_principal"].astype(str).tolist()

        self._cache: list[np.ndarray] | None = None
        if self.cache_in_memory:
            self._cache = [
                normalize_spectrogram(np.load(path)) for path in self.spec_paths
            ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        if self._cache is not None:
            spec = self._cache[idx]
        else:
            spec = normalize_spectrogram(np.load(self.spec_paths[idx]))

        if self.augment:
            spec = apply_spectrogram_augmentations(spec)

        tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, 64, 173)
        return tensor, self.labels[idx], self.recording_ids[idx]


def build_weighted_sampler(
    train_df: pd.DataFrame, class_to_idx: dict[str, int]
) -> WeightedRandomSampler:
    """Construye un muestreador ponderado para mitigar desbalanceo de clases y calidad."""
    species_counts = Counter(train_df["especie"].tolist())
    total_samples = len(train_df)
    num_classes = len(class_to_idx)

    quality_boost = {"A": 1.00, "B": 1.00, "C": 1.15}
    weights: list[float] = []
    for _, row in train_df.iterrows():
        sp_count = max(1, species_counts[row["especie"]])
        class_w = total_samples / (num_classes * sp_count)
        q_w = quality_boost.get(str(row["calidad"]).upper(), 1.0)
        weights.append(class_w * q_w)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
