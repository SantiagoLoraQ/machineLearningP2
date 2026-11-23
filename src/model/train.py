"""Entrenamiento supervisado de BirdCNN2D sobre espectrogramas de Mel en PyTorch.

Incluye partición sin fuga de datos, WeightedRandomSampler, SpecAugment en train,
Regularización (Dropout + weight_decay + label smoothing), ReduceLROnPlateau,
Early Stopping y guardado del mejor checkpoint.

Uso:
    python -m src.model.train
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import (
    BEST_MODEL_PATH,
    CLASS_MAPPING_PATH,
    CLIP_DURATION_SEC,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    FMAX,
    FMIN,
    HOP_LENGTH,
    MODEL_CONFIG_PATH,
    N_FFT,
    N_MELS,
    RANDOM_SEED,
    SAMPLE_RATE,
    SPEC_FRAMES,
    SPEC_INDEX_PATH,
    TRAINING_HISTORY_PATH,
    ensure_directories,
    to_project_relative_posix,
)
from src.model.cnn2d import BirdCNN2D
from src.model.dataset import BirdSpectrogramDataset, build_weighted_sampler


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fija semillas globales para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evalúa un conjunto agregando probabilidades Softmax tanto por ventana como por grabación."""
    model.eval()
    total_loss = 0.0
    total_windows = 0
    win_correct = 0

    rec_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    rec_targets: dict[str, int] = {}

    with torch.no_grad():
        for inputs, targets, rec_ids in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_size = inputs.size(0)
            total_loss += float(loss.item()) * batch_size
            total_windows += batch_size

            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            targets_np = targets.cpu().numpy()
            win_correct += int(np.sum(preds == targets_np))

            for rec_id, prob_vec, tgt in zip(rec_ids, probs, targets_np):
                rec_probs[str(rec_id)].append(prob_vec)
                rec_targets[str(rec_id)] = int(tgt)

    rec_losses: list[float] = []
    rec_correct = 0
    for rec_id, prob_list in rec_probs.items():
        mean_prob = np.mean(prob_list, axis=0)
        mean_prob = np.clip(mean_prob, 1e-7, 1.0)
        tgt = rec_targets[rec_id]
        rec_losses.append(float(-np.log(mean_prob[tgt])))
        if int(np.argmax(mean_prob)) == tgt:
            rec_correct += 1

    num_recs = max(1, len(rec_probs))
    return {
        "window_loss": total_loss / max(1, total_windows),
        "window_accuracy": win_correct / max(1, total_windows),
        "recording_loss": float(np.mean(rec_losses)) if rec_losses else 0.0,
        "recording_accuracy": rec_correct / num_recs,
    }


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    dropout: float = DEFAULT_DROPOUT,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> dict[str, Any]:
    """Ejecuta el entrenamiento completo de BirdCNN2D y guarda artefactos."""
    ensure_directories()
    set_seed(RANDOM_SEED)

    if not SPEC_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No existe {SPEC_INDEX_PATH}. Ejecuta primero: python -m src.data.extract_spectrograms"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)

    index_df = pd.read_csv(SPEC_INDEX_PATH, dtype={"id": str})
    classes = sorted(index_df["especie"].unique().tolist())
    species_to_idx = {sp: idx for idx, sp in enumerate(classes)}
    idx_to_species = {str(idx): sp for idx, sp in enumerate(classes)}

    class_mapping = {
        "num_classes": len(classes),
        "classes": classes,
        "species_to_idx": species_to_idx,
        "idx_to_species": idx_to_species,
    }
    with CLASS_MAPPING_PATH.open("w", encoding="utf-8") as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)

    train_df = index_df[index_df["split"] == "train"].reset_index(drop=True)
    val_df = index_df[index_df["split"] == "val"].reset_index(drop=True)

    train_ds = BirdSpectrogramDataset(train_df, species_to_idx, augment=True)
    val_ds = BirdSpectrogramDataset(val_df, species_to_idx, augment=False)

    sampler = build_weighted_sampler(train_df, species_to_idx)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = BirdCNN2D(num_classes=len(classes), dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    audio_config = {
        "sample_rate": SAMPLE_RATE,
        "clip_duration_sec": CLIP_DURATION_SEC,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "fmin": FMIN,
        "fmax": FMAX,
        "spec_frames": SPEC_FRAMES,
    }
    model_config = {
        **model.get_architecture_summary(),
        "optimizer": "AdamW",
        "initial_lr": lr,
        "weight_decay": weight_decay,
        "label_smoothing": 0.05,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "early_stopping_patience": patience,
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "audio_config": audio_config,
    }
    with MODEL_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    history_epochs: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    start_time = time.perf_counter()

    print(
        f"Iniciando entrenamiento en {device} | Clases: {len(classes)} | "
        f"Parámetros: {model.count_parameters():,} | "
        f"Train ventanas: {len(train_ds)} | Val ventanas: {len(val_ds)}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train = 0

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bs = inputs.size(0)
            running_loss += float(loss.item()) * bs
            preds = torch.argmax(logits, dim=-1)
            running_correct += int((preds == targets).sum().item())
            total_train += bs

        train_loss = running_loss / max(1, total_train)
        train_acc = running_correct / max(1, total_train)

        val_metrics = evaluate_epoch(model, val_loader, eval_criterion, device)
        val_rec_loss = val_metrics["recording_loss"]
        val_rec_acc = val_metrics["recording_accuracy"]
        val_win_loss = val_metrics["window_loss"]
        val_win_acc = val_metrics["window_accuracy"]

        scheduler.step(val_rec_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        ep_time = time.perf_counter() - ep_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(train_acc, 5),
            "val_loss": round(val_rec_loss, 5),
            "val_accuracy": round(val_rec_acc, 5),
            "val_window_loss": round(val_win_loss, 5),
            "val_window_accuracy": round(val_win_acc, 5),
            "learning_rate": current_lr,
            "duration_sec": round(ep_time, 2),
        }
        history_epochs.append(epoch_record)

        improved_checkpoint = (val_rec_acc > best_val_acc) or (
            abs(val_rec_acc - best_val_acc) < 1e-6 and val_rec_loss < best_val_loss
        )
        improved_loss = val_rec_loss < (best_val_loss - 1e-4)

        if improved_checkpoint:
            best_val_acc = val_rec_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_mapping": class_mapping,
                    "audio_config": audio_config,
                    "model_config": model_config,
                    "best_epoch": best_epoch,
                    "best_val_acc": round(best_val_acc, 5),
                    "best_val_loss": round(val_rec_loss, 5),
                },
                BEST_MODEL_PATH,
            )

        if improved_loss or improved_checkpoint:
            if val_rec_loss < best_val_loss:
                best_val_loss = val_rec_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        marker = " *BEST*" if improved_checkpoint else ""
        print(
            f"Época {epoch:02d}/{epochs:02d} ({ep_time:.1f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_rec_loss:.4f} val_acc={val_rec_acc:.4f} | "
            f"lr={current_lr:.2e}{marker}",
            flush=True,
        )

        # Guardar historial parcial tras cada época para monitoreo transparente
        partial_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_training_time_sec": round(time.perf_counter() - start_time, 2),
            "epochs_completed": len(history_epochs),
            "best_epoch": best_epoch,
            "best_val_accuracy": round(best_val_acc, 5),
            "best_val_loss": round(best_val_loss, 5),
            "model_checkpoint": to_project_relative_posix(BEST_MODEL_PATH),
            "history": history_epochs,
        }
        with TRAINING_HISTORY_PATH.open("w", encoding="utf-8") as f:
            json.dump(partial_payload, f, ensure_ascii=False, indent=2)

        if epochs_no_improve >= patience:
            print(
                f"Early stopping activado en época {epoch} "
                f"(sin mejora durante {patience} épocas; mejor época: {best_epoch}).",
                flush=True,
            )
            break

    total_time = time.perf_counter() - start_time
    history_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_training_time_sec": round(total_time, 2),
        "epochs_completed": len(history_epochs),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 5),
        "best_val_loss": round(best_val_loss, 5),
        "model_checkpoint": to_project_relative_posix(BEST_MODEL_PATH),
        "history": history_epochs,
    }
    with TRAINING_HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, ensure_ascii=False, indent=2)

    print(
        f"Entrenamiento finalizado en {total_time:.1f}s. "
        f"Mejor checkpoint en época {best_epoch} (val_acc={best_val_acc:.4f}).",
        flush=True,
    )
    return history_payload


def main() -> None:
    train_model()
    from src.model.evaluate import evaluate_model

    evaluate_model()


if __name__ == "__main__":
    main()
