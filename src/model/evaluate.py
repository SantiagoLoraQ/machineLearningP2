"""Evaluación independiente del mejor checkpoint en el conjunto de prueba (test).

Genera métricas globales, reporte por especie, matriz de confusión 39x39,
pares más confundidos y análisis de errores por calidad (A/B/C) y tipo de vocalización.

Uso:
    python -m src.model.evaluate
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.config import (
    BEST_MODEL_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    QUALITY_ANALYSIS_PATH,
    SPEC_INDEX_PATH,
    SPLITS_CSV_PATH,
    TEST_METRICS_PATH,
    ensure_directories,
    to_project_relative_posix,
)
from src.model.cnn2d import BirdCNN2D
from src.model.dataset import BirdSpectrogramDataset


def _topk_accuracy(probs_matrix: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Calcula Top-k accuracy a partir de matriz de probabilidades (N, C)."""
    k_eff = min(k, probs_matrix.shape[1])
    top_k_preds = np.argsort(probs_matrix, axis=1)[:, -k_eff:]
    hits = [int(true_cls in preds) for true_cls, preds in zip(y_true, top_k_preds)]
    return float(np.mean(hits)) if hits else 0.0


def evaluate_model() -> dict[str, Any]:
    """Evalúa el mejor checkpoint sobre el split test y guarda todos los JSON."""
    ensure_directories()
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe {BEST_MODEL_PATH}. Ejecuta primero: python -m src.model.train"
        )
    if not SPEC_INDEX_PATH.exists():
        raise FileNotFoundError(f"No existe {SPEC_INDEX_PATH}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    class_mapping = checkpoint["class_mapping"]
    model_cfg = checkpoint["model_config"]
    classes: list[str] = class_mapping["classes"]
    species_to_idx: dict[str, int] = class_mapping["species_to_idx"]

    model = BirdCNN2D(
        num_classes=len(classes),
        dropout=float(model_cfg.get("dropout", 0.40)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    index_df = pd.read_csv(SPEC_INDEX_PATH, dtype={"id": str})
    test_df = index_df[index_df["split"] == "test"].reset_index(drop=True)
    test_ds = BirdSpectrogramDataset(test_df, species_to_idx, augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    rec_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    rec_meta: dict[str, dict[str, Any]] = {}

    win_correct = 0
    win_total = 0

    row_cursor = 0
    with torch.no_grad():
        for inputs, targets, rec_ids in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            targets_np = targets.numpy()
            preds_np = np.argmax(probs, axis=1)

            win_correct += int(np.sum(preds_np == targets_np))
            win_total += len(targets_np)

            for prob_vec, tgt, rec_id in zip(probs, targets_np, rec_ids):
                rid = str(rec_id)
                rec_probs[rid].append(prob_vec)
                if rid not in rec_meta:
                    meta_row = test_df.iloc[row_cursor]
                    rec_meta[rid] = {
                        "id": rid,
                        "true_idx": int(tgt),
                        "true_species": str(meta_row["especie"]),
                        "calidad": str(meta_row["calidad"]),
                        "tipo_vocalizacion_principal": str(
                            meta_row["tipo_vocalizacion_principal"]
                        ),
                    }
                row_cursor += 1

    ordered_ids = sorted(rec_probs.keys())
    y_true: list[int] = []
    y_pred: list[int] = []
    mean_probs_list: list[np.ndarray] = []
    rec_losses: list[float] = []
    test_predictions: list[dict[str, Any]] = []

    for rid in ordered_ids:
        mean_prob = np.mean(rec_probs[rid], axis=0)
        mean_prob = np.clip(mean_prob, 1e-7, 1.0)
        pred_idx = int(np.argmax(mean_prob))
        true_idx = rec_meta[rid]["true_idx"]

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        mean_probs_list.append(mean_prob)
        rec_losses.append(float(-np.log(mean_prob[true_idx])))

        test_predictions.append(
            {
                "id": rid,
                "true_species": rec_meta[rid]["true_species"],
                "predicted_species": classes[pred_idx],
                "confidence": round(float(mean_prob[pred_idx]), 4),
                "correct": bool(pred_idx == true_idx),
                "calidad": rec_meta[rid]["calidad"],
                "tipo_vocalizacion_principal": rec_meta[rid][
                    "tipo_vocalizacion_principal"
                ],
                "windows_used": len(rec_probs[rid]),
            }
        )

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    probs_arr = np.vstack(mean_probs_list)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    top3_acc = _topk_accuracy(probs_arr, y_true_arr, k=3)
    top5_acc = _topk_accuracy(probs_arr, y_true_arr, k=5)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )

    splits_df = pd.read_csv(SPLITS_CSV_PATH, dtype={"id": str}) if SPLITS_CSV_PATH.exists() else None
    split_counts = (
        splits_df["split"].value_counts().to_dict() if splits_df is not None else {}
    )

    test_metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_loss": round(float(np.mean(rec_losses)), 5),
        "accuracy": round(acc, 5),
        "window_level_accuracy": round(win_correct / max(1, win_total), 5),
        "top3_accuracy": round(top3_acc, 5),
        "top5_accuracy": round(top5_acc, 5),
        "precision_macro": round(float(p_macro), 5),
        "recall_macro": round(float(r_macro), 5),
        "f1_macro": round(float(f1_macro), 5),
        "precision_weighted": round(float(p_weighted), 5),
        "recall_weighted": round(float(r_weighted), 5),
        "f1_weighted": round(float(f1_weighted), 5),
        "num_test_recordings": len(ordered_ids),
        "num_test_windows": len(test_ds),
        "num_classes": len(classes),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "model_params": int(model.count_parameters()),
        "device_used": str(device),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
    }
    with TEST_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # Reporte por clase
    cls_report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    per_class_report = [
        {
            "especie": sp,
            "precision": round(float(cls_report_dict[sp]["precision"]), 4),
            "recall": round(float(cls_report_dict[sp]["recall"]), 4),
            "f1_score": round(float(cls_report_dict[sp]["f1-score"]), 4),
            "support": int(cls_report_dict[sp]["support"]),
        }
        for sp in classes
    ]
    with CLASSIFICATION_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(per_class_report, f, ensure_ascii=False, indent=2)

    # Matriz de confusión 39x39 y top confusiones
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(classes))))
    confusions: list[dict[str, Any]] = []
    for i, true_sp in enumerate(classes):
        for j, pred_sp in enumerate(classes):
            if i != j and cm[i, j] > 0:
                confusions.append(
                    {
                        "true_species": true_sp,
                        "predicted_species": pred_sp,
                        "count": int(cm[i, j]),
                    }
                )
    confusions.sort(key=lambda item: item["count"], reverse=True)

    cm_payload = {
        "labels": classes,
        "matrix": cm.tolist(),
        "top_confusions": confusions[:15],
    }
    with CONFUSION_MATRIX_PATH.open("w", encoding="utf-8") as f:
        json.dump(cm_payload, f, ensure_ascii=False, indent=2)

    # Análisis por calidad (A/B/C) y tipo de vocalización
    pred_df = pd.DataFrame(test_predictions)
    by_quality: list[dict[str, Any]] = []
    for qual in ("A", "B", "C"):
        sub = pred_df[pred_df["calidad"] == qual]
        if len(sub) > 0:
            _, _, q_f1, _ = precision_recall_fscore_support(
                sub["true_species"],
                sub["predicted_species"],
                average="macro",
                zero_division=0,
            )
            by_quality.append(
                {
                    "calidad": qual,
                    "count": int(len(sub)),
                    "correct": int(sub["correct"].sum()),
                    "accuracy": round(float(sub["correct"].mean()), 4),
                    "f1_macro": round(float(q_f1), 4),
                    "avg_confidence": round(float(sub["confidence"].mean()), 4),
                }
            )
        else:
            by_quality.append(
                {
                    "calidad": qual,
                    "count": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                    "f1_macro": 0.0,
                    "avg_confidence": 0.0,
                }
            )

    by_vocalization: list[dict[str, Any]] = []
    for voc, sub in pred_df.groupby("tipo_vocalizacion_principal"):
        by_vocalization.append(
            {
                "tipo_vocalizacion": str(voc),
                "count": int(len(sub)),
                "correct": int(sub["correct"].sum()),
                "accuracy": round(float(sub["correct"].mean()), 4),
                "avg_confidence": round(float(sub["confidence"].mean()), 4),
            }
        )
    by_vocalization.sort(key=lambda item: item["count"], reverse=True)

    quality_payload = {
        "by_quality": by_quality,
        "by_vocalization": by_vocalization,
        "test_predictions": test_predictions,
    }
    with QUALITY_ANALYSIS_PATH.open("w", encoding="utf-8") as f:
        json.dump(quality_payload, f, ensure_ascii=False, indent=2)

    print("=== Evaluación en Conjunto de Prueba Independiente (Test) ===")
    print(f"Grabaciones evaluadas: {len(ordered_ids)} ({len(test_ds)} ventanas)")
    print(
        f"Test Loss: {test_metrics['test_loss']:.4f} | "
        f"Accuracy: {test_metrics['accuracy']*100:.2f}% | "
        f"Top-3: {test_metrics['top3_accuracy']*100:.2f}% | "
        f"Top-5: {test_metrics['top5_accuracy']*100:.2f}%"
    )
    print(
        f"F1 Macro: {test_metrics['f1_macro']:.4f} | "
        f"Precision Macro: {test_metrics['precision_macro']:.4f} | "
        f"Recall Macro: {test_metrics['recall_macro']:.4f}"
    )
    print(f"Métricas guardadas en: {to_project_relative_posix(TEST_METRICS_PATH)}")
    return test_metrics


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()
