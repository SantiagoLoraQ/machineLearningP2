"""API local FastAPI para exploración de dataset, inferencia CNN 2D y métricas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.audio_inference import BirdAudioInferenceEngine
from src.config import (
    AUDIOS_DIR,
    CLASSIFICATION_REPORT_PATH,
    CLEAN_CSV_PATH,
    CONFUSION_MATRIX_PATH,
    DATASET_STATS_PATH,
    MODEL_CONFIG_PATH,
    PROJECT_ROOT,
    QUALITY_ANALYSIS_PATH,
    SPLITS_CSV_PATH,
    TEST_METRICS_PATH,
    TRAINING_HISTORY_PATH,
)
from src.unsupervised import UNSUPERVISED_STATUS

app = FastAPI(
    title="API Bioacústica - Aves de Bolivia (CNN 2D)",
    description=(
        "Backend local de inferencia y análisis para la Primera Entrega de "
        "Reconocimiento de Voz (vocalizaciones de 39 especies de aves de Bolivia)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = BirdAudioInferenceEngine()


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_samples_df() -> pd.DataFrame:
    if SPLITS_CSV_PATH.exists():
        return pd.read_csv(SPLITS_CSV_PATH, dtype={"id": str})
    if CLEAN_CSV_PATH.exists():
        df = pd.read_csv(CLEAN_CSV_PATH, dtype={"id": str})
        df["split"] = "unassigned"
        return df
    raise HTTPException(
        status_code=503,
        detail="Dataset limpio no encontrado. Ejecuta primero python -m src.data.clean_csv.",
    )


def _resolve_safe_audio_path(rel_path: str) -> Path:
    candidate = (PROJECT_ROOT / Path(rel_path)).resolve()
    audios_root = AUDIOS_DIR.resolve()
    if audios_root not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Archivo de audio no encontrado.")
    return candidate


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """Verifica el estado del servicio y la disponibilidad del modelo entrenado."""
    if not engine.is_ready:
        engine.load()
    return {
        "status": "ok",
        "model_loaded": engine.is_ready,
        "num_classes": len(engine.classes),
        "device": str(engine.device),
        "best_epoch": engine.best_epoch,
        "best_val_acc": engine.best_val_acc,
        "dataset_ready": CLEAN_CSV_PATH.exists(),
    }


@app.get("/api/dataset/summary")
def get_dataset_summary() -> dict[str, Any]:
    """Devuelve estadísticas completas y auditoría del dataset limpio."""
    stats = _read_json(DATASET_STATS_PATH)
    if stats is None:
        raise HTTPException(
            status_code=404,
            detail="Estadísticas del dataset no disponibles. Ejecuta python -m src.data.clean_csv.",
        )
    return stats


@app.get("/api/dataset/samples")
def get_dataset_samples(
    species: str | None = Query(default=None),
    quality: str | None = Query(default=None),
    split: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=60, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Lista muestras del dataset depurado con filtros para inspección e inferencia."""
    df = _load_samples_df()
    all_species = sorted(df["especie"].unique().tolist())

    if species:
        df = df[df["especie"] == species]
    if quality:
        df = df[df["calidad"].str.upper() == quality.upper()]
    if split:
        df = df[df["split"] == split]
    if search:
        q = search.strip().lower()
        df = df[
            df["especie"].str.lower().str.contains(q, na=False)
            | df["id"].str.lower().str.contains(q, na=False)
            | df["tipo_vocalizacion"].str.lower().str.contains(q, na=False)
        ]

    total_filtered = int(len(df))
    page = df.iloc[offset : offset + limit]
    items = page[
        [
            "id",
            "especie",
            "calidad",
            "tipo_vocalizacion",
            "tipo_vocalizacion_principal",
            "duracion_seg",
            "sample_rate",
            "split",
            "ruta_relativa",
        ]
    ].to_dict(orient="records")

    return {
        "total": total_filtered,
        "limit": limit,
        "offset": offset,
        "species_list": all_species,
        "items": items,
    }


@app.get("/api/audio/{sample_id}")
def stream_dataset_audio(sample_id: str) -> FileResponse:
    """Sirve de forma segura un archivo de audio del dataset para reproducción web."""
    df = _load_samples_df()
    match = df[df["id"] == str(sample_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Muestra {sample_id} no encontrada.")

    row = match.iloc[0]
    audio_path = _resolve_safe_audio_path(str(row["ruta_relativa"]))
    media_type = "audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/mpeg"
    return FileResponse(path=audio_path, media_type=media_type, filename=audio_path.name)


@app.post("/api/predict")
async def predict_uploaded_audio(
    file: UploadFile = File(...),
    theme: str = Query(default="light"),
) -> dict[str, Any]:
    """Clasifica un archivo de audio (.wav/.mp3/.flac/.ogg) subido por el usuario."""
    if not engine.is_ready:
        engine.load()
    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="El modelo CNN 2D aún no está entrenado. Ejecuta python -m src.model.train.",
        )

    filename = file.filename or "audio_subido.wav"
    ext = Path(filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".ogg"}:
        raise HTTPException(
            status_code=400,
            detail="Formato no soportado. Usa archivos .wav, .mp3, .flac u .ogg.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío.")

    try:
        result = engine.predict_from_bytes(
            raw_bytes, filename=filename, top_k=5, theme=theme
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo decodificar o procesar el audio: {exc}",
        ) from exc

    result["mode"] = "upload"
    return result


@app.post("/api/predict/sample/{sample_id}")
def predict_dataset_sample(
    sample_id: str,
    theme: str = Query(default="light"),
) -> dict[str, Any]:
    """Ejecuta inferencia sobre una muestra del dataset y compara con su etiqueta real."""
    if not engine.is_ready:
        engine.load()
    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="El modelo CNN 2D aún no está entrenado.",
        )

    df = _load_samples_df()
    match = df[df["id"] == str(sample_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Muestra {sample_id} no encontrada.")

    row = match.iloc[0]
    audio_path = _resolve_safe_audio_path(str(row["ruta_relativa"]))

    try:
        result = engine.predict_from_path(audio_path, top_k=5, theme=theme)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Error al procesar la muestra {sample_id}: {exc}",
        ) from exc

    true_species = str(row["especie"])
    result["mode"] = "dataset_sample"
    result["sample_id"] = str(row["id"])
    result["true_species"] = true_species
    result["calidad"] = str(row["calidad"])
    result["tipo_vocalizacion"] = str(row["tipo_vocalizacion"])
    result["split"] = str(row["split"])
    result["is_correct"] = result["predicted_species"] == true_species
    return result


@app.get("/api/metrics/overview")
def get_metrics_overview() -> dict[str, Any]:
    """Retorna curvas de entrenamiento, métricas de prueba, matriz de confusión y análisis."""
    test_metrics = _read_json(TEST_METRICS_PATH)
    training_history = _read_json(TRAINING_HISTORY_PATH)
    cls_report = _read_json(CLASSIFICATION_REPORT_PATH)
    confusion = _read_json(CONFUSION_MATRIX_PATH)
    quality_analysis = _read_json(QUALITY_ANALYSIS_PATH)
    model_config = _read_json(MODEL_CONFIG_PATH)

    if test_metrics is None or training_history is None:
        raise HTTPException(
            status_code=404,
            detail="Métricas del modelo no disponibles. Ejecuta python -m src.model.train.",
        )

    return {
        "model_config": model_config,
        "training_history": training_history,
        "test_metrics": test_metrics,
        "classification_report": cls_report,
        "confusion_matrix": confusion,
        "quality_error_analysis": quality_analysis,
    }


@app.get("/api/unsupervised/status")
def get_unsupervised_status() -> dict[str, Any]:
    """Devuelve el estado reservado del módulo No Supervisado (Siguiente Entrega)."""
    return UNSUPERVISED_STATUS
