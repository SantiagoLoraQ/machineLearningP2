"""Limpieza automatizada y reproducible de manifest_verificado.csv.

Genera data/processed/dataset_limpio.csv con rutas relativas portables,
valida la existencia y legibilidad real de cada audio en Audios/ y guarda
reportes de auditoría y estadísticas descriptivas del dataset.

Uso:
    python -m src.data.clean_csv
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from src.config import (
    AUDIT_JSON_PATH,
    AUDIOS_DIR,
    CLEAN_CSV_PATH,
    DATASET_STATS_PATH,
    MIN_VALID_AUDIO_SEC,
    PROJECT_ROOT,
    RAW_CSV_PATH,
    ensure_directories,
    to_project_relative_posix,
)

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")
KEEP_COLUMNS = [
    "id",
    "especie",
    "especie_carpeta",
    "ruta_relativa",
    "calidad",
    "tipo_vocalizacion",
    "tipo_vocalizacion_principal",
    "duracion_seg",
    "sample_rate",
    "canales",
]


def normalize_species_name(raw_name: str) -> str:
    """Limpia espacios redundantes en el nombre científico de la especie."""
    return " ".join((raw_name or "").strip().split())


def extract_primary_vocalization(raw_voc: str) -> str:
    """Extrae una categoría principal determinística de tipo_vocalizacion."""
    cleaned = " ".join((raw_voc or "").strip().lower().split())
    if not cleaned:
        return "unknown"
    tokens = [t.strip() for t in cleaned.split(",") if t.strip()]
    if any(re.search(r"\bsongs?\b", t) for t in tokens):
        return "song"
    if any("call" in t for t in tokens):
        return "call"
    if any("duet" in t for t in tokens):
        return "duet"
    if any("subsong" in t for t in tokens):
        return "subsong"
    return tokens[0] if tokens else "unknown"


def resolve_audio_path(
    record_id: str, species_folder: str, raw_ruta_local: str
) -> Path | None:
    """Resuelve la ruta local del audio bajo Audios/<Especie>/ de forma portable.

    Soporta re-ejecuciones si el usuario limpió o convirtió un archivo (por ejemplo
    de .mp3 a .wav manteniendo el identificador XC<id> dentro de la carpeta).
    """
    species_dir = AUDIOS_DIR / species_folder
    if not species_dir.exists() or not species_dir.is_dir():
        return None

    filename = Path(raw_ruta_local.replace("\\", "/")).name if raw_ruta_local else ""
    if filename:
        candidate = species_dir / filename
        if candidate.is_file():
            return candidate

    # Búsqueda de respaldo por prefijo XC<id> tras limpieza manual de audios
    stem_prefix = f"XC{record_id}"
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        alt = species_dir / f"{stem_prefix}{ext}"
        if alt.is_file():
            return alt

    for path in sorted(species_dir.iterdir()):
        if path.is_file() and path.stem.upper() == stem_prefix.upper():
            if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                return path
    return None


def parse_manifest_duration(raw_dur: str) -> float | None:
    """Convierte cadenas mm:ss o hh:mm:ss del manifiesto a segundos."""
    text = (raw_dur or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60.0 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
    if len(nums) == 1:
        return nums[0]
    return None


def build_dataset_stats(df: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    """Calcula estadísticas detalladas del dataset depurado para métricas y UI."""
    species_counts = (
        df.groupby("especie")
        .agg(
            total=("id", "count"),
            quality_a=("calidad", lambda s: int((s == "A").sum())),
            quality_b=("calidad", lambda s: int((s == "B").sum())),
            quality_c=("calidad", lambda s: int((s == "C").sum())),
            avg_duration_sec=("duracion_seg", lambda s: round(float(s.mean()), 2)),
            median_duration_sec=("duracion_seg", lambda s: round(float(s.median()), 2)),
        )
        .reset_index()
        .sort_values(by=["total", "especie"], ascending=[False, True])
    )

    quality_counts = df["calidad"].value_counts().to_dict()
    voc_counts = df["tipo_vocalizacion_principal"].value_counts().to_dict()

    durations = df["duracion_seg"].to_numpy(dtype=float)
    bins = [0, 10, 20, 30, 45, 60, 90, 120, 240, 1000]
    bin_labels = [
        "0-10s",
        "10-20s",
        "20-30s",
        "30-45s",
        "45-60s",
        "60-90s",
        "90-120s",
        "120-240s",
        ">240s",
    ]
    hist_counts, _ = np.histogram(durations, bins=bins)
    duration_histogram = [
        {"range": label, "count": int(cnt)}
        for label, cnt in zip(bin_labels, hist_counts)
    ]

    formats = (
        df["ruta_relativa"]
        .apply(lambda p: Path(p).suffix.lower().lstrip("."))
        .value_counts()
        .to_dict()
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_clean_records": int(len(df)),
        "num_species": int(df["especie"].nunique()),
        "quality_distribution": {k: int(quality_counts.get(k, 0)) for k in ("A", "B", "C")},
        "vocalization_distribution": {str(k): int(v) for k, v in voc_counts.items()},
        "audio_format_distribution": {str(k): int(v) for k, v in formats.items()},
        "duration_stats": {
            "min_sec": round(float(np.min(durations)), 2),
            "p25_sec": round(float(np.percentile(durations, 25)), 2),
            "median_sec": round(float(np.median(durations)), 2),
            "mean_sec": round(float(np.mean(durations)), 2),
            "p75_sec": round(float(np.percentile(durations, 75)), 2),
            "max_sec": round(float(np.max(durations)), 2),
            "total_hours": round(float(np.sum(durations) / 3600.0), 2),
        },
        "duration_histogram": duration_histogram,
        "species_breakdown": species_counts.to_dict(orient="records"),
        "cleaning_audit": audit,
    }


def clean_manifest(
    raw_csv_path: Path = RAW_CSV_PATH,
    clean_csv_path: Path = CLEAN_CSV_PATH,
    audit_json_path: Path = AUDIT_JSON_PATH,
    stats_json_path: Path = DATASET_STATS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limpia el manifiesto base y valida la integridad física de los audios."""
    ensure_directories()
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV base: {raw_csv_path}")

    with raw_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)

    seen_ids: set[str] = set()
    clean_records: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    excluded_details: list[dict[str, str]] = []
    duration_mismatches: list[dict[str, Any]] = []

    for row in raw_rows:
        record_id = (row.get("id") or "").strip()
        species = normalize_species_name(row.get("especie") or "")

        if not record_id or not species:
            exclusion_counts["invalid_row"] += 1
            excluded_details.append(
                {"id": record_id, "especie": species, "reason": "invalid_row"}
            )
            continue

        if species.lower() == "mystery mystery":
            exclusion_counts["mystery_mystery"] += 1
            continue

        if record_id in seen_ids:
            exclusion_counts["duplicate_id"] += 1
            excluded_details.append(
                {"id": record_id, "especie": species, "reason": "duplicate_id"}
            )
            continue
        seen_ids.add(record_id)

        species_folder = species.replace(" ", "_")
        audio_path = resolve_audio_path(
            record_id=record_id,
            species_folder=species_folder,
            raw_ruta_local=(row.get("ruta_local") or "").strip(),
        )

        if audio_path is None:
            exclusion_counts["missing_file"] += 1
            excluded_details.append(
                {
                    "id": record_id,
                    "especie": species,
                    "reason": "missing_file",
                    "expected_raw": (row.get("ruta_local") or "").strip(),
                }
            )
            continue

        try:
            info = sf.info(str(audio_path))
            real_duration = float(info.duration)
            sr = int(info.samplerate)
            channels = int(info.channels)
            if info.frames <= 0 or sr <= 0:
                raise ValueError("Archivo sin muestras válidas")
        except Exception as exc:
            exclusion_counts["corrupted_or_unreadable"] += 1
            excluded_details.append(
                {
                    "id": record_id,
                    "especie": species,
                    "reason": "corrupted_or_unreadable",
                    "file": to_project_relative_posix(audio_path),
                    "error": str(exc),
                }
            )
            continue

        if real_duration < MIN_VALID_AUDIO_SEC:
            exclusion_counts["too_short"] += 1
            excluded_details.append(
                {
                    "id": record_id,
                    "especie": species,
                    "reason": "too_short",
                    "file": to_project_relative_posix(audio_path),
                    "duration_sec": str(round(real_duration, 3)),
                }
            )
            continue

        manifest_dur = parse_manifest_duration(row.get("duracion") or "")
        if manifest_dur is not None and abs(manifest_dur - real_duration) > 5.0:
            duration_mismatches.append(
                {
                    "id": record_id,
                    "especie": species,
                    "manifest_sec": round(manifest_dur, 2),
                    "real_sec": round(real_duration, 2),
                }
            )

        quality = (row.get("calidad") or "").strip().upper()
        if quality not in {"A", "B", "C"}:
            quality = "C"

        raw_voc = " ".join((row.get("tipo_vocalizacion") or "").strip().split())
        primary_voc = extract_primary_vocalization(raw_voc)

        clean_records.append(
            {
                "id": record_id,
                "especie": species,
                "especie_carpeta": species_folder,
                "ruta_relativa": to_project_relative_posix(audio_path),
                "calidad": quality,
                "tipo_vocalizacion": raw_voc if raw_voc else "unknown",
                "tipo_vocalizacion_principal": primary_voc,
                "duracion_seg": round(real_duration, 2),
                "sample_rate": sr,
                "canales": channels,
            }
        )

    df = pd.DataFrame(clean_records, columns=KEEP_COLUMNS)
    df = df.sort_values(by=["especie", "id"]).reset_index(drop=True)
    df.to_csv(clean_csv_path, index=False, encoding="utf-8")

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": to_project_relative_posix(raw_csv_path),
        "output_csv": to_project_relative_posix(clean_csv_path),
        "raw_total_rows": len(raw_rows),
        "clean_total_rows": int(len(df)),
        "species_count": int(df["especie"].nunique()),
        "dropped_columns": [
            "pais",
            "localidad",
            "fecha",
            "grabador",
            "duracion",
            "licencia",
            "url_audio",
            "nombre_archivo_origen",
            "url_ficha",
            "lat",
            "lon",
            "ruta_local",
        ],
        "exclusion_summary": dict(exclusion_counts),
        "excluded_non_mystery_records": excluded_details,
        "duration_updated_vs_manifest_count": len(duration_mismatches),
    }

    with audit_json_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    stats = build_dataset_stats(df, audit)
    with stats_json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return df, audit


def main() -> None:
    df, audit = clean_manifest()
    print("=== Limpieza automatizada del dataset ===")
    print(f"Filas originales: {audit['raw_total_rows']}")
    print(f"Exclusiones: {audit['exclusion_summary']}")
    print(f"Filas limpias finales: {audit['clean_total_rows']} ({audit['species_count']} especies)")
    print(f"CSV generado: {to_project_relative_posix(CLEAN_CSV_PATH)}")
    print(f"Auditoría: {to_project_relative_posix(AUDIT_JSON_PATH)}")
    print(f"Estadísticas: {to_project_relative_posix(DATASET_STATS_PATH)}")
    if audit["excluded_non_mystery_records"]:
        print("Registros excluidos (no Mystery):", audit["excluded_non_mystery_records"])


if __name__ == "__main__":
    main()
