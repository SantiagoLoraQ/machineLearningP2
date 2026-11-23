"""Extracción en lote de espectrogramas de Mel en dB y partición estratificada.

Divide las grabaciones limpias de forma estratificada por especie (70% train,
15% val, 15% test) evitando fuga de datos entre ventanas de un mismo audio,
estandariza las señales a 22,050 Hz mono, selecciona hasta 3 ventanas de 4 s
con mayor energía acústica y guarda espectrogramas Mel en float16 (.npy).

Uso:
    python -m src.data.extract_spectrograms
"""

from __future__ import annotations

import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly, stft
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_CSV_PATH,
    CLIP_SAMPLES,
    FMAX,
    FMIN,
    HOP_LENGTH,
    MAX_LOAD_SECONDS,
    N_FFT,
    N_MELS,
    PROJECT_ROOT,
    RANDOM_SEED,
    SAMPLE_RATE,
    SPEC_ERRORS_PATH,
    SPEC_FRAMES,
    SPEC_INDEX_PATH,
    SPECTROGRAMS_DIR,
    SPLITS_CSV_PATH,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    WINDOW_HOP_SEC,
    WINDOWS_PER_CLIP,
    ensure_directories,
    to_project_relative_posix,
)


def _resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Remuestrea de forma determinística usando resample_poly."""
    if orig_sr == target_sr:
        return y.astype(np.float32, copy=False)
    frac = Fraction(target_sr, orig_sr).limit_denominator(256)
    resampled = resample_poly(y, frac.numerator, frac.denominator)
    return resampled.astype(np.float32, copy=False)


def _read_audio_source(
    source: Path | bytes, max_seconds: float = MAX_LOAD_SECONDS
) -> tuple[np.ndarray, int, float]:
    """Lee un archivo de audio desde disco o bytes y devuelve (mono, orig_sr, real_duration)."""
    if isinstance(source, (bytes, bytearray)):
        bio_info = io.BytesIO(source)
        info = sf.info(bio_info)
        orig_sr = int(info.samplerate)
        real_duration = float(info.duration)
        max_frames = int(orig_sr * max_seconds) if max_seconds > 0 else -1
        bio_read = io.BytesIO(source)
        y, read_sr = sf.read(bio_read, frames=max_frames, dtype="float32", always_2d=True)
    else:
        info = sf.info(str(source))
        orig_sr = int(info.samplerate)
        real_duration = float(info.duration)
        max_frames = int(orig_sr * max_seconds) if max_seconds > 0 else -1
        y, read_sr = sf.read(str(source), frames=max_frames, dtype="float32", always_2d=True)

    if y.size == 0:
        raise ValueError("Señal de audio vacía")

    mono = np.mean(y, axis=1, dtype=np.float32)
    return mono, int(read_sr), real_duration


def load_and_standardize_audio(
    source: Path | bytes,
    max_seconds: float = MAX_LOAD_SECONDS,
    target_sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Carga, convierte a mono, remuestrea a 22,050 Hz, centra y normaliza amplitud."""
    mono, orig_sr, real_duration = _read_audio_source(source, max_seconds=max_seconds)
    y = _resample_audio(mono, orig_sr=orig_sr, target_sr=target_sr)

    # Eliminar offset DC y normalizar pico
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))
    if peak > 1e-8:
        y = y / peak
    else:
        raise ValueError("Audio silencioso (amplitud pico cercana a cero)")

    meta = {
        "orig_sample_rate": orig_sr,
        "target_sample_rate": target_sr,
        "duration_sec": round(real_duration, 2),
        "analyzed_sec": round(len(y) / float(target_sr), 2),
    }
    return y.astype(np.float32, copy=False), meta


def select_energetic_windows(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    clip_samples: int = CLIP_SAMPLES,
    max_windows: int = WINDOWS_PER_CLIP,
    hop_sec: float = WINDOW_HOP_SEC,
) -> list[np.ndarray]:
    """Selecciona hasta max_windows segmentos de 4s con mayor energía vocal."""
    n = len(y)
    if n == 0:
        raise ValueError("No hay muestras para extraer ventanas")

    if n < clip_samples:
        repeats = int(math.ceil(clip_samples / n))
        padded = np.tile(y, repeats)[:clip_samples]
        return [padded.astype(np.float32, copy=False)]

    if n == clip_samples:
        return [y.astype(np.float32, copy=False)]

    hop_samples = max(1, int(hop_sec * sr))
    starts = list(range(0, n - clip_samples + 1, hop_samples))
    last_start = n - clip_samples
    if starts[-1] != last_start:
        starts.append(last_start)

    # Pre-énfasis para atenuar ruido grave (viento/manejo) al medir energía de aves
    y_pe = np.empty_like(y)
    y_pe[0] = y[0]
    y_pe[1:] = y[1:] - 0.95 * y[:-1]

    scored: list[tuple[float, int]] = []
    for st in starts:
        seg = y_pe[st : st + clip_samples]
        rms = float(np.sqrt(np.mean(seg * seg)))
        scored.append((rms, st))

    scored.sort(key=lambda item: item[0], reverse=True)

    min_sep = clip_samples // 2  # máximo 50% de solapamiento entre ventanas seleccionadas
    chosen_starts: list[int] = []
    for _, st in scored:
        if all(abs(st - prev) >= min_sep for prev in chosen_starts):
            chosen_starts.append(st)
            if len(chosen_starts) >= max_windows:
                break

    chosen_starts.sort()
    return [y[st : st + clip_samples].astype(np.float32, copy=False) for st in chosen_starts]


@lru_cache(maxsize=8)
def build_mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: int = FMIN,
    fmax: int = FMAX,
) -> np.ndarray:
    """Construye el banco de filtros triangulares Mel con normalización Slaney."""
    mel_min = 2595.0 * np.log10(1.0 + float(fmin) / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + float(fmax) / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    fft_freqs = np.linspace(0.0, sr / 2.0, (n_fft // 2) + 1)

    fb = np.zeros((n_mels, (n_fft // 2) + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up_slope = (fft_freqs - left) / max(center - left, 1e-9)
        down_slope = (right - fft_freqs) / max(right - center, 1e-9)
        tri = np.maximum(0.0, np.minimum(up_slope, down_slope))
        enorm = 2.0 / max(right - left, 1e-9)
        fb[i] = (tri * enorm).astype(np.float32)
    return fb


def compute_mel_spectrogram_db(
    y_win: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmin: int = FMIN,
    fmax: int = FMAX,
    target_frames: int = SPEC_FRAMES,
) -> np.ndarray:
    """Calcula el espectrograma de Mel en dB con forma fija (64, 173) vía SciPy STFT."""
    _, _, zxx = stft(
        y_win,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    power_spec = (np.abs(zxx) ** 2).astype(np.float32, copy=False)
    mel_fb = build_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_power = mel_fb @ power_spec

    ref_power = max(float(np.max(mel_power)), 1e-10)
    mel_db = 10.0 * np.log10(np.maximum(mel_power, 1e-10) / ref_power)
    mel_db = np.clip(mel_db, -80.0, 0.0)

    if mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), constant_values=-80.0)
    elif mel_db.shape[1] > target_frames:
        mel_db = mel_db[:, :target_frames]

    if not np.all(np.isfinite(mel_db)):
        raise ValueError("Espectrograma contiene valores no finitos (NaN/Inf)")
    return mel_db.astype(np.float32, copy=False)


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Divide grabaciones de forma estratificada en train/val/test sin fuga de datos."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    train_idx, temp_idx = train_test_split(
        df.index.to_numpy(),
        test_size=(val_ratio + test_ratio),
        stratify=df["especie"],
        random_state=random_state,
    )
    temp_df = df.loc[temp_idx]
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_df.index.to_numpy(),
        test_size=relative_test,
        stratify=temp_df["especie"],
        random_state=random_state,
    )

    out = df.copy()
    out["split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    return out


def extract_all_spectrograms(
    clean_csv_path: Path = CLEAN_CSV_PATH,
    splits_csv_path: Path = SPLITS_CSV_PATH,
    spec_dir: Path = SPECTROGRAMS_DIR,
    spec_index_path: Path = SPEC_INDEX_PATH,
    errors_path: Path = SPEC_ERRORS_PATH,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Procesa en lote las grabaciones limpias y genera el índice de espectrogramas."""
    ensure_directories()
    if not clean_csv_path.exists():
        raise FileNotFoundError(
            f"No existe {clean_csv_path}. Ejecuta primero: python -m src.data.clean_csv"
        )

    df = pd.read_csv(clean_csv_path, dtype={"id": str})
    df_splits = create_stratified_splits(df)
    df_splits.to_csv(splits_csv_path, index=False, encoding="utf-8")

    # Limpiar archivos .npy huérfanos previos para mantener sincronía exacta
    for old_npy in spec_dir.glob("*.npy"):
        old_npy.unlink()

    index_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    split_win_counts: Counter[str] = Counter()

    total = len(df_splits)
    for idx, row in df_splits.iterrows():
        rec_id = str(row["id"])
        audio_path = PROJECT_ROOT / Path(row["ruta_relativa"])
        try:
            y, _ = load_and_standardize_audio(audio_path)
            windows = select_energetic_windows(y)
            for w_idx, win in enumerate(windows):
                spec_db = compute_mel_spectrogram_db(win)
                npy_path = spec_dir / f"{rec_id}_w{w_idx}.npy"
                np.save(npy_path, spec_db.astype(np.float16))
                index_rows.append(
                    {
                        "id": rec_id,
                        "window_idx": w_idx,
                        "especie": row["especie"],
                        "calidad": row["calidad"],
                        "tipo_vocalizacion_principal": row["tipo_vocalizacion_principal"],
                        "split": row["split"],
                        "spec_rel_path": to_project_relative_posix(npy_path),
                    }
                )
                split_win_counts[row["split"]] += 1
        except Exception as exc:
            errors.append(
                {
                    "id": rec_id,
                    "especie": row["especie"],
                    "ruta_relativa": row["ruta_relativa"],
                    "error": str(exc),
                }
            )

        if (idx + 1) % 200 == 0 or (idx + 1) == total:
            print(f"Procesados {idx + 1}/{total} audios -> {len(index_rows)} espectrogramas...")

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(spec_index_path, index=False, encoding="utf-8")

    error_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_recordings": int(total),
        "successful_recordings": int(index_df["id"].nunique()) if not index_df.empty else 0,
        "total_spectrogram_windows": int(len(index_df)),
        "windows_by_split": dict(split_win_counts),
        "error_count": len(errors),
        "errors": errors,
    }
    with errors_path.open("w", encoding="utf-8") as f:
        json.dump(error_report, f, ensure_ascii=False, indent=2)

    return index_df, errors


def main() -> None:
    index_df, errors = extract_all_spectrograms()
    print("=== Extracción en lote de espectrogramas de Mel ===")
    print(f"Particiones guardadas en: {to_project_relative_posix(SPLITS_CSV_PATH)}")
    print(f"Ventanas extraídas: {len(index_df)} ({index_df['id'].nunique()} grabaciones)")
    print("Ventanas por split:", index_df["split"].value_counts().to_dict())
    print(f"Errores registrados: {len(errors)}")
    print(f"Índice: {to_project_relative_posix(SPEC_INDEX_PATH)}")


if __name__ == "__main__":
    main()
