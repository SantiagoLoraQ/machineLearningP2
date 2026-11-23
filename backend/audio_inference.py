"""Motor de inferencia en PyTorch y renderizado de espectrogramas de Mel."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    BEST_MODEL_PATH,
    CLASS_MAPPING_PATH,
    CLIP_DURATION_SEC,
    FMAX,
    FMIN,
    SAMPLE_RATE,
)
from src.data.extract_spectrograms import (
    compute_mel_spectrogram_db,
    load_and_standardize_audio,
    select_energetic_windows,
)
from src.model.cnn2d import BirdCNN2D
from src.model.dataset import normalize_spectrogram


def build_waveform_preview(y: np.ndarray, num_points: int = 120) -> list[float]:
    """Genera una envolvente firmada de 120 puntos en [-1, 1] para visualizar en React."""
    if len(y) == 0:
        return [0.0] * num_points
    chunks = np.array_split(y, num_points)
    preview: list[float] = []
    for chunk in chunks:
        if chunk.size == 0:
            preview.append(0.0)
        else:
            idx = int(np.argmax(np.abs(chunk)))
            preview.append(round(float(chunk[idx]), 4))
    return preview


def render_spectrogram_base64(
    spec_db: np.ndarray,
    title: str = "Espectrograma de Mel (dB)",
    theme: str = "light",
) -> str:
    """Renderiza un espectrograma de Mel (64, 173) como imagen PNG codificada en Base64."""
    is_dark = str(theme).lower() == "dark"
    bg_color = "#1a231d" if is_dark else "#ffffff"
    text_color = "#e8efe9" if is_dark else "#1c2920"
    muted_color = "#a9bcb0" if is_dark else "#4b5e50"
    border_color = "#2e4034" if is_dark else "#c7d6cb"
    cmap_name = "viridis" if is_dark else "YlGn"

    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=140)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    img = ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap=cmap_name,
        extent=[0.0, CLIP_DURATION_SEC, FMIN / 1000.0, FMAX / 1000.0],
        vmin=-80.0,
        vmax=0.0,
    )
    ax.set_title(title, color=text_color, fontsize=10, pad=8, fontweight="bold")
    ax.set_xlabel("Tiempo en ventana analizada (s)", color=muted_color, fontsize=8.5)
    ax.set_ylabel("Frecuencia (kHz)", color=muted_color, fontsize=8.5)
    ax.tick_params(colors=muted_color, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.ax.tick_params(colors=muted_color, labelsize=7.5)
    cbar.outline.set_edgecolor(border_color)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


class BirdAudioInferenceEngine:
    """Carga el modelo entrenado y ejecuta inferencia reproducible sobre audio."""

    def __init__(
        self,
        model_path: Path = BEST_MODEL_PATH,
        class_mapping_path: Path = CLASS_MAPPING_PATH,
    ) -> None:
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: BirdCNN2D | None = None
        self.classes: list[str] = []
        self.model_config: dict[str, Any] = {}
        self.best_epoch: int | None = None
        self.best_val_acc: float | None = None
        self.load()

    def load(self) -> None:
        """Carga pesos y mapeo de clases si el checkpoint ya existe en disco."""
        if not self.model_path.exists():
            self.model = None
            return

        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        class_mapping = checkpoint.get("class_mapping")
        if class_mapping is None and self.class_mapping_path.exists():
            with self.class_mapping_path.open("r", encoding="utf-8") as f:
                class_mapping = json.load(f)

        if class_mapping is None:
            raise RuntimeError("No se encontró el mapeo de clases del modelo.")

        self.classes = list(class_mapping["classes"])
        self.model_config = dict(checkpoint.get("model_config", {}))
        self.best_epoch = checkpoint.get("best_epoch")
        self.best_val_acc = checkpoint.get("best_val_acc")

        dropout = float(self.model_config.get("dropout", 0.40))
        model = BirdCNN2D(num_classes=len(self.classes), dropout=dropout).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model

    @property
    def is_ready(self) -> bool:
        return self.model is not None and len(self.classes) > 0

    def _predict_standardized_audio(
        self,
        y: np.ndarray,
        audio_meta: dict[str, Any],
        filename: str,
        top_k: int = 5,
        theme: str = "light",
    ) -> dict[str, Any]:
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "El modelo aún no ha sido entrenado. Ejecuta: python -m src.model.train"
            )

        windows = select_energetic_windows(y)
        specs_db = [compute_mel_spectrogram_db(w) for w in windows]

        # Elegir la ventana con mayor energía para renderizar el espectrograma representativo
        energies = [float(np.mean(w * w)) for w in windows]
        best_win_idx = int(np.argmax(energies))
        spec_b64 = render_spectrogram_base64(
            specs_db[best_win_idx],
            title=f"Espectrograma de Mel (ventana {best_win_idx + 1}/{len(windows)} de {CLIP_DURATION_SEC:.0f}s)",
            theme=theme,
        )

        tensors = [
            torch.from_numpy(normalize_spectrogram(spec)).unsqueeze(0)
            for spec in specs_db
        ]
        batch = torch.stack(tensors, dim=0).to(self.device)  # (W, 1, 64, 173)

        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        mean_probs = np.mean(probs, axis=0)
        k_eff = min(max(1, top_k), len(self.classes))
        top_indices = np.argsort(mean_probs)[::-1][:k_eff]

        top_predictions = [
            {
                "species": self.classes[int(idx)],
                "probability": round(float(mean_probs[int(idx)]), 4),
            }
            for idx in top_indices
        ]

        return {
            "predicted_species": top_predictions[0]["species"],
            "confidence": top_predictions[0]["probability"],
            "top_k": top_predictions,
            "audio_info": {
                "filename": filename,
                "duration_sec": audio_meta["duration_sec"],
                "analyzed_sec": audio_meta["analyzed_sec"],
                "orig_sample_rate": audio_meta["orig_sample_rate"],
                "sample_rate": SAMPLE_RATE,
                "windows_analyzed": len(windows),
            },
            "spectrogram_base64": spec_b64,
            "waveform_preview": build_waveform_preview(y, num_points=120),
        }

    def predict_from_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        top_k: int = 5,
        theme: str = "light",
    ) -> dict[str, Any]:
        """Ejecuta inferencia desde bytes subidos al endpoint."""
        y, meta = load_and_standardize_audio(audio_bytes)
        return self._predict_standardized_audio(
            y, meta, filename=filename, top_k=top_k, theme=theme
        )

    def predict_from_path(
        self,
        audio_path: Path,
        top_k: int = 5,
        theme: str = "light",
    ) -> dict[str, Any]:
        """Ejecuta inferencia sobre un archivo de audio existente en disco."""
        y, meta = load_and_standardize_audio(audio_path)
        return self._predict_standardized_audio(
            y, meta, filename=audio_path.name, top_k=top_k, theme=theme
        )
