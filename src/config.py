"""Configuración centralizada y portable del proyecto mediante pathlib.

Todas las rutas se resuelven de forma relativa respecto a PROJECT_ROOT,
evitando rutas absolutas dependientes de máquina.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Carpetas de entrada (preservadas intactas)
AUDIOS_DIR: Path = PROJECT_ROOT / "Audios"
CSV_DIR: Path = PROJECT_ROOT / "CSV"
RAW_CSV_PATH: Path = CSV_DIR / "manifest_verificado.csv"

# Carpetas de datos procesados
DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CLEAN_CSV_PATH: Path = PROCESSED_DIR / "dataset_limpio.csv"
AUDIT_JSON_PATH: Path = PROCESSED_DIR / "audit_limpieza.json"
SPLITS_CSV_PATH: Path = PROCESSED_DIR / "splits.csv"

SPECTROGRAMS_DIR: Path = DATA_DIR / "spectrograms"
SPEC_INDEX_PATH: Path = SPECTROGRAMS_DIR / "spectrogram_index.csv"
SPEC_ERRORS_PATH: Path = SPECTROGRAMS_DIR / "extraction_errors.json"

# Artefactos del modelo y métricas
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"

BEST_MODEL_PATH: Path = MODELS_DIR / "best_bird_cnn.pt"
CLASS_MAPPING_PATH: Path = MODELS_DIR / "class_mapping.json"
MODEL_CONFIG_PATH: Path = MODELS_DIR / "model_config.json"

DATASET_STATS_PATH: Path = METRICS_DIR / "dataset_stats.json"
TRAINING_HISTORY_PATH: Path = METRICS_DIR / "training_history.json"
TEST_METRICS_PATH: Path = METRICS_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH: Path = METRICS_DIR / "confusion_matrix.json"
CLASSIFICATION_REPORT_PATH: Path = METRICS_DIR / "classification_report.json"
QUALITY_ANALYSIS_PATH: Path = METRICS_DIR / "quality_error_analysis.json"

# Parámetros de estandarización de audio y espectrogramas de Mel
SAMPLE_RATE: int = 22050
CLIP_DURATION_SEC: float = 4.0
CLIP_SAMPLES: int = int(SAMPLE_RATE * CLIP_DURATION_SEC)  # 88,200 muestras
MAX_LOAD_SECONDS: float = 60.0
WINDOWS_PER_CLIP: int = 3
WINDOW_HOP_SEC: float = 2.0

N_MELS: int = 64
N_FFT: int = 1024
HOP_LENGTH: int = 512
FMIN: int = 300
FMAX: int = 11025
SPEC_FRAMES: int = 1 + (CLIP_SAMPLES // HOP_LENGTH)  # 173 frames
MIN_VALID_AUDIO_SEC: float = 0.3

# Partición y entrenamiento supervisado
RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

DEFAULT_BATCH_SIZE: int = 32
DEFAULT_EPOCHS: int = 30
DEFAULT_LR: float = 2e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_DROPOUT: float = 0.35
EARLY_STOPPING_PATIENCE: int = 8


def ensure_directories() -> None:
    """Crea la estructura de directorios de salida si no existe."""
    for directory in (
        DATA_DIR,
        PROCESSED_DIR,
        SPECTROGRAMS_DIR,
        ARTIFACTS_DIR,
        MODELS_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def to_project_relative_posix(path: Path) -> str:
    """Convierte una ruta dentro del proyecto a una ruta relativa POSIX portable."""
    resolved = path.resolve()
    root_resolved = PROJECT_ROOT.resolve()
    try:
        return resolved.relative_to(root_resolved).as_posix()
    except ValueError:
        return path.as_posix()
