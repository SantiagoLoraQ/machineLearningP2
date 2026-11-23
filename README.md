# Reconocimiento Acústico de Vocalizaciones de Aves de Bolivia — Primera Entrega

Prototipo académico full-stack (**PyTorch + FastAPI + React JS**) para la **Práctica del Segundo Parcial** (Área: *Reconocimiento de Voz* — Subproblema: *Identificación acústica de 39 especies de aves de Bolivia*).

> **Alcance de esta Primera Entrega:** Implementa exclusivamente el modelo supervisado de Deep Learning (**CNN 2D sobre espectrogramas de Mel en dB**), el flujo reproducible de limpieza y extracción en lote, la API local de inferencia y la interfaz web interactiva en React. El segundo modelo de **Aprendizaje No Supervisado** queda estrictamente reservado para la siguiente entrega (`src/unsupervised/`).

---

## Arquitectura General

1. **Preprocesamiento Reproducible (`src/data/`):**
   - `clean_csv.py`: Depura `CSV/manifest_verificado.csv` (1,000 → **972 grabaciones válidas** de **39 especies**, excluyendo 27 registros de `"Mystery mystery"` y 1 archivo ausente `XC1630.mp3`), genera rutas relativas portables con `pathlib` y verifica duración/legibilidad real con `soundfile`.
   - `extract_spectrograms.py`: Divide de forma estratificada por grabación (**70% train / 15% val / 15% test**) sin fuga de datos, estandariza a **22,050 Hz mono**, selecciona hasta 3 ventanas energéticas de **4.0 s** y extrae en lote **2,813 espectrogramas de Mel en dB** de forma `(64, 173)` mediante `scipy.signal.stft`.
2. **Modelo Supervisado PyTorch (`src/model/`):**
   - `BirdCNN2D` (`cnn2d.py`): CNN 2D de 4 bloques (`24 → 48 → 96 → 192` canales, **248,911 parámetros**) con *BatchNorm*, *ReLU*, *MaxPool2d*, *AdaptiveAvgPool2d*, *Dropout2d/Dropout* y salida Softmax sobre 39 clases.
   - `train.py` & `evaluate.py`: `WeightedRandomSampler`, *SpecAugment* (enmascaramiento tiempo/frecuencia + *time roll* + ruido gaussiano en entrenamiento), `AdamW` con `weight_decay`, `ReduceLROnPlateau` y *Early Stopping*.
   - **Resultados verificados en prueba independiente (`145` grabaciones):** **65.52% Accuracy**, **81.38% Top-3**, **87.59% Top-5**, **0.6332 F1-Macro**, **1.3451 Test Loss**.
3. **Backend Local (`backend/`):**
   - API REST en **FastAPI** que sirve estadísticas del dataset, streaming de audios locales, inferencia unificada sobre archivos `.wav`/`.mp3`/`.flac`/`.ogg` o muestras de prueba, generación dinámica de espectrogramas PNG en Base64 y métricas de evaluación.
4. **Frontend Interactivo (`frontend/`):**
   - Aplicación web en **React (JavaScript) + Vite + Recharts** con estilo clásico en tonalidades verdes, soporte de **modo claro / modo oscuro** y 4 pestañas: *Identificar Ave* (con botones centrales para **subir archivo** o **escuchar en vivo por micrófono** grabando en `.wav` PCM, más selector opcional de muestras del dataset), *Dataset*, *Métricas del Modelo* (incluyendo matriz de confusión interactiva 39×39) y *No Supervisado (Próximamente)*.

---

## Estructura de Carpetas

```text
Sonidos de Aves/
├── Audios/                                 # Audios originales por especie (preservado)
├── CSV/                                    # Manifiesto base manifest_verificado.csv (preservado)
├── data/
│   ├── processed/                          # dataset_limpio.csv, audit_limpieza.json, splits.csv
│   └── spectrograms/                       # *.npy (64x173), spectrogram_index.csv, extraction_errors.json
├── artifacts/
│   ├── models/                             # best_bird_cnn.pt, class_mapping.json, model_config.json
│   └── metrics/                            # dataset_stats.json, training_history.json, test_metrics.json, ...
├── src/
│   ├── config.py                           # Única fuente de rutas portables (pathlib) e hiperparámetros
│   ├── data/
│   │   ├── clean_csv.py
│   │   └── extract_spectrograms.py
│   ├── model/
│   │   ├── cnn2d.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── unsupervised/                       # RESERVADO para la Segunda Entrega (sin lógica ML aún)
├── backend/
│   ├── audio_inference.py
│   └── main.py
├── frontend/                               # React + Vite (JavaScript)
├── documentacion/                          # 9 archivos .md por subtítulo listos para LaTeX
└── requirements.txt
```

---

## Guía de Ejecución en Local (Windows PowerShell)

### 1. Instalar dependencias de Python (`.venv`)

```powershell
cd "c:\Users\pixel\Downloads\Sonidos de Aves"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> Si deseas instalar explícitamente la variante CPU de PyTorch:
> `.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 2. Ejecutar el pipeline de datos y entrenamiento

*(Nota: Los artefactos ya vienen pre-generados y evaluados; vuelve a ejecutar estos pasos si realizas limpieza manual de archivos en `Audios/`.)*

```powershell
# Paso 1: Limpiar CSV y validar archivos físicos en Audios/
.\.venv\Scripts\python.exe -m src.data.clean_csv

# Paso 2: Dividir de forma estratificada y extraer espectrogramas Mel en lote
.\.venv\Scripts\python.exe -m src.data.extract_spectrograms

# Paso 3: Entrenar CNN 2D y evaluar automáticamente sobre el conjunto test
.\.venv\Scripts\python.exe -u -m src.model.train
```

### 3. Levantar el Backend Local (FastAPI)

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

La documentación interactiva Swagger estará disponible en `http://127.0.0.1:8000/docs`.

### 4. Instalar y levantar el Frontend (React + Vite)

En una segunda terminal PowerShell:

```powershell
cd "c:\Users\pixel\Downloads\Sonidos de Aves\frontend"
npm install
npm run dev
```

Abre en tu navegador **`http://localhost:5173`**.
