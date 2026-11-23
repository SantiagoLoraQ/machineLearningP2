# V. Ingeniería — (b) Entrenamiento y Ejecución del Modelo

## 1. Arquitectura de la Red Neuronal Convolucional (`BirdCNN2D`)

En [`src/model/cnn2d.py`](../src/model/cnn2d.py) se implementó la arquitectura supervisada **`BirdCNN2D`** en **PyTorch** (`torch 2.15.0+cpu`), diseñada específicamente para procesar tensores tiempo-frecuencia monocanal de dimensión **`(1, 64, 173)`** con eficiencia en CPU local sin sacrificar capacidad de extracción armónica.

### 1.1 Capas de Extracción Espacial y Clasificador

| Bloque / Capa | Operación Principal | Salida `(C, F, T)` | Regularización |
| :--- | :--- | :--- | :--- |
| **Entrada** | Espectrograma Mel dB normalizado | `(1, 64, 173)` | Normalización por muestra ($\mu=0, \sigma=1$) |
| **Bloque 1 (Stem)** | `Conv2d(1→24, k=5, s=2, p=2)` + `BN` + `ReLU` + `MaxPool(2)` | `(24, 16, 43)` | `Dropout2d(0.10)` |
| **Bloque 2** | `Conv2d(24→48, k=3, p=1)` + `BN` + `ReLU` + `MaxPool(2)` | `(48, 8, 21)` | `Dropout2d(0.15)` |
| **Bloque 3** | `Conv2d(48→96, k=3, p=1)` + `BN` + `ReLU` + `MaxPool(2)` | `(96, 4, 10)` | `Dropout2d(0.20)` |
| **Bloque 4** | `Conv2d(96→192, k=3, p=1)` + `BN` + `ReLU` + `AdaptiveAvgPool(1, 1)` | `(192, 1, 1)` | — |
| **Clasificador Denso** | `Flatten` → `Linear(192→128)` + `BN1d` + `ReLU` → `Linear(128→39)` | `(39,)` *logits* | `Dropout(0.35)` y `Dropout(0.25)` |

- **Total de parámetros entrenables:** **248,911 parámetros**.
- **Salida probabilística:** Durante evaluación e inferencia se aplica la función **Softmax** sobre los 39 *logits* de salida y se promedian las probabilidades entre las ventanas de una misma grabación.

## 2. Estrategias contra Desbalanceo, Ruido y Sobreajuste

En [`src/model/dataset.py`](../src/model/dataset.py) y [`src/model/train.py`](../src/model/train.py) se configuraron las siguientes técnicas de control:

1. **Partición estratificada sin fuga de datos:** Entrenamiento con **1,956 ventanas** (677 grabaciones válidas), validación con **428 ventanas** (146 grabaciones) y prueba independiente con **429 ventanas** (145 grabaciones), garantizando intersección nula de `id` entre particiones.
2. **Muestreo ponderado (`WeightedRandomSampler`):** Solo sobre *train*, compensando diferencias entre especies (21 a 27 grabaciones) e incrementando el peso de grabaciones de calidad `C` (`factor = 1.15`) frente a `A`/`B` (`factor = 1.00`).
3. **Aumento de datos solo en entrenamiento:**
   - Desplazamiento temporal circular aleatorio ($\pm 30$ tramas).
   - Ajuste aleatorio de contraste espectral ($\text{gain} \sim U(0.85, 1.15)$).
   - **SpecAugment:** 1 a 2 máscaras en frecuencia (2–8 bandas) y 1 a 2 máscaras temporales (4–22 tramas).
   - Ruido gaussiano aditivo ($\sigma \sim U(0.02, 0.12)$).
4. **Optimización y control de entrenamiento:**
   - Función de pérdida: `CrossEntropyLoss(label_smoothing=0.05)`.
   - Optimizador: `AdamW` con `lr = 2e-3` y regularización L2 `weight_decay = 1e-4`.
   - Planificador: `ReduceLROnPlateau(mode="min", factor=0.5, patience=3)` sobre `val_loss`.
   - Recorte de gradiente (`clip_grad_norm_`, `max_norm = 2.0`).
   - **Early Stopping** con paciencia de `8` épocas y guardado automático del mejor checkpoint (`artifacts/models/best_bird_cnn.pt`) junto a `class_mapping.json` y `model_config.json`.

## 3. Ejecución y Evolución del Entrenamiento en CPU

El entrenamiento se ejecutó localmente en Windows sobre CPU (`torch 2.15.0+cpu`, 4 hilos) mediante:

```powershell
.\.venv\Scripts\python.exe -u -m src.model.train
```

### 3.1 Hitos de Convergencia Registrados (`training_history.json`)

| Época | Train Loss | Train Acc | Val Loss (Grab.) | Val Acc (Grab.) | Duración (s) | Estado |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 3.6232 | 5.98% | 3.4753 | 7.53% | 27.65 | Inicio |
| **5** | 2.7184 | 28.12% | 2.3183 | 37.67% | 28.7 | Mejora sostenida |
| **10** | 2.3057 | 37.88% | 1.8644 | 52.74% | 22.9 | Checkpoint mejorado |
| **18** | 1.9918 | 48.98% | 1.5805 | 58.22% | 29.4 | Checkpoint mejorado |
| **27** | 1.6559 | 59.82% | 1.4923 | 60.27% | 26.87 | Empate en Val Acc |
| **28** | 1.6700 | 58.23% | **1.4707** | **60.27%** | 21.70 | **Mejor Checkpoint Guardado** |
| **30** | 1.6443 | 60.12% | 1.5502 | 59.59% | 26.73 | Fin de 30 épocas |

- **Tiempo total de entrenamiento (30 épocas):** **767.75 segundos** (~12.8 minutos en CPU, ~25.6 s/época en promedio).
- **Mejor checkpoint:** **Época 28** (`val_accuracy = 60.27%`, `val_loss = 1.4707`), almacenado en `artifacts/models/best_bird_cnn.pt`.

[CAPTURA DE PANTALLA: Terminal PowerShell ejecutando `python -u -m src.model.train` mostrando el progreso por épocas, el guardado de `*BEST*` en la época 28 y el tiempo total de entrenamiento.]

## 4. Especificaciones de Hardware y Software de Entrenamiento

| Componente | Especificación Utilizada |
| :--- | :--- |
| **Sistema Operativo** | Windows (ejecución local en PowerShell) |
| **Procesador / Memoria** | AMD A10-7700K Radeon R7 (4 núcleos CPU) / 16 GB RAM |
| **Framework Deep Learning** | PyTorch `2.15.0+cpu` |
| **Entorno Python** | Python `3.14.7` en entorno virtual `.venv` |
| **Tamaño del Checkpoint** | `best_bird_cnn.pt` (~1.0 MB para 248,911 parámetros + metadatos) |
