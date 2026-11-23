# V. Ingeniería — (c) Validación del Modelo, Gráficas e Interfaz Web

## 1. Validación en el Conjunto de Prueba Independiente (`test`)

Al finalizar el entrenamiento, [`src/model/evaluate.py`](../src/model/evaluate.py) cargó el mejor *checkpoint* (época 28) y evaluó las **145 grabaciones independientes** del conjunto de prueba (**429 ventanas** de 4 segundos), agregando las probabilidades **Softmax** por grabación (`id`). Los resultados se almacenan en `artifacts/metrics/test_metrics.json`.

### 1.1 Métricas Globales de Generalización

| Métrica de Evaluación (Test Set) | Valor Obtenido | Observación |
| :--- | :---: | :--- |
| **Test Loss (NLL por grabación)** | **1.34513** | Coherente con validación (`1.4707`) |
| **Accuracy (por grabación)** | **65.52%** | **25.6× superior al azar uniforme** ($1/39 = 2.56\%$) |
| **Accuracy (por ventana individual)** | **63.40%** | La agregación multi-ventana aporta $+2.12\%$ |
| **Top-3 Accuracy** | **81.38%** | En 4 de cada 5 grabaciones la especie real está en el Top-3 |
| **Top-5 Accuracy** | **87.59%** | Alto poder de descarte como asistente bioacústico |
| **Precision Macro / Ponderada** | **0.6659 / 0.6639** | Buen equilibrio entre especies |
| **Recall Macro / Ponderada** | **0.6581 / 0.6552** | Cobertura consistente en las 39 clases |
| **F1-Score Macro / Ponderado** | **0.6332 / 0.6308** | Generalización sólida en 39 clases |

## 2. Análisis de Errores por Calidad, Tipo de Vocalización y Especies

### 2.1 Rendimiento por Calidad de Grabación (`A`, `B`, `C`)

| Calidad | Grabaciones en Test | Aciertos | Accuracy | F1 Macro | Confianza Promedio |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A (Alta)** | 64 | 41 | **64.06%** | 0.5527 | 63.07% |
| **B (Buena)** | 68 | 48 | **70.59%** | 0.6245 | 52.94% |
| **C (Ruidosa)** | 13 | 6 | **46.15%** | 0.3111 | 50.16% |

Se confirma cuantitativamente que las grabaciones de calidad `A` y `B` superan el 64%–70% de exactitud, mientras que las de calidad `C` (apenas 10.8% del dataset y con mayor ruido ambiental) sufren una caída aproximada de ~20 puntos porcentuales, reflejando además menor confianza promedio (`50.16%`).

### 2.2 Rendimiento por Tipo de Vocalización Principal

- **Cantos (`song`, n=101):** `65.35%` de accuracy (`66/101`).
- **Llamados (`call`, n=38):** `65.79%` de accuracy (`25/38`), demostrando que el modelo no depende únicamente de cantos largos sino que también identifica llamados breves.
- **Dúos (`duet`, n=2), subcanto (`subsong`, n=1) y `stuttersong` (n=1):** `100.0%` de acierto.
- **Emisiones atípicas únicas (`rattles` n=1, `scold` n=1):** `0/2` aciertos, con baja confianza (`14.69%` y `40.40%`), al ser variantes vocales subrepresentadas en entrenamiento.

### 2.3 Especies con Mayor Acierto y Mayores Confusiones (`confusion_matrix.json`)

- **Especies con desempeño sobresaliente:** *Psarocolius atrovirens* (`F1 = 1.00`), *Saltator aurantiirostris* (`F1 = 1.00`), *Elaenia parvirostris* (`F1 = 0.8889`), *Psittacara mitratus* (`F1 = 0.8889`), *Pyrrhura molinae* (`F1 = 0.8889`), *Myrmoborus myotherinus* (`F1 = 0.8571`), *Scytalopus bolivianus* (`F1 = 0.8571`), *Scytalopus schulenbergi* (`F1 = 0.8571`) y *Taraba major* (`F1 = 0.8571`).
- **Pares más confundidos:**
  - *Cantorchilus guarayanus* → *Henicorhina leucophrys* (`2` casos): ambas pertenecen a la familia *Troglodytidae* y comparten frases melódicas rápidas en bandas frecuenciales similares.
  - *Chiroxiphia boliviana* → *Scytalopus zimmeri* (`2` casos).
  - *Troglodytes aedon* → *Myiothlypis bivittata* (`2` casos).
- **Especies críticas en prueba (`F1 = 0.0`):** *Atlapetes rufinucha* (`n=3`), *Cantorchilus guarayanus* (`n=4`) y *Troglodytes aedon* (`n=4`), debido principalmente a repertorios vocales amplios (llamados cortos vs. cantos complejos) y similitud espectral con otras especies del sotobosque.

[CAPTURA DE PANTALLA: Pestaña "Métricas del Modelo CNN 2D" en la aplicación React mostrando las tarjetas KPI (Accuracy 65.52%, Top-3 81.38%, F1 Macro 63.32%), las curvas Loss y Accuracy por época, las gráficas por calidad A/B/C y la Matriz de Confusión interactiva 39x39.]

## 3. Servicio Backend Local en FastAPI

El backend implementado en [`backend/main.py`](../backend/main.py) y [`backend/audio_inference.py`](../backend/audio_inference.py) garantiza **cero divergencia entre entrenamiento e inferencia** al reutilizar exactamente `load_and_standardize_audio`, `select_energetic_windows`, `compute_mel_spectrogram_db` y `normalize_spectrogram`.

| Endpoint | Método | Función Principal |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Estado del servidor, modelo cargado, clases y dispositivo |
| `/api/dataset/summary` | `GET` | Auditoría y estadísticas de `dataset_stats.json` |
| `/api/dataset/samples` | `GET` | Listado paginado con filtros (`species`, `quality`, `split`, `search`) |
| `/api/audio/{sample_id}` | `GET` | Reproducción segura en streaming de grabaciones de `Audios/` |
| `/api/predict` | `POST` | Clasificación de archivo subido (`.wav`, `.mp3`, `.flac`, `.ogg`) |
| `/api/predict/sample/{sample_id}` | `POST` | Inferencia sobre muestra etiquetada + comparación con etiqueta real |
| `/api/metrics/overview` | `GET` | Historial de entrenamiento, métricas test, reporte y matriz 39×39 |
| `/api/unsupervised/status` | `GET` | Estado reservado del módulo No Supervisado (`implemented: false`) |

Todas las rutas fueron validadas mediante `fastapi.testclient.TestClient`, comprobando inferencia real tanto sobre archivos `.mp3` (`XC150547.mp3` → *Asthenes harterti*, `65.0%`) como `.wav` (`XC989053.wav` → *Asthenes dorbignyi*, `56.9%`), generación dinámica del espectrograma PNG en Base64 y envolvente de onda de 120 puntos.

## 4. Interfaz Web Interactiva en React (JavaScript)

La aplicación web construida en `frontend/` con **React 18**, **Vite** y **Recharts** adopta un diseño clásico sobrio en tonalidades de verde con selector de **modo claro / modo oscuro** (reflejado también en el renderizado del espectrograma de Mel mediante el parámetro `theme` del backend) y organiza la presentación y el análisis bioacústico en cuatro vistas:

1. **Identificar Ave (vista principal):** Presenta en el centro de la pantalla dos acciones principales lado a lado —**Subir archivo de audio** (`.wav`, `.mp3`, `.flac`, `.ogg`) y **Escuchar por micrófono** (captura en vivo en el navegador codificada directamente a `.wav` PCM mono de 16 bits mediante Web Audio API, con medidor de nivel y temporizador hasta 15 s)—, junto a una sección desplegable opcional para probar grabaciones del split de prueba. Muestra el reproductor `<audio controls>`, la forma de onda, el **Espectrograma de Mel (dB)** adaptado al tema activo, la especie predicha y el gráfico **Top-5 probabilidades Softmax**.
2. **Dataset:** Auditoría de limpieza (`1,000 → 972`), KPIs, distribución de calidades `A`/`B`/`C`, tipos de vocalización, histograma de duraciones reales y catálogo filtrable de las 39 especies.
3. **Métricas del Modelo:** Curvas interactivas de entrenamiento/validación, rendimiento por calidad (`A`, `B`, `C`) y vocalización, pares más confundidos, **Matriz de Confusión 39×39 interactiva** con resaltado por especie y tabla ordenable del reporte por clase.
4. **No Supervisado (Próximamente):** Pestaña reservada vinculada a `/api/unsupervised/status` y `src/unsupervised/`, dejando explícito que no se ejecuta agrupamiento en esta primera entrega.

[CAPTURA DE PANTALLA: Pestaña "Identificar Ave" en React (modo claro y modo oscuro) mostrando los dos botones centrales de subida de archivo y grabación por micrófono, junto al espectrograma de Mel en dB, predicción y gráfico Top-5.]

[CAPTURA DE PANTALLA: Pestaña "No Supervisado (Próximamente)" indicando el estado reservado del segundo modelo para la siguiente entrega.]
