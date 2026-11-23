# V. Ingeniería — (a) Preprocesamiento del Dataset

## 1. Limpieza Automatizada del CSV (`dataset_limpio.csv`)

El archivo base proporcionado en `CSV/manifest_verificado.csv` contiene **1,000 filas** con 16 columnas, incluyendo metadatos editoriales y URLs externas (`url_audio`, `url_ficha`, `licencia`, `grabador`, `pais`, `localidad`, `fecha`, `nombre_archivo_origen`, `lat`, `lon`), rutas absolutas heredadas de un entorno Linux/Colab (`/content/aves_bolivia_1000/audios/...`) y **27 registros** correspondientes a la categoría indeterminada `"Mystery mystery"`.

Para garantizar reproducibilidad en Windows y escalabilidad para todo el equipo, se implementó el script modular [`src/data/clean_csv.py`](../src/data/clean_csv.py), ejecutable mediante:

```powershell
.\.venv\Scripts\python.exe -m src.data.clean_csv
```

El proceso ejecuta las siguientes etapas automatizadas:

1. **Exclusión de clase indeterminada y filas inválidas:** Se descartan todos los registros con `especie` vacía o igual a `"Mystery mystery"` (27 filas excluidas) y se valida unicidad por `id` (0 duplicados encontrados).
2. **Normalización de nombres y rutas portables:** Se normalizan los nombres científicos (`especie`), se genera `especie_carpeta` reemplazando espacios por guion bajo (`_`) y se construye `ruta_relativa` en formato POSIX portable relativa a la raíz del proyecto (`Audios/<Especie>/<archivo>`), prohibiendo cualquier ruta absoluta dependiente de máquina mediante [`src/config.py`](../src/config.py) y `pathlib`.
3. **Verificación física y compatibilidad con limpieza manual de audios:** Para cada registro se comprueba su existencia real en disco. Si el usuario limpia o convierte un audio manualmente (por ejemplo de `.mp3` a `.wav` conservando el prefijo `XC<id>`), `resolve_audio_path()` localiza automáticamente la nueva versión en `Audios/<Especie>/`. Posteriormente `soundfile.info()` verifica legibilidad, número de canales, frecuencia de muestreo y duración real (`duracion_seg >= 0.3 s`). En esta verificación se detectó y excluyó 1 archivo ausente en disco (`id=1630`, `Patagioenas plumbea`, ruta esperada `XC1630.mp3`) y se corrigió la duración real frente al manifiesto en 6 registros con diferencia superior a 5 segundos.
4. **Selección estricta de columnas útiles:** `data/processed/dataset_limpio.csv` conserva únicamente las 10 columnas necesarias para modelado y análisis:
   `id`, `especie`, `especie_carpeta`, `ruta_relativa`, `calidad`, `tipo_vocalizacion`, `tipo_vocalizacion_principal`, `duracion_seg`, `sample_rate`, `canales`.
5. **Auditoría y estadísticas:** El script genera simultáneamente `data/processed/audit_limpieza.json` y `artifacts/metrics/dataset_stats.json`, registrando el paso de **1,000 registros originales** a **972 grabaciones válidas** de **39 especies**.

[CAPTURA DE PANTALLA: Terminal PowerShell ejecutando `python -m src.data.clean_csv` mostrando los 1,000 registros originales, las exclusiones (27 mystery_mystery, 1 missing_file) y los 972 registros limpios finales, junto a las primeras filas de `data/processed/dataset_limpio.csv`.]

## 2. Protocolo de Re-ejecución tras Limpieza Directa de Audios

Cuando el equipo realice depuración directa sobre archivos individuales dentro de `Audios/<Especie>/` (recorte de silencios, filtrado pasa-banda o conversión a `.wav`), basta con volver a ejecutar en orden:

```powershell
.\.venv\Scripts\python.exe -m src.data.clean_csv
.\.venv\Scripts\python.exe -m src.data.extract_spectrograms
```

De esta forma se recalculan automáticamente `duracion_seg`, `sample_rate`, `ruta_relativa`, particiones estratificadas y todos los espectrogramas en lote sin editar manualmente el CSV.

## 3. Estandarización de Señales y Extracción en Lote de Espectrogramas de Mel

El procesamiento acústico en lote se implementa en [`src/data/extract_spectrograms.py`](../src/data/extract_spectrograms.py) bajo bloques `try/except` por grabación para evitar que archivos con tramas corruptas detengan el flujo:

1. **División estratificada sin fuga de datos (`data/processed/splits.csv`):** Antes de segmentar ventanas, las 972 grabaciones limpias se dividen de manera estratificada por especie en **70% entrenamiento (680 audios)**, **15% validación (146 audios)** y **15% prueba (146 audios)** con semilla fija `42`. Esto garantiza que ninguna grabación comparta ventanas entre distintos subconjuntos.
2. **Estandarización de onda:** Cada archivo se decodifica de forma nativa con `soundfile` (hasta un máximo de 60.0 s para mantener eficiencia en pistas de >10 min), se mezcla a **mono**, se remuestrea determinísticamente a **22,050 Hz** mediante `scipy.signal.resample_poly`, se elimina el desplazamiento DC y se normaliza por amplitud pico.
3. **Selección energética de ventanas de 4.0 s:** Para grabaciones menores a 4.0 s (`88,200` muestras) se aplica repetición cíclica (*tiling*) hasta 4.0 s; para grabaciones mayores se evalúan segmentos candidatos con salto de `2.0 s` tras un filtro de pre-énfasis ($y[n] - 0.95\,y[n-1]$) que atenúa viento grave, seleccionando hasta **3 ventanas de 4.0 s** de mayor energía RMS con solapamiento máximo del 50%.
4. **Cálculo de espectrogramas de Mel en dB:** Se aplica una STFT con ventana Hann (`n_fft = 1024`, `hop_length = 512`), proyectada sobre un banco de **64 filtros triangulares Mel** entre **300 Hz y 11,025 Hz** (normalización Slaney) y convertida a decibelios en el rango `[-80.0, 0.0] dB`, generando matrices fijas de dimensión **`(64, 173)`** almacenadas en formato `float16` (`.npy`) bajo `data/spectrograms/`.

### 3.1 Resultados de la Extracción en Lote

- **Grabaciones procesadas exitosamente:** **968 de 972** (`677` en entrenamiento, `146` en validación y `145` en prueba).
- **Ventanas de espectrogramas generadas:** **2,813 matrices `(64, 173)`** (`1,956` en *train*, `428` en *val* y `429` en *test*).
- **Manejo de errores sin interrupción:** El bloque `try/except` detectó y aisló **4 archivos `.mp3` con tramas internas dañadas** (`XC1702.mp3`, `XC2540.mp3`, `XC2541.mp3`, `XC1837.mp3`), registrándolos en `data/spectrograms/extraction_errors.json` sin interrumpir el lote.

## 4. Especificaciones de Herramientas de Preprocesamiento

| Herramienta / Librería | Versión / Rol en Preprocesamiento |
| :--- | :--- |
| **Python (`cp314`) + `pathlib`** | Lenguaje base (`3.14.7`) y resolución portable de rutas relativas |
| **SoundFile (`libsndfile`)** | Lectura nativa de `.mp3` y `.wav`, inspección de duración, SR y canales |
| **SciPy (`scipy.signal`)** | Remuestreo polifásico (`resample_poly`) y STFT (`stft`) con ventana Hann |
| **NumPy & Pandas** | Banco de filtros Mel, conversión logarítmica dB y manipulación tabular |
| **Scikit-learn** | División estratificada por especie (`train_test_split` 70/15/15) |

[CAPTURA DE PANTALLA: Terminal PowerShell tras ejecutar `python -m src.data.extract_spectrograms` mostrando las 2,813 ventanas extraídas por split y el archivo `data/spectrograms/extraction_errors.json` con los 4 audios aislados.]
