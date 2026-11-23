# IV. Marco Referencial

## 1. Planteamiento y Sustentación del Problema

El monitoreo acústico pasivo de avifauna permite detectar especies en ecosistemas complejos donde la observación visual directa resulta limitada por la densidad vegetal, el comportamiento esquivo o la baja luminosidad del sotobosque. En **Bolivia**, país que alberga gradientes ecológicos desde el altiplano y valles secos hasta los bosques montanos de los Yungas y la llanura amazónica, automatizar la clasificación de cantos y llamados contribuye tanto al inventario de biodiversidad como a estudios de conservación frente a la fragmentación de hábitat.

Sin embargo, el problema presenta desafíos técnicos concretos en el ámbito del reconocimiento de señales acústicas:
- **Heterogeneidad de formatos y frecuencias de muestreo:** Las grabaciones de campo provienen de distintos equipos y épocas (en este conjunto de datos, desde `1984-01-08` hasta `2026-08-24`, aportadas por `32` grabadores distintos), almacenadas en formatos `.mp3` (`858` archivos) y `.wav` (`114` archivos) con frecuencias de muestreo diversas (principalmente 44,100 Hz y 48,000 Hz).
- **Variabilidad extrema de duración:** Las pistas abarcan desde fragmentos muy cortos (`2.48 s`) hasta registros continuos superiores a once minutos (`710.46 s`), con una mediana de `35.91 s` y un acumulado total de `14.95 horas` de audio.
- **Ruido ambiental y niveles de calidad (`A`, `B`, `C`):** Las grabaciones incluyen ruido de viento, agua, insectos y otras aves en segundo plano, categorizado en el estándar de Xeno-canto mediante las calidades `A` (excelente), `B` (buena/intermedia) y `C` (moderada o con interferencia de fondo).
- **Diversidad de repertorios vocales:** Una misma especie emite distintos tipos de señales acústicas —cantos territoriales (*song*), llamados cortos (*call*), dúos (*duet*) o subcantos (*subsong*)— cuyas firmas espectrales difieren notablemente entre sí.

## 2. Descripción y Diagnóstico del Dataset

El punto de partida del proyecto es el archivo `CSV/manifest_verificado.csv`, compuesto originalmente por **1,000 registros** asociados a grabaciones almacenadas en `Audios/`. Tras ejecutar la depuración automatizada (`src/data/clean_csv.py`), el conjunto válido consolidado en `data/processed/dataset_limpio.csv` queda conformado por **972 grabaciones válidas** correspondientes a **39 especies de aves de Bolivia**.

### 2.1 Resumen Cuantitativo del Conjunto Depurado

| Indicador del Dataset | Valor Verificado |
| :--- | :--- |
| Registros iniciales en `manifest_verificado.csv` | 1,000 |
| Registros excluidos de clase indeterminada (`Mystery mystery`) | 27 |
| Registros excluidos por ausencia física en disco (`XC1630.mp3`) | 1 |
| **Grabaciones válidas finales (`dataset_limpio.csv`)** | **972** |
| **Número de especies clasificables** | **39** |
| Rango de muestras por especie | 21 a 27 grabaciones |
| Distribución por formato de audio | 858 MP3 / 114 WAV |
| Distribución por calidad (`A` / `B` / `C`) | 423 (43.5%) / 444 (45.7%) / 105 (10.8%) |
| Duración mínima / mediana / media / máxima | 2.48 s / 35.91 s / 55.39 s / 710.46 s |
| Duración acumulada total | 14.95 horas |

### 2.2 Cobertura Geográfica y Taxonómica

Todos los registros conservados pertenecen a localidades de **Bolivia**, cubriendo ecorregiones distribuidas principalmente en los departamentos de **Santa Cruz**, **La Paz**, **Cochabamba**, **Beni**, **Chuquisaca**, **Pando** y **Tarija**. Entre las familias representadas destacan horneros y colaespinas (*Asthenes dorbignyi*, *Asthenes harterti*, *Cranioleuca albiceps*), tapaculos andinos de vocalización repetitiva (*Scytalopus parvirostris*, *Scytalopus schulenbergi*, *Scytalopus bolivianus*, *Scytalopus zimmeri*), hormigueros (*Thamnophilus aethiops*, *Thamnophilus caerulescens*, *Pyriglena maura*), trogloditas (*Cantorchilus guarayanus*, *Pheugopedius genibarbis*, *Troglodytes aedon*), zorzales (*Turdus hauxwelli*, *Turdus leucomelas*), loros (*Psittacara mitratus*, *Pyrrhura molinae*) y palomas (*Patagioenas plumbea*).

### 2.3 Distribución por Tipo de Vocalización Principal

Al normalizar la columna `tipo_vocalizacion`, predominan las emisiones de canto (`song`: 615 grabaciones) y llamado (`call`: 319), seguidas por dúos (`duet`: 11), subcantos (`subsong`: 10) y variantes específicas (`stuttersong`: 4, `vocalizations`: 4, entre otras 9 ocurrencias minoritarias).

[CAPTURA DE PANTALLA: Pestaña "Explorador del Dataset" en la interfaz web React mostrando las tarjetas KPI, el banner de auditoría de limpieza, las gráficas de distribución por calidad A/B/C, tipos de vocalización, histograma de duraciones y la distribución de las 39 especies.]
