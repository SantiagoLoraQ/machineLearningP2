# Módulo Reservado: Aprendizaje No Supervisado (Siguiente Entrega)

> **Estado:** Reservado — **No implementado en la Primera Entrega**.

De acuerdo con los lineamientos de la **Práctica del Segundo Parcial** (Sección 3: *Planteamiento de Investigación*), la solución completa contempla dos modelos:
1. **Modelo de Aprendizaje Profundo Supervisado** (implementado en esta **Primera Entrega**): CNN 2D en PyTorch entrenada sobre espectrogramas de Mel para clasificar 39 especies de aves de Bolivia.
2. **Modelo de Aprendizaje No Supervisado** (correspondiente a la **Siguiente Entrega**): análisis exploratorio y agrupamiento (*clustering*) de vocalizaciones sin utilizar etiquetas durante el ajuste.

## Alcance planificado para la Segunda Entrega
- Extracción de vectores de características / *embeddings* acústicos a partir de los audios depurados en `data/processed/dataset_limpio.csv`.
- Reducción de dimensionalidad para inspección de estructura acústica por especie, calidad o tipo de vocalización (*song*, *call*, *duet*).
- Entrenamiento y evaluación de algoritmos no supervisados (por ejemplo, K-Means, agrupamiento jerárquico o basado en densidad) con métricas internas (Silhouette, Davies-Bouldin).
- Integración de los resultados y visualizaciones en la pestaña reservada del frontend React y el endpoint correspondiente del backend FastAPI.
