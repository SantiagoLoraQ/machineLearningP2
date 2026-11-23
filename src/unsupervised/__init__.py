"""Espacio reservado para el módulo de Aprendizaje No Supervisado (Siguiente Entrega).

NOTA IMPORTANTE:
En la Primera Entrega NO se implementa ni entrena lógica no supervisada
(clustering, embeddings, UMAP, t-SNE, etc.). Este módulo únicamente expone
información descriptiva del alcance futuro del proyecto.
"""

UNSUPERVISED_STATUS: dict = {
    "status": "reserved",
    "implemented": False,
    "delivery": "Segunda Entrega",
    "title": "Próximamente: Aprendizaje No Supervisado (Siguiente Entrega)",
    "description": (
        "Módulo reservado para el segundo modelo requerido por la práctica: "
        "agrupamiento acústico no supervisado sobre representaciones latentes de "
        "vocalizaciones de aves de Bolivia (embeddings de espectrogramas, reducción "
        "dimensional y descubrimiento de patrones vocales/tipos de canto sin etiquetas)."
    ),
    "planned_directory": "src/unsupervised/",
    "planned_components": [
        "Extracción de embeddings acústicos (sin etiquetas de clase)",
        "Reducción dimensional exploratoria (PCA / UMAP)",
        "Agrupamiento no supervisado (K-Means / HDBSCAN) y evaluación con Silhouette Score",
        "Visualización interactiva de clústeres acústicos en la aplicación web",
    ],
}
