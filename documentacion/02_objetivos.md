# II. Objetivos

## Objetivo General

Desarrollar e implementar un sistema de aprendizaje profundo basado en una **Red Neuronal Convolucional Bidimensional (CNN 2D)** en **PyTorch**, acompañado de una API local de inferencia y una interfaz web interactiva en **React**, capaz de reconocer vocalizaciones de 39 especies de aves de Bolivia de manera robusta frente a variaciones de duración, calidad de grabación y ruido ambiental.

## Objetivos Específicos

1. **Depuración reproducible y representación tiempo-frecuencia:** Construir un flujo automatizado de limpieza sobre el manifiesto verificado que descarte registros indeterminados (`Mystery mystery`) o sin audio válido, genere rutas relativas portables y extraiga en lote **espectrogramas de Mel en decibelios (dB)** que preserven las estructuras armónicas distintivas de cada especie.
2. **Mitigación de desbalanceo, ruido y sobreajuste:** Incorporar estrategias de muestreo ponderado por especie y calidad (`A`, `B`, `C`), aumento de datos sobre espectrogramas (*SpecAugment*, desplazamientos temporales y ruido gaussiano controlado solo en entrenamiento), normalización por lotes (*BatchNorm*), *Dropout*, regularización L2 (`weight_decay`) y detención temprana (*Early Stopping*).
3. **Validación rigurosa y análisis de errores:** Evaluar la capacidad de generalización del modelo sobre un conjunto de prueba independiente sin fuga de datos entre ventanas de una misma grabación, midiendo *loss*, *accuracy*, *Top-k accuracy*, *precision*, *recall*, *F1-score*, matriz de confusión multiclase y degradación por calidad o tipo de vocalización.
4. **Integración full-stack y escalabilidad modular:** Desplegar un backend local en **FastAPI** y un cliente web en **React (JavaScript)** para explorar el dataset, inspeccionar espectrogramas, clasificar grabaciones en tiempo real y visualizar métricas, dejando reservada explícitamente la estructura del futuro módulo de **Aprendizaje No Supervisado** para la segunda entrega del proyecto.
