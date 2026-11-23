# I. Introducción

El reconocimiento automático de señales acústicas constituye una de las ramas más dinámicas dentro del **Reconocimiento de Voz** y el aprendizaje automático moderno. Si bien sus aplicaciones más conocidas se orientan al habla humana, las mismas bases de procesamiento tiempo-frecuencia y aprendizaje profundo permiten abordar un subproblema científico de alto valor ambiental: el **reconocimiento acústico de especies de aves a partir de sus vocalizaciones**.

Bolivia se ubica entre los países con mayor diversidad avifaunística del mundo, abarcando ecorregiones contrastantes que incluyen los valles secos interandinos, los bosques nublados de los Yungas, la llanura amazónica, la Chiquitanía y el Chaco. El monitoreo pasivo mediante grabaciones de campo permite registrar la presencia de especies crípticas o de difícil observación visual; sin embargo, la inspección manual de cientos de grabaciones heterogéneas resulta costosa, lenta y dependiente de especialistas.

En el marco de la **Práctica del Segundo Parcial** de la asignatura de Machine Learning, este proyecto desarrolla un prototipo integral de software en entorno local Windows enfocado en vocalizaciones de **39 especies de aves registradas en Bolivia**. La **Primera Entrega** comprende íntegramente la rama de **Aprendizaje Profundo Supervisado**, implementando:

1. Un flujo automatizado y reproducible de depuración del manifiesto base (`manifest_verificado.csv` hacia `dataset_limpio.csv`) con verificación física de los archivos en disco y rutas relativas portables mediante `pathlib`.
2. Un extractor en lote de **espectrogramas de Mel en decibelios (dB)** con selección energética de ventanas, mitigando diferencias de duración, frecuencia de muestreo y ruido ambiental.
3. Una **Red Neuronal Convolucional Bidimensional (CNN 2D)** entrenada en **PyTorch**, incorporando muestreo ponderado, aumento de datos (*SpecAugment*), *Batch Normalization*, *Dropout*, regularización L2 y detención temprana (*Early Stopping*).
4. Un servicio backend local en **FastAPI** y una aplicación web interactiva en **React (JavaScript)** para explorar el dataset, reproducir e inspeccionar grabaciones, visualizar espectrogramas de Mel, clasificar nuevos audios en tiempo real y analizar curvas de entrenamiento y matriz de confusión.

Conforme a los lineamientos de la práctica, la solución final contemplará un segundo modelo basado en **Aprendizaje No Supervisado**. De forma estricta con el alcance de esta primera entrega, dicho modelo **no se implementa ni entrena en esta fase**, manteniendo reservada la carpeta `src/unsupervised/`, un endpoint informativo y una pestaña dedicada en la interfaz web bajo el rótulo **"Próximamente: Aprendizaje No Supervisado (Siguiente Entrega)"**.
