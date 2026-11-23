# III. Marco Teórico

## 1. Reconocimiento de Voz y Bioacústica Computacional

Dentro del aprendizaje automático, el área de **Reconocimiento de Voz** aborda la extracción de patrones acústicos en señales temporales de audio para identificar unidades lingüísticas, locutores o fuentes sonoras estructuradas. Como subproblema aplicado, el **reconocimiento acústico de especies de aves** comparte exactamente la misma base matemática de procesamiento digital de señales: la transformación de ondas de presión sonora unidimensionales en representaciones espectrales bidimensionales y su posterior clasificación mediante modelos profundos.

A diferencia del habla humana en entornos controlados, las vocalizaciones aviares presentan un rango frecuencial amplio (desde graves por debajo de 500 Hz en palomas como *Patagioenas plumbea* hasta armónicos agudos superiores a 8 kHz en paseriformes), superposición de ruido ambiental (viento, lluvia, insectos, cursos de agua) y gran variabilidad entre tipos de emisión (cantos territoriales o *song*, llamados cortos o *call*, dúos y subcantos).

## 2. Espectrogramas de Mel en Decibelios

Una señal de audio muestreada discreta $y[n]$ a frecuencia $f_s$ contiene información en el dominio del tiempo, pero oculta su estructura armónica. Para revelar cómo varía el contenido frecuencial a lo largo del tiempo se emplea la **Transformada de Fourier de Tiempo Corto (STFT)**, aplicando ventanas solapadas de longitud $N_{\text{fft}}$ con salto $H$:

$$X(m, k) = \sum_{n=0}^{N_{\text{fft}}-1} y[n + mH]\, w[n]\, e^{-j 2\pi k n / N_{\text{fft}}}$$

Posteriormente, el espectro de potencia $|X(m, k)|^2$ se proyecta sobre un banco de filtros triangulares espaciados en la **escala perceptual de Mel**, la cual ofrece mayor resolución en frecuencias bajas y medias (donde reside gran parte de la energía fundamental y armónica de las aves) y comprime frecuencias muy altas:

$$m(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

Finalmente, la energía en cada banda de Mel se convierte a escala logarítmica en **decibelios (dB)** respecto a la energía máxima de la ventana:

$$S_{\text{dB}}(m, b) = 10 \cdot \log_{10}\left(\frac{\max(\varepsilon, S_{\text{Mel}}(m, b))}{\max(S_{\text{Mel}})}\right)$$

El resultado es una matriz bidimensional tiempo-frecuencia (en este proyecto de dimensiones $64 \times 173$ para ventanas de $4\,\text{s}$ a $22{,}050\,\text{Hz}$), análoga a una imagen monocanal donde las tramas verticales representan pulsos breves y las trazas horizontales u oblicuas reflejan silbidos, trinos y armónicos.

## 3. Redes Neuronales Convolucionales (CNN 2D)

### 3.1 Fundamento y Principios Arquitectónicos

Las **Redes Neuronales Convolucionales (CNN)** fueron motivadas por los descubrimientos de Hubel y Wiesel sobre la organización jerárquica del córtex visual. Se trata de redes *feedforward* especializadas en procesar datos con topología de rejilla (imágenes o espectrogramas 2D) sin depender de una extracción manual rígida de descriptores. Sus dos pilares conceptuales son:

- **Campo receptivo local:** Las correlaciones locales entre celdas vecinas en tiempo y frecuencia son mucho más informativas que las relaciones entre puntos distantes del espectrograma. Cada neurona convolucional observa únicamente una vecindad local definida por el tamaño del *kernel* ($3 \times 3$), y las capas profundas combinan estos rasgos locales (bordes espectrales, armónicos cortos) en representaciones globales de un canto completo.
- **Compartición de parámetros:** Cada filtro/kernel utiliza exactamente los mismos pesos al deslizarse sobre toda la matriz tiempo-frecuencia. Esto reduce drásticamente el número de parámetros entrenables frente a un perceptrón multicapa tradicional y otorga invarianza parcial ante desplazamientos temporales del canto dentro de la ventana.

### 3.2 Componentes de la Arquitectura

1. **Capa convolucional (`Conv2d`):** Realiza una convolución multicanal deslizando $C_{\text{out}}$ filtros aprendibles sobre los mapas de entrada, sumando un sesgo (o normalizando mediante *Batch Normalization*) y aplicando una función de activación no lineal.
2. **Normalización por lotes (`BatchNorm2d` / `BatchNorm1d`):** Estandariza las activaciones intermedias por mini-lote, estabilizando el flujo de gradientes y permitiendo tasas de aprendizaje más eficientes.
3. **Función de activación ReLU:** Definida como $\text{ReLU}(z) = \max(0, z)$. Su derivada constante igual a $1$ en la región positiva evita la saturación que padecen funciones clásicas como la sigmoide.
4. **Capa de agrupamiento (*Pooling*):** Reduce progresivamente las dimensiones espaciales de los mapas de características. El proyecto combina **Max Pooling** ($2 \times 2$), que selecciona el valor máximo en cada subregión conservando las activaciones espectrales más intensas, y **Adaptive Average Pooling**, que resume cada canal en un escalar global antes del clasificador.
5. **Capa totalmente conectada y Softmax:** Al finalizar los bloques convolucionales, las características extraídas se vectorizan (*Flatten*) y atraviesan capas densas hasta producir un vector de *logits* $\mathbf{z} \in \mathbb{R}^K$ ($K = 39$ especies). La función **Softmax** transforma los *logits* en una distribución de probabilidad multiclase:

$$\sigma(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}, \qquad j = 1, \dots, K$$

## 4. Problemas Comunes en el Entrenamiento y Soluciones Aplicadas

En el desarrollo de redes neuronales profundas surgen tres problemas clásicos que condicionan la capacidad de generalización del modelo:

### 4.1 Desequilibrio de Datos

Ocurre cuando la cantidad de muestras varía entre categorías o subgrupos (por ejemplo, especies con 21 grabaciones frente a 27, y marcada asimetría en la calidad de grabación: 424 de calidad `A`, 444 de calidad `B` y apenas 105 de calidad `C`). Sin compensación, el modelo tiende a priorizar las clases o condiciones mayoritarias. Para mitigarlo, se implementa un **muestreo ponderado aleatorio (`WeightedRandomSampler`)** que equilibra la probabilidad de muestreo por clase y otorga un ligero refuerzo a grabaciones de calidad `C` para mejorar la tolerancia al ruido.

### 4.2 Desvanecimiento y Explosión de Gradiente

Al aumentar la profundidad de una red, la regla de la cadena multiplica derivadas locales desde la capa de salida hacia la entrada. Si se emplean activaciones sigmoideas cuya derivada máxima es $\sigma'(x) \le \frac{1}{4}$ y pesos $|w| < 1$, el producto acumulado decrece exponencialmente (*desvanecimiento*); si los pesos son grandes, crece sin control (*explosión*). En este sistema se previene mediante tres mecanismos complementarios:
- Uso de activaciones **ReLU** en todas las capas ocultas.
- Inclusión de **BatchNorm** tras cada convolución y capa densa intermedia.
- **Recorte de gradiente (*gradient clipping*)** con norma máxima $2.0$ antes de cada actualización de pesos del optimizador `AdamW`.

### 4.3 Sobreajuste (*Overfitting*)

Se manifiesta cuando el modelo memoriza los ejemplos y el ruido particular del conjunto de entrenamiento (alcanzando baja pérdida en *train*), pero pierde precisión al evaluar grabaciones nuevas en *validation* o *test*. Dado el tamaño acotado de grabaciones por especie frente a la dimensionalidad del espectrograma, se aplican simultáneamente las cuatro estrategias fundamentales de regularización:
1. **Aumento de datos (*Data Augmentation*):** Aplicado exclusivamente en entrenamiento mediante desplazamiento temporal circular, ajuste aleatorio de contraste/ganancia, ruido gaussiano moderado y **SpecAugment** (enmascaramiento de bandas de frecuencia y bloques temporales).
2. **Regularización de pesos y etiquetas:** Decaimiento de pesos L2 (`weight_decay = 1e-4`) en `AdamW` y *label smoothing* ($0.05$) en la pérdida de entropía cruzada.
3. **Dropout (`Dropout2d` y `Dropout`):** Desactivación aleatoria de mapas de características y neuronas densas durante el entrenamiento para evitar coadaptación excesiva.
4. **Detención temprana (*Early Stopping*):** Monitoreo continuo sobre el conjunto de validación para interrumpir el entrenamiento cuando deja de existir mejora y restaurar el mejor *checkpoint*.

## 5. Proyección Conceptual al Aprendizaje No Supervisado (Siguiente Entrega)

En concordancia con los lineamientos de la práctica, la solución global contempla complementar el clasificador supervisado con un segundo enfoque de **Aprendizaje No Supervisado**, reservado exclusivamente para la **segunda entrega**. A nivel conceptual, dicho enfoque buscará extraer representaciones latentes (*embeddings*) de los espectrogramas sin utilizar etiquetas de especie durante el ajuste, aplicar reducción de dimensionalidad y ejecutar algoritmos de agrupamiento (*clustering*) para descubrir similitudes acústicas entre cantos, llamados y condiciones de grabación. En esta primera entrega únicamente se deja reservada la estructura de directorio e interfaz correspondiente, sin entrenar ni ejecutar algoritmos no supervisados.
