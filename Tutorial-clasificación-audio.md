# Tutorial TFG: Clasificación de Audio - De Baseline a Transformers

## 1. Introducción

Este tutorial describe paso a paso el desarrollo de un sistema de clasificación de audio orientado a un Trabajo de Fin de Grado (TFG). Se cubre desde un enfoque baseline clásico hasta el fine-tuning de modelos basados en Transformers para detectar patologías respiratorias.

## 2. Preprocesamiento

El preprocesamiento es crítico en señales biomédicas debido al ruido ambiental, variabilidad en equipos de grabación y artefactos.

### Pasos principales

**Conversión a mono:** Los estetoscopios digitales suelen grabar en mono, pero si hay grabaciones estéreo, se unifican a un solo canal para simplificar el procesamiento.

**Sample rate fijo (16 kHz):** Frecuencia estándar que captura adecuadamente los sonidos respiratorios (rango típico: 100-2000 Hz). Algunos trabajos usan 4-8 kHz, pero 16 kHz es compatible con modelos preentrenados.

**Normalización:** Ajusta diferencias de volumen entre grabaciones realizadas con diferentes equipos o posiciones del estetoscopio.

### Técnicas específicas por ejemplo para dominio médico 

**Filtrado de ruido cardíaco:** Los latidos pueden interferir con el análisis respiratorio. Se aplican filtros paso-banda (100-2000 Hz) o técnicas de separación de fuentes.

**Segmentación de ciclos respiratorios:** Dividir grabaciones largas en ciclos individuales (inspiración-espiración) para análisis más preciso. Requiere detección de inicio/fin de cada ciclo.

**Eliminación de silencios:** Recortar segmentos sin actividad respiratoria reduce ruido y enfoca el modelo en información relevante.

**Data augmentation médicamente válido:**

- Time stretching leve (±10%) para simular diferentes frecuencias respiratorias
- Pitch shifting mínimo (±2 semitonos) sin alterar características diagnósticas
- Adición de ruido ambiental realista (conversaciones, roce de ropa)
- Evitar transformaciones que distorsionen patrones clínicos

**Consideraciones importantes:**

- No aplicar augmentation agresivo que cree artefactos irreales
- Consultar literatura médica sobre rangos aceptables de variación
- Validar con expertos clínicos que las transformaciones preservan información diagnóstica

**Librería recomendada:** [Librosa](https://librosa.org/doc/latest/index.html)

**Herramientas adicionales:**

- [SciPy](https://scipy.org/) para filtrado de señales
- [PyDub](https://github.com/jiaaro/pydub) para manipulación básica
- [Essentia](https://essentia.upf.edu/) para análisis avanzado

## 3. Baseline: Features + Machine Learning

El enfoque baseline extrae características acústicas diseñadas manualmente y utiliza algoritmos de ML tradicionales. Este paso es fundamental para establecer un rendimiento mínimo aceptable.

### ¿Por qué empezar con un baseline?

- **Referencia de comparación:** Establece el piso de rendimiento que métodos avanzados deben superar
- **Interpretabilidad:** Las características tienen significado clínico directo
- **Recursos computacionales:** Entrenamiento rápido en CPU, ideal para iteración inicial
- **Validación del dataset:** Si el baseline falla, puede indicar problemas en los datos
- **Publicaciones médicas:** Muchos trabajos clínicos aún usan estos métodos

### Características acústicas relevantes

**MFCCs (Mel-Frequency Cepstral Coefficients):** Representación compacta del espectro de potencia. Ampliamente utilizados en clasificación de sonidos respiratorios. Se extraen típicamente 13-40 coeficientes.

**Características espectrales:**

- **Spectral Centroid:** Indica el "centro de gravedad" del espectro, útil para diferenciar sibilancias
- **Spectral Rolloff:** Frecuencia bajo la cual está el 85% de la energía
- **Spectral Flux:** Cambios temporales en el espectro, detecta transiciones en ciclos respiratorios
- **Zero Crossing Rate:** Tasa de cruces por cero, relacionada con contenido de altas frecuencias

**Características temporales:**

- **Energía RMS:** Potencia de la señal, varía entre inspiración/espiración
- **Duración de eventos:** Tiempo de inspiración vs espiración
- **Autocorrelación:** Periodicidad de patrones respiratorios

**Características específicas de dominio médico:**

- **Frecuencia dominante:** Identifica tonos de sibilancias (típicamente 400-1600 Hz)
- **Ancho de banda espectral:** Dispersión del espectro de frecuencias
- **Kurtosis espectral:** Forma de la distribución espectral

### Modelos comunes

**SVM (Support Vector Machine):** Efectivo en espacios de alta dimensionalidad con datasets pequeños. Kernel RBF suele funcionar bien. Buena generalización con pocos datos.

**Random Forest:** Robusto ante overfitting, proporciona importancia de características. Útil para identificar qué features son más discriminativas clínicamente.

**Logistic Regression:** Modelo simple y rápido, buena baseline inicial. Ofrece probabilidades calibradas útiles en contexto médico.

**Gradient Boosting (XGBoost/LightGBM):** A menudo superior a Random Forest en tablas de features. Requiere tuning cuidadoso de hiperparámetros.

### Estrategia de validación

**K-Fold Cross-Validation estratificado:** Asegura proporción de clases en cada fold. Típicamente k=5 o k=10.

**Leave-One-Subject-Out (LOSO):** Evalúa generalización a nuevos pacientes. Más realista clínicamente pero más exigente.

**Validación temporal:** Si hay grabaciones longitudinales, entrenar con datos antiguos y validar con recientes.

### Este baseline sirve como referencia mínima

Un accuracy del 70-80% en clasificación binaria (normal/patológico) es típico en literatura médica con features tradicionales. Métodos más avanzados deben demostrar mejora estadísticamente significativa.

**Recursos:**

- [Tutorial Scikit-learn](https://scikit-learn.org/stable/tutorial/index.html)
- [Guía sobre feature engineering en audio](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)

## 4. Deep Learning con CNNs

El audio se transforma en representaciones visuales (espectrogramas) y se utilizan Convolutional Neural Networks para aprendizaje automático de patrones.

### ¿Por qué Deep Learning?

**Ventajas:**

- **Aprendizaje automático de patrones:** No requiere diseño manual de características, la red descubre representaciones óptimas
- **Captura de dependencias complejas:** Detecta relaciones no lineales entre frecuencias y tiempo
- **Escalabilidad:** Mejora con más datos, a diferencia de métodos tradicionales
- **Transfer learning:** Aprovecha modelos preentrenados en grandes datasets

**Limitaciones:**

- **Necesidad de mayor cantidad de datos:** Típicamente requiere cientos o miles de ejemplos
- **Caja negra:** Menor interpretabilidad que características explícitas
- **Recursos computacionales:** Requiere GPU para entrenamiento eficiente
- **Riesgo de overfitting:** Especialmente con datasets médicos pequeños

### Representaciones visuales del audio

**Mel-Spectrograma:** Representación tiempo-frecuencia con escala Mel (perceptualmente motivada). Visualiza la energía en diferentes frecuencias a lo largo del tiempo. Es la entrada más común para CNNs de audio.

**Espectrograma STFT:** Transformada de Fourier de corto tiempo. Más resolución frecuencial que Mel, pero más grande.

**Chromagrama:** Representa contenido tonal, menos usado en respiración pero útil si hay componentes armónicos.

**Scalograma (CWT):** Transformada Wavelet Continua, buena resolución tiempo-frecuencia adaptativa.

### Arquitecturas CNN populares

**CNN Custom:** Arquitecturas diseñadas específicamente para espectrogramas de audio respiratorio. Típicamente 3-5 capas convolucionales con pooling progresivo.

**VGG adaptado:** Arquitectura profunda con filtros 3×3. Pre-entrenada en ImageNet y fine-tuneada en espectrogramas médicos.

**ResNet:** Conexiones residuales permiten redes más profundas sin degradación. ResNet-18 o ResNet-50 son populares.

**EfficientNet:** Balance óptimo entre precisión y eficiencia computacional. Escalado compuesto de profundidad, anchura y resolución.

**MobileNet:** Arquitectura ligera para deployment en dispositivos móviles. Útil para aplicaciones de telemedicina.

### Estrategias para datasets pequeños

**Transfer Learning:** Usar modelos preentrenados en ImageNet y fine-tunear solo las últimas capas. Reduce dramáticamente la necesidad de datos.

**Data Augmentation:** Aumentar variabilidad sintética (time stretching, mixup, SpecAugment). Debe ser clínicamente válido.

**Regularización:** Dropout, weight decay, batch normalization para prevenir overfitting.

**Ensembles:** Combinar predicciones de múltiples modelos entrenados con diferentes semillas aleatorias.

**Few-shot Learning:** Técnicas como Siamese Networks o Prototypical Networks para aprender con pocos ejemplos por clase.

### Consideraciones específicas para audio médico

- **Resolución temporal adecuada:** Capturar eventos breves como crepitantes (50-200 ms)
- **Rango de frecuencias relevante:** Enfocarse en 100-2000 Hz donde ocurren sonidos respiratorios
- **Balanceo de clases:** Datasets médicos suelen estar desbalanceados (más normales que patológicos)
- **Validación cruzada por paciente:** Evitar que muestras del mismo paciente estén en train y test

**Recursos:**

- [Tutorial CNN para clasificación de audio](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [SpecAugment paper](https://arxiv.org/abs/1904.08779)

## 5. Transformers para Audio

Los modelos Transformer permiten modelar dependencias temporales largas mediante mecanismos de atención, superando limitaciones de CNNs y RNNs.

### ¿Por qué Transformers en audio médico?

**Modelado de contexto global:** A diferencia de CNNs que tienen campos receptivos locales, los Transformers pueden relacionar cualquier parte de la señal con otra, capturando patrones respiratorios complejos que se extienden en toda la grabación.

**Atención selectiva:** El mecanismo de atención aprende automáticamente qué partes del audio son más relevantes para el diagnóstico (por ejemplo, focalizar en sibilancias o crepitantes).

**Transfer learning efectivo:** Modelos preentrenados en millones de horas de audio genérico aprenden representaciones robustas que se transfieren bien a dominios específicos con fine-tuning mínimo.

**State-of-the-art:** Dominan benchmarks actuales de clasificación de audio en múltiples dominios.

### Modelos Transformer para audio

**[Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2):** Desarrollado por Meta/Facebook. Preentrenado mediante aprendizaje auto-supervisado en audio crudo (sin etiquetas). Aprende representaciones contextuales de la forma de onda directamente. Ideal para señales donde la información temporal es crítica.

**[HuBERT](https://huggingface.co/docs/transformers/model_doc/hubert) (Hidden-Unit BERT):** Similar a Wav2Vec2 pero con estrategia de enmascaramiento diferente. Preentrenado en Librispeech (habla). Excelente para audio con estructura temporal compleja.

**[Wav2Vec2-BERT](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert):** Versión mejorada que combina ideas de Wav2Vec2 y W2V-BERT. Mejor rendimiento en tareas downstream. Arquitectura más moderna y eficiente.

**[Audio Spectrogram Transformer (AST)](https://github.com/YuanGongND/ast):** Procesa espectrogramas directamente (no forma de onda). Basado en Vision Transformer (ViT). Preentrenado en AudioSet (2M+ videos de YouTube). Captura tanto dependencias temporales como espectrales.

**[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper):** Modelo de OpenAI preentrenado en 680,000 horas de audio multilingüe. Originalmente para transcripción, pero sus encoders son útiles para clasificación.

### ¿Cuál elegir para audio respiratorio?

**Wav2Vec2:** Primera opción para señales respiratorias por:

- Procesa audio crudo sin necesidad de espectrogramas
- Captura detalles temporales finos (crepitantes de 50-200 ms)
- Modelos preentrenados disponibles en múltiples idiomas/dominios
- Arquitectura probada en aplicaciones médicas

**AST:** Alternativa si prefieres trabajar con espectrogramas o tienes experiencia previa con imágenes. Útil si combinas información visual con audio.

**HuBERT:** Similar a Wav2Vec2, puede ofrecer ligera mejora según el dataset.

### Se emplea fine-tuning sobre modelos preentrenados

**Ventaja clave:** Los modelos ya entrenados en millones de horas de audio tienen conocimiento general sobre estructuras acústicas. El fine-tuning adapta este conocimiento a la tarea específica de clasificación respiratoria con relativamente pocos ejemplos.

**Estrategia típica:**

- Congelar capas iniciales (mantienen representaciones generales)
- Entrenar solo capas finales y capa de clasificación
- Opcionalmente, descongelar gradualmente más capas si hay suficientes datos

**Recursos:**

- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction)
- [Tutorial: Fine-tuning para clasificación de audio](https://huggingface.co/docs/transformers/tasks/audio_classification)

## 6. Fine-tuning de Wav2Vec2

Proceso detallado para adaptar un modelo Wav2Vec2 preentrenado a tu dataset de sonidos respiratorios.

### Selección del modelo base

**[facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base):** 95M parámetros, preentrenado en 960h de Librispeech. Buen balance velocidad/rendimiento.

**[facebook/wav2vec2-large-robust](https://huggingface.co/facebook/wav2vec2-large-robust):** 317M parámetros, más robusto ante ruido. Recomendado si hay variabilidad en calidad de grabaciones.

**[ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition):** Ya fine-tuneado en reconocimiento de emociones, puede transferir bien a patrones respiratorios.

**Modelos específicos de dominio:** Buscar en Hugging Face Hub modelos fine-tuneados en audio médico si existen.

### Pasos del fine-tuning

**1. Cargar modelo preentrenado:** Descargar el modelo base desde Hugging Face Hub. Se cargan pesos de todas las capas Transformer pero se reemplaza el clasificador final.

**2. Ajustar capa de clasificación:** Añadir una nueva capa densa que mapea las representaciones de Wav2Vec2 (típicamente 768 o 1024 dimensiones) al número de clases de tu problema (por ejemplo, 4 clases: normal, asma, EPOC, neumonía).

**3. Preparar dataset:** Convertir archivos de audio al formato esperado (arrays NumPy de 16kHz). Crear labels numéricos para cada clase. Dividir en train/validation/test manteniendo pacientes separados.

**4. Configurar hiperparámetros:** Learning rate bajo (1e-5 a 5e-5), batch size pequeño (8-16 por limitaciones de memoria GPU), epochs limitados (10-30).

**5. Entrenar con learning rate bajo:** Tasa de aprendizaje reducida evita destruir representaciones preentrenadas. Usar learning rate scheduler (linear decay o cosine annealing).

**6. Monitorización:** Tracking de métricas en validation set cada epoch. Early stopping si no hay mejora en 3-5 epochs. Guardar mejor modelo según F1-score o AUC-ROC.

**7. Evaluación final:** Test en conjunto separado nunca visto durante entrenamiento. Análisis de errores y matriz de confusión.

### Optimización y técnicas avanzadas

**Gradient Accumulation:** Si la GPU tiene poca memoria, acumular gradientes de varios mini-batches antes de actualizar pesos.

**Mixed Precision Training:** Usar FP16 en lugar de FP32 reduce uso de memoria y acelera entrenamiento.

**Layer-wise Learning Rate Decay:** Aplicar learning rates progresivamente menores a capas más profundas (las más generales).

**Weighted Loss:** Si hay desbalanceo de clases, ponderar la función de pérdida inversamente proporcional a la frecuencia de cada clase.

**Ensemble de modelos:** Entrenar múltiples Wav2Vec2 con diferentes seeds y promediar predicciones.

## 7. Evaluación

Métricas utilizadas para evaluar y comparar todos los enfoques de forma rigurosa.

### Métricas básicas

**Accuracy (Exactitud):** Porcentaje de predicciones correctas sobre el total. Útil solo si las clases están balanceadas. En datasets médicos desbalanceados puede ser engañosa.

**Precision (Precisión):** De todos los casos que el modelo predijo como positivos, ¿cuántos realmente lo eran? Crítica cuando los falsos positivos tienen consecuencias graves (pruebas innecesarias, ansiedad del paciente).

**Recall (Sensibilidad/Exhaustividad):** De todos los casos realmente positivos, ¿cuántos detectó el modelo? Crítica cuando los falsos negativos son peligrosos (pacientes enfermos no detectados).

**F1-Score:** Media armónica de Precision y Recall. Balance entre ambas métricas. Útil cuando necesitas un único número para optimización.