"""
Definición de modelos: CNN puro y CRNN (contrapropuesta).

=== ARQUITECTURA CNN (Enfoque principal) ===

La CNN sigue el patrón clásico para clasificación de imágenes
adaptado a espectrogramas de audio:

    Input (128 × 1292 × 1)   ← Mel spectrogram como "imagen"
         │
         ▼
    ┌─────────────────────────────────┐
    │  Bloque Conv 1:                 │
    │  Conv2D(32, 3×3) → BatchNorm   │  ← Detecta patrones locales
    │  → ReLU → MaxPool(2×2)         │    (bordes, texturas de audio)
    └─────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────┐
    │  Bloque Conv 2:                 │
    │  Conv2D(64, 3×3) → BatchNorm   │  ← Combina patrones en
    │  → ReLU → MaxPool(2×2)         │    estructuras más complejas
    └─────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────┐
    │  Bloque Conv 3:                 │
    │  Conv2D(128, 3×3) → BatchNorm  │  ← Detecta motivos musicales
    │  → ReLU → MaxPool(2×2)         │    (patrones rítmicos, armónicos)
    └─────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────┐
    │  Bloque Conv 4:                 │
    │  Conv2D(256, 3×3) → BatchNorm  │  ← Features de alto nivel
    │  → ReLU → MaxPool(2×2)         │    (estilo, género)
    └─────────────────────────────────┘
         │
         ▼
    GlobalAveragePooling2D            ← Promedia cada feature map
         │                              en un solo valor (reduce
         ▼                              overfitting vs Flatten)
    Dense(256) → Dropout(0.5)
         │
         ▼
    Dense(128) → Dropout(0.5)
         │
         ▼
    Dense(10, softmax)                ← 10 géneros GTZAN

=== POR QUÉ CADA COMPONENTE ===

Conv2D: Aplica filtros que detectan patrones espaciales 2D.
    En un espectrograma, patrones 2D = combinaciones de
    frecuencia × tiempo. Un acorde es un patrón vertical
    (múltiples frecuencias simultáneas). Un ritmo es un
    patrón horizontal (energía repetida en el tiempo).

BatchNormalization: Normaliza las activaciones entre capas.
    Esto estabiliza el entrenamiento, permite learning rates
    más altos, y actúa como regularización ligera.

MaxPooling2D: Reduce dimensiones a la mitad, forzando al modelo
    a aprender representaciones más abstractas y reduciendo
    el costo computacional.

GlobalAveragePooling2D: En vez de Flatten (que crea un vector
    enorme y propenso a overfitting), GAP promedia cada feature
    map en un solo número. Esto reduce drásticamente los
    parámetros de la capa Dense siguiente.

Dropout: "Apaga" neuronas aleatorias durante training. Fuerza
    al modelo a no depender de neuronas específicas, mejorando
    generalización. 0.5 es agresivo pero adecuado para GTZAN
    dado su tamaño pequeño.

=== ARQUITECTURA CRNN (Contrapropuesta) ===

La CRNN es la contrapropuesta directa a la CNN pura:

    Input (128 × 1292 × 1)
         │
         ▼
    CNN Blocks (3 bloques)            ← Extrae features espaciales
         │
         ▼
    Reshape: (time_steps, features)   ← Reorganiza para la RNN
         │
         ▼
    Bidirectional GRU(128)            ← Captura dependencias
         │                              temporales en ambas
         ▼                              direcciones
    Dense(128) → Dropout(0.4)
         │
         ▼
    Dense(10, softmax)

=== CNN vs CRNN: LA COMPARACIÓN PARA TU TESIS ===

CNN pura:
  + Más simple de implementar y entrenar.
  + Menos parámetros → menos overfitting con pocos datos.
  + Buena para capturar texturas estáticas (timbre, armonía).
  - Pierde dependencias temporales largas.
  - Un acorde de jazz y uno de rock pueden verse iguales localmente.

CRNN:
  + Captura evolución temporal (intro → verso → coro).
  + Mejor para géneros que se distinguen por estructura temporal.
  + Bidireccional: "ve" el contexto futuro y pasado.
  - Más parámetros → mayor riesgo de overfitting.
  - Training más lento (RNN es secuencial, no paralelizable).
  - Más compleja de debuggear.

Hipótesis para tu tesis: la CNN pura debería superar 80% en GTZAN
porque las diferencias tímbricas entre géneros son fuertes. La CRNN
podría mejorar en géneros confusos (rock vs country, jazz vs blues)
donde la estructura temporal importa más.
"""
import logging

from config import CRNNConfig, ModelConfig

logger = logging.getLogger(__name__)


# ===================================================================
# SECCIÓN 1: Bloques reutilizables
# ===================================================================
# TODO: Implementar conv_block()
#   - Parámetros: input_tensor, filters, kernel_size, pool_size,
#     use_batch_norm
#   - Secuencia: Conv2D → [BatchNormalization] → Activation → MaxPool
#   - Usar padding='same' para mantener dimensiones antes del pool
#   - Retornar tensor de salida
#
# NOTA SOBRE PADDING:
#   'same' agrega ceros para que la salida de Conv2D tenga las
#   mismas dimensiones que la entrada. Esto simplifica el cálculo
#   de dimensiones y evita que se pierda información en los bordes.
#   'valid' no agrega padding y reduce dimensiones, lo que puede
#   ser útil pero requiere calcular dimensiones manualmente.


# ===================================================================
# SECCIÓN 2: Modelo CNN
# ===================================================================
# TODO: Implementar build_cnn_model()
#   - Recibe ModelConfig
#   - Construye secuencialmente usando conv_block()
#   - Agrega GlobalAveragePooling2D o Flatten según config
#   - Agrega capas Dense con Dropout
#   - Capa final: Dense(num_classes, activation='softmax')
#   - Retornar tf.keras.Model
#
# CONSIDERACIONES:
#   - Usar API Funcional de Keras (no Sequential) para flexibilidad
#   - Esto permite inspeccionar capas intermedias fácilmente
#   - model.summary() debe mostrar ~500K-2M parámetros para GTZAN
#   - Si tienes más de 5M parámetros, el modelo es muy grande


# ===================================================================
# SECCIÓN 3: Modelo CRNN (contrapropuesta)
# ===================================================================
# TODO: Implementar build_crnn_model()
#   - Recibe CRNNConfig
#   - Bloques CNN (menos profundos que la CNN pura)
#   - Reshape: después del último Conv+Pool, reorganizar el tensor
#     de (batch, height, width, channels) a (batch, width, height*channels)
#     para que cada "columna" del spectrogram sea un timestep
#   - Capa recurrente: GRU o LSTM según config
#   - Si bidirectional=True, envolver con Bidirectional()
#   - Capas Dense finales con Dropout
#   - Retornar tf.keras.Model
#
# DETALLE DEL RESHAPE:
#   Después de los bloques CNN, el tensor tiene shape:
#   (batch, freq_reduced, time_reduced, last_filters)
#
#   Para la RNN necesitamos: (batch, time_steps, features)
#   Entonces: Permute(2, 1, 3) para poner tiempo primero,
#   luego Reshape para combinar freq y channels en features.
#
# NOTA SOBRE GRU vs LSTM:
#   GRU tiene menos parámetros (2 gates vs 3 en LSTM).
#   Para datasets pequeños como GTZAN, GRU suele ser mejor
#   porque hay menos riesgo de overfitting.


# ===================================================================
# SECCIÓN 4: Utilidades del modelo
# ===================================================================
# TODO: Implementar get_model_summary()
#   - Recibe un modelo compilado
#   - Retorna string con el summary
#   - Útil para logging y para la documentación de tu tesis
#
# TODO: Implementar count_parameters()
#   - Total params, trainable params, non-trainable params
#   - Esto va directo a la tabla de tu tesis
#
# TODO: Implementar compile_model()
#   - Recibe modelo y TrainingConfig
#   - Configura optimizer (Adam con learning_rate)
#   - Configura loss (categorical_crossentropy)
#   - Configura metrics: accuracy + F1Score
#   - Para F1, puedes usar tfa.metrics.F1Score o implementar
#     un callback custom que lo calcule al final de cada epoch
