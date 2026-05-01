"""
Módulo de carga y preprocesamiento de audio.

Este módulo se encarga de:
1. Cargar archivos de audio WAV desde el dataset.
2. Convertir el audio crudo a espectrogramas Mel.
3. Aplicar data augmentation (opcional).
4. Preparar los datos para alimentar la CNN.

=== FLUJO DE DATOS ===

    Audio WAV (30s, 22050 Hz)
         │
         ▼
    Normalización de longitud (padding/truncate)
         │
         ▼
    [Augmentation opcional: noise, stretch, shift]
         │
         ▼
    Mel Spectrogram (128 bandas × ~1292 frames)
         │
         ▼
    Normalización (0-1 o z-score)
         │
         ▼
    Tensor listo para CNN (128, 1292, 1)

=== POR QUÉ MEL SPECTROGRAMS ===

Un espectrograma Mel convierte audio en una "imagen" 2D donde:
- Eje Y = frecuencia (en escala Mel, que imita percepción humana)
- Eje X = tiempo
- Intensidad del pixel = energía en esa frecuencia/tiempo

Esto permite usar CNNs (diseñadas para imágenes) directamente
sobre representaciones de audio. La escala Mel comprime frecuencias
altas y expande frecuencias bajas, alineándose con cómo el oído
humano percibe la música.

=== SOBRE DATA AUGMENTATION ===

GTZAN solo tiene 1000 muestras (100 por género). Esto es MUY poco
para deep learning. Data augmentation genera variaciones artificiales
del audio para que el modelo vea más diversidad sin necesitar más
datos reales. Las técnicas usadas:

- Ruido gaussiano: simula grabaciones en diferentes condiciones.
- Time stretch: cambia velocidad sin cambiar pitch.
- Pitch shift: cambia pitch sin cambiar velocidad.
- Time shift: desplaza el audio en el tiempo (circular).
"""
import logging
from pathlib import Path

import librosa
import numpy as np

from config import AudioConfig, AugmentationConfig, ProjectConfig

logger = logging.getLogger(__name__)


# ===================================================================
# SECCIÓN 1: Carga de audio
# ===================================================================
# TODO: Implementar load_audio_file()
#   - Usar librosa.load() con sr=config.audio.sample_rate
#   - Manejar archivos corruptos con try/except
#   - Retornar np.ndarray con la señal de audio
#
# TODO: Implementar normalize_length()
#   - Si el audio es más corto que target_samples: pad con ceros
#   - Si es más largo: truncar
#   - Esto garantiza que todos los audios tengan la misma longitud


# ===================================================================
# SECCIÓN 2: Data augmentation
# ===================================================================
# TODO: Implementar add_noise()
#   - Generar ruido gaussiano: np.random.randn(len(audio))
#   - Escalar por noise_factor y sumar al audio
#   - Aplicar con probabilidad noise_probability
#
# TODO: Implementar time_stretch()
#   - Usar librosa.effects.time_stretch()
#   - Rate aleatorio en time_stretch_range
#   - Re-normalizar longitud después del stretch
#
# TODO: Implementar pitch_shift()
#   - Usar librosa.effects.pitch_shift()
#   - Steps aleatorio en pitch_shift_range
#
# TODO: Implementar time_shift()
#   - np.roll() para desplazar circularmente
#   - Shift aleatorio en milisegundos → muestras
#
# TODO: Implementar apply_augmentation()
#   - Recibe audio y AugmentationConfig
#   - Aplica cada transformación según su probabilidad
#   - Retorna audio aumentado


# ===================================================================
# SECCIÓN 3: Extracción de features
# ===================================================================
# TODO: Implementar audio_to_mel_spectrogram()
#   - Usar librosa.feature.melspectrogram()
#   - Parámetros desde AudioConfig: sr, n_mels, hop_length, n_fft
#   - Convertir a escala logarítmica con librosa.power_to_db()
#   - Retornar np.ndarray de shape (n_mels, frames)
#
# TODO: Implementar normalize_spectrogram()
#   - Opción 1 (recomendada): min-max a rango [0, 1]
#   - Opción 2: z-score (media=0, std=1)
#   - La normalización es CRUCIAL para convergencia del training


# ===================================================================
# SECCIÓN 4: Pipeline completo
# ===================================================================
# TODO: Implementar process_single_file()
#   - Carga audio → normaliza longitud → [augmentation] →
#     mel spectrogram → normaliza → reshape para CNN
#   - Retorna tensor de shape (n_mels, frames, 1)
#
# TODO: Implementar load_dataset()
#   - Recorre el directorio del dataset (estructura: genero/archivo.wav)
#   - Procesa cada archivo
#   - Retorna X (spectrogramas), y (etiquetas), label_map (dict)
#   - Mostrar progreso con tqdm o logging
#
# TODO: Implementar save_processed_data()
#   - Guardar X, y como .npy para no reprocesar cada vez
#   - Incluir label_map como JSON


# ===================================================================
# SECCIÓN 5: Generador de datos (opcional pero recomendado)
# ===================================================================
# TODO: Implementar DataGenerator (hereda de tf.keras.utils.Sequence)
#   - Genera batches on-the-fly para no cargar todo en RAM
#   - Aplica augmentation en tiempo real durante training
#   - Métodos: __init__, __len__, __getitem__, on_epoch_end
#   - Esto es especialmente útil si cambias a un dataset más grande
