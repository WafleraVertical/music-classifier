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

from music_classifier.config import AudioConfig

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_audio_file(file_path: Path, config: AudioConfig) -> np.ndarray | None:
    """
    Loads audio from file and convers it to numpy array for audio processing.

    Args:
        file_path: The path of the audio file.
        config: The configuration of the audio file.

    Returns:
        y: np.ndarray: Audio time series.
    """
    try:
        y, _ = librosa.load(file_path, sr=config.sample_rate, duration=config.duration)
        return y

    except Exception as e:
        logger.error(f"Something weird happened with {file_path}: {e}")
        return None


def normalize_length(y: np.ndarray, config: AudioConfig) -> np.ndarray:
    """
    This method normalizes the audio time series to the Audio config sample target defined.

    Args:
        y: The audio time series.
        config: The configuration of the audio file.

    Returns:
        y: The normalized audio time series.
    """
    if len(y) > config.target_samples:
        y = y[: config.target_samples]
    elif len(y) < config.target_samples:
        y = np.pad(y, (0, config.target_samples - len(y)), "constant")

    return y


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


def audio_to_mel_spectrogram(y: np.ndarray, config: AudioConfig) -> np.ndarray:
    """
    Create the mel spectrogram using the audio time series and process to a logarithmic scale.

    Args:
        y: The audio time series.
        config: The configuration of the audio file.

    Returns:
        mel_spec: np.ndarray  (n_mels,frames) of the mel spectrogram on a logarithmic scale.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
    )
    mel_spec = librosa.power_to_db(y, ref=np.max)

    return mel_spec


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
