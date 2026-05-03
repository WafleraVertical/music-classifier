# ==============================================================================
# Author: Luis Eduardo Polaco
# Description: Data loading and preprocessing pipeline for audio classification.
#
# Handles audio loading, length normalization, mel spectrogram extraction,
# data augmentation (noise, time stretch, pitch shift, time shift),
# and dataset preparation for CNN input.
#
# Pipeline: WAV → normalize length → [augmentation] → mel spectrogram → normalize → (n_mels, frames, 1)
# ==============================================================================

import json
import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from tqdm import tqdm

from config import ProjectConfig

logger = logging.getLogger(__name__)


# ===================================================================
# SECCIÓN 1: Carga de audio
# ===================================================================


def load_audio_file(file_path: Path, config: ProjectConfig) -> np.ndarray | None:
    """
    Loads audio from file and converts it to numpy array for audio processing.

    Args:
        file_path: The path of the audio file.
        config: The project configuration.

    Returns:
        y: Audio time series as np.ndarray, or None if loading fails.
    """
    try:
        y, _ = librosa.load(
            file_path,
            sr=config.audio.sample_rate,
            duration=config.audio.duration,
        )
        return y

    except Exception as e:
        logger.error(f"Something weird happened with {file_path}: {e}")
        return None


def normalize_length(y: np.ndarray, config: ProjectConfig) -> np.ndarray:
    """
    Normalizes the audio time series length to the target samples defined in config.

    If the audio is longer than target, it gets truncated.
    If shorter, it gets zero-padded at the end (silence).

    Args:
        y: The audio time series.
        config: The project configuration.

    Returns:
        y: The length-normalized audio time series of shape (target_samples,).
    """
    if len(y) > config.audio.target_samples:
        y = y[: config.audio.target_samples]
    elif len(y) < config.audio.target_samples:
        y = np.pad(y, (0, config.audio.target_samples - len(y)), "constant")

    return y


# ===================================================================
# SECCIÓN 2: Data augmentation
# ===================================================================


def add_noise(y: np.ndarray, config: ProjectConfig) -> np.ndarray:
    """
    Adds gaussian noise to the audio time series with a given probability.

    Args:
        y: The audio time series.
        config: The project configuration.

    Returns:
        y: The audio time series with noise added (or unchanged).
    """
    if np.random.rand() < config.augmentation.noise_probability:
        y = y + config.augmentation.noise_factor * np.random.normal(size=y.shape)

    return y


def time_stretch(y: np.ndarray, config: ProjectConfig) -> np.ndarray:
    """
    Applies a random time stretch to the audio time series with a given probability.

    Args:
        y: The audio time series.
        config: The project configuration.

    Returns:
        y: The audio time series with time stretch applied (or unchanged).
    """
    if np.random.rand() < config.augmentation.time_stretch_probability:
        rate = np.random.uniform(
            config.augmentation.time_stretch_range[0],
            config.augmentation.time_stretch_range[1],
        )
        y = librosa.effects.time_stretch(y, rate=rate)
        y = normalize_length(y, config)

    return y


def pitch_shift(y: np.ndarray, config: ProjectConfig) -> np.ndarray:
    """
    Applies a random pitch shift to the audio time series with a given probability.

    Args:
        y: The audio time series.
        config: The project configuration.

    Returns:
        y: The audio time series with pitch shift applied (or unchanged).
    """
    if np.random.rand() < config.augmentation.pitch_shift_probability:
        steps = np.random.randint(
            config.augmentation.pitch_shift_range[0],
            config.augmentation.pitch_shift_range[1],
        )
        y = librosa.effects.pitch_shift(
            y,
            sr=config.audio.sample_rate,
            n_steps=steps,
        )

    return y


def time_shift(y: np.ndarray, config: ProjectConfig) -> np.ndarray:
    """
    Applies a random circular time shift to the audio time series with a given probability.

    Args:
        y: The audio time series.
        config: The project configuration.

    Returns:
        y: The audio time series with time shift applied (or unchanged).
    """
    if np.random.rand() < config.augmentation.time_shift_probability:
        samples = int(
            (config.augmentation.time_shift_max_ms / 1000) * config.audio.sample_rate
        )
        shift = np.random.randint(-samples, samples)
        y = np.roll(y, shift)

    return y


def apply_augmentation(y: np.ndarray, config: ProjectConfig) -> np.ndarray:
    """
    Orchestrator that applies multiple augmentations to the audio time series.

    Only runs if config.augmentation.enable is True.

    Args:
        y: The audio time series.
        config: The project configuration.

    Returns:
        y: The audio time series with augmentations applied (or unchanged).
    """
    if config.augmentation.enable:
        y = add_noise(y, config)
        y = time_stretch(y, config)
        y = pitch_shift(y, config)
        y = time_shift(y, config)

    return y


# ===================================================================
# SECCIÓN 3: Extracción de features
# ===================================================================


def audio_to_mel_spectrogram(y: np.ndarray, config: ProjectConfig) -> np.ndarray:
    """
    Converts audio time series to a Mel spectrogram in logarithmic scale.

    Args:
        y: The audio time series.
        config: The project configuration.

    Returns:
        mel_spec: np.ndarray of shape (n_mels, frames) in dB scale.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=config.audio.sample_rate,
        n_mels=config.audio.n_mels,
        hop_length=config.audio.hop_length,
        n_fft=config.audio.n_fft,
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    Applies min-max scaling to a spectrogram, mapping values to [0, 1].

    Args:
        spectrogram: np.ndarray of shape (n_mels, frames).

    Returns:
        normalized_spectrogram: np.ndarray of shape (n_mels, frames) with values in [0, 1].
    """
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)

    normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val + 1e-8)

    return normalized_spectrogram


# ===================================================================
# SECCIÓN 4: Pipeline completo
# ===================================================================


def process_single_file(
    file_path: Path,
    config: ProjectConfig,
    augment: bool = False,
) -> np.ndarray | None:
    """
    Full preprocessing pipeline for a single audio file.

    Loads audio → normalizes length → [augmentation] →
    mel spectrogram → normalizes → adds channel dimension.

    Args:
        file_path: Path to the audio file.
        config: The project configuration.
        augment: Whether to apply data augmentation.

    Returns:
        Tensor of shape (n_mels, frames, 1) ready for CNN, or None if loading fails.
    """
    audio = load_audio_file(file_path, config)
    if audio is None:
        return None

    audio = normalize_length(audio, config)

    if augment:
        audio = apply_augmentation(audio, config)

    spectrogram = audio_to_mel_spectrogram(audio, config)
    spectrogram = normalize_spectrogram(spectrogram)
    spectrogram = np.expand_dims(spectrogram, axis=-1)

    return spectrogram


# ===================================================================
# SECCIÓN 4 (continuación): Carga del dataset completo
# ===================================================================
# TODO: Implementar load_dataset()
#   - Recorre el directorio del dataset (estructura: genero/archivo.wav)
#   - Procesa cada archivo
#   - Retorna X (spectrogramas), y (etiquetas), label_map (dict)
#   - Mostrar progreso con tqdm o logging


def load_dataset(config: ProjectConfig) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Loads the dataset dynamically by iterating through genre folders.

    Args:
        config (ProjectConfig): Project configuration containing paths and parameters.

    Returns:
        Tuple containing:
        - X_array (np.ndarray): Array of spectrograms of shape (samples, n_mels, frames, 1).
        - y_array (np.ndarray): Array of integer labels of shape (samples,).
        - label_map (dict): Dictionary mapping genre string names to integer indices.
    """
    X = []
    y = []

    base_path = config.paths.raw_audio / "genres_original"

    genre_folders = sorted(
        [folder for folder in base_path.iterdir() if folder.is_dir()]
    )
    label_map = {folder.name: idx for idx, folder in enumerate(genre_folders)}

    logger.info(f"Detected classes: {label_map}")

    for folder in genre_folders:
        genre_name = folder.name
        class_index = label_map[genre_name]

        wav_files = list(folder.glob("*.wav"))

        for file_path in tqdm(wav_files, desc=f"Processing {genre_name}"):

            spectrogram = process_single_file(file_path, config, augment=False)

            if spectrogram is not None:
                X.append(spectrogram)
                y.append(class_index)

    logger.info("Converting lists to numpy arrays...")
    X_array = np.array(X)
    y_array = np.array(y)
    logger.info(f"Dataset loaded: {X_array.shape[0]} samples, shape {X_array.shape}")

    return X_array, y_array, label_map


def save_processed_data(
    X_array: np.ndarray, y_array: np.ndarray, label_map: dict, config: ProjectConfig
):
    """
    Saves processed data to disk.

    Args:
        X_array: Tensor of shape (n_mels, frames, 1).
        y_array: List of integer labels of shape (samples,).
        label_map: Tensor of integer labels of shape (samples,).
        config: The project configuration.

    """
    np.save(config.paths.processed / "X.npy", X_array)
    np.save(config.paths.processed / "y.npy", y_array)

    with open(config.paths.processed / "label_map.json", "w") as f:
        json.dump(label_map, f)

    logger.info(f"Processed data saved to {config.paths.processed}")


# ===================================================================
# SECCIÓN 5: Generador de datos (opcional pero recomendado)
# ===================================================================
# TODO: Implementar DataGenerator (hereda de tf.keras.utils.Sequence)
#   - Genera batches on-the-fly para no cargar todo en RAM
#   - Aplica augmentation en tiempo real durante training
#   - Métodos: __init__, __len__, __getitem__, on_epoch_end
#   - Esto es especialmente útil si cambias a un dataset más grande
