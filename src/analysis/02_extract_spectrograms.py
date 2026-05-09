# ==============================================================================
# Author: Luis Eduardo Polaco
# Description: Extraction and visualization of Mel spectrograms from GTZAN.
#
# Generates representative mel spectrograms per genre and comparative
# visualizations to support thesis section 2.6 (Musical traits analysis).
#
# Output: PNG files with spectrograms and feature distributions.
# ==============================================================================

import logging
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ProjectConfig

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


N_MELS = 128
FMAX = 8000
SR = 22050
HOP_LENGTH = 512
N_FFT = 2048
DPI = 100


def extract_representative_spectrograms(
    gtzan_path: Path, output_path: Path, sample_idx: str = "middle"
) -> None:
    """
    Extracts one representative mel spectrogram per genre.

    For each genre folder, selects a sample (middle by default) and
    generates its mel spectrogram visualization.

    Args:
        gtzan_path: Path to genres_original directory.
        output_path: Directory to save spectrogram PNGs.
        sample_idx: Selection strategy - 'middle' (default) or 'first'.
    """
    logger.info("=" * 70)
    logger.info("EXTRACTING REPRESENTATIVE SPECTROGRAMS PER GENRE")
    logger.info("=" * 70)

    for genre_dir in sorted(gtzan_path.iterdir()):
        if not genre_dir.is_dir():
            continue

        wav_files = sorted(list(genre_dir.glob("*.wav")))

        if not wav_files:
            logger.warning(f"No .wav files found in {genre_dir.name}")
            continue

        if sample_idx == "middle":
            selected_idx = len(wav_files) // 2
        else:
            selected_idx = 0

        selected_file = wav_files[selected_idx]
        genre_name = genre_dir.name

        logger.info(f"\n{genre_name.upper()}:")
        logger.info(f"  File: {selected_file.name}")

        try:
            y, sr = librosa.load(str(selected_file), sr=SR, mono=True)

            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, hop_length=HOP_LENGTH, n_fft=N_FFT
            )
            S_db = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(figsize=(12, 4))

            img = librosa.display.specshow(
                S_db,
                sr=sr,
                hop_length=HOP_LENGTH,
                x_axis="time",
                y_axis="mel",
                fmax=FMAX,
                ax=ax,
                cmap="viridis",
            )

            ax.set_title(
                f"Mel Spectrogram - {genre_name.capitalize()} ({selected_file.name})",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")

            fig.colorbar(img, ax=ax, format="%+2.0f dB")

            output_file = output_path / f"{genre_name}_example.png"
            plt.tight_layout()
            plt.savefig(str(output_file), dpi=DPI, bbox_inches="tight")
            plt.close()

            logger.info(f"  ✓ Saved: {output_file.name}")

        except Exception as e:
            logger.error(f"  ✗ Error processing {selected_file.name}: {e}")


def create_trait_comparisons(gtzan_path: Path, output_path: Path) -> None:
    """
    Creates comparative visualizations of genres for different musical traits.

    Comparisons:
    - RITMO: Blues (slow) vs Hip-Hop (fast)
    - TONO: Classical (multiple notes) vs Rock (direct)
    - TIMBRE: Jazz (smooth) vs Rock (raw)

    Args:
        gtzan_path: Path to genres_original directory.
        output_path: Directory to save comparison PNGs.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAIT COMPARISONS")
    logger.info("=" * 70)

    comparisons = {
        "ritmo": ["blues", "hiphop"],
        "tono": ["classical", "rock"],
        "timbre": ["jazz", "rock"],
    }

    for trait_name, genres in comparisons.items():
        logger.info(f"\nCreating comparison: {trait_name.upper()}")

        fig, axes = plt.subplots(1, len(genres), figsize=(14, 4))

        if len(genres) == 1:
            axes = [axes]

        for idx, genre in enumerate(genres):
            genre_path = gtzan_path / genre

            if not genre_path.exists():
                logger.warning(f"  ⚠ Genre directory not found: {genre}")
                continue

            wav_files = sorted(list(genre_path.glob("*.wav")))

            if not wav_files:
                logger.warning(f"  ⚠ No files in {genre}")
                continue

            selected_file = wav_files[len(wav_files) // 2]

            try:
                y, sr = librosa.load(str(selected_file), sr=SR, mono=True)
                S = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_mels=N_MELS,
                    fmax=FMAX,
                    hop_length=HOP_LENGTH,
                    n_fft=N_FFT,
                )
                S_db = librosa.power_to_db(S, ref=np.max)

                # Plot
                img = librosa.display.specshow(
                    S_db,
                    sr=sr,
                    hop_length=HOP_LENGTH,
                    x_axis="time",
                    y_axis="mel",
                    fmax=FMAX,
                    ax=axes[idx],
                    cmap="viridis",
                )

                axes[idx].set_title(f"{genre.capitalize()}", fontweight="bold")
                if idx == 0:
                    axes[idx].set_ylabel("Frequency (Hz)")

            except Exception as e:
                logger.error(f"  ✗ Error with {genre}: {e}")

        plt.suptitle(
            f"Trait Comparison - {trait_name.upper()}: {', '.join([g.capitalize() for g in genres])}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        output_file = output_path / f"comparison_{trait_name}.png"
        plt.savefig(str(output_file), dpi=DPI, bbox_inches="tight")
        plt.close()

        logger.info(f"  ✓ Saved: {output_file.name}")


def plot_feature_distributions(csv_path: Path, output_path: Path) -> None:
    """
    Creates box plots showing feature distributions across genres.

    Features visualized:
    - Tempo (BPM) - captures RITMO trait
    - Spectral Centroid (Hz) - captures TONO trait
    - RMS Energy - captures TIMBRE trait intensity
    - Zero Crossing Rate - captures BRILLO trait

    Args:
        csv_path: Path to directory containing features_30_sec.csv.
        output_path: Directory to save distribution PNGs.
    """
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE DISTRIBUTIONS PER GENRE")
    logger.info("=" * 70)

    csv_file = csv_path / "features_30_sec.csv"

    if not csv_file.exists():
        logger.error(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    features_config = [
        ("tempo", "Tempo (BPM)", "BPM"),
        ("spectral_centroid_mean", "Spectral Centroid (Hz)", "Hz"),
        ("rms_mean", "RMS Energy", "Energy"),
        ("zero_crossing_rate_mean", "Zero Crossing Rate", "ZCR"),
    ]

    for feature_col, title, ylabel in features_config:
        logger.info(f"\n  Plotting: {title}")

        fig, ax = plt.subplots(figsize=(12, 5))

        genres = sorted(df["label"].unique())
        data_by_genre = [df[df["label"] == g][feature_col].values for g in genres]

        bp = ax.boxplot(
            data_by_genre,
            labels=[g.capitalize() for g in genres],
            patch_artist=True,
        )

        colors = plt.cm.Set3(np.linspace(0, 1, len(genres)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title(f"{title} per Genre", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45)

        plt.tight_layout()

        output_file = output_path / f"distribution_{feature_col}.png"
        plt.savefig(str(output_file), dpi=DPI, bbox_inches="tight")
        plt.close()

        logger.info(f"    ✓ Saved: {output_file.name}")


def extract_spectrograms(config: ProjectConfig) -> None:
    """
    Full spectrogram extraction pipeline.

    Extracts representative spectrograms per genre → creates trait
    comparisons → plots feature distributions.

    Args:
        config: Project configuration containing paths.
    """
    gtzan_path = config.paths.raw_audio / "genres_original"
    output_path = config.paths.processed / "spectrograms"

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("\n")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " MEL SPECTROGRAM EXTRACTION AND VISUALIZATION ".center(68) + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info(f"\nMel Spectrogram Parameters:")
    logger.info(f"  N_MELS: {N_MELS}")
    logger.info(f"  FMAX: {FMAX} Hz")
    logger.info(f"  SR: {SR} Hz")
    logger.info(f"  HOP_LENGTH: {HOP_LENGTH}")
    logger.info(f"  N_FFT: {N_FFT}\n")

    if not gtzan_path.exists():
        logger.error(f"ERROR: {gtzan_path} does not exist")
        return

    if not (config.paths.raw_audio / "features_30_sec.csv").exists():
        logger.error(f"ERROR: features_30_sec.csv not found")
        return

    extract_representative_spectrograms(gtzan_path, output_path)
    create_trait_comparisons(gtzan_path, output_path)
    plot_feature_distributions(config.paths.raw_audio, output_path)

    logger.info("\n" + "=" * 70)
    logger.info("✓ EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nSpectrograms saved to: {output_path}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = ProjectConfig()
    extract_spectrograms(config)
