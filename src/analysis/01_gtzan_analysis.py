# ==============================================================================
# Author: Luis Eduardo Polaco
# Description: Analysis and validation of GTZAN dataset for thesis documentation.
#
# This script validates GTZAN structure, detects duplicates (Sturm 2013),
# and extracts feature statistics per genre to support thesis sections 2.6.
#
# Output: CSV files with summary statistics and feature characterization.
# ==============================================================================

import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
from tqdm import tqdm

from src.config import ProjectConfig

logger = logging.getLogger(__name__)


def validate_gtzan_structure(base_path: Path) -> Dict[str, int]:
    """
    Validates that GTZAN has the expected structure:
    - 10 genre folders
    - ~100 .wav files per genre

    Args:
        base_path: Path to genres_original directory.

    Returns:
        Dictionary mapping genre names to file counts.
    """
    logger.info("=" * 70)
    logger.info("GTZAN STRUCTURE VALIDATION")
    logger.info("=" * 70)

    genres_found = {}

    for genre_dir in sorted(base_path.iterdir()):
        if not genre_dir.is_dir():
            continue

        wav_files = list(genre_dir.glob("*.wav"))
        genres_found[genre_dir.name] = len(wav_files)

        logger.info(f"✓ {genre_dir.name:15s} : {len(wav_files):3d} files")

    logger.info(f"\nTotal genres: {len(genres_found)}")
    logger.info(f"Total files: {sum(genres_found.values())}")

    return genres_found


def find_duplicate_files(base_path: Path) -> Dict:
    """
    Detects duplicate audio files using MD5 hash of first 10KB.

    Sturm (2013) documented that GTZAN contains duplicate tracks,
    which can bias model evaluation if not detected.

    Args:
        base_path: Path to genres_original directory.

    Returns:
        Dictionary with duplicate groups (hash -> list of file paths).
    """
    logger.info("\n" + "=" * 70)
    logger.info("DUPLICATE DETECTION (Sturm 2013)")
    logger.info("=" * 70)

    hash_to_files = {}

    for genre_dir in base_path.iterdir():
        if not genre_dir.is_dir():
            continue

        for wav_file in tqdm(
            genre_dir.glob("*.wav"), desc=f"Checking {genre_dir.name}"
        ):
            try:
                with open(wav_file, "rb") as f:
                    file_hash = hashlib.md5(f.read(10240)).hexdigest()

                if file_hash not in hash_to_files:
                    hash_to_files[file_hash] = []

                file_path = f"{genre_dir.name}/{wav_file.name}"
                hash_to_files[file_hash].append(file_path)

            except Exception as e:
                logger.error(f"Error reading {wav_file}: {e}")

    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}

    if duplicates:
        logger.warning(f"\n⚠ Found {len(duplicates)} duplicate groups:")
        for i, (h, files) in enumerate(duplicates.items(), 1):
            logger.warning(f"\n  Duplicate group {i}:")
            for f in files:
                logger.warning(f"    - {f}")
    else:
        logger.info("\n✓ No duplicates detected (clean GTZAN version)")

    return duplicates


def create_gtzan_summary_table(csv_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Creates summary statistics table from features_30_sec.csv.

    Computes per-genre aggregates for:
    - Sample count
    - Duration
    - Tempo
    - Spectral features (centroid, bandwidth)
    - Energy features (RMS, ZCR)

    Args:
        csv_path: Path to features_30_sec.csv file.
        output_path: Directory to save output CSV.

    Returns:
        DataFrame with summary statistics per genre.
    """
    logger.info("\n" + "=" * 70)
    logger.info("GTZAN SUMMARY TABLE")
    logger.info("=" * 70)

    csv_file = csv_path / "features_30_sec.csv"

    if not csv_file.exists():
        logger.error(f"File not found: {csv_file}")
        return None

    df = pd.read_csv(csv_file)

    summary_data = []

    for genre in sorted(df["label"].unique()):
        genre_data = df[df["label"] == genre]

        summary_data.append(
            {
                "Género": genre,
                "Muestras": len(genre_data),
                "Duración (s)": round(genre_data["length"].mean() / 22050, 2),
                "Tempo (BPM)": round(genre_data["tempo"].mean(), 2),
                "Tempo std": round(genre_data["tempo"].std(), 2),
                "Centroide espectral (Hz)": round(
                    genre_data["spectral_centroid_mean"].mean(), 2
                ),
                "Ancho de banda (Hz)": round(
                    genre_data["spectral_bandwidth_mean"].mean(), 2
                ),
                "Zero Crossing Rate": round(
                    genre_data["zero_crossing_rate_mean"].mean(), 4
                ),
                "RMS Energy": round(genre_data["rms_mean"].mean(), 4),
            }
        )

    summary_df = pd.DataFrame(summary_data)

    logger.info("\n" + summary_df.to_string(index=False))

    output_file = output_path / "gtzan_summary_table.csv"
    summary_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Summary table saved: {output_file}")

    return summary_df


def characterize_by_trait(csv_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Creates feature characterization table grouped by musical traits:
    - RITMO: Tempo (BPM)
    - TONO: Spectral Centroid (Hz)
    - ARMONÍA: Chroma STFT
    - TIMBRE: MFCC1 + RMS Energy
    - BRILLO: Zero Crossing Rate

    This table supports the analysis in thesis section 2.6
    (Ritmo, Tono, Armonía, Timbre).

    Args:
        csv_path: Path to features_30_sec.csv file.
        output_path: Directory to save output CSV.

    Returns:
        DataFrame with trait characteristics per genre.
    """
    logger.info("\n" + "=" * 70)
    logger.info("CHARACTERIZATION BY MUSICAL TRAIT")
    logger.info("=" * 70)

    csv_file = csv_path / "features_30_sec.csv"

    if not csv_file.exists():
        logger.error(f"File not found: {csv_file}")
        return None

    df = pd.read_csv(csv_file)

    trait_data = []

    for genre in sorted(df["label"].unique()):
        genre_data = df[df["label"] == genre]

        trait_data.append(
            {
                "Género": genre,
                "Tempo (BPM)": round(genre_data["tempo"].mean(), 2),
                "Tempo std": round(genre_data["tempo"].std(), 2),
                "Centroide espectral (Hz)": round(
                    genre_data["spectral_centroid_mean"].mean(), 2
                ),
                "Chroma STFT": round(genre_data["chroma_stft_mean"].mean(), 4),
                "RMS Energy": round(genre_data["rms_mean"].mean(), 4),
                "MFCC1 (timbre)": round(genre_data["mfcc1_mean"].mean(), 2),
                "Zero Crossing Rate": round(
                    genre_data["zero_crossing_rate_mean"].mean(), 4
                ),
            }
        )

    trait_df = pd.DataFrame(trait_data)

    logger.info("\n" + trait_df.to_string(index=False))

    output_file = output_path / "features_by_trait.csv"
    trait_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Traits table saved: {output_file}")

    return trait_df


def analyze_gtzan(
    config: ProjectConfig,
) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Full GTZAN analysis pipeline.

    Validates structure → detects duplicates → creates summary table →
    characterizes by trait.

    Args:
        config: Project configuration containing paths.

    Returns:
        Tuple containing:
        - genres_found (dict): Genre -> file count mapping
        - duplicates (dict): Duplicate groups detected
        - summary_df (pd.DataFrame): Summary statistics
        - traits_df (pd.DataFrame): Trait characterization
    """
    base_path = config.paths.raw_audio / "genres_original"
    output_path = config.paths.processed / "analysis"

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("\n")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " GTZAN DATASET ANALYSIS AND VALIDATION ".center(68) + "║")
    logger.info("╚" + "═" * 68 + "╝")

    if not base_path.exists():
        logger.error(f"ERROR: {base_path} does not exist")
        return None, None, None, None

    genres_found = validate_gtzan_structure(base_path)

    duplicates = find_duplicate_files(base_path)

    summary_df = create_gtzan_summary_table(config.paths.raw_audio, output_path)

    traits_df = characterize_by_trait(config.paths.raw_audio, output_path)

    logger.info("\n" + "=" * 70)
    logger.info("✓ ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nOutput files saved to: {output_path}\n")

    return genres_found, duplicates, summary_df, traits_df


if __name__ == "__main__":

    from src.config import ProjectConfig

    logging.basicConfig(level=logging.INFO)

    config = ProjectConfig()
    analyze_gtzan(config)
