# ==============================================================================
# Author: Luis Eduardo Polaco
# Description: scratch
# ==============================================================================

from pathlib import Path

raw = Path("data/raw/genres_original")

print(list(raw.iterdir()))

print(list((raw / "blues").glob("*.wav")))
