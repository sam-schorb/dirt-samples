from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf


def compute_peak_rms_dbfs(audio_path: Path) -> Dict[str, float]:
    """Compute peak, RMS, and crest dBFS for a file."""
    eps = 1e-12
    try:
        data, _ = sf.read(audio_path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Could not read audio: {audio_path}") from exc

    if data.size == 0:
        raise RuntimeError(f"Empty audio data: {audio_path}")

    mono = np.mean(data, axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(math.sqrt(float(np.mean(np.square(mono)))))

    peak_dbfs = 20.0 * math.log10(peak + eps)
    rms_dbfs = 20.0 * math.log10(rms + eps)
    crest_db = peak_dbfs - rms_dbfs

    return {
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "crest_db": crest_db,
    }
