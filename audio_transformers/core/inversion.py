import numpy as np
from numpy.typing import NDArray

from audio_transformers.core.transform import Transform


class Inversion(Transform):
    """Inverse waveform polarity by multiplying it by -1."""

    def __call__(self, signal: NDArray[np.float32], rate: int) -> NDArray[np.float32]:
        return -signal
